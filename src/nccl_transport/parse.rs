//! Pure parser for NCCL's INFO log (NCCL 2.18 through 2.30 formats).
//!
//! Lines look like `host:pid:tid [dev] NCCL INFO <message>`; only the
//! message after `NCCL INFO ` matters, so prefixes (and any timestamp a
//! newer NCCL puts in front) are ignored. Recognized messages:
//!
//! - `NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/RoCE [RO]; OOB ...` — the
//!   HCA ports of NCCL's IB transport (or an IB-based plugin).
//! - `NET/Socket : Using [0]bond0:10.1.1.10<0> ...` — the socket
//!   transport's interfaces.
//! - `NET/Plugin: Loaded net plugin <name> (vN)` — an external net plugin.
//! - `Using network <name>` — the network NCCL selected (authoritative).
//! - `Channel NN/N : a[x] -> b[y] [send] via <transport>/...` — one peer
//!   connection, counted per transport (`P2P`, `SHM`, `NET`, `COLLNET`;
//!   `.../GDRDMA` marks GPUDirect RDMA).
//! - `NCCL version <version>` — the version banner.
//!
//! Everything else (WARN lines included) is ignored. Parsing never fails:
//! unknown shapes are skipped, and `parse_log` returns `None` only when the
//! text has no transport line at all.

use super::{ChannelTransports, IbLinkLayer, IbPort, NcclTransportInfo, NetTransport, SocketIface};

const INFO_MARKER: &str = "NCCL INFO ";
const VERSION_MARKER: &str = "NCCL version ";
const PLUGIN_MARKER: &str = "Loaded net plugin ";
const NETWORK_MARKER: &str = "Using network ";

/// Everything one log says about transports, or `None` when it names no
/// network and no peer connection.
pub fn parse_log(text: &str) -> Option<NcclTransportInfo> {
    let mut state = ParseState::default();
    for line in text.lines() {
        state.line(line);
    }
    state.finish()
}

#[derive(Default)]
struct ParseState {
    nccl_version: Option<String>,
    ib_ports: Vec<IbPort>,
    socket_ifaces: Vec<SocketIface>,
    plugin: Option<String>,
    network: Option<String>,
    channels: ChannelTransports,
    ib_no_device: bool,
}

impl ParseState {
    fn line(&mut self, line: &str) {
        if self.nccl_version.is_none()
            && let Some(at) = line.find(VERSION_MARKER)
        {
            self.nccl_version = line[at + VERSION_MARKER.len()..]
                .split_whitespace()
                .next()
                .map(str::to_string);
        }
        let Some(at) = line.find(INFO_MARKER) else {
            return;
        };
        let message = line[at + INFO_MARKER.len()..].trim();
        if message.starts_with("NET/IB") && message.contains("No device found") {
            self.ib_no_device = true;
        } else if let Some(devices) = using_list(message, "NET/IB") {
            for port in devices.filter_map(parse_ib_port) {
                if !self.ib_ports.contains(&port) {
                    self.ib_ports.push(port);
                }
            }
        } else if let Some(devices) = using_list(message, "NET/Socket") {
            for iface in devices.filter_map(parse_socket_iface) {
                if !self.socket_ifaces.contains(&iface) {
                    self.socket_ifaces.push(iface);
                }
            }
        } else if let Some(at) = message.find(PLUGIN_MARKER) {
            let name = strip_plugin_version(&message[at + PLUGIN_MARKER.len()..]);
            if !name.is_empty() {
                self.plugin = Some(name.to_string());
            }
        } else if let Some(name) = message.strip_prefix(NETWORK_MARKER) {
            let name = name.trim();
            if !name.is_empty() {
                self.network = Some(name.to_string());
            }
        } else if let Some(via) = connection_transport(message) {
            self.count(via);
        }
    }

    fn count(&mut self, via: &str) {
        let kind = via.split('/').next().unwrap_or_default();
        match kind {
            "P2P" => self.channels.p2p += 1,
            "SHM" => self.channels.shm += 1,
            "NET" => {
                self.channels.net += 1;
                if via.split('/').any(|part| part == "GDRDMA") {
                    self.channels.net_gdr += 1;
                }
            }
            "COLLNET" => self.channels.collnet += 1,
            _ => {}
        }
    }

    fn finish(self) -> Option<NcclTransportInfo> {
        let net = match self.network.as_deref() {
            Some("IB") => Some(NetTransport::Ib {
                ports: self.ib_ports,
            }),
            Some("Socket") => Some(NetTransport::Socket {
                ifaces: self.socket_ifaces,
            }),
            Some(name) => Some(NetTransport::Plugin {
                name: name.to_string(),
                ports: self.ib_ports,
            }),
            // No `Using network` line: infer from the device lines, in
            // NCCL's own preference order (plugin, IB, then sockets).
            None => match &self.plugin {
                Some(name) => Some(NetTransport::Plugin {
                    name: name.clone(),
                    ports: self.ib_ports,
                }),
                None if !self.ib_ports.is_empty() => Some(NetTransport::Ib {
                    ports: self.ib_ports,
                }),
                None if !self.socket_ifaces.is_empty() => Some(NetTransport::Socket {
                    ifaces: self.socket_ifaces,
                }),
                None => None,
            },
        };
        if net.is_none() && self.channels.total() == 0 {
            return None;
        }
        Some(NcclTransportInfo {
            nccl_version: self.nccl_version,
            net,
            net_plugin: self.plugin,
            channels: self.channels,
            ib_no_device: self.ib_no_device,
        })
    }
}

/// The device tokens of `<prefix> : Using ...` (any spacing around the
/// colon), up to the `; OOB ...` sideband part.
fn using_list<'a>(message: &'a str, prefix: &str) -> Option<impl Iterator<Item = &'a str>> {
    let rest = message.strip_prefix(prefix)?.trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix("Using")?;
    let devices = rest.split(';').next().unwrap_or_default();
    Some(devices.split_whitespace())
}

/// `[0]mlx5_0:1/IB` -> mlx5_0 port 1, InfiniBand. Tokens without the
/// `[index]` prefix (`[RO]`, words) are skipped.
fn parse_ib_port(token: &str) -> Option<IbPort> {
    let spec = strip_index(token)?;
    let (device_port, link) = spec.split_once('/')?;
    let (device, port) = device_port.rsplit_once(':')?;
    let link_layer = match link {
        "IB" => IbLinkLayer::Infiniband,
        "RoCE" => IbLinkLayer::Roce,
        _ => return None,
    };
    Some(IbPort {
        device: device.to_string(),
        port: port.parse().ok()?,
        link_layer,
    })
}

/// `[0]bond0:10.1.1.10<0>` -> bond0 at 10.1.1.10 (IPv6 addresses keep
/// their colons: the name ends at the first one).
fn parse_socket_iface(token: &str) -> Option<SocketIface> {
    let spec = strip_index(token)?;
    let (name, addr) = spec.split_once(':')?;
    let addr = addr.split('<').next().unwrap_or(addr);
    if name.is_empty() || addr.is_empty() {
        return None;
    }
    Some(SocketIface {
        name: name.to_string(),
        addr: addr.to_string(),
    })
}

/// `[12]rest` -> `rest`; `None` unless the bracket holds an index.
fn strip_index(token: &str) -> Option<&str> {
    let rest = token.strip_prefix('[')?;
    let (index, spec) = rest.split_once(']')?;
    (!index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()) && !spec.is_empty())
        .then_some(spec)
}

/// `Libfabric (v7)` -> `Libfabric`; also drops a trailing period.
fn strip_plugin_version(text: &str) -> &str {
    let text = text.trim().trim_end_matches('.');
    match text.rfind(" (v") {
        Some(at) if text.ends_with(')') => text[..at].trim(),
        _ => text,
    }
}

/// The `<transport>/...` word after ` via ` of a peer-connection line
/// (`Channel ...` or `CollNet ...`).
fn connection_transport(message: &str) -> Option<&str> {
    if !(message.starts_with("Channel ") || message.starts_with("CollNet ")) {
        return None;
    }
    let (_, via) = message.split_once(" via ")?;
    via.split_whitespace().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    const IB_2_18: &str = include_str!("fixtures/ib_2_18.log");
    const ROCE_2_21: &str = include_str!("fixtures/roce_2_21.log");
    const SOCKET_2_23: &str = include_str!("fixtures/socket_fallback_2_23.log");
    const LIBFABRIC_2_19: &str = include_str!("fixtures/plugin_libfabric_2_19.log");
    const IBEXT_2_27: &str = include_str!("fixtures/plugin_ibext_2_27.log");
    const INTRANODE_2_30: &str = include_str!("fixtures/intranode_2_30.log");

    fn ib(device: &str) -> IbPort {
        IbPort {
            device: device.into(),
            port: 1,
            link_layer: IbLinkLayer::Infiniband,
        }
    }

    fn roce(device: &str) -> IbPort {
        IbPort {
            device: device.into(),
            port: 1,
            link_layer: IbLinkLayer::Roce,
        }
    }

    #[test]
    fn infiniband_2_18() {
        let info = parse_log(IB_2_18).expect("transport lines");
        assert_eq!(info.nccl_version.as_deref(), Some("2.18.5+cuda12.2"));
        assert!(!info.ib_no_device);
        assert_eq!(
            info.net,
            Some(NetTransport::Ib {
                ports: vec![ib("mlx5_0"), ib("mlx5_1")]
            })
        );
        assert_eq!(info.net_plugin, None, "'No plugin found' is not a plugin");
        assert_eq!(
            info.channels,
            ChannelTransports {
                p2p: 4,
                shm: 0,
                net: 4,
                net_gdr: 4,
                collnet: 0
            }
        );
    }

    #[test]
    fn roce_2_21() {
        let info = parse_log(ROCE_2_21).expect("transport lines");
        assert_eq!(info.nccl_version.as_deref(), Some("2.21.5+cuda12.4"));
        assert_eq!(
            info.net,
            Some(NetTransport::Ib {
                ports: vec![roce("mlx5_0"), roce("mlx5_1")]
            }),
            "the [RO] flag and the OOB part are not ports"
        );
        assert_eq!(
            info.net.as_ref().map(NetTransport::summary).as_deref(),
            Some("RoCE mlx5_0:1,mlx5_1:1")
        );
        // One NET connection without GPUDirect RDMA.
        assert_eq!(info.channels.p2p, 3);
        assert_eq!(info.channels.net, 4);
        assert_eq!(info.channels.net_gdr, 3);
    }

    #[test]
    fn socket_fallback_2_23() {
        let info = parse_log(SOCKET_2_23).expect("transport lines");
        assert_eq!(info.nccl_version.as_deref(), Some("2.23.4+cuda12.6"));
        assert_eq!(
            info.net,
            Some(NetTransport::Socket {
                ifaces: vec![SocketIface {
                    name: "bond0".into(),
                    addr: "10.20.4.12".into()
                }]
            })
        );
        assert!(info.net.as_ref().is_some_and(NetTransport::is_socket));
        assert_eq!(info.channels.net, 4);
        assert_eq!(info.channels.net_gdr, 0);
        assert_eq!(info.channels.p2p, 2);
        assert!(info.ib_no_device, "NET/IB : No device found");
    }

    #[test]
    fn libfabric_plugin_2_19() {
        let info = parse_log(LIBFABRIC_2_19).expect("transport lines");
        assert_eq!(info.net_plugin.as_deref(), Some("Libfabric"));
        assert_eq!(
            info.net,
            Some(NetTransport::Plugin {
                name: "Libfabric".into(),
                ports: vec![]
            })
        );
        assert_eq!(info.channels.net, 2);
        assert_eq!(info.channels.net_gdr, 2);
        assert_eq!(info.channels.p2p, 1);
    }

    #[test]
    fn ibext_plugin_with_ib_ports_2_27() {
        let info = parse_log(IBEXT_2_27).expect("transport lines");
        assert_eq!(info.nccl_version.as_deref(), Some("2.27.3+cuda12.8"));
        assert_eq!(info.net_plugin.as_deref(), Some("IBext_v8"));
        assert_eq!(
            info.net,
            Some(NetTransport::Plugin {
                name: "IBext_v8".into(),
                ports: vec![ib("mlx5_0"), ib("mlx5_3"), ib("mlx5_4"), ib("mlx5_5")]
            })
        );
    }

    #[test]
    fn intranode_2_30_counts_p2p_and_shm() {
        let info = parse_log(INTRANODE_2_30).expect("transport lines");
        assert_eq!(info.nccl_version.as_deref(), Some("2.30.7+cuda13.0"));
        // The network is initialized even for a single-node communicator.
        assert_eq!(
            info.net,
            Some(NetTransport::Ib {
                ports: vec![ib("mlx5_0"), ib("mlx5_1")]
            })
        );
        assert_eq!(
            info.channels,
            ChannelTransports {
                p2p: 2,
                shm: 2,
                net: 0,
                net_gdr: 0,
                collnet: 0
            }
        );
    }

    #[test]
    fn without_using_network_the_device_lines_decide() {
        let socket_only =
            "h:1:1 [0] NCCL INFO NET/Socket : Using [0]eth0:fe80::1%eth0<0> [1]eth1:10.0.0.2<0>\n";
        assert_eq!(
            parse_log(socket_only).and_then(|info| info.net),
            Some(NetTransport::Socket {
                ifaces: vec![
                    SocketIface {
                        name: "eth0".into(),
                        addr: "fe80::1%eth0".into()
                    },
                    SocketIface {
                        name: "eth1".into(),
                        addr: "10.0.0.2".into()
                    },
                ]
            })
        );
        let both = "h:1:1 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/IB ; OOB ib0:10.0.0.1<0>\n\
                    h:1:1 [0] NCCL INFO NET/Socket : Using [0]eth0:10.0.0.2<0>\n";
        assert_eq!(
            parse_log(both).and_then(|info| info.net),
            Some(NetTransport::Ib {
                ports: vec![ib("mlx5_0")]
            })
        );
    }

    #[test]
    fn using_network_wins_over_device_lines() {
        // IB devices initialized, but NCCL_NET=Socket made NCCL pick sockets.
        let text = "h:1:1 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/IB ; OOB ib0:10.0.0.1<0>\n\
                    h:1:1 [0] NCCL INFO NET/Socket : Using [0]ib0:10.0.0.1<0>\n\
                    h:1:1 [0] NCCL INFO Using network Socket\n";
        assert!(
            parse_log(text)
                .and_then(|info| info.net)
                .is_some_and(|net| net.is_socket())
        );
    }

    #[test]
    fn logs_without_transport_lines_parse_to_none() {
        assert_eq!(parse_log(""), None);
        let noise = "NCCL version 2.18.5+cuda12.2\n\
                     h:1:1 [0] NCCL INFO Bootstrap : Using bond0:10.1.1.10<0>\n\
                     h:1:1 [0] NCCL INFO NET/IB : No device found.\n\
                     h:1:1 [0] NCCL WARN NET/Socket : no interface found\n\
                     h:1:1 [0] NCCL INFO Channel 00/02 :    0   1\n\
                     h:1:1 [0] NCCL INFO Connected all rings\n";
        assert_eq!(parse_log(noise), None);
    }

    #[test]
    fn warn_lines_and_malformed_tokens_are_ignored() {
        let text = "h:1:1 [0] NCCL WARN Using network Socket\n\
                    h:1:1 [0] NCCL INFO NET/IB : Using [x]mlx5_0:1/IB [0]mlx5_1:one/IB [1]mlx5_2:1/XX [2]mlx5_3:1/IB\n";
        let info = parse_log(text).expect("one valid port");
        assert_eq!(
            info.net,
            Some(NetTransport::Ib {
                ports: vec![ib("mlx5_3")]
            })
        );
    }

    #[test]
    fn collnet_and_unknown_connection_kinds() {
        let text = "h:1:1 [0] NCCL INFO CollNet 00/0 : 0 [send] via COLLNET/IBext/0/GDRDMA\n\
                    h:1:1 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via NVLS/whatever\n";
        let info = parse_log(text).expect("a collnet connection");
        assert_eq!(info.channels.collnet, 1);
        assert_eq!(info.channels.total(), 1);
        assert_eq!(info.net, None);
    }

    #[test]
    fn plugin_names_lose_their_version_suffix() {
        assert_eq!(strip_plugin_version("Libfabric (v7)"), "Libfabric");
        assert_eq!(strip_plugin_version("AWS Libfabric (v6)"), "AWS Libfabric");
        assert_eq!(strip_plugin_version("IBext_v8 (v8)."), "IBext_v8");
        assert_eq!(strip_plugin_version("custom"), "custom");
    }
}
