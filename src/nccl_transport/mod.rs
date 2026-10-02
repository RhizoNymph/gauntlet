//! Which transport NCCL actually used, per communicator.
//!
//! NCCL has no API that reports its network or peer transports, so the
//! record comes from NCCL's own INFO log: the orchestrator starts every
//! NCCL-hosting agent process with `NCCL_DEBUG=INFO`,
//! `NCCL_DEBUG_SUBSYS=INIT,NET` and a per-host `NCCL_DEBUG_FILE` (merged
//! with, never overriding, what the config sets: `debug`), and the agent
//! parses that file once its workload has run (`parse`) and reports a typed
//! `NcclTransportReport`.
//!
//! The point is diagnosis: a silent fallback from InfiniBand to TCP sockets
//! (a wrong NCCL_IB_HCA, a missing GID index, a down port) is about 10x
//! slower and otherwise just looks like low bandwidth. The report flags a
//! multi-host communicator whose network transport is `Socket` unless the
//! level's env disabled IB on purpose (`socket_is_deliberate`).

pub mod debug;
pub mod parse;

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::nccl_level::NcclLevel;

/// Link layer of an IB-verbs port as NCCL names it (`/IB` or `/RoCE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IbLinkLayer {
    Infiniband,
    Roce,
}

/// One HCA port NCCL's IB transport (or an IB-based plugin) uses, from
/// `NET/IB : Using [0]mlx5_0:1/IB ...`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IbPort {
    pub device: String,
    pub port: u32,
    pub link_layer: IbLinkLayer,
}

/// One interface NCCL's socket transport uses, from
/// `NET/Socket : Using [0]bond0:10.1.1.10<0>`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SocketIface {
    pub name: String,
    pub addr: String,
}

/// The network transport NCCL selected for inter-node traffic
/// (`Using network <name>`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NetTransport {
    /// NCCL's internal IB-verbs transport, over InfiniBand or RoCE ports.
    Ib { ports: Vec<IbPort> },
    /// NCCL's internal TCP socket transport.
    Socket { ifaces: Vec<SocketIface> },
    /// An external net plugin, by the network name NCCL selected (e.g.
    /// `Libfabric`, `IBext_v8`); `ports` lists any IB ports the plugin
    /// reported in NCCL's `NET/IB` format, empty otherwise.
    Plugin { name: String, ports: Vec<IbPort> },
}

impl NetTransport {
    /// Whether this is NCCL's TCP socket fallback.
    pub fn is_socket(&self) -> bool {
        matches!(self, NetTransport::Socket { .. })
    }

    /// One-line human form: "IB mlx5_0:1,mlx5_1:1", "RoCE mlx5_0:1",
    /// "Socket bond0(10.1.1.10)", "plugin Libfabric".
    pub fn summary(&self) -> String {
        match self {
            NetTransport::Ib { ports } => format!("{} {}", port_family(ports), port_list(ports)),
            NetTransport::Socket { ifaces } => {
                let list = ifaces
                    .iter()
                    .map(|iface| format!("{}({})", iface.name, iface.addr))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("Socket {list}").trim_end().to_string()
            }
            NetTransport::Plugin { name, ports } if ports.is_empty() => format!("plugin {name}"),
            NetTransport::Plugin { name, ports } => {
                format!("plugin {name} {} {}", port_family(ports), port_list(ports))
            }
        }
    }
}

fn port_family(ports: &[IbPort]) -> &'static str {
    if !ports.is_empty()
        && ports
            .iter()
            .all(|port| port.link_layer == IbLinkLayer::Roce)
    {
        "RoCE"
    } else {
        "IB"
    }
}

fn port_list(ports: &[IbPort]) -> String {
    ports
        .iter()
        .map(|port| format!("{}:{}", port.device, port.port))
        .collect::<Vec<_>>()
        .join(",")
}

/// How many peer connections (`Channel NN/N : a -> b via <transport>`
/// lines) each transport carried in this process. Informational: the
/// shape of the intra-node path (P2P vs SHM vs NET) and whether NET
/// connections used GPUDirect RDMA.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelTransports {
    /// CUDA peer-to-peer (NVLink or PCIe): `via P2P/...`.
    pub p2p: u32,
    /// Host shared memory: `via SHM/...`.
    pub shm: u32,
    /// Network (IB, socket or plugin): `via NET/...`.
    pub net: u32,
    /// The subset of `net` using GPUDirect RDMA (`.../GDRDMA`).
    pub net_gdr: u32,
    /// In-network collectives (SHARP etc.): `via COLLNET/...`.
    pub collnet: u32,
}

impl ChannelTransports {
    pub fn total(&self) -> u32 {
        self.p2p + self.shm + self.net + self.collnet
    }

    /// "P2P 4 SHM 2 NET 8 (GDR 8)", omitting zero counts; "-" when empty.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.p2p > 0 {
            parts.push(format!("P2P {}", self.p2p));
        }
        if self.shm > 0 {
            parts.push(format!("SHM {}", self.shm));
        }
        if self.net > 0 {
            parts.push(format!("NET {} (GDR {})", self.net, self.net_gdr));
        }
        if self.collnet > 0 {
            parts.push(format!("COLLNET {}", self.collnet));
        }
        if parts.is_empty() {
            "-".to_string()
        } else {
            parts.join(" ")
        }
    }
}

/// What one NCCL-hosting process's INFO log says about its transports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcclTransportInfo {
    /// "2.18.5+cuda12.2", when the log carries the version banner.
    pub nccl_version: Option<String>,
    /// `None` when the log never named a network (no `Using network` and
    /// no `NET/IB` / `NET/Socket` device line).
    pub net: Option<NetTransport>,
    /// The external net plugin NCCL loaded (`Loaded net plugin <name>`),
    /// `None` for NCCL's internal transports.
    pub net_plugin: Option<String>,
    pub channels: ChannelTransports,
    /// NCCL's IB transport reported `NET/IB : No device found`: the host
    /// had no usable IB/RoCE device (or NCCL_IB_HCA matched none). Used as
    /// a "no IB here" signal when the host has no inventory.
    #[serde(default)]
    pub ib_no_device: bool,
}

/// Why no transport could be recorded for a communicator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum UnknownTransport {
    /// NCCL_DEBUG is unset or below INFO in the process env (the config set
    /// it, so gauntlet left it alone): NCCL logs nothing parsable.
    DebugLevel { level: Option<String> },
    /// NCCL_DEBUG_SUBSYS (set by the config) masks out INIT, which carries
    /// every line the parser reads.
    SubsysExcludesInit { subsys: String },
    /// NCCL_DEBUG_FILE is unset: the log went to the agent's stderr.
    NoDebugFile,
    /// The expanded NCCL_DEBUG_FILE could not be read.
    Unreadable { path: String, error: String },
    /// A previous log at the path could not be removed before NCCL init, so
    /// what is there afterwards may not be this process's.
    Uncleared { path: String, error: String },
    /// The file holds no transport line at all.
    NoTransportLines { path: String },
}

impl fmt::Display for UnknownTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnknownTransport::DebugLevel { level: None } => {
                f.write_str("NCCL_DEBUG is unset, so NCCL wrote no INFO log")
            }
            UnknownTransport::DebugLevel { level: Some(level) } => write!(
                f,
                "NCCL_DEBUG={level} (from the config) is below INFO, so NCCL wrote no INFO log"
            ),
            UnknownTransport::SubsysExcludesInit { subsys } => write!(
                f,
                "NCCL_DEBUG_SUBSYS={subsys} (from the config) masks out INIT"
            ),
            UnknownTransport::NoDebugFile => {
                f.write_str("NCCL_DEBUG_FILE is unset, so the log went to stderr")
            }
            UnknownTransport::Unreadable { path, error } => {
                write!(f, "cannot read NCCL debug file {path}: {error}")
            }
            UnknownTransport::Uncleared { path, error } => write!(
                f,
                "cannot clear the previous NCCL debug file {path} before init: {error}"
            ),
            UnknownTransport::NoTransportLines { path } => {
                write!(f, "NCCL debug file {path} names no transport")
            }
        }
    }
}

/// The transport record of one communicator, or why there is none.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum TransportCapture {
    Captured { info: NcclTransportInfo },
    Unknown { reason: UnknownTransport },
}

/// Whether a communicator's ranks live on one host or several. Only a
/// multi-host communicator's network transport carries traffic, so only
/// it can be flagged for a socket fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommSpan {
    SingleHost,
    MultiHost,
}

impl CommSpan {
    /// From a rank-per-GPU assignment: one host owns one contiguous block,
    /// so the world spans hosts exactly when it is larger than this host's
    /// block.
    pub fn from_world(local_ranks: u32, world_size: u32) -> Self {
        if world_size > local_ranks {
            CommSpan::MultiHost
        } else {
            CommSpan::SingleHost
        }
    }
}

/// One agent process's transport record for the communicator it hosted at
/// `level`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcclTransportReport {
    pub level: NcclLevel,
    pub span: CommSpan,
    pub capture: TransportCapture,
}

/// Whether a level's env asks NCCL for sockets on purpose:
/// `NCCL_IB_DISABLE` set to a non-zero integer (NCCL's own parse), or
/// `NCCL_NET=Socket` (case-insensitive, as NCCL compares it). A socket
/// transport under such an env is the requested configuration, not a
/// fallback.
pub fn socket_is_deliberate(env: &BTreeMap<String, String>) -> bool {
    let ib_disabled = env
        .get("NCCL_IB_DISABLE")
        .and_then(|value| value.trim().parse::<i64>().ok())
        .is_some_and(|value| value != 0);
    let socket_net = env
        .get("NCCL_NET")
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("socket"));
    ib_disabled || socket_net
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    #[test]
    fn sockets_are_deliberate_only_when_the_env_says_so() {
        assert!(!socket_is_deliberate(&env(&[])));
        assert!(!socket_is_deliberate(&env(&[("NCCL_IB_DISABLE", "0")])));
        assert!(!socket_is_deliberate(&env(&[("NCCL_IB_HCA", "mlx5_9")])));
        // NCCL parses the knob as an integer; garbage reads as 0.
        assert!(!socket_is_deliberate(&env(&[("NCCL_IB_DISABLE", "yes")])));
        assert!(socket_is_deliberate(&env(&[("NCCL_IB_DISABLE", "1")])));
        assert!(socket_is_deliberate(&env(&[("NCCL_IB_DISABLE", " 1 ")])));
        assert!(socket_is_deliberate(&env(&[("NCCL_NET", "Socket")])));
        assert!(socket_is_deliberate(&env(&[("NCCL_NET", "socket")])));
        assert!(!socket_is_deliberate(&env(&[("NCCL_NET", "IB")])));
    }

    #[test]
    fn comm_span_follows_the_rank_block() {
        assert_eq!(CommSpan::from_world(8, 8), CommSpan::SingleHost);
        assert_eq!(CommSpan::from_world(8, 16), CommSpan::MultiHost);
        assert_eq!(CommSpan::from_world(1, 2), CommSpan::MultiHost);
    }

    fn port(device: &str, link_layer: IbLinkLayer) -> IbPort {
        IbPort {
            device: device.into(),
            port: 1,
            link_layer,
        }
    }

    #[test]
    fn summaries_name_the_transport_and_its_devices() {
        let ib = NetTransport::Ib {
            ports: vec![
                port("mlx5_0", IbLinkLayer::Infiniband),
                port("mlx5_1", IbLinkLayer::Infiniband),
            ],
        };
        assert_eq!(ib.summary(), "IB mlx5_0:1,mlx5_1:1");
        let roce = NetTransport::Ib {
            ports: vec![port("mlx5_0", IbLinkLayer::Roce)],
        };
        assert_eq!(roce.summary(), "RoCE mlx5_0:1");
        let socket = NetTransport::Socket {
            ifaces: vec![SocketIface {
                name: "bond0".into(),
                addr: "10.1.1.10".into(),
            }],
        };
        assert_eq!(socket.summary(), "Socket bond0(10.1.1.10)");
        assert!(socket.is_socket() && !ib.is_socket());
        let plugin = NetTransport::Plugin {
            name: "Libfabric".into(),
            ports: vec![],
        };
        assert_eq!(plugin.summary(), "plugin Libfabric");
        let ibext = NetTransport::Plugin {
            name: "IBext_v8".into(),
            ports: vec![port("mlx5_0", IbLinkLayer::Infiniband)],
        };
        assert_eq!(ibext.summary(), "plugin IBext_v8 IB mlx5_0:1");
    }

    #[test]
    fn channel_summary_omits_zero_counts() {
        assert_eq!(ChannelTransports::default().summary(), "-");
        let channels = ChannelTransports {
            p2p: 4,
            shm: 0,
            net: 8,
            net_gdr: 6,
            collnet: 0,
        };
        assert_eq!(channels.summary(), "P2P 4 NET 8 (GDR 6)");
        assert_eq!(channels.total(), 12);
    }

    #[test]
    fn reports_survive_the_wire() {
        let report = NcclTransportReport {
            level: NcclLevel::Fleet,
            span: CommSpan::MultiHost,
            capture: TransportCapture::Captured {
                info: NcclTransportInfo {
                    nccl_version: Some("2.21.5+cuda12.4".into()),
                    net: Some(NetTransport::Socket {
                        ifaces: vec![SocketIface {
                            name: "bond0".into(),
                            addr: "10.1.1.10".into(),
                        }],
                    }),
                    net_plugin: None,
                    channels: ChannelTransports::default(),
                    ib_no_device: false,
                },
            },
        };
        let json = serde_json::to_string(&report).expect("serialize");
        let back: NcclTransportReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, report);

        let unknown = NcclTransportReport {
            level: NcclLevel::Intranode,
            span: CommSpan::SingleHost,
            capture: TransportCapture::Unknown {
                reason: UnknownTransport::DebugLevel {
                    level: Some("WARN".into()),
                },
            },
        };
        let json = serde_json::to_string(&unknown).expect("serialize");
        assert_eq!(
            serde_json::from_str::<NcclTransportReport>(&json).ok(),
            Some(unknown)
        );
    }
}
