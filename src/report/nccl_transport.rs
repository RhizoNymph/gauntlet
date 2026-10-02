//! Transport records in the results document: the socket-fallback finding
//! and the terminal rendering.
//!
//! Each NCCL-hosting agent process reports which transports its
//! communicator used (`HostObservations.nccl_transports`). A multi-host
//! communicator whose network transport is NCCL's TCP socket fallback is a
//! finding — about 10x slower than IB and otherwise indistinguishable from
//! "low bandwidth" — unless the level's env asked for sockets on purpose
//! (`nccl_transport::socket_is_deliberate`).

use std::collections::BTreeMap;
use std::io::Write;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::nccl_env::LevelEnvMap;
use super::{RunResults, new_table, section};
use crate::nccl_level::NcclLevel;
use crate::nccl_transport::{
    CommSpan, NcclTransportReport, NetTransport, SocketIface, TransportCapture,
    socket_is_deliberate,
};
use crate::orchestrator::collect::HostObservations;

/// A multi-host communicator at `level` that fell back to TCP sockets on a
/// host whose level env did not disable IB.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SocketFallback {
    pub level: NcclLevel,
    /// The interfaces the socket transport used.
    pub ifaces: Vec<SocketIface>,
}

impl SocketFallback {
    /// One-line reason for the findings section.
    pub fn describe(&self) -> String {
        let ifaces = if self.ifaces.is_empty() {
            "unknown interfaces".to_string()
        } else {
            self.ifaces
                .iter()
                .map(|iface| format!("{} ({})", iface.name, iface.addr))
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "{} communicator spans hosts but NCCL fell back to TCP sockets over {ifaces}; \
             IB was not disabled (check NCCL_IB_HCA, NCCL_IB_GID_INDEX and the IB port state)",
            self.level
        )
    }
}

/// What a host's phase-0 inventory says about its IB-verbs ports (IB or
/// RoCE, from `/sys/class/infiniband`). A socket transport is only a
/// *fallback* on a host that had a usable port to fall back from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IbAvailability {
    /// At least one port reported state ACTIVE.
    ActivePort,
    /// Ports absent, or none ACTIVE: sockets are the only transport.
    NoActivePort,
    /// No inventory, but NCCL itself logged `NET/IB : No device found` while
    /// the level's env did not restrict NCCL_IB_HCA: no IB device exists.
    NoDeviceSeenByNccl,
    /// No inventory and no NCCL signal: nothing is known, nothing flagged.
    Unknown,
}

impl IbAvailability {
    /// From the host's inventory; without one, from NCCL's own
    /// `No device found` (only when NCCL_IB_HCA was unset for that level —
    /// with a filter set, "no device" may be exactly the misconfiguration).
    pub fn of(obs: &HostObservations, level_env: Option<&LevelEnvMap>) -> Self {
        let hca_unset = |level: NcclLevel| {
            level_env
                .and_then(|envs| envs.get(&level))
                .is_none_or(|env| !env.contains_key("NCCL_IB_HCA"))
        };
        match &obs.inventory {
            None if obs.nccl_transports.iter().any(|report| {
                hca_unset(report.level)
                    && matches!(
                        &report.capture,
                        TransportCapture::Captured { info } if info.ib_no_device
                    )
            }) =>
            {
                IbAvailability::NoDeviceSeenByNccl
            }
            None => IbAvailability::Unknown,
            Some(inventory)
                if inventory
                    .ib_ports
                    .iter()
                    .any(|port| port.state.trim().eq_ignore_ascii_case("ACTIVE")) =>
            {
                IbAvailability::ActivePort
            }
            Some(_) => IbAvailability::NoActivePort,
        }
    }
}

/// Socket fallbacks per host: every multi-host communicator whose captured
/// network transport is `Socket` on a host whose inventory shows at least
/// one ACTIVE IB/RoCE port, at most one finding per (host, level) however
/// many repeats saw it. Never flagged: a level whose recorded env disables
/// IB on purpose (`NCCL_IB_DISABLE=1`, `NCCL_NET=Socket`), a host without
/// an active IB/RoCE port (an Ethernet-only fleet has nothing to fall back
/// from) or without an inventory, unknown captures, and single-host
/// communicators. The transport record itself is kept either way.
pub fn socket_fallbacks(
    observations: &BTreeMap<String, HostObservations>,
    level_env: Option<&LevelEnvMap>,
) -> BTreeMap<String, Vec<SocketFallback>> {
    let deliberate = |level: NcclLevel| {
        level_env
            .and_then(|envs| envs.get(&level))
            .is_some_and(socket_is_deliberate)
    };
    let mut findings = BTreeMap::new();
    for (host, obs) in observations {
        if IbAvailability::of(obs, level_env) != IbAvailability::ActivePort {
            continue;
        }
        let mut flagged: BTreeMap<NcclLevel, SocketFallback> = BTreeMap::new();
        for report in &obs.nccl_transports {
            if report.span != CommSpan::MultiHost || deliberate(report.level) {
                continue;
            }
            let TransportCapture::Captured { info } = &report.capture else {
                continue;
            };
            if let Some(NetTransport::Socket { ifaces }) = &info.net {
                flagged
                    .entry(report.level)
                    .or_insert_with(|| SocketFallback {
                        level: report.level,
                        ifaces: ifaces.clone(),
                    });
            }
        }
        if !flagged.is_empty() {
            findings.insert(host.clone(), flagged.into_values().collect());
        }
    }
    findings
}

/// The latest report per (host, level): what the table shows.
fn latest_reports(
    observations: &BTreeMap<String, HostObservations>,
) -> Vec<(&str, &NcclTransportReport)> {
    let mut rows = Vec::new();
    for (host, obs) in observations {
        let mut latest: BTreeMap<NcclLevel, &NcclTransportReport> = BTreeMap::new();
        for report in &obs.nccl_transports {
            latest.insert(report.level, report);
        }
        rows.extend(latest.into_values().map(|report| (host.as_str(), report)));
    }
    rows
}

/// The network column: the transport summary, or why it is unknown.
pub fn net_cell(capture: &TransportCapture) -> String {
    match capture {
        TransportCapture::Captured { info } => match &info.net {
            Some(net) => net.summary(),
            None => "(not named in the log)".to_string(),
        },
        TransportCapture::Unknown { reason } => format!("unknown: {reason}"),
    }
}

/// "nccl transports" (one row per host and level) and, when any, the
/// "nccl socket fallback" findings section.
pub fn render(results: &RunResults, out: &mut dyn Write) -> Result<()> {
    let rows = latest_reports(&results.hosts);
    if !rows.is_empty() {
        let mut table = new_table(&["host", "level", "span", "net", "peer channels", "nccl"]);
        for (host, report) in rows {
            let (channels, version) = match &report.capture {
                TransportCapture::Captured { info } => (
                    info.channels.summary(),
                    info.nccl_version.clone().unwrap_or_else(|| "-".into()),
                ),
                TransportCapture::Unknown { .. } => ("-".into(), "-".into()),
            };
            let span = match report.span {
                CommSpan::SingleHost => "single host",
                CommSpan::MultiHost => "multi host",
            };
            table.add_row(vec![
                host.to_string(),
                report.level.to_string(),
                span.to_string(),
                net_cell(&report.capture),
                channels,
                version,
            ]);
        }
        section(out, "nccl transports", &table)?;
    }
    if !results.fleet.socket_fallbacks.is_empty() {
        let mut table = new_table(&["host", "finding"]);
        for (host, findings) in &results.fleet.socket_fallbacks {
            for finding in findings {
                table.add_row(vec![host.clone(), finding.describe()]);
            }
        }
        section(out, "nccl socket fallback", &table)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nccl_transport::{
        ChannelTransports, IbLinkLayer, IbPort, NcclTransportInfo, UnknownTransport,
    };

    fn captured(net: NetTransport) -> TransportCapture {
        TransportCapture::Captured {
            info: NcclTransportInfo {
                nccl_version: Some("2.23.4+cuda12.6".into()),
                net: Some(net),
                net_plugin: None,
                channels: ChannelTransports::default(),
                ib_no_device: false,
            },
        }
    }

    fn socket() -> NetTransport {
        NetTransport::Socket {
            ifaces: vec![SocketIface {
                name: "bond0".into(),
                addr: "10.1.1.10".into(),
            }],
        }
    }

    fn ib() -> NetTransport {
        NetTransport::Ib {
            ports: vec![IbPort {
                device: "mlx5_0".into(),
                port: 1,
                link_layer: IbLinkLayer::Infiniband,
            }],
        }
    }

    fn inventory(ib_states: &[&str]) -> crate::proto::InventorySnapshot {
        let mut inventory: crate::proto::InventorySnapshot =
            serde_json::from_value(serde_json::json!({
                "hostname": "n", "kernel": "6.8", "cpu_model": "x", "logical_cores": 8,
                "numa_nodes": 1, "mem_total_bytes": 1, "cpu_governor": null,
                "clock_offset_ms": null, "nvidia_driver": null, "cuda_version": null,
                "gpus": [], "nics": [], "ib_ports": [], "xid_errors": []
            }))
            .expect("inventory");
        inventory.ib_ports = ib_states
            .iter()
            .enumerate()
            .map(|(index, state)| crate::proto::IbPortInventory {
                device: format!("mlx5_{index}"),
                port: 1,
                state: state.to_string(),
                rate_gbps: Some(200.0),
                link_downed_count: None,
            })
            .collect();
        inventory
    }

    /// A host with one ACTIVE IB port: a socket transport there is a
    /// fallback.
    fn host(reports: Vec<(NcclLevel, CommSpan, TransportCapture)>) -> HostObservations {
        host_with(Some(inventory(&["ACTIVE"])), reports)
    }

    fn host_with(
        inventory: Option<crate::proto::InventorySnapshot>,
        reports: Vec<(NcclLevel, CommSpan, TransportCapture)>,
    ) -> HostObservations {
        HostObservations {
            inventory,
            nccl_transports: reports
                .into_iter()
                .map(|(level, span, capture)| NcclTransportReport {
                    level,
                    span,
                    capture,
                })
                .collect(),
            ..HostObservations::default()
        }
    }

    #[test]
    fn a_multi_host_socket_transport_is_a_finding() {
        let observations = BTreeMap::from([
            (
                "n1".to_string(),
                host(vec![(
                    NcclLevel::Fleet,
                    CommSpan::MultiHost,
                    captured(socket()),
                )]),
            ),
            (
                "n2".to_string(),
                host(vec![(
                    NcclLevel::Fleet,
                    CommSpan::MultiHost,
                    captured(ib()),
                )]),
            ),
        ]);
        let findings = socket_fallbacks(&observations, None);
        assert_eq!(findings.len(), 1);
        let finding = &findings["n1"][0];
        assert_eq!(finding.level, NcclLevel::Fleet);
        assert_eq!(finding.ifaces[0].name, "bond0");
        let text = finding.describe();
        assert!(text.contains("fleet") && text.contains("bond0"), "{text}");
        assert!(text.contains("NCCL_IB_HCA"), "{text}");
    }

    #[test]
    fn single_host_and_unknown_captures_are_never_findings() {
        let observations = BTreeMap::from([(
            "n1".to_string(),
            host(vec![
                // Intra-node: the network carries nothing.
                (
                    NcclLevel::Intranode,
                    CommSpan::SingleHost,
                    captured(socket()),
                ),
                // A one-host fleet world (2 GPUs on one node).
                (NcclLevel::Fleet, CommSpan::SingleHost, captured(socket())),
                (
                    NcclLevel::OverlapFleet,
                    CommSpan::MultiHost,
                    TransportCapture::Unknown {
                        reason: UnknownTransport::NoDebugFile,
                    },
                ),
            ]),
        )]);
        assert!(socket_fallbacks(&observations, None).is_empty());
    }

    #[test]
    fn deliberately_disabled_ib_is_not_a_finding() {
        let observations = BTreeMap::from([(
            "n1".to_string(),
            host(vec![
                (NcclLevel::Fleet, CommSpan::MultiHost, captured(socket())),
                (
                    NcclLevel::OverlapFleet,
                    CommSpan::MultiHost,
                    captured(socket()),
                ),
            ]),
        )]);
        // IB disabled for the fleet sweep only: the overlap step's socket
        // transport is still a fallback.
        let level_env: LevelEnvMap = BTreeMap::from([
            (
                NcclLevel::Fleet,
                BTreeMap::from([("NCCL_IB_DISABLE".to_string(), "1".to_string())]),
            ),
            (NcclLevel::OverlapFleet, BTreeMap::new()),
        ]);
        let findings = socket_fallbacks(&observations, Some(&level_env));
        assert_eq!(
            findings["n1"]
                .iter()
                .map(|finding| finding.level)
                .collect::<Vec<_>>(),
            vec![NcclLevel::OverlapFleet]
        );
    }

    fn fleet_socket() -> Vec<(NcclLevel, CommSpan, TransportCapture)> {
        vec![(NcclLevel::Fleet, CommSpan::MultiHost, captured(socket()))]
    }

    #[test]
    fn only_a_host_with_an_active_ib_port_can_fall_back() {
        let fleet_ib = vec![(NcclLevel::Fleet, CommSpan::MultiHost, captured(ib()))];
        let observations = BTreeMap::from([
            // IB present + socket: the fallback this finding exists for.
            ("ib-socket".to_string(), host(fleet_socket())),
            // No IB at all (Ethernet bond) + socket: the only transport.
            (
                "eth-only".to_string(),
                host_with(Some(inventory(&[])), fleet_socket()),
            ),
            // Ports present but none ACTIVE: nothing usable to fall back from.
            (
                "ib-down".to_string(),
                host_with(Some(inventory(&["DOWN", "INIT"])), fleet_socket()),
            ),
            // IB present + IB: healthy.
            ("ib-ib".to_string(), host(fleet_ib)),
            // No inventory: unknown, never flagged.
            ("no-inventory".to_string(), host_with(None, fleet_socket())),
        ]);
        let findings = socket_fallbacks(&observations, None);
        assert_eq!(findings.keys().collect::<Vec<_>>(), vec!["ib-socket"]);
        // The transport record itself is kept on every host.
        assert!(
            observations
                .values()
                .all(|obs| obs.nccl_transports.len() == 1)
        );

        // IB present + socket + NCCL_IB_DISABLE=1: requested, not a fallback.
        let level_env: LevelEnvMap = BTreeMap::from([(
            NcclLevel::Fleet,
            BTreeMap::from([("NCCL_IB_DISABLE".to_string(), "1".to_string())]),
        )]);
        assert!(socket_fallbacks(&observations, Some(&level_env)).is_empty());
    }

    #[test]
    fn ib_availability_reads_the_inventory() {
        assert_eq!(
            IbAvailability::of(&host_with(None, vec![]), None),
            IbAvailability::Unknown
        );
        assert_eq!(
            IbAvailability::of(&host_with(Some(inventory(&[])), vec![]), None),
            IbAvailability::NoActivePort
        );
        assert_eq!(
            IbAvailability::of(
                &host_with(Some(inventory(&["DOWN", "unknown"])), vec![]),
                None
            ),
            IbAvailability::NoActivePort
        );
        assert_eq!(
            IbAvailability::of(
                &host_with(Some(inventory(&["DOWN", "active"])), vec![]),
                None
            ),
            IbAvailability::ActivePort
        );
    }

    #[test]
    fn without_inventory_nccls_no_device_line_means_no_ib() {
        let no_device = TransportCapture::Captured {
            info: NcclTransportInfo {
                nccl_version: None,
                net: Some(socket()),
                net_plugin: None,
                channels: ChannelTransports::default(),
                ib_no_device: true,
            },
        };
        let obs = host_with(
            None,
            vec![(NcclLevel::Fleet, CommSpan::MultiHost, no_device)],
        );
        assert_eq!(
            IbAvailability::of(&obs, None),
            IbAvailability::NoDeviceSeenByNccl
        );
        // NCCL_IB_HCA set: "no device" may be the filter matching nothing,
        // i.e. the misconfiguration itself — not proof that IB is absent.
        let hca: LevelEnvMap = BTreeMap::from([(
            NcclLevel::Fleet,
            BTreeMap::from([("NCCL_IB_HCA".to_string(), "mlx5_9".to_string())]),
        )]);
        assert_eq!(
            IbAvailability::of(&obs, Some(&hca)),
            IbAvailability::Unknown
        );
        // Neither case is flagged: there is no inventory proving a usable port.
        let observations = BTreeMap::from([("n1".to_string(), obs)]);
        assert!(socket_fallbacks(&observations, None).is_empty());
        assert!(socket_fallbacks(&observations, Some(&hca)).is_empty());
    }

    #[test]
    fn repeats_flag_a_level_once() {
        let observations = BTreeMap::from([(
            "n1".to_string(),
            host(vec![
                (NcclLevel::Fleet, CommSpan::MultiHost, captured(ib())),
                (NcclLevel::Fleet, CommSpan::MultiHost, captured(socket())),
                (NcclLevel::Fleet, CommSpan::MultiHost, captured(socket())),
            ]),
        )]);
        assert_eq!(socket_fallbacks(&observations, None)["n1"].len(), 1);
    }

    #[test]
    fn the_table_shows_the_latest_report_per_level() {
        let observations = BTreeMap::from([(
            "n1".to_string(),
            host(vec![
                (NcclLevel::Fleet, CommSpan::MultiHost, captured(socket())),
                (NcclLevel::Fleet, CommSpan::MultiHost, captured(ib())),
            ]),
        )]);
        let rows = latest_reports(&observations);
        assert_eq!(rows.len(), 1);
        assert_eq!(net_cell(&rows[0].1.capture), "IB mlx5_0:1");
        assert_eq!(
            net_cell(&TransportCapture::Unknown {
                reason: UnknownTransport::DebugLevel {
                    level: Some("WARN".into())
                }
            }),
            "unknown: NCCL_DEBUG=WARN (from the config) is below INFO, so NCCL wrote no INFO log"
        );
    }
}
