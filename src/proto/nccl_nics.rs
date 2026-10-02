//! Which IB/RoCE ports NCCL would use on a host, and what they add up to.
//!
//! The result of replaying NCCL's IB device scan over one host's inventory
//! (`crate::nccl_ib::summarize`). It records the env inputs it was derived
//! under (`hca`, `net`, `ib_disabled`), so the outcome, the metric, the
//! consistency fields and the report section are all functions of this one
//! value: none of them re-derives anything, and they cannot drift apart.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{
    LinkLayer, MetricRecord, PciLocality, PortRate, Scope, TestId, TestOutcome, Unit,
    gbps_to_gib_per_sec,
};

pub const IB_HCA: &str = "NCCL_IB_HCA";
pub const IB_DISABLE: &str = "NCCL_IB_DISABLE";
pub const NET: &str = "NCCL_NET";

/// Metric name of the per-node ceiling (`nccl_nics.ceiling_gib_per_sec`).
pub const CEILING_METRIC: &str = "ceiling_gib_per_sec";

/// Consistency-field names contributed per host.
pub mod consistency {
    pub const PORTS: &str = "nccl_ib_ports";
    pub const LINK_LAYER: &str = "nccl_ib_link_layer";
    pub const CEILING_GBPS: &str = "nccl_ib_ceiling_gbps";
}

/// "NCCL_IB_HCA=<value>" or "NCCL_IB_HCA unset" — the one spelling used by
/// outcome reasons and the report header.
pub fn describe_hca(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("{IB_HCA}={value}"),
        None => format!("{IB_HCA} unset"),
    }
}

/// `NCCL_NET`: NCCL compares it case-insensitively (`strcasecmp`) with
/// each network's name — an external plugin first, then the built-in "IB"
/// and "Socket".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NetChoice {
    /// Unset: plugin if any, else IB if it has devices, else sockets.
    #[default]
    Default,
    /// "IB": the built-in IB transport.
    Ib,
    /// "Socket": the built-in socket transport; no IB port is used.
    Socket,
    /// Anything else names an external net plugin. Not modelled: the IB
    /// selection is kept as if the IB transport ran.
    Plugin { name: String },
}

impl NetChoice {
    pub fn parse(value: Option<&str>) -> Self {
        match value {
            None => Self::Default,
            Some(name) if name.eq_ignore_ascii_case("ib") => Self::Ib,
            Some(name) if name.eq_ignore_ascii_case("socket") => Self::Socket,
            Some(name) => Self::Plugin {
                name: name.to_string(),
            },
        }
    }
}

/// Why NCCL would not use a port. Checked in this order; the first that
/// applies is recorded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum PortExclusion {
    /// `NCCL_NET=Socket` selected the socket transport.
    NetSocket,
    /// `NCCL_IB_DISABLE` turned the IB transport off.
    IbDisabled,
    /// Logical state is not ACTIVE; carries the state as inventoried.
    NotActive { state: String },
    /// Link layer is neither InfiniBand nor Ethernet.
    UnsupportedLinkLayer,
    /// `NCCL_IB_HCA` filters it out.
    FilteredByHca,
}

impl PortExclusion {
    pub fn describe(&self) -> String {
        match self {
            Self::NetSocket => format!("{NET}=Socket forces sockets"),
            Self::IbDisabled => format!("{IB_DISABLE} set"),
            Self::NotActive { state } => format!("not active ({state})"),
            Self::UnsupportedLinkLayer => "unsupported link layer".to_string(),
            Self::FilteredByHca => format!("filtered by {IB_HCA}"),
        }
    }
}

/// A port NCCL would not use, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExcludedPort {
    pub device: String,
    pub port: u32,
    #[serde(flatten)]
    pub reason: PortExclusion,
}

impl ExcludedPort {
    /// "mlx5_2:1 filtered by NCCL_IB_HCA" — shared by the outcome reason
    /// and the report table.
    pub fn describe(&self) -> String {
        format!("{}:{} {}", self.device, self.port, self.reason.describe())
    }
}

/// A port NCCL would use. ACTIVE with an IB/Ethernet link layer by
/// construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectedPort {
    pub device: String,
    pub port: u32,
    pub link_layer: LinkLayer,
    pub rate: PortRate,
    pub netdevs: Vec<String>,
    /// The device's NUMA node, when known.
    pub numa_node: Option<u32>,
    /// PCI distance from this NIC to every GPU with a known placement, in
    /// GPU index order. A list, not a map: integer map keys do not survive
    /// the buffered decode of an internally tagged `AgentEvent`.
    pub gpu_locality: Vec<GpuLocality>,
}

/// One GPU's PCI distance from a NIC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuLocality {
    pub gpu: u32,
    pub locality: PciLocality,
}

impl SelectedPort {
    /// This NIC's PCI distance from GPU `gpu`, when its placement is known.
    pub fn locality_to(&self, gpu: u32) -> Option<PciLocality> {
        self.gpu_locality
            .iter()
            .find(|entry| entry.gpu == gpu)
            .map(|entry| entry.locality)
    }

    /// The nearest known locality and the GPUs at it, e.g.
    /// `(PcieSwitch, [0, 1])`. `None` when no GPU placement is known.
    pub fn nearest_gpus(&self) -> Option<(PciLocality, Vec<u32>)> {
        let nearest = self
            .gpu_locality
            .iter()
            .map(|entry| entry.locality)
            .filter(|locality| *locality != PciLocality::Unknown)
            .min()?;
        let gpus = self
            .gpu_locality
            .iter()
            .filter(|entry| entry.locality == nearest)
            .map(|entry| entry.gpu)
            .collect();
        Some((nearest, gpus))
    }

    /// Encoding-corrected payload rate, Gb/s.
    pub fn payload_gbps(&self) -> Option<f64> {
        self.rate.payload_gbps(&self.link_layer)
    }
}

/// Link layer(s) of the selected ports, as one fleet-comparable value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionLinkLayer {
    None,
    Infiniband,
    Ethernet,
    Mixed,
}

impl SelectionLinkLayer {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Infiniband => "infiniband",
            Self::Ethernet => "ethernet",
            Self::Mixed => "mixed",
        }
    }
}

/// One host's NCCL NIC picture under the run's env.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NcclNicSummary {
    /// `NCCL_IB_HCA` in effect (`None` = unset).
    pub hca: Option<String>,
    /// `NCCL_NET` in effect.
    pub net: NetChoice,
    /// `NCCL_IB_DISABLE` evaluated as NCCL does (non-zero integer).
    pub ib_disabled: bool,
    pub selected: Vec<SelectedPort>,
    pub excluded: Vec<ExcludedPort>,
}

impl NcclNicSummary {
    /// The host has no IB/RoCE ports at all.
    pub fn has_no_ports(&self) -> bool {
        self.selected.is_empty() && self.excluded.is_empty()
    }

    /// Why the IB transport is off by configuration, if it is.
    pub fn transport_off(&self) -> Option<String> {
        if self.net == NetChoice::Socket {
            Some(format!("{NET}=Socket; NCCL uses sockets"))
        } else if self.ib_disabled {
            Some(format!("{IB_DISABLE} set; NCCL uses sockets"))
        } else {
            None
        }
    }

    /// A caveat for the reader when NCCL_NET names a plugin.
    pub fn net_caveat(&self) -> Option<String> {
        match &self.net {
            NetChoice::Plugin { name } => Some(format!(
                "{NET}={name} is an external plugin (not modelled; IB selection shown)"
            )),
            _ => None,
        }
    }

    pub fn link_layer(&self) -> SelectionLinkLayer {
        let infiniband = self
            .selected
            .iter()
            .any(|port| port.link_layer == LinkLayer::Infiniband);
        let ethernet = self
            .selected
            .iter()
            .any(|port| matches!(port.link_layer, LinkLayer::Ethernet { .. }));
        match (infiniband, ethernet) {
            (false, false) => SelectionLinkLayer::None,
            (true, false) => SelectionLinkLayer::Infiniband,
            (false, true) => SelectionLinkLayer::Ethernet,
            (true, true) => SelectionLinkLayer::Mixed,
        }
    }

    /// Sum of the selected ports' payload rates in Gb/s: `Some(0.0)` when
    /// nothing is selected, `None` when a selected rate is unknown (a
    /// partial sum would understate the ceiling).
    pub fn ceiling_gbps(&self) -> Option<f64> {
        self.selected
            .iter()
            .map(SelectedPort::payload_gbps)
            .sum::<Option<f64>>()
    }

    /// [`Self::ceiling_gbps`] in GiB/s.
    pub fn ceiling_gib_per_sec(&self) -> Option<f64> {
        self.ceiling_gbps().map(gbps_to_gib_per_sec)
    }

    /// Why the ceiling is unknown: every selected port whose rate is not
    /// known, with its typed reason. `None` when the ceiling is known.
    pub fn ceiling_unknown_reason(&self) -> Option<String> {
        let unknown: Vec<String> = self
            .selected
            .iter()
            .filter(|port| port.payload_gbps().is_none())
            .map(|port| format!("{}:{} {}", port.device, port.port, port.rate.describe()))
            .collect();
        (!unknown.is_empty()).then(|| format!("ceiling unknown: {}", unknown.join(", ")))
    }

    /// The `nccl_nics` outcome:
    /// - no IB/RoCE ports at all: Skipped (sockets by design);
    /// - transport off by NCCL_NET=Socket or NCCL_IB_DISABLE: Skipped;
    /// - ports present but none selected: Failed — NCCL would silently
    ///   fall back to sockets;
    /// - otherwise Passed.
    pub fn outcome(&self) -> TestOutcome {
        if self.has_no_ports() {
            return TestOutcome::Skipped {
                reason: "no InfiniBand/RoCE ports; NCCL uses sockets".to_string(),
            };
        }
        if let Some(reason) = self.transport_off() {
            return TestOutcome::Skipped { reason };
        }
        if !self.selected.is_empty() {
            return TestOutcome::Passed;
        }
        let why: Vec<String> = self.excluded.iter().map(ExcludedPort::describe).collect();
        TestOutcome::Failed {
            reason: format!(
                "{} selects no active IB/RoCE port ({}); NCCL would fall back to sockets",
                describe_hca(self.hca.as_deref()),
                why.join(", ")
            ),
        }
    }

    /// The fleet-comparable node-scope metric, when it means something:
    /// the host has IB/RoCE ports, the transport is on, and every selected
    /// rate is known.
    pub fn ceiling_metric(&self) -> Option<MetricRecord> {
        if self.has_no_ports() || self.transport_off().is_some() {
            return None;
        }
        Some(MetricRecord {
            test: TestId::NcclNics,
            scope: Scope::Node,
            name: CEILING_METRIC.to_string(),
            value: self.ceiling_gib_per_sec()?,
            unit: Unit::GibPerSec,
            repeat: 0,
        })
    }

    /// Consistency fields: selected-port count, selected link layer(s),
    /// and the ceiling in Gb/s (exact-match, so one slower host dissents
    /// even when the MAD of an otherwise uniform fleet is zero).
    pub fn consistency_fields(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            (
                consistency::PORTS.to_string(),
                self.selected.len().to_string(),
            ),
            (
                consistency::LINK_LAYER.to_string(),
                self.link_layer().label().to_string(),
            ),
            (
                consistency::CEILING_GBPS.to_string(),
                self.ceiling_gbps()
                    .map_or_else(|| "unknown".to_string(), |gbps| format!("{gbps}")),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn net_choice_is_case_insensitive() {
        assert_eq!(NetChoice::parse(None), NetChoice::Default);
        assert_eq!(NetChoice::parse(Some("Socket")), NetChoice::Socket);
        assert_eq!(NetChoice::parse(Some("SOCKET")), NetChoice::Socket);
        assert_eq!(NetChoice::parse(Some("ib")), NetChoice::Ib);
        assert_eq!(
            NetChoice::parse(Some("AWS Libfabric")),
            NetChoice::Plugin {
                name: "AWS Libfabric".into()
            }
        );
    }

    #[test]
    fn hca_is_described_one_way() {
        assert_eq!(describe_hca(Some("^mlx5_2")), "NCCL_IB_HCA=^mlx5_2");
        assert_eq!(describe_hca(None), "NCCL_IB_HCA unset");
    }

    #[test]
    fn excluded_ports_describe_themselves() {
        let port = ExcludedPort {
            device: "mlx5_3".into(),
            port: 1,
            reason: PortExclusion::NotActive {
                state: "DOWN".into(),
            },
        };
        assert_eq!(port.describe(), "mlx5_3:1 not active (DOWN)");
        let json = serde_json::to_string(&port).expect("serialize");
        assert_eq!(
            json,
            r#"{"device":"mlx5_3","port":1,"reason":"not_active","state":"DOWN"}"#
        );
        assert_eq!(
            serde_json::from_str::<ExcludedPort>(&json).expect("decode"),
            port
        );
    }
}
