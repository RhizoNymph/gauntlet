//! Which IB/RoCE ports NCCL would use, and what they add up to.
//!
//! The agent inventories ports as facts (`proto::IbPortInventory`); the
//! orchestrator knows the run's resolved NCCL environment. This module joins
//! the two, purely: [`NcclIbConfig`] is the slice of the env that decides
//! the IB transport's port list, [`summarize`] replays NCCL's device scan
//! over one host's inventory, and [`NcclNicSummary`] is the per-host result
//! (selected ports, excluded ports with the reason, and the summed line
//! rate: the "NCCL NIC ceiling").
//!
//! What is modelled, in NCCL's order (IB transport init, `net_ib.cc`):
//! 1. `NCCL_IB_DISABLE` set to a non-zero integer turns the IB transport
//!    off: no port is used.
//! 2. Only ports whose logical state is ACTIVE are considered.
//! 3. Only InfiniBand and Ethernet (RoCE) link layers are considered.
//! 4. The port must pass the `NCCL_IB_HCA` filter (see [`hca`]); unset
//!    means every port passes.
//!
//! Not modelled: `NCCL_NET` / external net plugins (a plugin replaces the
//! IB transport entirely), `NCCL_IB_HCA` values coming from the node's own
//! environment or `/etc/nccl.conf` (only the `[nccl] env` map is known
//! here), NIC fusion (`NCCL_IB_MERGE_NICS`), and per-GPU NIC assignment
//! (NCCL picks NICs per rank by topology; the ceiling is the node total).

pub mod hca;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::nccl_env::NcclEnv;
use crate::proto::{
    AgentEvent, InventorySnapshot, LinkLayer, MetricRecord, PciLocality, PortState, Scope, TestId,
    TestOutcome, Unit, gbps_to_gib_per_sec, pci_locality,
};
use hca::HcaFilter;

pub const IB_HCA: &str = "NCCL_IB_HCA";
pub const IB_DISABLE: &str = "NCCL_IB_DISABLE";

/// Metric name of the per-node ceiling (`nccl_nics.ceiling_gib_per_sec`).
pub const CEILING_METRIC: &str = "ceiling_gib_per_sec";

/// Consistency-field names contributed per host.
pub mod consistency {
    pub const PORTS: &str = "nccl_ib_ports";
    pub const LINK_LAYER: &str = "nccl_ib_link_layer";
    pub const CEILING_GBPS: &str = "nccl_ib_ceiling_gbps";
}

/// The NCCL knobs that decide the IB transport's port list.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NcclIbConfig {
    /// The raw `NCCL_IB_HCA` value, for messages; `None` = unset.
    hca_value: Option<String>,
    hca: HcaFilter,
    ib_disabled: bool,
}

impl NcclIbConfig {
    pub fn new(hca: Option<&str>, ib_disable: Option<&str>) -> Self {
        Self {
            hca_value: hca.map(str::to_string),
            hca: hca.map_or_else(HcaFilter::unset, HcaFilter::parse),
            ib_disabled: ib_disable.is_some_and(ib_disable_is_set),
        }
    }

    pub fn from_env(env: &NcclEnv) -> Self {
        Self::new(env.get(IB_HCA), env.get(IB_DISABLE))
    }

    pub fn hca_value(&self) -> Option<&str> {
        self.hca_value.as_deref()
    }

    pub fn hca(&self) -> &HcaFilter {
        &self.hca
    }

    pub fn ib_disabled(&self) -> bool {
        self.ib_disabled
    }

    /// "NCCL_IB_HCA=<value>" or "NCCL_IB_HCA unset".
    pub fn describe_hca(&self) -> String {
        match &self.hca_value {
            Some(value) => format!("{IB_HCA}={value}"),
            None => format!("{IB_HCA} unset"),
        }
    }
}

/// NCCL reads integer params with `strtoll` and keeps the default (0) when
/// the text is not a whole integer; any non-zero value disables.
fn ib_disable_is_set(raw: &str) -> bool {
    raw.trim().parse::<i64>().is_ok_and(|value| value != 0)
}

/// Why NCCL would not use a port. Checked in this order; the first that
/// applies is recorded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum PortExclusion {
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
            Self::IbDisabled => "IB transport disabled".to_string(),
            Self::NotActive { state } => format!("not active ({state})"),
            Self::UnsupportedLinkLayer => "unsupported link layer".to_string(),
            Self::FilteredByHca => format!("filtered by {IB_HCA}"),
        }
    }
}

/// A port NCCL would use. Selected ports are ACTIVE by construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectedPort {
    pub device: String,
    pub port: u32,
    pub link_layer: LinkLayer,
    /// The rate as printed by sysfs, Gb/s.
    pub rate_gbps: Option<f64>,
    /// Encoding-corrected payload rate, Gb/s (`proto::payload_gbps`).
    pub payload_gbps: Option<f64>,
    pub netdevs: Vec<String>,
    /// The device's NUMA node, when known.
    pub numa_node: Option<u32>,
    /// PCI distance from this NIC to every GPU with a known placement,
    /// keyed by GPU index.
    pub gpu_locality: BTreeMap<u32, PciLocality>,
}

/// A port NCCL would not use, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExcludedPort {
    pub device: String,
    pub port: u32,
    #[serde(flatten)]
    pub reason: PortExclusion,
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
    pub selected: Vec<SelectedPort>,
    pub excluded: Vec<ExcludedPort>,
    /// Sum of the selected ports' payload rates in GiB/s. `Some(0.0)` when
    /// nothing is selected; `None` when a selected port's rate is unknown
    /// (a partial sum would understate the ceiling).
    pub ceiling_gib_per_sec: Option<f64>,
}

impl NcclNicSummary {
    /// The host has no IB/RoCE ports at all.
    pub fn has_no_ports(&self) -> bool {
        self.selected.is_empty() && self.excluded.is_empty()
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

    /// Sum of the selected ports' payload rates in Gb/s (`None` when any is
    /// unknown).
    pub fn ceiling_gbps(&self) -> Option<f64> {
        self.selected
            .iter()
            .map(|port| port.payload_gbps)
            .sum::<Option<f64>>()
    }

    /// The `nccl_nics` outcome for this host:
    /// - no IB/RoCE ports at all: Skipped (NCCL uses sockets by design);
    /// - IB transport disabled: Skipped;
    /// - ports present but none selected: Failed — NCCL would silently fall
    ///   back to sockets;
    /// - otherwise Passed.
    pub fn outcome(&self, config: &NcclIbConfig) -> TestOutcome {
        if self.has_no_ports() {
            return TestOutcome::Skipped {
                reason: "no InfiniBand/RoCE ports; NCCL uses sockets".to_string(),
            };
        }
        if config.ib_disabled() {
            return TestOutcome::Skipped {
                reason: format!("{IB_DISABLE} set; NCCL uses sockets"),
            };
        }
        if !self.selected.is_empty() {
            return TestOutcome::Passed;
        }
        let why: Vec<String> = self
            .excluded
            .iter()
            .map(|port| format!("{}:{} {}", port.device, port.port, port.reason.describe()))
            .collect();
        TestOutcome::Failed {
            reason: format!(
                "{} selects no active IB/RoCE port ({}); NCCL would fall back to sockets",
                config.describe_hca(),
                why.join(", ")
            ),
        }
    }

    /// The fleet-comparable node-scope metric, when it means something: the
    /// host has IB/RoCE ports, the transport is enabled, and every selected
    /// rate is known.
    pub fn ceiling_metric(&self, config: &NcclIbConfig) -> Option<MetricRecord> {
        if self.has_no_ports() || config.ib_disabled() {
            return None;
        }
        Some(MetricRecord {
            test: TestId::NcclNics,
            scope: Scope::Node,
            name: CEILING_METRIC.to_string(),
            value: self.ceiling_gib_per_sec?,
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

/// Classify one port the way NCCL's device scan would.
pub fn classify_port(
    config: &NcclIbConfig,
    port: &crate::proto::IbPortInventory,
) -> Result<(), PortExclusion> {
    if config.ib_disabled() {
        return Err(PortExclusion::IbDisabled);
    }
    if port.logical_state() != PortState::Active {
        return Err(PortExclusion::NotActive {
            state: port.state.clone(),
        });
    }
    if !port.link_layer.nccl_capable() {
        return Err(PortExclusion::UnsupportedLinkLayer);
    }
    if !config.hca().matches(&port.device, port.port) {
        return Err(PortExclusion::FilteredByHca);
    }
    Ok(())
}

/// The events the orchestrator derives from a host's inventory as it
/// arrives: one Node-scope `nccl_nics` outcome, then the ceiling metric
/// when it applies (see [`NcclNicSummary::ceiling_metric`]).
pub fn nccl_nic_events(inventory: &InventorySnapshot, config: &NcclIbConfig) -> Vec<AgentEvent> {
    let summary = summarize(inventory, config);
    let mut events = vec![AgentEvent::Outcome {
        test: TestId::NcclNics,
        scope: Scope::Node,
        outcome: summary.outcome(config),
    }];
    if let Some(record) = summary.ceiling_metric(config) {
        events.push(AgentEvent::Metric { record });
    }
    events
}

/// Replay NCCL's IB device scan over one host's inventory.
pub fn summarize(inventory: &InventorySnapshot, config: &NcclIbConfig) -> NcclNicSummary {
    let mut selected = Vec::new();
    let mut excluded = Vec::new();
    for port in &inventory.ib_ports {
        match classify_port(config, port) {
            Ok(()) => {
                let pci = inventory
                    .ib_devices
                    .iter()
                    .find(|device| device.name == port.device)
                    .and_then(|device| device.pci.as_ref());
                let gpu_locality = pci
                    .map(|nic| {
                        inventory
                            .gpus
                            .iter()
                            .filter_map(|gpu| {
                                gpu.pci
                                    .as_ref()
                                    .map(|placement| (gpu.index, pci_locality(nic, placement)))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                selected.push(SelectedPort {
                    device: port.device.clone(),
                    port: port.port,
                    link_layer: port.link_layer.clone(),
                    rate_gbps: port.rate_gbps,
                    payload_gbps: port.payload_gbps(),
                    netdevs: port.netdevs.clone(),
                    numa_node: pci.and_then(|nic| nic.numa_node),
                    gpu_locality,
                });
            }
            Err(reason) => excluded.push(ExcludedPort {
                device: port.device.clone(),
                port: port.port,
                reason,
            }),
        }
    }
    let ceiling_gib_per_sec = selected
        .iter()
        .map(|port| port.payload_gbps)
        .sum::<Option<f64>>()
        .map(gbps_to_gib_per_sec);
    NcclNicSummary {
        selected,
        excluded,
        ceiling_gib_per_sec,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ib_disable_follows_ncclparam_integer_rules() {
        assert!(ib_disable_is_set("1"));
        assert!(ib_disable_is_set("2"));
        assert!(ib_disable_is_set("-1"));
        assert!(!ib_disable_is_set("0"));
        assert!(!ib_disable_is_set("yes"));
        assert!(!ib_disable_is_set("1x"));
        assert!(NcclIbConfig::new(None, Some("1")).ib_disabled());
        assert!(!NcclIbConfig::new(None, None).ib_disabled());
    }

    #[test]
    fn hca_is_described_for_messages() {
        assert_eq!(
            NcclIbConfig::new(Some("^mlx5_2"), None).describe_hca(),
            "NCCL_IB_HCA=^mlx5_2"
        );
        assert_eq!(
            NcclIbConfig::new(None, None).describe_hca(),
            "NCCL_IB_HCA unset"
        );
    }
}
