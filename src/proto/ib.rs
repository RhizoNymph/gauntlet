//! InfiniBand / RoCE inventory types (proto v11).
//!
//! Everything here is a *fact* the agent reads from sysfs, typed:
//! `/sys/class/infiniband/<dev>/ports/<p>/{state,phys_state,rate,
//! link_layer,gid_attrs/*}` and the device's PCI placement. Which ports NCCL
//! would use under a given `NCCL_IB_HCA` is decided orchestrator-side
//! (`crate::nccl_ib`), never by the agent.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Bits per byte, for line-rate conversions.
const BITS_PER_BYTE: f64 = 8.0;
/// Bytes per GiB.
const BYTES_PER_GIB: f64 = (1u64 << 30) as f64;
/// sysfs reports rates in decimal gigabits.
const BITS_PER_GBIT: f64 = 1e9;

/// Convert a decimal Gb/s line rate to GiB/s (binary bytes): 200 Gb/s is
/// 200e9 / 8 / 2^30 = 23.28 GiB/s.
pub fn gbps_to_gib_per_sec(gbps: f64) -> f64 {
    gbps * BITS_PER_GBIT / BITS_PER_BYTE / BYTES_PER_GIB
}

// ---------------------------------------------------------------------------
// Port state
// ---------------------------------------------------------------------------

/// Logical port state (`ports/<p>/state`, e.g. "4: ACTIVE"). NCCL only uses
/// `Active` ports. `IbPortInventory.state` keeps the symbolic string on the
/// wire (unchanged since v1); this is its typed reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortState {
    Nop,
    Down,
    Init,
    Armed,
    Active,
    ActiveDefer,
    Unknown,
}

impl PortState {
    /// Accepts the raw sysfs text ("4: ACTIVE") or its symbolic half
    /// ("ACTIVE", any case). Anything else is `Unknown`.
    pub fn parse(raw: &str) -> Self {
        let symbolic = raw.split_once(':').map_or(raw, |(_, name)| name).trim();
        match symbolic.to_ascii_uppercase().as_str() {
            "NOP" => Self::Nop,
            "DOWN" => Self::Down,
            "INIT" => Self::Init,
            "ARMED" => Self::Armed,
            "ACTIVE" => Self::Active,
            "ACTIVE_DEFER" => Self::ActiveDefer,
            _ => Self::Unknown,
        }
    }
}

/// Physical port state (`ports/<p>/phys_state`, e.g. "5: LinkUp"), keyed
/// on the numeric code. `Polling` on a port that should be cabled is the
/// classic "no peer / bad cable" signature.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysState {
    Sleep,
    Polling,
    Disabled,
    PortConfigurationTraining,
    LinkUp,
    LinkErrorRecovery,
    PhyTest,
    #[default]
    Unknown,
}

impl PhysState {
    pub fn parse(raw: &str) -> Self {
        let code = raw.split_once(':').map_or(raw, |(code, _)| code).trim();
        match code.parse::<u32>() {
            Ok(1) => Self::Sleep,
            Ok(2) => Self::Polling,
            Ok(3) => Self::Disabled,
            Ok(4) => Self::PortConfigurationTraining,
            Ok(5) => Self::LinkUp,
            Ok(6) => Self::LinkErrorRecovery,
            Ok(7) => Self::PhyTest,
            _ => Self::Unknown,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Sleep => "sleep",
            Self::Polling => "polling",
            Self::Disabled => "disabled",
            Self::PortConfigurationTraining => "training",
            Self::LinkUp => "link_up",
            Self::LinkErrorRecovery => "link_error_recovery",
            Self::PhyTest => "phy_test",
            Self::Unknown => "unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// Link layer
// ---------------------------------------------------------------------------

/// RoCE version of a populated GID (`gid_attrs/types/<i>`): "IB/RoCE v1"
/// on an Ethernet port is RoCE v1, "RoCE v2" is RoCE v2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoceVersion {
    V1,
    V2,
}

impl RoceVersion {
    pub fn parse_gid_type(raw: &str) -> Option<Self> {
        match raw.trim() {
            "IB/RoCE v1" | "RoCE v1" => Some(Self::V1),
            "RoCE v2" => Some(Self::V2),
            _ => None,
        }
    }
}

/// `ports/<p>/link_layer`. RoCE versions only exist on Ethernet, so they
/// live inside that variant. `Unknown` is what a pre-v11 inventory decodes
/// to, and what an unreadable or unrecognised file reads as.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LinkLayer {
    Infiniband,
    Ethernet {
        /// Distinct RoCE versions among the port's populated GIDs, sorted.
        /// Empty when no GID is populated (or the types were unreadable).
        #[serde(default)]
        roce_versions: Vec<RoceVersion>,
    },
    #[default]
    Unknown,
}

impl LinkLayer {
    /// Parse the sysfs `link_layer` text ("InfiniBand" / "Ethernet").
    /// `roce_versions` is only kept for Ethernet.
    pub fn parse(raw: &str, mut roce_versions: Vec<RoceVersion>) -> Self {
        match raw.trim() {
            "InfiniBand" => Self::Infiniband,
            "Ethernet" => {
                roce_versions.sort();
                roce_versions.dedup();
                Self::Ethernet { roce_versions }
            }
            _ => Self::Unknown,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Infiniband => "infiniband",
            Self::Ethernet { .. } => "ethernet",
            Self::Unknown => "unknown",
        }
    }

    /// NCCL's IB transport accepts InfiniBand and Ethernet (RoCE) ports
    /// only.
    pub fn nccl_capable(&self) -> bool {
        matches!(self, Self::Infiniband | Self::Ethernet { .. })
    }
}

// ---------------------------------------------------------------------------
// Rate
// ---------------------------------------------------------------------------

/// The speed name in the sysfs rate string's parenthetical. The kernel
/// prints SDR either as " SDR" or (older kernels) as nothing at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IbSpeed {
    Sdr,
    Ddr,
    Qdr,
    Fdr10,
    Fdr,
    Edr,
    Hdr,
    Ndr,
    Xdr,
}

impl IbSpeed {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "" | "SDR" => Some(Self::Sdr),
            "DDR" => Some(Self::Ddr),
            "QDR" => Some(Self::Qdr),
            "FDR10" => Some(Self::Fdr10),
            "FDR" => Some(Self::Fdr),
            "EDR" => Some(Self::Edr),
            "HDR" => Some(Self::Hdr),
            "NDR" => Some(Self::Ndr),
            "XDR" => Some(Self::Xdr),
            _ => None,
        }
    }

    /// Payload bits per reported bit on an InfiniBand link. The kernel
    /// reports SDR/DDR/QDR at their 8b/10b *signalling* rate (2.5/5/10 Gb/s
    /// per lane), and FDR at 14 Gb/s per lane, which carries 64b/66b
    /// encoding. FDR10 and EDR onwards are reported at their data rate.
    pub fn encoding_efficiency(self) -> f64 {
        match self {
            Self::Sdr | Self::Ddr | Self::Qdr => 0.8,
            Self::Fdr => 64.0 / 66.0,
            Self::Fdr10 | Self::Edr | Self::Hdr | Self::Ndr | Self::Xdr => 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum IbRateError {
    #[error("ib rate {raw:?} does not start with a number")]
    Number { raw: String },
    #[error("ib rate {raw:?} is not in Gb/sec")]
    Unit { raw: String },
}

/// A parsed `ports/<p>/rate` string: "200 Gb/sec (4X HDR)",
/// "2.5 Gb/sec (1X SDR)", "10 Gb/sec (4X)".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IbRate {
    /// The number the kernel printed (decimal Gb/s).
    pub gbps: f64,
    /// Link width in lanes ("4X" -> 4), when printed.
    pub lanes: Option<u32>,
    /// Speed name, when printed and recognised.
    pub speed: Option<IbSpeed>,
}

impl IbRate {
    pub fn parse(raw: &str) -> Result<Self, IbRateError> {
        let trimmed = raw.trim();
        let mut words = trimmed.split_whitespace();
        let gbps = words
            .next()
            .and_then(|number| number.parse::<f64>().ok())
            .filter(|gbps| gbps.is_finite() && *gbps >= 0.0)
            .ok_or_else(|| IbRateError::Number {
                raw: raw.to_string(),
            })?;
        match words.next() {
            Some("Gb/sec" | "Gb/s") => {}
            _ => {
                return Err(IbRateError::Unit {
                    raw: raw.to_string(),
                });
            }
        }
        let (lanes, speed) = match trimmed.split_once('(') {
            Some((_, rest)) => {
                let inner = rest.trim_end().trim_end_matches(')').trim();
                let (width, name) = inner.split_once(' ').unwrap_or((inner, ""));
                let lanes = width
                    .strip_suffix(['X', 'x'])
                    .and_then(|lanes| lanes.parse::<u32>().ok());
                (lanes, IbSpeed::parse(name.trim()))
            }
            None => (None, None),
        };
        Ok(Self { gbps, lanes, speed })
    }
}

/// Payload line rate of a port in Gb/s. On InfiniBand the reported number
/// is corrected for the speed's encoding (see
/// [`IbSpeed::encoding_efficiency`]); an unknown speed is taken at face
/// value. On Ethernet (RoCE) the kernel maps the Ethernet speed onto an IB
/// speed/width pair (40GbE reads "40 Gb/sec (4X QDR)") but the number is
/// already the Ethernet data rate, so it is never corrected.
pub fn payload_gbps(rate_gbps: f64, speed: Option<IbSpeed>, link_layer: &LinkLayer) -> f64 {
    match (link_layer, speed) {
        (LinkLayer::Infiniband, Some(speed)) => rate_gbps * speed.encoding_efficiency(),
        _ => rate_gbps,
    }
}

// ---------------------------------------------------------------------------
// PCI placement
// ---------------------------------------------------------------------------

/// Where a PCI function sits: its address, NUMA node, and the bridges above
/// it (host bridge first: `["pci0000:3a", "0000:3a:00.0", ...]`), read from
/// the canonical `/sys/devices/...` path. An empty `upstream` means the
/// path was not resolved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PciLocation {
    /// sysfs form, "0000:3b:00.0".
    pub address: String,
    /// `numa_node`; the kernel's -1 ("no affinity") is `None`.
    #[serde(default)]
    pub numa_node: Option<u32>,
    #[serde(default)]
    pub upstream: Vec<String>,
}

impl PciLocation {
    /// A location known only by address (path and NUMA node unresolved).
    pub fn address_only(address: impl Into<String>) -> Self {
        Self {
            address: address.into(),
            numa_node: None,
            upstream: Vec::new(),
        }
    }
}

/// How close two PCI functions are, nearest first. Mirrors NCCL's path
/// types closely enough to reason about GPUDirect RDMA:
/// `PcieSwitch` ~ PIX/PXB (traffic stays below the CPU), `HostBridge` ~
/// PHB, `SameNuma` ~ NODE, `CrossNuma` ~ SYS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PciLocality {
    /// Both sit below a common bridge under the host bridge: a shared PCIe
    /// switch (or root port with a switch beneath it).
    PcieSwitch,
    /// Same host bridge (root complex), no shared bridge below it.
    HostBridge,
    /// Different host bridges (or unresolved paths) on the same NUMA node.
    SameNuma,
    /// Different NUMA nodes.
    CrossNuma,
    /// Not enough information to say.
    Unknown,
}

impl PciLocality {
    pub fn label(self) -> &'static str {
        match self {
            Self::PcieSwitch => "pcie_switch",
            Self::HostBridge => "host_bridge",
            Self::SameNuma => "same_numa",
            Self::CrossNuma => "cross_numa",
            Self::Unknown => "unknown",
        }
    }
}

/// Classify the PCI distance between two functions. Shared upstream
/// bridges decide first (they are the stronger statement); NUMA nodes
/// decide when the paths do not share a host bridge or were not resolved.
pub fn pci_locality(a: &PciLocation, b: &PciLocation) -> PciLocality {
    if !a.upstream.is_empty() && !b.upstream.is_empty() {
        let shared = a
            .upstream
            .iter()
            .zip(&b.upstream)
            .take_while(|(x, y)| x == y)
            .count();
        match shared {
            0 => {}
            1 => return PciLocality::HostBridge,
            _ => return PciLocality::PcieSwitch,
        }
    }
    match (a.numa_node, b.numa_node) {
        (Some(x), Some(y)) if x == y => PciLocality::SameNuma,
        (Some(_), Some(_)) => PciLocality::CrossNuma,
        _ => PciLocality::Unknown,
    }
}

/// One RDMA device (`/sys/class/infiniband/<name>`): its PCI placement.
/// Ports are listed separately in `InventorySnapshot.ib_ports`, joined by
/// device name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IbDeviceInventory {
    pub name: String,
    /// `None` for devices without a PCI parent (soft RoCE, siw) or when
    /// the `device` link is unreadable.
    #[serde(default)]
    pub pci: Option<PciLocation>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_states_parse_raw_and_symbolic_forms() {
        assert_eq!(PortState::parse("4: ACTIVE"), PortState::Active);
        assert_eq!(PortState::parse("ACTIVE"), PortState::Active);
        assert_eq!(PortState::parse("Active"), PortState::Active);
        assert_eq!(PortState::parse("1: DOWN"), PortState::Down);
        assert_eq!(PortState::parse("5: ACTIVE_DEFER"), PortState::ActiveDefer);
        assert_eq!(PortState::parse("2: INIT"), PortState::Init);
        assert_eq!(PortState::parse("3: ARMED"), PortState::Armed);
        assert_eq!(PortState::parse("unknown"), PortState::Unknown);
        assert_eq!(PortState::parse(""), PortState::Unknown);
    }

    #[test]
    fn phys_states_key_on_the_code() {
        assert_eq!(PhysState::parse("5: LinkUp"), PhysState::LinkUp);
        assert_eq!(PhysState::parse("2: Polling"), PhysState::Polling);
        assert_eq!(PhysState::parse("3: Disabled"), PhysState::Disabled);
        assert_eq!(
            PhysState::parse("4: PortConfigurationTraining"),
            PhysState::PortConfigurationTraining
        );
        assert_eq!(PhysState::parse("9: Whatever"), PhysState::Unknown);
        assert_eq!(PhysState::parse("LinkUp"), PhysState::Unknown);
    }

    #[test]
    fn link_layers_parse_and_keep_roce_only_on_ethernet() {
        assert_eq!(
            LinkLayer::parse("InfiniBand\n", vec![RoceVersion::V2]),
            LinkLayer::Infiniband
        );
        assert_eq!(
            LinkLayer::parse(
                "Ethernet",
                vec![RoceVersion::V2, RoceVersion::V1, RoceVersion::V2]
            ),
            LinkLayer::Ethernet {
                roce_versions: vec![RoceVersion::V1, RoceVersion::V2]
            }
        );
        assert_eq!(LinkLayer::parse("Omni-Path", vec![]), LinkLayer::Unknown);
        assert!(LinkLayer::Infiniband.nccl_capable());
        assert!(
            LinkLayer::Ethernet {
                roce_versions: vec![]
            }
            .nccl_capable()
        );
        assert!(!LinkLayer::Unknown.nccl_capable());
        assert_eq!(
            RoceVersion::parse_gid_type("IB/RoCE v1"),
            Some(RoceVersion::V1)
        );
        assert_eq!(
            RoceVersion::parse_gid_type("RoCE v2\n"),
            Some(RoceVersion::V2)
        );
        assert_eq!(RoceVersion::parse_gid_type(""), None);
    }

    #[test]
    fn link_layer_serde_shape() {
        let ethernet = LinkLayer::Ethernet {
            roce_versions: vec![RoceVersion::V2],
        };
        let json = serde_json::to_string(&ethernet).expect("serialize");
        assert_eq!(json, r#"{"kind":"ethernet","roce_versions":["v2"]}"#);
        assert_eq!(
            serde_json::from_str::<LinkLayer>(&json).expect("decode"),
            ethernet
        );
        assert_eq!(
            serde_json::from_str::<LinkLayer>(r#"{"kind":"ethernet"}"#).expect("decode"),
            LinkLayer::Ethernet {
                roce_versions: vec![]
            }
        );
        assert_eq!(
            serde_json::to_string(&LinkLayer::Infiniband).expect("serialize"),
            r#"{"kind":"infiniband"}"#
        );
    }

    #[test]
    fn rate_strings_parse() {
        let hdr = IbRate::parse("200 Gb/sec (4X HDR)\n").expect("hdr");
        assert_eq!(hdr.gbps, 200.0);
        assert_eq!(hdr.lanes, Some(4));
        assert_eq!(hdr.speed, Some(IbSpeed::Hdr));

        let sdr = IbRate::parse("2.5 Gb/sec (1X SDR)").expect("sdr");
        assert_eq!(
            (sdr.gbps, sdr.lanes, sdr.speed),
            (2.5, Some(1), Some(IbSpeed::Sdr))
        );
        // Older kernels print SDR as a bare width.
        let bare = IbRate::parse("10 Gb/sec (4X)").expect("bare");
        assert_eq!((bare.lanes, bare.speed), (Some(4), Some(IbSpeed::Sdr)));

        let ndr = IbRate::parse("400 Gb/sec (4X NDR)").expect("ndr");
        assert_eq!(ndr.speed, Some(IbSpeed::Ndr));
        let fdr10 = IbRate::parse("40 Gb/sec (4X FDR10)").expect("fdr10");
        assert_eq!(fdr10.speed, Some(IbSpeed::Fdr10));
        let edr1x = IbRate::parse("25 Gb/sec (1X EDR)").expect("edr");
        assert_eq!((edr1x.gbps, edr1x.lanes), (25.0, Some(1)));

        // No parenthetical: the number still counts.
        let plain = IbRate::parse("100 Gb/sec").expect("plain");
        assert_eq!((plain.gbps, plain.lanes, plain.speed), (100.0, None, None));
        // An unknown speed name keeps the number.
        let odd = IbRate::parse("800 Gb/sec (8X GDR)").expect("odd");
        assert_eq!((odd.gbps, odd.lanes, odd.speed), (800.0, Some(8), None));

        assert!(matches!(
            IbRate::parse("fast"),
            Err(IbRateError::Number { .. })
        ));
        assert!(matches!(IbRate::parse(""), Err(IbRateError::Number { .. })));
        assert!(matches!(
            IbRate::parse("-5 Gb/sec"),
            Err(IbRateError::Number { .. })
        ));
        assert!(matches!(
            IbRate::parse("200 MB/sec (4X HDR)"),
            Err(IbRateError::Unit { .. })
        ));
        assert!(matches!(
            IbRate::parse("200"),
            Err(IbRateError::Unit { .. })
        ));
    }

    #[test]
    fn payload_rates_correct_ib_encoding_only() {
        let ib = LinkLayer::Infiniband;
        let eth = LinkLayer::Ethernet {
            roce_versions: vec![RoceVersion::V2],
        };
        assert_eq!(payload_gbps(200.0, Some(IbSpeed::Hdr), &ib), 200.0);
        assert_eq!(payload_gbps(40.0, Some(IbSpeed::Qdr), &ib), 32.0);
        assert_eq!(payload_gbps(10.0, Some(IbSpeed::Sdr), &ib), 8.0);
        assert!((payload_gbps(56.0, Some(IbSpeed::Fdr), &ib) - 56.0 * 64.0 / 66.0).abs() < 1e-9);
        assert_eq!(payload_gbps(40.0, Some(IbSpeed::Fdr10), &ib), 40.0);
        // 40GbE reads as 4X QDR but is already a data rate.
        assert_eq!(payload_gbps(40.0, Some(IbSpeed::Qdr), &eth), 40.0);
        assert_eq!(payload_gbps(100.0, None, &ib), 100.0);
    }

    #[test]
    fn gbps_convert_to_binary_gib() {
        assert!((gbps_to_gib_per_sec(200.0) - 23.283_064_365).abs() < 1e-6);
        assert!((gbps_to_gib_per_sec(8.0 * 1.073_741_824) - 1.0).abs() < 1e-12);
        assert_eq!(gbps_to_gib_per_sec(0.0), 0.0);
    }

    fn at(address: &str, numa: Option<u32>, upstream: &[&str]) -> PciLocation {
        PciLocation {
            address: address.into(),
            numa_node: numa,
            upstream: upstream.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn locality_prefers_shared_bridges_then_numa() {
        let gpu = at(
            "0000:1b:00.0",
            Some(0),
            &["pci0000:16", "0000:16:02.0", "0000:17:00.0", "0000:18:08.0"],
        );
        let nic_switch = at(
            "0000:1c:00.0",
            Some(0),
            &["pci0000:16", "0000:16:02.0", "0000:17:00.0", "0000:18:10.0"],
        );
        let nic_root = at("0000:20:00.0", Some(0), &["pci0000:16", "0000:16:04.0"]);
        let nic_numa = at("0000:3b:00.0", Some(0), &["pci0000:3a", "0000:3a:00.0"]);
        let nic_far = at("0000:9b:00.0", Some(1), &["pci0000:9a", "0000:9a:00.0"]);
        let nic_unresolved = at("0000:5b:00.0", Some(1), &[]);
        let nic_blind = at("0000:5c:00.0", None, &[]);

        assert_eq!(pci_locality(&gpu, &nic_switch), PciLocality::PcieSwitch);
        assert_eq!(pci_locality(&nic_switch, &gpu), PciLocality::PcieSwitch);
        assert_eq!(pci_locality(&gpu, &nic_root), PciLocality::HostBridge);
        assert_eq!(pci_locality(&gpu, &nic_numa), PciLocality::SameNuma);
        assert_eq!(pci_locality(&gpu, &nic_far), PciLocality::CrossNuma);
        assert_eq!(pci_locality(&gpu, &nic_unresolved), PciLocality::CrossNuma);
        assert_eq!(pci_locality(&gpu, &nic_blind), PciLocality::Unknown);
        assert!(PciLocality::PcieSwitch < PciLocality::CrossNuma);
    }
}
