//! Which IB/RoCE ports NCCL would use: a pure replay of NCCL's IB device
//! scan over one host's inventory under the run's resolved NCCL env.
//!
//! The agent inventories ports as facts (`proto::IbPortInventory`); the
//! orchestrator knows the env. [`NcclIbConfig`] is the slice of the env
//! that decides the IB transport's port list; [`summarize`] produces the
//! per-host [`NcclNicSummary`] (types in `proto::nccl_nics`).
//!
//! Modelled, in NCCL's order:
//! 1. `NCCL_NET` (case-insensitive): "Socket" selects the socket
//!    transport, so no IB port is used. "IB" or unset keep the IB
//!    transport. Any other value names an external net plugin, which is
//!    not modelled: the IB selection is reported as if IB ran, and the
//!    summary carries a caveat.
//! 2. `NCCL_IB_DISABLE`, read like NCCL's `ncclLoadParam`: C
//!    `strtoll(s, &end, 0)` (hex `0x`, octal `0` prefixes, trailing text
//!    ignored); no digits or overflow keeps the default 0. Non-zero
//!    disables the IB transport.
//! 3. Only ports whose logical state is ACTIVE.
//! 4. Only InfiniBand and Ethernet (RoCE) link layers.
//! 5. The `NCCL_IB_HCA` filter (see [`hca`]); unset passes everything.
//!
//! Not modelled: external plugin behaviour, `NCCL_IB_HCA` values from the
//! node's own environment or `/etc/nccl.conf` (only `[nccl] env` is known
//! here), NIC fusion (`NCCL_IB_MERGE_NICS`), and per-rank NIC assignment.

pub mod hca;

use crate::nccl_env::NcclEnv;
use crate::proto::nccl_nics::{
    ExcludedPort, GpuLocality, IB_DISABLE, IB_HCA, NET, NetChoice, PortExclusion, SelectedPort,
};
use crate::proto::{IbPortInventory, InventorySnapshot, NcclNicSummary, PortState, pci_locality};
use hca::HcaFilter;

/// The NCCL knobs that decide the IB transport's port list.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NcclIbConfig {
    /// The raw `NCCL_IB_HCA` value, recorded on summaries; `None` = unset.
    hca_value: Option<String>,
    hca: HcaFilter,
    ib_disabled: bool,
    net: NetChoice,
}

impl NcclIbConfig {
    pub fn new(hca: Option<&str>, ib_disable: Option<&str>, net: Option<&str>) -> Self {
        Self {
            hca_value: hca.map(str::to_string),
            hca: hca.map_or_else(HcaFilter::unset, HcaFilter::parse),
            ib_disabled: ib_disable.is_some_and(ib_disable_is_set),
            net: NetChoice::parse(net),
        }
    }

    pub fn from_env(env: &NcclEnv) -> Self {
        Self::new(env.get(IB_HCA), env.get(IB_DISABLE), env.get(NET))
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

    pub fn net(&self) -> &NetChoice {
        &self.net
    }
}

/// `NCCL_IB_DISABLE` as `ncclLoadParam` reads it: non-zero after C
/// `strtoll(s, &end, 0)`; unparseable (no digits) or out of range keeps the
/// default 0.
fn ib_disable_is_set(raw: &str) -> bool {
    strtoll_base0(raw).is_some_and(|value| value != 0)
}

/// C `strtoll(s, &end, 0)` as NCCL checks it: `None` when no digits were
/// consumed (`end == s`) or the value is out of range (`errno == ERANGE`).
/// Leading C whitespace, an optional sign, then `0x`/`0X` + hex digit for
/// base 16, a leading `0` for base 8, else base 10; parsing stops at the
/// first character not valid in the base.
fn strtoll_base0(text: &str) -> Option<i64> {
    let bytes = text.as_bytes();
    let mut index = bytes
        .iter()
        .take_while(|byte| matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r'))
        .count();
    let negative = match bytes.get(index) {
        Some(b'-') => {
            index += 1;
            true
        }
        Some(b'+') => {
            index += 1;
            false
        }
        _ => false,
    };
    let hex_prefix = bytes.get(index) == Some(&b'0')
        && matches!(bytes.get(index + 1), Some(b'x' | b'X'))
        && bytes.get(index + 2).is_some_and(u8::is_ascii_hexdigit);
    let (radix, start) = if hex_prefix {
        (16, index + 2)
    } else if bytes.get(index) == Some(&b'0') {
        (8, index)
    } else {
        (10, index)
    };
    let digits: Vec<u32> = bytes[start..]
        .iter()
        .map_while(|byte| char::from(*byte).to_digit(radix))
        .collect();
    if digits.is_empty() {
        return None;
    }
    let limit = if negative {
        u128::from(i64::MAX.unsigned_abs()) + 1
    } else {
        u128::from(i64::MAX.unsigned_abs())
    };
    let mut magnitude: u128 = 0;
    for digit in digits {
        magnitude = magnitude * u128::from(radix) + u128::from(digit);
        if magnitude > limit {
            return None;
        }
    }
    let magnitude = i128::try_from(magnitude).ok()?;
    i64::try_from(if negative { -magnitude } else { magnitude }).ok()
}

/// Classify one port the way NCCL's device scan would.
pub fn classify_port(config: &NcclIbConfig, port: &IbPortInventory) -> Result<(), PortExclusion> {
    if config.net() == &NetChoice::Socket {
        return Err(PortExclusion::NetSocket);
    }
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
                let mut gpu_locality: Vec<GpuLocality> = pci
                    .map(|nic| {
                        inventory
                            .gpus
                            .iter()
                            .filter_map(|gpu| {
                                gpu.pci.as_ref().map(|placement| GpuLocality {
                                    gpu: gpu.index,
                                    locality: pci_locality(nic, placement),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                gpu_locality.sort_by_key(|entry| entry.gpu);
                selected.push(SelectedPort {
                    device: port.device.clone(),
                    port: port.port,
                    link_layer: port.link_layer.clone(),
                    rate: port.rate.clone(),
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
    NcclNicSummary {
        hca: config.hca_value().map(str::to_string),
        net: config.net().clone(),
        ib_disabled: config.ib_disabled(),
        selected,
        excluded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strtoll_mirrors_c_base_zero() {
        assert_eq!(strtoll_base0("1"), Some(1));
        assert_eq!(strtoll_base0("0"), Some(0));
        assert_eq!(strtoll_base0("0x1"), Some(1));
        assert_eq!(strtoll_base0("0X1f"), Some(31));
        assert_eq!(strtoll_base0("010"), Some(8));
        assert_eq!(strtoll_base0("08"), Some(0), "octal stops at 8");
        assert_eq!(strtoll_base0("0x"), Some(0), "the 0 is consumed");
        assert_eq!(strtoll_base0("1x"), Some(1), "trailing text ignored");
        assert_eq!(strtoll_base0("  -1"), Some(-1));
        assert_eq!(strtoll_base0("+5"), Some(5));
        assert_eq!(strtoll_base0("x1"), None);
        assert_eq!(strtoll_base0("yes"), None);
        assert_eq!(strtoll_base0(""), None);
        assert_eq!(strtoll_base0("-"), None);
        assert_eq!(strtoll_base0("9223372036854775807"), Some(i64::MAX));
        assert_eq!(strtoll_base0("-9223372036854775808"), Some(i64::MIN));
        assert_eq!(strtoll_base0("9223372036854775808"), None, "ERANGE");
        assert_eq!(strtoll_base0("99999999999999999999999"), None, "ERANGE");
    }

    #[test]
    fn ib_disable_follows_ncclloadparam() {
        for disabled in ["1", "2", "-1", "1x", "0x1", "010", " 1"] {
            assert!(ib_disable_is_set(disabled), "{disabled:?}");
        }
        for enabled in [
            "0",
            "0x0",
            "00",
            "08",
            "yes",
            "x1",
            "",
            "99999999999999999999",
        ] {
            assert!(!ib_disable_is_set(enabled), "{enabled:?}");
        }
        assert!(NcclIbConfig::new(None, Some("1"), None).ib_disabled());
        assert!(!NcclIbConfig::new(None, None, None).ib_disabled());
    }
}
