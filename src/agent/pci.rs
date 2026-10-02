//! PCI placement from sysfs: address, NUMA node and the bridges above a
//! device, read from the canonical `/sys/devices/pci<domain>:<bus>/...`
//! path. Roots are injectable so tests run against fixture trees.

use std::path::Path;

use crate::agent::gpu_occupancy::PciBusId;
use crate::proto::PciLocation;

/// Real sysfs root of PCI functions by address.
pub const PCI_DEVICES_ROOT: &str = "/sys/bus/pci/devices";

/// Placement of the PCI function that `device_link` (a sysfs symlink such
/// as `/sys/class/infiniband/mlx5_0/device` or
/// `/sys/bus/pci/devices/0000:3b:00.0`) resolves to. `None` when the link
/// does not resolve under a PCI host bridge (soft RoCE, virtual devices).
pub fn locate(device_link: &Path) -> Option<PciLocation> {
    let canonical = std::fs::canonicalize(device_link).ok()?;
    let components: Vec<String> = canonical
        .iter()
        .map(|component| component.to_string_lossy().into_owned())
        .collect();
    let host_bridge = components.iter().position(|name| is_host_bridge(name))?;
    let (address, upstream) = components[host_bridge..].split_last()?;
    if upstream.is_empty() || address.parse::<PciBusId>().is_err() {
        return None;
    }
    Some(PciLocation {
        address: address.clone(),
        numa_node: read_numa_node(&canonical),
        upstream: upstream.to_vec(),
    })
}

/// Resolve a location known only by address (from nvidia-smi) against
/// `pci_root`. Leaves it untouched when sysfs has nothing better.
pub fn resolve(pci_root: &Path, location: &mut PciLocation) {
    if let Some(resolved) = locate(&pci_root.join(&location.address)) {
        *location = resolved;
    }
}

/// `pci0000:3a` / `pci10000:00` (VMD): the host-bridge directory.
fn is_host_bridge(name: &str) -> bool {
    name.strip_prefix("pci")
        .and_then(|rest| rest.split_once(':'))
        .is_some_and(|(domain, bus)| {
            !domain.is_empty()
                && !bus.is_empty()
                && domain.bytes().all(|b| b.is_ascii_hexdigit())
                && bus.bytes().all(|b| b.is_ascii_hexdigit())
        })
}

/// `numa_node` holds -1 when the platform reports no affinity.
fn read_numa_node(device_dir: &Path) -> Option<u32> {
    let raw = std::fs::read_to_string(device_dir.join("numa_node")).ok()?;
    let node: i64 = raw.trim().parse().ok()?;
    u32::try_from(node).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_bridge_names() {
        assert!(is_host_bridge("pci0000:00"));
        assert!(is_host_bridge("pci10000:e0"));
        assert!(!is_host_bridge("pci"));
        assert!(!is_host_bridge("pci_express"));
        assert!(!is_host_bridge("0000:00:01.0"));
        assert!(!is_host_bridge("pcizz:00"));
    }

    #[test]
    fn missing_links_are_none() {
        assert_eq!(locate(Path::new("/nonexistent/gauntlet/device")), None);
    }
}
