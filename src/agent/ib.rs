//! InfiniBand / RoCE inventory from `/sys/class/infiniband` (injectable
//! root, so tests run against fixture trees).
//!
//! Per device: PCI placement via the `device` link. Per port
//! (`ports/<p>/`): `state`, `phys_state`, `rate`, `link_layer`,
//! `counters/link_downed`, the bound netdevs, and on Ethernet the RoCE
//! versions of the populated GIDs. Every read is best effort: a missing
//! file degrades that field to unknown, never fails the probe.

use std::collections::BTreeSet;
use std::path::Path;

use crate::agent::pci;
use crate::proto::{IbDeviceInventory, IbPortInventory, IbRate, LinkLayer, PhysState, RoceVersion};

/// Real sysfs root.
pub const INFINIBAND_ROOT: &str = "/sys/class/infiniband";

/// Devices and ports, each sorted by name (ports by (device, port)).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IbInventory {
    pub devices: Vec<IbDeviceInventory>,
    pub ports: Vec<IbPortInventory>,
}

pub fn probe(ib_root: &Path) -> IbInventory {
    let Ok(entries) = std::fs::read_dir(ib_root) else {
        return IbInventory::default();
    };
    let mut inventory = IbInventory::default();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let device_dir = entry.path();
        inventory.devices.push(IbDeviceInventory {
            name: name.clone(),
            pci: pci::locate(&device_dir.join("device")),
        });
        let Ok(port_dirs) = std::fs::read_dir(device_dir.join("ports")) else {
            continue;
        };
        for port_dir in port_dirs.flatten() {
            let Ok(port) = port_dir.file_name().to_string_lossy().parse::<u32>() else {
                continue;
            };
            inventory
                .ports
                .push(probe_port(&device_dir, &name, port, &port_dir.path()));
        }
    }
    inventory.devices.sort_by(|a, b| a.name.cmp(&b.name));
    inventory
        .ports
        .sort_by(|a, b| (&a.device, a.port).cmp(&(&b.device, b.port)));
    inventory
}

fn probe_port(device_dir: &Path, device: &str, port: u32, port_dir: &Path) -> IbPortInventory {
    // `state` reads as "4: ACTIVE"; the wire keeps the symbolic half.
    let state = read_trimmed(&port_dir.join("state"))
        .map(|raw| {
            raw.split_once(':')
                .map(|(_, name)| name.trim().to_string())
                .unwrap_or(raw)
        })
        .unwrap_or_else(|| "unknown".to_string());
    let rate = read_trimmed(&port_dir.join("rate")).and_then(|raw| IbRate::parse(&raw).ok());
    let link_layer = match read_trimmed(&port_dir.join("link_layer")) {
        Some(raw) => {
            let roce = if raw == "Ethernet" {
                roce_versions(port_dir)
            } else {
                Vec::new()
            };
            LinkLayer::parse(&raw, roce)
        }
        None => LinkLayer::Unknown,
    };
    IbPortInventory {
        device: device.to_string(),
        port,
        state,
        rate_gbps: rate.map(|rate| rate.gbps),
        link_downed_count: read_trimmed(&port_dir.join("counters/link_downed"))
            .and_then(|raw| raw.parse::<u64>().ok()),
        lanes: rate.and_then(|rate| rate.lanes),
        speed: rate.and_then(|rate| rate.speed),
        link_layer,
        phys_state: read_trimmed(&port_dir.join("phys_state"))
            .map_or(PhysState::Unknown, |raw| PhysState::parse(&raw)),
        netdevs: netdevs(device_dir, port, port_dir),
    }
}

/// RoCE versions of the populated GIDs (`gid_attrs/types/<i>`; unpopulated
/// entries are unreadable or empty).
fn roce_versions(port_dir: &Path) -> Vec<RoceVersion> {
    let Ok(entries) = std::fs::read_dir(port_dir.join("gid_attrs/types")) else {
        return Vec::new();
    };
    let versions: BTreeSet<RoceVersion> = entries
        .flatten()
        .filter_map(|entry| read_trimmed(&entry.path()))
        .filter_map(|raw| RoceVersion::parse_gid_type(&raw))
        .collect();
    versions.into_iter().collect()
}

/// Netdevs bound to `port`: the PCI function's `net/*` entries whose
/// `dev_port` (0-based) is `port - 1` (an entry without `dev_port` counts
/// for every port); failing that, the names in the port's GID table
/// (`gid_attrs/ndevs/<i>`). Sorted, deduplicated.
fn netdevs(device_dir: &Path, port: u32, port_dir: &Path) -> Vec<String> {
    let mut names = BTreeSet::new();
    if let Ok(entries) = std::fs::read_dir(device_dir.join("device/net")) {
        for entry in entries.flatten() {
            let bound = match read_trimmed(&entry.path().join("dev_port")) {
                Some(raw) => raw
                    .parse::<u32>()
                    .is_ok_and(|dev_port| dev_port.checked_add(1) == Some(port)),
                None => true,
            };
            if bound {
                names.insert(entry.file_name().to_string_lossy().into_owned());
            }
        }
    }
    if names.is_empty()
        && let Ok(entries) = std::fs::read_dir(port_dir.join("gid_attrs/ndevs"))
    {
        names.extend(
            entries
                .flatten()
                .filter_map(|entry| read_trimmed(&entry.path())),
        );
    }
    names.into_iter().collect()
}

fn read_trimmed(path: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_root_is_empty_not_an_error() {
        assert_eq!(
            probe(Path::new("/nonexistent/gauntlet/infiniband")),
            IbInventory::default()
        );
    }
}
