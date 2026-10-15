//! InfiniBand / RoCE inventory from `/sys/class/infiniband` (injectable
//! root, so tests run against fixture trees).
//!
//! Per device: PCI placement via the `device` link. Per port
//! (`ports/<p>/`): `state`, `phys_state`, `rate`, `link_layer`,
//! `counters/link_downed`, the bound netdevs, and on Ethernet the RoCE
//! versions of the populated GIDs. Every read is best effort: a missing
//! file degrades that field to unknown, never fails the probe. A rate that
//! is present but does not parse is kept with its typed error (and logged).

use std::collections::BTreeSet;
use std::path::Path;

use tracing::warn;

use crate::agent::pci;
use crate::agent::sysfs::read_trimmed;
use crate::proto::{
    IbDeviceInventory, IbPortInventory, LinkLayer, PhysState, PortRate, RoceVersion,
};

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
    let state = read_trimmed(port_dir.join("state"))
        .map(|raw| {
            raw.split_once(':')
                .map(|(_, name)| name.trim().to_string())
                .unwrap_or(raw)
        })
        .unwrap_or_else(|| "unknown".to_string());
    let rate = PortRate::from_sysfs(read_trimmed(port_dir.join("rate")).as_deref());
    if let PortRate::Unparseable { error } = &rate {
        warn!(device, port, %error, "unparseable ib port rate");
    }
    let link_layer = match read_trimmed(port_dir.join("link_layer")) {
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
    let netdevs = netdevs(device_dir, port, port_dir, &link_layer);
    IbPortInventory {
        device: device.to_string(),
        port,
        state,
        rate,
        link_downed_count: read_trimmed(port_dir.join("counters/link_downed"))
            .and_then(|raw| raw.parse::<u64>().ok()),
        link_layer,
        phys_state: read_trimmed(port_dir.join("phys_state"))
            .map_or(PhysState::Unknown, |raw| PhysState::parse(&raw)),
        netdevs,
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
        .filter_map(|entry| read_trimmed(entry.path()))
        .filter_map(|raw| RoceVersion::parse_gid_type(&raw))
        .collect();
    versions.into_iter().collect()
}

/// Netdevs bound to `port`, sorted and deduplicated.
///
/// RoCE (Ethernet link layer): the names in the port's GID table
/// (`gid_attrs/ndevs/<i>`, non-empty). That is the interface RoCE traffic
/// actually uses: for a LAG device (`mlx5_bond_0`) it is the bond, not a
/// member PF, and in switchdev mode it is the uplink, never a VF
/// representor.
///
/// Otherwise (IPoIB on InfiniBand, or a RoCE port with an empty GID
/// table): the PCI function's `net/*` entries whose `dev_port` (0-based)
/// equals `port - 1`, skipping VF/SF representors (`phys_port_name` like
/// `pf0vf3` / `pf0sf1`). An entry without `dev_port` is not attributed.
fn netdevs(device_dir: &Path, port: u32, port_dir: &Path, link_layer: &LinkLayer) -> Vec<String> {
    if matches!(link_layer, LinkLayer::Ethernet { .. }) {
        let from_gids = gid_ndevs(port_dir);
        if !from_gids.is_empty() {
            return from_gids;
        }
    }
    let Ok(entries) = std::fs::read_dir(device_dir.join("device/net")) else {
        return Vec::new();
    };
    let names: BTreeSet<String> = entries
        .flatten()
        .filter(|entry| {
            let dir = entry.path();
            let bound = read_trimmed(dir.join("dev_port"))
                .and_then(|raw| raw.parse::<u32>().ok())
                .is_some_and(|dev_port| dev_port.checked_add(1) == Some(port));
            bound && !is_representor(&dir)
        })
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.into_iter().collect()
}

fn gid_ndevs(port_dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(port_dir.join("gid_attrs/ndevs")) else {
        return Vec::new();
    };
    let names: BTreeSet<String> = entries
        .flatten()
        .filter_map(|entry| read_trimmed(entry.path()))
        .collect();
    names.into_iter().collect()
}

/// A switchdev VF/SF representor: `phys_port_name` names a function
/// (`pf0vf3`, `c1pf0vf3`, `pf0sf1`), where an uplink reads `p0`.
fn is_representor(netdev_dir: &Path) -> bool {
    read_trimmed(netdev_dir.join("phys_port_name"))
        .is_some_and(|name| name.contains("vf") || name.contains("sf"))
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
