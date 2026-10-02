//! InfiniBand inventory and the NCCL NIC ceiling: sysfs fixture trees
//! (IB HDR, RoCE, a storage NIC excluded via `^`, a down port, mixed
//! rates, VMD, RoCE LAG, switchdev), NCCL_IB_HCA / NCCL_NET /
//! NCCL_IB_DISABLE selection, the derived nccl_nics results, report
//! integration, and serde compatibility. Nothing here needs RDMA hardware.

mod common;

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

use gauntlet::agent::{ib, pci};
use gauntlet::config::FleetConfig;
use gauntlet::nccl_ib::{NcclIbConfig, summarize};
use gauntlet::orchestrator::collect::Collector;
use gauntlet::orchestrator::derive::{InventoryDerivation, derive_inventory_events};
use gauntlet::proto::nccl_nics::{
    CEILING_METRIC, NetChoice, PortExclusion, SelectionLinkLayer, consistency,
};
use gauntlet::proto::{
    AgentEvent, GpuInventory, GpuOccupancy, IbRateError, IbSpeed, InventorySnapshot, LinkLayer,
    PciLocality, PciLocation, PhysState, PortRate, RoceVersion, Scope, TestId, TestOutcome, Unit,
    decode_event, encode_event, gbps_to_gib_per_sec, pci_locality,
};
use gauntlet::report::{self, RunResults, Verdict};

// ---------------------------------------------------------------------------
// Fixture sysfs
// ---------------------------------------------------------------------------

fn write(path: &Path, content: &str) {
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    fs::write(path, content).expect("write fixture");
}

/// A PCI function under `<root>/devices/<chain...>` with a numa_node file.
fn pci_device(root: &Path, chain: &[&str], numa: i32) -> PathBuf {
    let dir = chain
        .iter()
        .fold(root.join("devices"), |dir, part| dir.join(part));
    write(&dir.join("numa_node"), &format!("{numa}\n"));
    // The by-address view, as /sys/bus/pci/devices has it.
    let by_address = root.join("bus/pci/devices");
    fs::create_dir_all(&by_address).expect("mkdir");
    symlink(&dir, by_address.join(chain.last().expect("address"))).expect("symlink");
    dir
}

/// A netdev under a PCI function: `net/<name>/{dev_port,phys_port_name}`.
fn netdev(pci_dir: &Path, name: &str, dev_port: Option<u32>, phys_port_name: Option<&str>) {
    let dir = pci_dir.join("net").join(name);
    fs::create_dir_all(&dir).expect("mkdir");
    if let Some(dev_port) = dev_port {
        write(&dir.join("dev_port"), &format!("{dev_port}\n"));
    }
    if let Some(phys) = phys_port_name {
        write(&dir.join("phys_port_name"), &format!("{phys}\n"));
    }
}

struct Port<'a> {
    state: &'a str,
    phys_state: &'a str,
    rate: &'a str,
    link_layer: &'a str,
    gid_types: &'a [&'a str],
    gid_ndevs: &'a [&'a str],
}

impl Port<'_> {
    fn active_ib(rate: &str) -> Port<'_> {
        Port {
            state: "4: ACTIVE",
            phys_state: "5: LinkUp",
            rate,
            link_layer: "InfiniBand",
            gid_types: &[],
            gid_ndevs: &[],
        }
    }
}

fn ib_device(root: &Path, name: &str, pci_dir: &Path, port: Port) {
    let device = root.join("class/infiniband").join(name);
    fs::create_dir_all(&device).expect("mkdir");
    symlink(pci_dir, device.join("device")).expect("symlink");
    let port_dir = device.join("ports/1");
    write(&port_dir.join("state"), &format!("{}\n", port.state));
    write(
        &port_dir.join("phys_state"),
        &format!("{}\n", port.phys_state),
    );
    write(&port_dir.join("rate"), &format!("{}\n", port.rate));
    write(
        &port_dir.join("link_layer"),
        &format!("{}\n", port.link_layer),
    );
    write(&port_dir.join("counters/link_downed"), "0\n");
    for (index, gid_type) in port.gid_types.iter().enumerate() {
        write(&port_dir.join(format!("gid_attrs/types/{index}")), gid_type);
    }
    for (index, ndev) in port.gid_ndevs.iter().enumerate() {
        write(&port_dir.join(format!("gid_attrs/ndevs/{index}")), ndev);
    }
}

/// A scratch root that cleans up after itself.
struct Root(PathBuf);

impl Root {
    fn new(tag: &str) -> Self {
        Self(common::scratch_dir(tag))
    }

    fn ib_root(&self) -> PathBuf {
        self.0.join("class/infiniband")
    }

    fn pci_root(&self) -> PathBuf {
        self.0.join("bus/pci/devices")
    }
}

impl Drop for Root {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// One host's sysfs:
/// - GPU 0 at 0000:1b:00.0, NUMA 0, behind PCIe switch 0000:17:00.0.
/// - GPU 1 at 0000:9c:00.0, NUMA 1, its own host bridge.
/// - mlx5_0: IB HDR 200G, active, under GPU 0's switch, IPoIB netdev.
/// - mlx5_1: RoCE (Ethernet) 100G, active, NUMA 0 on another host bridge.
/// - mlx5_2: IB EDR 100G storage NIC, active, NUMA 1, IPoIB netdev.
/// - mlx5_3: IB port that is DOWN (Polling: no peer).
struct Fixture {
    root: Root,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let root = Root::new(tag);
        let base = &root.0;
        let switch = ["pci0000:16", "0000:16:02.0", "0000:17:00.0"];
        pci_device(
            base,
            &[
                switch[0],
                switch[1],
                switch[2],
                "0000:18:08.0",
                "0000:1b:00.0",
            ],
            0,
        );
        pci_device(base, &["pci0000:9a", "0000:9a:02.0", "0000:9c:00.0"], 1);

        let mlx5_0 = pci_device(
            base,
            &[
                switch[0],
                switch[1],
                switch[2],
                "0000:18:10.0",
                "0000:1c:00.0",
            ],
            0,
        );
        netdev(&mlx5_0, "ibp28s0", Some(0), None);
        ib_device(
            base,
            "mlx5_0",
            &mlx5_0,
            Port::active_ib("200 Gb/sec (4X HDR)"),
        );

        let mlx5_1 = pci_device(base, &["pci0000:3a", "0000:3a:00.0", "0000:3b:00.0"], 0);
        netdev(&mlx5_1, "ens3np0", Some(0), Some("p0"));
        ib_device(
            base,
            "mlx5_1",
            &mlx5_1,
            Port {
                state: "4: ACTIVE",
                phys_state: "5: LinkUp",
                rate: "100 Gb/sec (4X EDR)",
                link_layer: "Ethernet",
                gid_types: &["IB/RoCE v1\n", "RoCE v2\n", "IB/RoCE v1\n", "RoCE v2\n", ""],
                gid_ndevs: &["ens3np0\n", "ens3np0\n", ""],
            },
        );

        let mlx5_2 = pci_device(base, &["pci0000:9a", "0000:9a:00.0", "0000:9b:00.0"], 1);
        netdev(&mlx5_2, "ibp155s0", Some(0), None);
        // A netdev without dev_port is not attributed to any port.
        netdev(&mlx5_2, "mystery0", None, None);
        ib_device(
            base,
            "mlx5_2",
            &mlx5_2,
            Port::active_ib("100 Gb/sec (4X EDR)"),
        );

        let mlx5_3 = pci_device(base, &["pci0000:9a", "0000:9a:04.0", "0000:9d:00.0"], 1);
        ib_device(
            base,
            "mlx5_3",
            &mlx5_3,
            Port {
                state: "1: DOWN",
                phys_state: "2: Polling",
                rate: "10 Gb/sec (4X SDR)",
                link_layer: "InfiniBand",
                gid_types: &[],
                gid_ndevs: &[],
            },
        );
        Self { root }
    }

    fn gpu(&self, index: u32, address: &str) -> GpuInventory {
        let mut location = PciLocation::address_only(address);
        pci::resolve(&self.root.pci_root(), &mut location);
        GpuInventory {
            index,
            name: "NVIDIA H100".into(),
            uuid: format!("GPU-{index}"),
            vbios: "96.00".into(),
            mem_total_bytes: 80 << 30,
            ecc_volatile_errors: Some(0),
            remapped_rows_pending: Some(false),
            pcie_gen_current: Some(5),
            pcie_gen_max: Some(5),
            pcie_width_current: Some(16),
            pcie_width_max: Some(16),
            nvlinks_active: None,
            persistence_mode: Some(true),
            occupancy: GpuOccupancy::default(),
            pci: Some(location),
        }
    }

    fn inventory(&self, host: &str) -> InventorySnapshot {
        let probed = ib::probe(&self.root.ib_root());
        InventorySnapshot {
            hostname: host.into(),
            kernel: "6.8.0".into(),
            cpu_model: "TestCPU".into(),
            logical_cores: 64,
            numa_nodes: 2,
            mem_total_bytes: 1 << 40,
            cpu_governor: Some("performance".into()),
            clock_offset_ms: None,
            nvidia_driver: Some("560.35.03".into()),
            cuda_version: Some("12.6".into()),
            gpus: vec![self.gpu(0, "0000:1b:00.0"), self.gpu(1, "0000:9c:00.0")],
            nics: vec![],
            ib_ports: probed.ports,
            ib_devices: probed.devices,
            xid_errors: vec![],
            gpu_libs: BTreeMap::new(),
            cuda_visible_gpus: Some(2),
        }
    }
}

fn hca(value: &str) -> NcclIbConfig {
    NcclIbConfig::new(Some(value), None, None)
}

fn unset() -> NcclIbConfig {
    NcclIbConfig::new(None, None, None)
}

fn selected_ids(summary: &gauntlet::proto::NcclNicSummary) -> Vec<String> {
    summary
        .selected
        .iter()
        .map(|port| format!("{}:{}", port.device, port.port))
        .collect()
}

fn known(gbps: f64, lanes: u32, speed: IbSpeed) -> PortRate {
    PortRate::Known {
        gbps,
        lanes: Some(lanes),
        speed: Some(speed),
    }
}

// ---------------------------------------------------------------------------
// Agent-side probe
// ---------------------------------------------------------------------------

#[test]
fn the_probe_reads_every_new_port_fact() {
    let fixture = Fixture::new("ib-probe");
    let probed = ib::probe(&fixture.root.ib_root());

    let names: Vec<&str> = probed.ports.iter().map(|p| p.device.as_str()).collect();
    assert_eq!(names, ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3"]);

    let hdr = &probed.ports[0];
    assert_eq!(hdr.state, "ACTIVE");
    assert_eq!(hdr.rate, known(200.0, 4, IbSpeed::Hdr));
    assert_eq!(hdr.link_layer, LinkLayer::Infiniband);
    assert_eq!(hdr.phys_state, PhysState::LinkUp);
    assert_eq!(hdr.netdevs, ["ibp28s0"]);
    assert_eq!(hdr.link_downed_count, Some(0));

    let roce = &probed.ports[1];
    assert_eq!(
        roce.link_layer,
        LinkLayer::Ethernet {
            roce_versions: vec![RoceVersion::V1, RoceVersion::V2]
        }
    );
    // GID-table ndevs, deduplicated, empties dropped.
    assert_eq!(roce.netdevs, ["ens3np0"]);

    // IPoIB by dev_port; the netdev without dev_port is not attributed.
    assert_eq!(probed.ports[2].netdevs, ["ibp155s0"]);

    let down = &probed.ports[3];
    assert_eq!(down.state, "DOWN");
    assert_eq!(down.phys_state, PhysState::Polling);
    assert_eq!(down.rate, known(10.0, 4, IbSpeed::Sdr));
}

#[test]
fn the_probe_places_devices_on_the_pci_tree() {
    let fixture = Fixture::new("ib-pci");
    let probed = ib::probe(&fixture.root.ib_root());
    let mlx5_0 = probed.devices[0].pci.as_ref().expect("pci");
    assert_eq!(mlx5_0.address, "0000:1c:00.0");
    assert_eq!(mlx5_0.numa_node, Some(0));
    assert_eq!(
        mlx5_0.upstream,
        ["pci0000:16", "0000:16:02.0", "0000:17:00.0", "0000:18:10.0"]
    );
    let mlx5_2 = probed.devices[2].pci.as_ref().expect("pci");
    assert_eq!(mlx5_2.numa_node, Some(1));
    assert_eq!(mlx5_2.upstream, ["pci0000:9a", "0000:9a:00.0"]);

    // GPUs resolve through /sys/bus/pci/devices.
    let placement = fixture.gpu(0, "0000:1b:00.0").pci.expect("pci");
    assert_eq!(placement.numa_node, Some(0));
    assert_eq!(placement.upstream.len(), 4);

    // An address sysfs does not know stays address-only.
    let mut unknown = PciLocation::address_only("0000:ff:00.0");
    pci::resolve(&fixture.root.pci_root(), &mut unknown);
    assert_eq!(unknown, PciLocation::address_only("0000:ff:00.0"));
}

#[test]
fn vmd_domains_anchor_at_the_innermost_host_bridge() {
    // The review's paths: a GPU and a NIC on different root ports of one
    // Intel VMD domain (pci10000:e0), itself behind 0000:00:0e.0.
    let root = Root::new("ib-vmd");
    let vmd = ["pci0000:00", "0000:00:0e.0", "pci10000:e0"];
    let gpu = pci_device(
        &root.0,
        &[vmd[0], vmd[1], vmd[2], "10000:e0:1d.0", "10000:e1:00.0"],
        0,
    );
    let nic = pci_device(
        &root.0,
        &[vmd[0], vmd[1], vmd[2], "10000:e0:1b.0", "10000:e2:00.0"],
        0,
    );
    let gpu = pci::locate(&gpu).expect("gpu placement");
    let nic = pci::locate(&nic).expect("nic placement");
    assert_eq!(gpu.address, "10000:e1:00.0");
    assert_eq!(gpu.upstream, ["pci10000:e0", "10000:e0:1d.0"]);
    assert_eq!(nic.upstream, ["pci10000:e0", "10000:e0:1b.0"]);
    let locality = pci_locality(&gpu, &nic);
    assert_ne!(locality, PciLocality::PcieSwitch);
    assert_eq!(locality, PciLocality::HostBridge);

    // Two functions under one switch inside the VMD domain still share it.
    let a = pci_device(
        &root.0,
        &[
            vmd[0],
            vmd[1],
            vmd[2],
            "10000:e0:1c.0",
            "10000:e3:00.0",
            "10000:e4:00.0",
        ],
        0,
    );
    let b = pci_device(
        &root.0,
        &[
            vmd[0],
            vmd[1],
            vmd[2],
            "10000:e0:1c.0",
            "10000:e3:00.0",
            "10000:e5:00.0",
        ],
        0,
    );
    let (a, b) = (pci::locate(&a).expect("a"), pci::locate(&b).expect("b"));
    assert_eq!(pci_locality(&a, &b), PciLocality::PcieSwitch);
}

#[test]
fn roce_lag_reports_the_bond_not_a_member_pf() {
    let root = Root::new("ib-lag");
    let pf0 = pci_device(&root.0, &["pci0000:3a", "0000:3a:00.0", "0000:3b:00.0"], 0);
    // The LAG device sits on PF0; its function also exposes the member PF.
    netdev(&pf0, "ens1f0np0", Some(0), Some("p0"));
    ib_device(
        &root.0,
        "mlx5_bond_0",
        &pf0,
        Port {
            state: "4: ACTIVE",
            phys_state: "5: LinkUp",
            rate: "200 Gb/sec (4X HDR)",
            link_layer: "Ethernet",
            gid_types: &["IB/RoCE v1\n", "RoCE v2\n"],
            gid_ndevs: &["bond0\n", "bond0\n"],
        },
    );
    let probed = ib::probe(&root.ib_root());
    assert_eq!(probed.ports[0].netdevs, ["bond0"]);
}

#[test]
fn switchdev_representors_are_never_attributed_to_a_port() {
    let root = Root::new("ib-switchdev");
    let pf = pci_device(&root.0, &["pci0000:3a", "0000:3a:00.0", "0000:3b:00.0"], 0);
    netdev(&pf, "ens1f0np0", Some(0), Some("p0"));
    netdev(&pf, "eth4", Some(0), Some("pf0vf0"));
    netdev(&pf, "eth5", Some(0), Some("pf0vf1"));
    netdev(&pf, "eth6", Some(0), Some("pf0sf8"));
    ib_device(
        &root.0,
        "mlx5_0",
        &pf,
        Port {
            state: "4: ACTIVE",
            phys_state: "5: LinkUp",
            rate: "100 Gb/sec (4X EDR)",
            link_layer: "Ethernet",
            gid_types: &["RoCE v2\n"],
            gid_ndevs: &["ens1f0np0\n"],
        },
    );
    // A second RoCE device whose GID table is empty: the device/net
    // fallback must still skip the representors.
    let pf1 = pci_device(&root.0, &["pci0000:3a", "0000:3a:00.0", "0000:3b:00.1"], 0);
    netdev(&pf1, "ens1f1np1", Some(0), Some("p1"));
    netdev(&pf1, "eth7", Some(0), Some("pf1vf0"));
    ib_device(
        &root.0,
        "mlx5_1",
        &pf1,
        Port {
            state: "4: ACTIVE",
            phys_state: "5: LinkUp",
            rate: "100 Gb/sec (4X EDR)",
            link_layer: "Ethernet",
            gid_types: &[],
            gid_ndevs: &[],
        },
    );
    let probed = ib::probe(&root.ib_root());
    assert_eq!(probed.ports[0].netdevs, ["ens1f0np0"]);
    assert_eq!(probed.ports[1].netdevs, ["ens1f1np1"]);
}

#[test]
fn an_unparseable_rate_keeps_its_typed_error() {
    let root = Root::new("ib-badrate");
    let pf = pci_device(&root.0, &["pci0000:3a", "0000:3a:00.0", "0000:3b:00.0"], 0);
    ib_device(&root.0, "mlx5_0", &pf, Port::active_ib("fast (4X HDR)"));
    let empty = pci_device(&root.0, &["pci0000:3a", "0000:3a:01.0", "0000:3c:00.0"], 0);
    ib_device(&root.0, "mlx5_1", &empty, Port::active_ib(""));
    let probed = ib::probe(&root.ib_root());
    assert_eq!(
        probed.ports[0].rate,
        PortRate::Unparseable {
            error: IbRateError::Number {
                raw: "fast (4X HDR)".into()
            }
        }
    );
    // An empty rate file is "unreadable", not a parse error.
    assert_eq!(probed.ports[1].rate, PortRate::Unreadable);
}

#[test]
fn a_device_without_a_pci_parent_has_no_placement() {
    let root = Root::new("ib-rxe");
    let port = root.0.join("rxe0/ports/1");
    write(&port.join("state"), "4: ACTIVE\n");
    write(&port.join("rate"), "10 Gb/sec (1X QDR)\n");
    write(&port.join("link_layer"), "Ethernet\n");
    let probed = ib::probe(&root.0);
    assert_eq!(probed.devices.len(), 1);
    assert_eq!(probed.devices[0].pci, None);
    assert_eq!(probed.ports[0].phys_state, PhysState::Unknown);
}

// ---------------------------------------------------------------------------
// Selection, ceiling, outcome
// ---------------------------------------------------------------------------

#[test]
fn caret_excludes_the_storage_nic() {
    let fixture = Fixture::new("ib-select");
    let summary = summarize(&fixture.inventory("n1"), &hca("^mlx5_2"));

    assert_eq!(selected_ids(&summary), ["mlx5_0:1", "mlx5_1:1"]);
    let excluded: BTreeMap<String, PortExclusion> = summary
        .excluded
        .iter()
        .map(|p| (format!("{}:{}", p.device, p.port), p.reason.clone()))
        .collect();
    assert_eq!(excluded["mlx5_2:1"], PortExclusion::FilteredByHca);
    assert_eq!(
        excluded["mlx5_3:1"],
        PortExclusion::NotActive {
            state: "DOWN".into()
        }
    );
    assert_eq!(summary.hca.as_deref(), Some("^mlx5_2"));

    // 200G HDR + 100G RoCE (no IB encoding correction on Ethernet).
    assert_eq!(summary.ceiling_gbps(), Some(300.0));
    let ceiling = summary.ceiling_gib_per_sec().expect("ceiling");
    assert!((ceiling - gbps_to_gib_per_sec(300.0)).abs() < 1e-12);
    assert!((ceiling - 34.924_596_548).abs() < 1e-6, "{ceiling}");
    assert_eq!(summary.ceiling_unknown_reason(), None);
    assert_eq!(summary.link_layer(), SelectionLinkLayer::Mixed);
    assert_eq!(summary.outcome(), TestOutcome::Passed);

    // Locality: mlx5_0 shares GPU 0's switch; the RoCE NIC only its NUMA
    // node; both are a NUMA hop from GPU 1.
    let hdr = &summary.selected[0];
    assert_eq!(hdr.numa_node, Some(0));
    assert_eq!(hdr.locality_to(0).expect("gpu0"), PciLocality::PcieSwitch);
    assert_eq!(hdr.locality_to(1).expect("gpu1"), PciLocality::CrossNuma);
    assert_eq!(
        summary.selected[1].locality_to(0).expect("gpu0"),
        PciLocality::SameNuma
    );
}

#[test]
fn unset_hca_takes_every_active_port() {
    let fixture = Fixture::new("ib-unset");
    let summary = summarize(&fixture.inventory("n1"), &unset());
    assert_eq!(selected_ids(&summary), ["mlx5_0:1", "mlx5_1:1", "mlx5_2:1"]);
    assert_eq!(summary.ceiling_gbps(), Some(400.0));
    assert_eq!(summary.excluded.len(), 1);
}

#[test]
fn exact_and_dev_port_forms_select_precisely() {
    let fixture = Fixture::new("ib-exact");
    let inventory = fixture.inventory("n1");
    let summary = summarize(&inventory, &hca("=mlx5_0:1,mlx5_1:2"));
    assert_eq!(selected_ids(&summary), ["mlx5_0:1"]);
    assert_eq!(summary.link_layer(), SelectionLinkLayer::Infiniband);
    // Prefix "mlx5_" without '=' would take all three active ports.
    assert_eq!(summarize(&inventory, &hca("mlx5_")).selected.len(), 3);
    assert_eq!(summarize(&inventory, &hca("=mlx5_")).selected.len(), 0);
}

fn derivation(nccl_ib: NcclIbConfig) -> InventoryDerivation {
    InventoryDerivation {
        gpu_idle_max_used_mib: 1024,
        nccl_ib: Some(nccl_ib),
    }
}

fn nccl_events(events: &[AgentEvent]) -> Vec<&AgentEvent> {
    events
        .iter()
        .filter(|event| match event {
            AgentEvent::Outcome { test, .. } => *test == TestId::NcclNics,
            AgentEvent::Metric { record } => record.test == TestId::NcclNics,
            AgentEvent::NcclNics { .. } => true,
            _ => false,
        })
        .collect()
}

#[test]
fn selecting_only_a_down_port_is_a_socket_fallback_finding() {
    let fixture = Fixture::new("ib-down");
    let config = hca("mlx5_3");
    let inventory = fixture.inventory("n1");
    let summary = summarize(&inventory, &config);
    assert!(summary.selected.is_empty());
    assert_eq!(summary.ceiling_gib_per_sec(), Some(0.0));
    let TestOutcome::Failed { reason } = summary.outcome() else {
        panic!("zero selected active ports must fail");
    };
    assert!(reason.contains("NCCL_IB_HCA=mlx5_3"), "{reason}");
    assert!(reason.contains("mlx5_3:1 not active (DOWN)"), "{reason}");
    assert!(
        reason.contains("mlx5_0:1 filtered by NCCL_IB_HCA"),
        "{reason}"
    );
    assert!(reason.contains("fall back to sockets"), "{reason}");

    let events = derive_inventory_events(&inventory, &derivation(config));
    let events = nccl_events(&events);
    assert_eq!(events.len(), 3, "outcome, zero metric, summary");
    assert!(matches!(
        events[0],
        AgentEvent::Outcome {
            test: TestId::NcclNics,
            scope: Scope::Node,
            outcome: TestOutcome::Failed { .. }
        }
    ));
    let AgentEvent::Metric { record } = events[1] else {
        panic!("the zero ceiling is still a fleet-comparable reading");
    };
    assert_eq!(record.value, 0.0);
    assert!(matches!(events[2], AgentEvent::NcclNics { .. }));
}

#[test]
fn ib_disable_and_ib_less_hosts_are_skipped_without_a_metric() {
    let fixture = Fixture::new("ib-disabled");
    let inventory = fixture.inventory("n1");
    // strtoll(.., 0): "0x1" and "1x" both disable, as in NCCL.
    for value in ["1", "0x1", "1x"] {
        let disabled = NcclIbConfig::new(None, Some(value), None);
        let summary = summarize(&inventory, &disabled);
        assert!(summary.selected.is_empty(), "{value}");
        assert!(
            summary
                .excluded
                .iter()
                .all(|port| port.reason == PortExclusion::IbDisabled)
        );
        assert!(matches!(summary.outcome(), TestOutcome::Skipped { .. }));
        assert_eq!(summary.ceiling_metric(), None);
    }
    let enabled = NcclIbConfig::new(None, Some("0x0"), None);
    assert_eq!(summarize(&inventory, &enabled).selected.len(), 3);

    let mut bare = inventory.clone();
    bare.ib_ports.clear();
    bare.ib_devices.clear();
    let summary = summarize(&bare, &unset());
    assert!(summary.has_no_ports());
    let TestOutcome::Skipped { reason } = summary.outcome() else {
        panic!("no IB is not a finding");
    };
    assert!(reason.contains("no InfiniBand/RoCE ports"), "{reason}");
    assert_eq!(summary.ceiling_metric(), None);
}

#[test]
fn nccl_net_socket_forces_sockets_and_plugins_keep_the_ib_view() {
    let fixture = Fixture::new("ib-net");
    let inventory = fixture.inventory("n1");

    let socket = summarize(&inventory, &NcclIbConfig::new(None, None, Some("Socket")));
    assert_eq!(socket.net, NetChoice::Socket);
    assert!(socket.selected.is_empty());
    assert!(
        socket
            .excluded
            .iter()
            .all(|port| port.reason == PortExclusion::NetSocket)
    );
    assert_eq!(
        socket.excluded[0].describe(),
        "mlx5_0:1 NCCL_NET=Socket forces sockets"
    );
    let TestOutcome::Skipped { reason } = socket.outcome() else {
        panic!("an explicit socket transport is not a finding");
    };
    assert!(reason.contains("NCCL_NET=Socket"), "{reason}");
    assert_eq!(socket.ceiling_metric(), None);
    // NCCL_NET wins over NCCL_IB_DISABLE in the recorded reason.
    let both = summarize(
        &inventory,
        &NcclIbConfig::new(None, Some("1"), Some("socket")),
    );
    assert_eq!(both.excluded[0].reason, PortExclusion::NetSocket);

    let ib = summarize(&inventory, &NcclIbConfig::new(None, None, Some("IB")));
    assert_eq!(ib.net, NetChoice::Ib);
    assert_eq!(ib.selected.len(), 3);
    assert_eq!(ib.net_caveat(), None);

    let plugin = summarize(
        &inventory,
        &NcclIbConfig::new(None, None, Some("AWS Libfabric")),
    );
    assert_eq!(
        plugin.net,
        NetChoice::Plugin {
            name: "AWS Libfabric".into()
        }
    );
    assert_eq!(plugin.selected.len(), 3, "IB selection kept");
    assert_eq!(plugin.outcome(), TestOutcome::Passed);
    assert!(
        plugin
            .net_caveat()
            .is_some_and(|caveat| caveat.contains("not modelled")),
        "{:?}",
        plugin.net_caveat()
    );
}

#[test]
fn passed_hosts_emit_the_ceiling_metric_and_gpu_idle_rides_the_same_hook() {
    let fixture = Fixture::new("ib-metric");
    let events = derive_inventory_events(&fixture.inventory("n1"), &derivation(hca("^mlx5_2")));
    // gpu_idle first (one per GPU), then outcome, metric, summary.
    let gpu_idle = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AgentEvent::Outcome {
                    test: TestId::GpuIdle,
                    ..
                }
            )
        })
        .count();
    assert_eq!(gpu_idle, 2);
    let nccl = nccl_events(&events);
    assert_eq!(nccl.len(), 3);
    let AgentEvent::Metric { record } = nccl[1] else {
        panic!("metric expected");
    };
    assert_eq!(record.scope, Scope::Node);
    assert_eq!(record.name, CEILING_METRIC);
    assert_eq!(record.unit, Unit::GibPerSec);
    assert!((record.value - gbps_to_gib_per_sec(300.0)).abs() < 1e-12);
    assert_eq!(
        report::metric_key(record.test, &record.name),
        "nccl_nics.ceiling_gib_per_sec"
    );

    // Without a known env only gpu_idle is derived.
    let no_env = InventoryDerivation {
        gpu_idle_max_used_mib: 1024,
        nccl_ib: None,
    };
    let events = derive_inventory_events(&fixture.inventory("n1"), &no_env);
    assert_eq!(events.len(), 2);
    assert!(nccl_events(&events).is_empty());
}

#[test]
fn an_unknown_selected_rate_leaves_the_ceiling_unknown_and_says_why() {
    let fixture = Fixture::new("ib-norate");
    let mut inventory = fixture.inventory("n1");
    inventory.ib_ports[0].rate = PortRate::Unparseable {
        error: IbRateError::Unit {
            raw: "200 MB/sec".into(),
        },
    };
    let summary = summarize(&inventory, &hca("^mlx5_2"));
    assert_eq!(summary.ceiling_gib_per_sec(), None);
    assert_eq!(
        summary.consistency_fields()[consistency::CEILING_GBPS],
        "unknown"
    );
    assert_eq!(summary.ceiling_metric(), None);
    let reason = summary.ceiling_unknown_reason().expect("explained");
    assert!(reason.contains("mlx5_0:1 unparseable rate"), "{reason}");
    assert!(reason.contains("200 MB/sec"), "{reason}");
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

fn config(hosts: &[&str], nccl: &str) -> FleetConfig {
    let list = hosts
        .iter()
        .map(|h| format!("\"{h}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let config: FleetConfig =
        toml::from_str(&format!("hosts = [{list}]\n{nccl}")).expect("test config");
    config.validate().expect("valid");
    config
}

/// Feed each host's inventory plus the derived events through the real
/// collector, exactly as `node_phase` does.
fn run(config: &FleetConfig, inventories: Vec<InventorySnapshot>) -> RunResults {
    let derivation = InventoryDerivation::from_config(config);
    let mut collector = Collector::new();
    for inventory in inventories {
        let host = inventory.hostname.clone();
        let derived = derive_inventory_events(&inventory, &derivation);
        collector.ingest(
            &host,
            AgentEvent::Inventory {
                snapshot: Box::new(inventory),
            },
        );
        for event in derived {
            collector.ingest(&host, event);
        }
    }
    report::build(config, collector.into_observations(), 1, 2)
}

fn rendered(results: &RunResults) -> String {
    let mut table = Vec::new();
    report::render_table(results, &mut table).expect("render");
    String::from_utf8(table).expect("utf8")
}

#[test]
fn the_report_reads_recorded_summaries_and_flags_mismatched_hosts() {
    let fixture = Fixture::new("ib-report");
    let hosts = ["n1", "n2", "n3"];
    let config = config(&hosts, "[nccl]\nenv = { NCCL_IB_HCA = \"^mlx5_2\" }");
    let mut inventories: Vec<InventorySnapshot> =
        hosts.iter().map(|host| fixture.inventory(host)).collect();
    // n3's HDR port is down: one selected port, Ethernet only, 100G.
    inventories[2].ib_ports[0].state = "DOWN".into();

    let results = run(&config, inventories);
    let summary = |host: &str| results.hosts[host].nccl_nics.as_ref().expect("recorded");
    assert_eq!(summary("n1").selected.len(), 2);
    assert_eq!(summary("n3").selected.len(), 1);

    let consistency_findings = &results.fleet.consistency;
    let ports = &consistency_findings[consistency::PORTS];
    assert_eq!(ports.majority_value, "2");
    assert_eq!(ports.dissenters["n3"], "1");
    let layer = &consistency_findings[consistency::LINK_LAYER];
    assert_eq!(layer.majority_value, "mixed");
    assert_eq!(layer.dissenters["n3"], "ethernet");
    let ceiling = &consistency_findings[consistency::CEILING_GBPS];
    assert_eq!(ceiling.majority_value, "300");
    assert_eq!(ceiling.dissenters["n3"], "100");

    // n3 still has a usable port: not a finding, so the run is clean.
    assert_eq!(report::verdict(&results), Verdict::Clean);
    assert_eq!(results.aggregates["nccl_nics.ceiling_gib_per_sec"].len(), 3);

    let table = rendered(&results);
    assert!(table.contains("nccl nics (NCCL_IB_HCA=^mlx5_2)"), "{table}");
    assert!(table.contains("mlx5_0:1 infiniband 200G"), "{table}");
    assert!(table.contains("gpu0 pcie_switch"), "{table}");
    assert!(
        table.contains("mlx5_2:1 filtered by NCCL_IB_HCA"),
        "{table}"
    );
    assert!(table.contains("34.92"), "{table}");
    assert!(table.contains("nccl_ib_ports"), "{table}");
}

#[test]
fn the_report_does_not_rederive_from_the_inventory() {
    // An inventory with no recorded summary (no derivation ran): no
    // section, no NIC consistency fields, nothing guessed.
    let fixture = Fixture::new("ib-noderive");
    let config = config(&["n1", "n2"], "");
    let mut collector = Collector::new();
    for host in ["n1", "n2"] {
        collector.ingest(
            host,
            AgentEvent::Inventory {
                snapshot: Box::new(fixture.inventory(host)),
            },
        );
    }
    let results = report::build(&config, collector.into_observations(), 1, 2);
    assert!(results.hosts.values().all(|obs| obs.nccl_nics.is_none()));
    assert!(!rendered(&results).contains("nccl nics"));
}

#[test]
fn a_host_falling_back_to_sockets_makes_the_verdict_stragglers() {
    let fixture = Fixture::new("ib-fallback");
    let hosts = ["n1", "n2"];
    let config = config(&hosts, "[nccl]\nenv = { NCCL_IB_HCA = \"=mlx5_0\" }");
    let mut inventories: Vec<InventorySnapshot> =
        hosts.iter().map(|host| fixture.inventory(host)).collect();
    inventories[1].ib_ports[0].state = "DOWN".into();

    let results = run(&config, inventories);
    assert_eq!(report::verdict(&results), Verdict::Stragglers);
    let failed: Vec<&String> = results.hosts["n2"]
        .outcomes
        .iter()
        .filter_map(|(test, _, outcome)| match (test, outcome) {
            (TestId::NcclNics, TestOutcome::Failed { reason }) => Some(reason),
            _ => None,
        })
        .collect();
    assert_eq!(failed.len(), 1);
    let table = rendered(&results);
    assert!(
        table.contains("FAIL: NCCL_IB_HCA==mlx5_0 selects no"),
        "{table}"
    );
}

#[test]
fn the_report_explains_an_unknown_ceiling_and_a_plugin() {
    let fixture = Fixture::new("ib-explain");
    let config = config(&["n1"], "[nccl]\nenv = { NCCL_NET = \"AWS Libfabric\" }");
    let mut inventory = fixture.inventory("n1");
    inventory.ib_ports[0].rate = PortRate::from_sysfs(Some("fast"));
    let results = run(&config, vec![inventory]);
    let table = rendered(&results);
    assert!(
        table.contains("ceiling unknown: mlx5_0:1 unparseable rate"),
        "{table}"
    );
    assert!(
        table.contains("NCCL_NET=AWS Libfabric is an external plugin"),
        "{table}"
    );
}

#[test]
fn an_ib_less_fleet_renders_no_nic_section_and_stays_clean() {
    let fixture = Fixture::new("ib-none");
    let hosts = ["n1", "n2"];
    let config = config(&hosts, "");
    let inventories: Vec<InventorySnapshot> = hosts
        .iter()
        .map(|host| {
            let mut inventory = fixture.inventory(host);
            inventory.ib_ports.clear();
            inventory.ib_devices.clear();
            inventory
        })
        .collect();
    let results = run(&config, inventories);
    assert_eq!(report::verdict(&results), Verdict::Clean);
    assert!(results.fleet.consistency.is_empty());
    assert!(
        !results
            .aggregates
            .contains_key("nccl_nics.ceiling_gib_per_sec")
    );
    assert!(!rendered(&results).contains("nccl nics"));
}

// ---------------------------------------------------------------------------
// Serde
// ---------------------------------------------------------------------------

#[test]
fn inventory_events_round_trip_with_the_new_fields() {
    let fixture = Fixture::new("ib-serde");
    let event = AgentEvent::Inventory {
        snapshot: Box::new(fixture.inventory("n1")),
    };
    let line = encode_event(&event);
    assert!(line.contains(r#""link_layer":{"kind":"ethernet","roce_versions":["v1","v2"]}"#));
    assert!(line.contains(r#""phys_state":"polling""#), "{line}");
    assert!(
        line.contains(r#""rate":{"status":"known","gbps":200.0,"lanes":4,"speed":"hdr"}"#),
        "{line}"
    );
    assert_eq!(decode_event(&line).expect("decode"), event);

    let mut inventory = fixture.inventory("n1");
    inventory.ib_ports[0].rate = PortRate::from_sysfs(Some("fast"));
    let summary = summarize(&inventory, &unset());
    let event = AgentEvent::NcclNics {
        summary: Box::new(summary),
    };
    let line = encode_event(&event);
    assert!(line.contains(r#""status":"unparseable""#), "{line}");
    assert_eq!(decode_event(&line).expect("decode"), event);
}

#[test]
fn pre_v11_inventories_decode_with_unknowns() {
    let fixture = Fixture::new("ib-old");
    let mut value = serde_json::to_value(fixture.inventory("n1")).expect("to value");
    let object = value.as_object_mut().expect("object");
    object.remove("ib_devices");
    for gpu in object["gpus"].as_array_mut().expect("gpus") {
        gpu.as_object_mut().expect("gpu").remove("pci");
    }
    for (index, port) in object["ib_ports"]
        .as_array_mut()
        .expect("ports")
        .iter_mut()
        .enumerate()
    {
        let port = port.as_object_mut().expect("port");
        for field in ["rate", "link_layer", "phys_state", "netdevs"] {
            port.remove(field);
        }
        // v10 carried the bare first number of the rate string (or null).
        let legacy = if index == 0 {
            serde_json::json!(200.0)
        } else {
            serde_json::Value::Null
        };
        port.insert("rate_gbps".into(), legacy);
    }
    let old: InventorySnapshot = serde_json::from_value(value).expect("decode v10 inventory");
    assert!(old.ib_devices.is_empty());
    assert!(old.gpus.iter().all(|gpu| gpu.pci.is_none()));
    let port = &old.ib_ports[0];
    assert_eq!(
        port.rate,
        PortRate::Known {
            gbps: 200.0,
            lanes: None,
            speed: None
        }
    );
    assert_eq!(old.ib_ports[1].rate, PortRate::Unreadable);
    assert_eq!(port.link_layer, LinkLayer::Unknown);
    assert_eq!(port.phys_state, PhysState::Unknown);
    assert!(port.netdevs.is_empty());
    // An unknown link layer is not something NCCL's IB transport takes.
    let summary = summarize(&old, &unset());
    assert!(summary.selected.is_empty());
    assert!(
        summary
            .excluded
            .iter()
            .any(|port| port.reason == PortExclusion::UnsupportedLinkLayer)
    );

    // The typed rate wins over a stray legacy number; unknown fields still
    // fail loudly through the compatibility decoder.
    let both = serde_json::json!({
        "device": "mlx5_0", "port": 1, "state": "ACTIVE",
        "rate": {"status": "unreadable"}, "rate_gbps": 1.0
    });
    let port: gauntlet::proto::IbPortInventory = serde_json::from_value(both).expect("decode");
    assert_eq!(port.rate, PortRate::Unreadable);
    let bogus = serde_json::json!({
        "device": "mlx5_0", "port": 1, "state": "ACTIVE", "bogus": 1
    });
    assert!(serde_json::from_value::<gauntlet::proto::IbPortInventory>(bogus).is_err());
}

#[test]
fn results_round_trip_and_pre_v13_documents_decode() {
    let fixture = Fixture::new("ib-results");
    let hosts = ["n1", "n2"];
    let config = config(&hosts, "[nccl]\nenv = { NCCL_IB_HCA = \"^mlx5_2\" }");
    let results = run(
        &config,
        hosts.iter().map(|host| fixture.inventory(host)).collect(),
    );
    assert_eq!(results.schema_version, report::SCHEMA_VERSION);
    let json = serde_json::to_string(&results).expect("serialize");
    let back: RunResults = serde_json::from_str(&json).expect("decode");
    assert_eq!(back, results);

    let mut value = serde_json::to_value(&results).expect("to value");
    for host in value["hosts"].as_object_mut().expect("hosts").values_mut() {
        host.as_object_mut().expect("host").remove("nccl_nics");
    }
    let old: RunResults = serde_json::from_value(value).expect("decode v12 document");
    assert!(old.hosts.values().all(|obs| obs.nccl_nics.is_none()));
}
