//! InfiniBand inventory and the NCCL NIC ceiling: sysfs fixture trees
//! (IB HDR, RoCE, a storage NIC excluded via `^`, a down port, mixed
//! rates), NCCL_IB_HCA selection, the nccl_nics outcome/metric, report
//! integration, and serde compatibility. Nothing here needs RDMA hardware.

mod common;

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

use gauntlet::agent::{ib, pci};
use gauntlet::config::FleetConfig;
use gauntlet::nccl_ib::{
    self, CEILING_METRIC, NcclIbConfig, PortExclusion, SelectionLinkLayer, nccl_nic_events,
    summarize,
};
use gauntlet::orchestrator::collect::Collector;
use gauntlet::proto::{
    AgentEvent, GpuInventory, GpuOccupancy, IbSpeed, InventorySnapshot, LinkLayer, PciLocality,
    PciLocation, PhysState, RoceVersion, Scope, TestId, TestOutcome, Unit, decode_event,
    encode_event, gbps_to_gib_per_sec,
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

struct Port<'a> {
    state: &'a str,
    phys_state: &'a str,
    rate: &'a str,
    link_layer: &'a str,
    gid_types: &'a [&'a str],
    gid_ndevs: &'a [&'a str],
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

/// One host's sysfs:
/// - GPU 0 at 0000:1b:00.0, NUMA 0, behind PCIe switch 0000:17:00.0.
/// - GPU 1 at 0000:9c:00.0, NUMA 1, its own host bridge.
/// - mlx5_0: IB HDR 200G, active, under GPU 0's switch, IPoIB netdev.
/// - mlx5_1: RoCE (Ethernet) 100G, active, NUMA 0 on another host bridge.
/// - mlx5_2: IB EDR 100G storage NIC, active, NUMA 1, GID-table netdev.
/// - mlx5_3: IB port that is DOWN (Polling: no peer).
struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        let root = common::scratch_dir(tag);
        let switch = ["pci0000:16", "0000:16:02.0", "0000:17:00.0"];
        pci_device(
            &root,
            &[
                switch[0],
                switch[1],
                switch[2],
                "0000:18:08.0",
                "0000:1b:00.0",
            ],
            0,
        );
        pci_device(&root, &["pci0000:9a", "0000:9a:02.0", "0000:9c:00.0"], 1);

        let mlx5_0 = pci_device(
            &root,
            &[
                switch[0],
                switch[1],
                switch[2],
                "0000:18:10.0",
                "0000:1c:00.0",
            ],
            0,
        );
        write(&mlx5_0.join("net/ibp28s0/dev_port"), "0\n");
        ib_device(
            &root,
            "mlx5_0",
            &mlx5_0,
            Port {
                state: "4: ACTIVE",
                phys_state: "5: LinkUp",
                rate: "200 Gb/sec (4X HDR)",
                link_layer: "InfiniBand",
                gid_types: &[],
                gid_ndevs: &[],
            },
        );

        let mlx5_1 = pci_device(&root, &["pci0000:3a", "0000:3a:00.0", "0000:3b:00.0"], 0);
        write(&mlx5_1.join("net/ens3np0/dev_port"), "0\n");
        // A second netdev bound to a different port of the function.
        write(&mlx5_1.join("net/ens3np1/dev_port"), "1\n");
        ib_device(
            &root,
            "mlx5_1",
            &mlx5_1,
            Port {
                state: "4: ACTIVE",
                phys_state: "5: LinkUp",
                rate: "100 Gb/sec (4X EDR)",
                link_layer: "Ethernet",
                gid_types: &["IB/RoCE v1\n", "RoCE v2\n", "IB/RoCE v1\n", "RoCE v2\n", ""],
                gid_ndevs: &["ens3np0\n", "ens3np0\n"],
            },
        );

        let mlx5_2 = pci_device(&root, &["pci0000:9a", "0000:9a:00.0", "0000:9b:00.0"], 1);
        ib_device(
            &root,
            "mlx5_2",
            &mlx5_2,
            Port {
                state: "4: ACTIVE",
                phys_state: "5: LinkUp",
                rate: "100 Gb/sec (4X EDR)",
                link_layer: "InfiniBand",
                gid_types: &[],
                gid_ndevs: &["ib_stor\n", ""],
            },
        );

        let mlx5_3 = pci_device(&root, &["pci0000:9a", "0000:9a:04.0", "0000:9d:00.0"], 1);
        ib_device(
            &root,
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

    fn ib_root(&self) -> PathBuf {
        self.root.join("class/infiniband")
    }

    fn pci_root(&self) -> PathBuf {
        self.root.join("bus/pci/devices")
    }

    fn gpu(&self, index: u32, address: &str) -> GpuInventory {
        let mut location = PciLocation::address_only(address);
        pci::resolve(&self.pci_root(), &mut location);
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
        let probed = ib::probe(&self.ib_root());
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

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn hca(value: &str) -> NcclIbConfig {
    NcclIbConfig::new(Some(value), None)
}

fn port_ids(ports: impl Iterator<Item = (String, u32)>) -> Vec<String> {
    ports
        .map(|(device, port)| format!("{device}:{port}"))
        .collect()
}

// ---------------------------------------------------------------------------
// Agent-side probe
// ---------------------------------------------------------------------------

#[test]
fn the_probe_reads_every_new_port_fact() {
    let fixture = Fixture::new("ib-probe");
    let probed = ib::probe(&fixture.ib_root());

    let names: Vec<&str> = probed.ports.iter().map(|p| p.device.as_str()).collect();
    assert_eq!(names, ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3"]);

    let hdr = &probed.ports[0];
    assert_eq!(hdr.state, "ACTIVE");
    assert_eq!(hdr.rate_gbps, Some(200.0));
    assert_eq!(hdr.lanes, Some(4));
    assert_eq!(hdr.speed, Some(IbSpeed::Hdr));
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
    // dev_port filters the function's netdevs to this port.
    assert_eq!(roce.netdevs, ["ens3np0"]);

    // No device/net: the GID table's ndevs, deduplicated, empties dropped.
    let storage = &probed.ports[2];
    assert_eq!(storage.netdevs, ["ib_stor"]);

    let down = &probed.ports[3];
    assert_eq!(down.state, "DOWN");
    assert_eq!(down.phys_state, PhysState::Polling);
    assert_eq!(down.speed, Some(IbSpeed::Sdr));
}

#[test]
fn the_probe_places_devices_on_the_pci_tree() {
    let fixture = Fixture::new("ib-pci");
    let probed = ib::probe(&fixture.ib_root());
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
    let gpu = fixture.gpu(0, "0000:1b:00.0");
    let placement = gpu.pci.expect("pci");
    assert_eq!(placement.numa_node, Some(0));
    assert_eq!(placement.upstream.len(), 4);

    // An address sysfs does not know stays address-only.
    let mut unknown = PciLocation::address_only("0000:ff:00.0");
    pci::resolve(&fixture.pci_root(), &mut unknown);
    assert_eq!(unknown, PciLocation::address_only("0000:ff:00.0"));
}

#[test]
fn a_device_without_a_pci_parent_has_no_placement() {
    let root = common::scratch_dir("ib-rxe");
    let port = root.join("rxe0/ports/1");
    write(&port.join("state"), "4: ACTIVE\n");
    write(&port.join("rate"), "10 Gb/sec (1X QDR)\n");
    write(&port.join("link_layer"), "Ethernet\n");
    let probed = ib::probe(&root);
    assert_eq!(probed.devices.len(), 1);
    assert_eq!(probed.devices[0].pci, None);
    assert_eq!(probed.ports[0].phys_state, PhysState::Unknown);
    let _ = fs::remove_dir_all(&root);
}

// ---------------------------------------------------------------------------
// Selection, ceiling, outcome
// ---------------------------------------------------------------------------

#[test]
fn caret_excludes_the_storage_nic() {
    let fixture = Fixture::new("ib-select");
    let summary = summarize(&fixture.inventory("n1"), &hca("^mlx5_2"));

    assert_eq!(
        port_ids(summary.selected.iter().map(|p| (p.device.clone(), p.port))),
        ["mlx5_0:1", "mlx5_1:1"]
    );
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

    // 200G HDR + 100G RoCE (no IB encoding correction on Ethernet).
    assert_eq!(summary.ceiling_gbps(), Some(300.0));
    let ceiling = summary.ceiling_gib_per_sec.expect("ceiling");
    assert!((ceiling - gbps_to_gib_per_sec(300.0)).abs() < 1e-12);
    assert!((ceiling - 34.924_596_548).abs() < 1e-6, "{ceiling}");
    assert_eq!(summary.link_layer(), SelectionLinkLayer::Mixed);
    assert_eq!(summary.outcome(&hca("^mlx5_2")), TestOutcome::Passed);

    // Locality: mlx5_0 shares GPU 0's switch; the RoCE NIC only its NUMA
    // node; both are a NUMA hop from GPU 1.
    let hdr = &summary.selected[0];
    assert_eq!(hdr.numa_node, Some(0));
    assert_eq!(hdr.gpu_locality[&0], PciLocality::PcieSwitch);
    assert_eq!(hdr.gpu_locality[&1], PciLocality::CrossNuma);
    assert_eq!(summary.selected[1].gpu_locality[&0], PciLocality::SameNuma);
}

#[test]
fn unset_hca_takes_every_active_port() {
    let fixture = Fixture::new("ib-unset");
    let config = NcclIbConfig::new(None, None);
    let summary = summarize(&fixture.inventory("n1"), &config);
    assert_eq!(
        port_ids(summary.selected.iter().map(|p| (p.device.clone(), p.port))),
        ["mlx5_0:1", "mlx5_1:1", "mlx5_2:1"]
    );
    assert_eq!(summary.ceiling_gbps(), Some(400.0));
    assert_eq!(summary.excluded.len(), 1);
}

#[test]
fn exact_and_dev_port_forms_select_precisely() {
    let fixture = Fixture::new("ib-exact");
    let inventory = fixture.inventory("n1");
    let summary = summarize(&inventory, &hca("=mlx5_0:1,mlx5_1:2"));
    assert_eq!(
        port_ids(summary.selected.iter().map(|p| (p.device.clone(), p.port))),
        ["mlx5_0:1"]
    );
    assert_eq!(summary.link_layer(), SelectionLinkLayer::Infiniband);
    // Prefix "mlx5_" without '=' would take all three active ports.
    assert_eq!(summarize(&inventory, &hca("mlx5_")).selected.len(), 3);
    assert_eq!(summarize(&inventory, &hca("=mlx5_")).selected.len(), 0);
}

#[test]
fn selecting_only_a_down_port_is_a_socket_fallback_finding() {
    let fixture = Fixture::new("ib-down");
    let config = hca("mlx5_3");
    let inventory = fixture.inventory("n1");
    let summary = summarize(&inventory, &config);
    assert!(summary.selected.is_empty());
    assert_eq!(summary.ceiling_gib_per_sec, Some(0.0));
    let TestOutcome::Failed { reason } = summary.outcome(&config) else {
        panic!("zero selected active ports must fail");
    };
    assert!(reason.contains("NCCL_IB_HCA=mlx5_3"), "{reason}");
    assert!(reason.contains("mlx5_3:1 not active (DOWN)"), "{reason}");
    assert!(
        reason.contains("mlx5_0:1 filtered by NCCL_IB_HCA"),
        "{reason}"
    );
    assert!(reason.contains("fall back to sockets"), "{reason}");

    let events = nccl_nic_events(&inventory, &config);
    assert!(matches!(
        &events[0],
        AgentEvent::Outcome {
            test: TestId::NcclNics,
            scope: Scope::Node,
            outcome: TestOutcome::Failed { .. }
        }
    ));
    let AgentEvent::Metric { record } = &events[1] else {
        panic!("the zero ceiling is still a fleet-comparable reading");
    };
    assert_eq!(record.value, 0.0);
}

#[test]
fn ib_disable_and_ib_less_hosts_are_skipped_without_a_metric() {
    let fixture = Fixture::new("ib-disabled");
    let disabled = NcclIbConfig::new(None, Some("1"));
    let inventory = fixture.inventory("n1");
    let summary = summarize(&inventory, &disabled);
    assert!(summary.selected.is_empty());
    assert!(
        summary
            .excluded
            .iter()
            .all(|port| port.reason == PortExclusion::IbDisabled)
    );
    assert!(matches!(
        summary.outcome(&disabled),
        TestOutcome::Skipped { .. }
    ));
    assert_eq!(nccl_nic_events(&inventory, &disabled).len(), 1);

    let mut bare = inventory.clone();
    bare.ib_ports.clear();
    bare.ib_devices.clear();
    let unset = NcclIbConfig::new(None, None);
    let summary = summarize(&bare, &unset);
    assert!(summary.has_no_ports());
    let TestOutcome::Skipped { reason } = summary.outcome(&unset) else {
        panic!("no IB is not a finding");
    };
    assert!(reason.contains("no InfiniBand/RoCE ports"), "{reason}");
    assert_eq!(nccl_nic_events(&bare, &unset).len(), 1);
}

#[test]
fn passed_hosts_emit_the_ceiling_metric() {
    let fixture = Fixture::new("ib-metric");
    let config = hca("^mlx5_2");
    let events = nccl_nic_events(&fixture.inventory("n1"), &config);
    assert_eq!(events.len(), 2);
    let AgentEvent::Metric { record } = &events[1] else {
        panic!("metric expected");
    };
    assert_eq!(record.test, TestId::NcclNics);
    assert_eq!(record.scope, Scope::Node);
    assert_eq!(record.name, CEILING_METRIC);
    assert_eq!(record.unit, Unit::GibPerSec);
    assert!((record.value - gbps_to_gib_per_sec(300.0)).abs() < 1e-12);
    assert_eq!(
        report::metric_key(record.test, &record.name),
        "nccl_nics.ceiling_gib_per_sec"
    );
}

#[test]
fn an_unknown_selected_rate_leaves_the_ceiling_unknown() {
    let fixture = Fixture::new("ib-norate");
    let mut inventory = fixture.inventory("n1");
    inventory.ib_ports[0].rate_gbps = None;
    let config = hca("^mlx5_2");
    let summary = summarize(&inventory, &config);
    assert_eq!(summary.ceiling_gib_per_sec, None);
    assert_eq!(
        summary.consistency_fields()["nccl_ib_ceiling_gbps"],
        "unknown"
    );
    assert_eq!(nccl_nic_events(&inventory, &config).len(), 1);
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
/// collector, as `node_phase` does.
fn run(config: &FleetConfig, inventories: Vec<InventorySnapshot>) -> RunResults {
    let nccl_ib = NcclIbConfig::from_env(config.nccl_env().expect("env"));
    let mut collector = Collector::new();
    for inventory in inventories {
        let host = inventory.hostname.clone();
        let derived = nccl_nic_events(&inventory, &nccl_ib);
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

#[test]
fn the_report_summarizes_and_flags_mismatched_hosts() {
    let fixture = Fixture::new("ib-report");
    let hosts = ["n1", "n2", "n3"];
    let config = config(&hosts, "[nccl]\nenv = { NCCL_IB_HCA = \"^mlx5_2\" }");
    let mut inventories: Vec<InventorySnapshot> =
        hosts.iter().map(|host| fixture.inventory(host)).collect();
    // n3's HDR port is down: one selected port, Ethernet only, 100G.
    inventories[2].ib_ports[0].state = "DOWN".into();

    let results = run(&config, inventories);
    assert_eq!(results.calibration.nccl_nics.len(), 3);
    assert_eq!(results.calibration.nccl_nics["n1"].selected.len(), 2);
    assert_eq!(results.calibration.nccl_nics["n3"].selected.len(), 1);

    let consistency = &results.fleet.consistency;
    let ports = &consistency[nccl_ib::consistency::PORTS];
    assert_eq!(ports.majority_value, "2");
    assert_eq!(ports.dissenters["n3"], "1");
    let layer = &consistency[nccl_ib::consistency::LINK_LAYER];
    assert_eq!(layer.majority_value, "mixed");
    assert_eq!(layer.dissenters["n3"], "ethernet");
    let ceiling = &consistency[nccl_ib::consistency::CEILING_GBPS];
    assert_eq!(ceiling.majority_value, "300");
    assert_eq!(ceiling.dissenters["n3"], "100");

    // n3 still has a usable port: not a finding, so the run is clean.
    assert_eq!(report::verdict(&results), Verdict::Clean);
    let aggregate = &results.aggregates["nccl_nics.ceiling_gib_per_sec"];
    assert_eq!(aggregate.len(), 3);

    let mut table = Vec::new();
    report::render_table(&results, &mut table).expect("render");
    let table = String::from_utf8(table).expect("utf8");
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
    let mut table = Vec::new();
    report::render_table(&results, &mut table).expect("render");
    let table = String::from_utf8(table).expect("utf8");
    assert!(
        table.contains("FAIL: NCCL_IB_HCA==mlx5_0 selects no"),
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
    let mut table = Vec::new();
    report::render_table(&results, &mut table).expect("render");
    assert!(
        !String::from_utf8(table)
            .expect("utf8")
            .contains("nccl nics")
    );
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
    for port in object["ib_ports"].as_array_mut().expect("ports") {
        let port = port.as_object_mut().expect("port");
        for field in ["lanes", "speed", "link_layer", "phys_state", "netdevs"] {
            port.remove(field);
        }
    }
    let old: InventorySnapshot = serde_json::from_value(value).expect("decode v10 inventory");
    assert!(old.ib_devices.is_empty());
    assert!(old.gpus.iter().all(|gpu| gpu.pci.is_none()));
    let port = &old.ib_ports[0];
    assert_eq!(port.link_layer, LinkLayer::Unknown);
    assert_eq!(port.phys_state, PhysState::Unknown);
    assert_eq!((port.lanes, port.speed), (None, None));
    assert!(port.netdevs.is_empty());
    // An unknown link layer is not something NCCL's IB transport takes.
    let summary = summarize(&old, &NcclIbConfig::new(None, None));
    assert!(summary.selected.is_empty());
    assert!(
        summary
            .excluded
            .iter()
            .any(|port| port.reason == PortExclusion::UnsupportedLinkLayer)
    );
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
    value["calibration"]
        .as_object_mut()
        .expect("calibration")
        .remove("nccl_nics");
    let old: RunResults = serde_json::from_value(value).expect("decode v12 document");
    assert!(old.calibration.nccl_nics.is_empty());
}
