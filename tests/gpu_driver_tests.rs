//! GPU driver state and the 2026-10-15 fleet run: a node rebooted into a
//! kernel without the NVIDIA module (RTX 3090 on the bus, nvidia-smi
//! failing, CUDA_ERROR_NO_DEVICE). Sysfs fixture trees for the PCI/module
//! probes, the orchestrator-derived `gpu_driver` outcome, the verdict
//! (one root cause, one hard finding), the SDC section (never
//! infrastructure errors) and the consistency vote (a missing driver never
//! wins). Nothing here needs a GPU.

mod common;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use gauntlet::agent::gpu_driver::{self, DriverProbeRoots};
use gauntlet::config::FleetConfig;
use gauntlet::orchestrator::collect::HostObservations;
use gauntlet::orchestrator::derive::{InventoryDerivation, derive_inventory_events};
use gauntlet::proto::{
    AgentEvent, FailureKind, GpuDriverState, GpuInventory, InventorySnapshot, ModuleAvailability,
    NvidiaModule, NvidiaSmiFailure, PciBusId, Scope, TestId, TestOutcome, consistency_fields,
};
use gauntlet::report::{self, Verdict};

// ---------------------------------------------------------------------------
// Fixture sysfs
// ---------------------------------------------------------------------------

struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new(tag: &str) -> Self {
        Self {
            root: common::scratch_dir(tag),
        }
    }

    fn roots(&self) -> DriverProbeRoots {
        DriverProbeRoots {
            pci_devices: self.root.join("sys/bus/pci/devices"),
            sys_module: self.root.join("sys/module"),
            lib_modules: self.root.join("lib/modules"),
        }
    }

    fn pci_function(&self, address: &str, vendor: &str, class: &str) {
        let dir = self.root.join("sys/bus/pci/devices").join(address);
        write(&dir.join("vendor"), &format!("{vendor}\n"));
        write(&dir.join("class"), &format!("{class}\n"));
    }

    /// The consumer card from the report: VGA function plus its HDMI audio
    /// function, next to a Mellanox NIC and the host bridge.
    fn rtx3090_host(&self) {
        self.pci_function("0000:00:00.0", "0x8086", "0x060000");
        self.pci_function("0000:01:00.0", "0x10de", "0x030000");
        self.pci_function("0000:01:00.1", "0x10de", "0x040300");
        self.pci_function("0000:41:00.0", "0x15b3", "0x020700");
    }

    fn module_loaded(&self) {
        fs::create_dir_all(self.root.join("sys/module/nvidia")).expect("mkdir");
    }

    fn modules_dep(&self, kernel: &str, text: &str) {
        write(
            &self
                .root
                .join("lib/modules")
                .join(kernel)
                .join("modules.dep"),
            text,
        );
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn write(path: &Path, content: &str) {
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    fs::write(path, content).expect("write fixture");
}

fn bus(raw: &str) -> PciBusId {
    raw.parse().expect("bus id")
}

const NEW_KERNEL: &str = "6.17.0-5-generic";

fn smi_failed() -> NvidiaSmiFailure {
    NvidiaSmiFailure::Failed {
        message: "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. \
                  Make sure that the latest NVIDIA driver is installed and running."
            .into(),
    }
}

// ---------------------------------------------------------------------------
// Probes
// ---------------------------------------------------------------------------

#[test]
fn the_pci_walk_finds_display_controllers_only() {
    let fixture = Fixture::new("pci-walk");
    fixture.rtx3090_host();
    // A datacenter GPU is a 3D controller; a garbage entry is skipped.
    fixture.pci_function("0000:c1:00.0", "0x10de", "0x030200");
    fixture.pci_function("not-an-address", "0x10de", "0x030000");
    fixture.pci_function("0000:c2:00.0", "0x10de", "garbage");
    assert_eq!(
        gpu_driver::nvidia_pci_gpus(&fixture.roots().pci_devices),
        [bus("0000:01:00.0"), bus("0000:c1:00.0")]
    );
    assert!(gpu_driver::nvidia_pci_gpus(Path::new("/nonexistent/gauntlet/pci")).is_empty());
}

#[test]
fn the_rebooted_node_has_no_module_for_its_kernel() {
    let fixture = Fixture::new("rebooted");
    fixture.rtx3090_host();
    // The new kernel's modules.dep has no nvidia module (DKMS never ran).
    fixture.modules_dep(
        NEW_KERNEL,
        "kernel/drivers/gpu/drm/nouveau/nouveau.ko.zst: kernel/drivers/gpu/drm/drm.ko.zst\n",
    );
    let state = gpu_driver::probe(&fixture.roots(), Err(smi_failed()), NEW_KERNEL);
    let GpuDriverState::DriverUnavailable {
        pci_gpus,
        module,
        nvidia_smi,
    } = &state
    else {
        panic!("expected DriverUnavailable, got {state:?}");
    };
    assert_eq!(pci_gpus.as_slice(), [bus("0000:01:00.0")]);
    assert_eq!(
        *module,
        NvidiaModule::NotLoaded {
            kernel: NEW_KERNEL.into(),
            availability: ModuleAvailability::MissingForKernel,
        }
    );
    assert_eq!(*nvidia_smi, smi_failed());
}

#[test]
fn module_states_are_distinguished() {
    let fixture = Fixture::new("modules");
    fixture.rtx3090_host();
    // No modules.dep at all: availability unknown.
    let unknown = gpu_driver::nvidia_module(&fixture.roots(), NEW_KERNEL);
    assert_eq!(
        unknown,
        NvidiaModule::NotLoaded {
            kernel: NEW_KERNEL.into(),
            availability: ModuleAvailability::Unknown,
        }
    );
    // Installed for this kernel but not loaded.
    fixture.modules_dep(NEW_KERNEL, "updates/dkms/nvidia.ko.zst:\n");
    let installed = gpu_driver::nvidia_module(&fixture.roots(), NEW_KERNEL);
    assert_eq!(
        installed,
        NvidiaModule::NotLoaded {
            kernel: NEW_KERNEL.into(),
            availability: ModuleAvailability::Installed,
        }
    );
    // Loaded, yet nvidia-smi lists nothing.
    fixture.module_loaded();
    let state = gpu_driver::probe(&fixture.roots(), Ok(0), NEW_KERNEL);
    assert!(matches!(
        state,
        GpuDriverState::DriverUnavailable {
            module: NvidiaModule::Loaded,
            nvidia_smi: NvidiaSmiFailure::ListedNone,
            ..
        }
    ));
}

#[test]
fn working_drivers_and_cpu_only_hosts() {
    let gpu_host = Fixture::new("driver-ok");
    gpu_host.rtx3090_host();
    gpu_host.module_loaded();
    assert_eq!(
        gpu_driver::probe(&gpu_host.roots(), Ok(1), NEW_KERNEL),
        GpuDriverState::DriverOk
    );

    let cpu_host = Fixture::new("cpu-only");
    cpu_host.pci_function("0000:00:00.0", "0x8086", "0x060000");
    cpu_host.pci_function("0000:41:00.0", "0x15b3", "0x020700");
    for smi in [Err(NvidiaSmiFailure::NotFound), Ok(0)] {
        assert_eq!(
            gpu_driver::probe(&cpu_host.roots(), smi, NEW_KERNEL),
            GpuDriverState::NoNvidiaHardware
        );
    }
}

// ---------------------------------------------------------------------------
// The 2026-10-15 fleet, end to end through derive + report
// ---------------------------------------------------------------------------

fn inventory(host: &str, driver: Option<&str>, gpu_driver: GpuDriverState) -> InventorySnapshot {
    let gpus: Vec<GpuInventory> = match driver {
        None => vec![],
        Some(_) => vec![
            serde_json::from_value(serde_json::json!({
                "index": 0, "name": "NVIDIA GeForce RTX 3090", "uuid": "u", "vbios": "v",
                "mem_total_bytes": 1, "ecc_volatile_errors": null,
                "remapped_rows_pending": null, "pcie_gen_current": null,
                "pcie_gen_max": null, "pcie_width_current": null,
                "pcie_width_max": null, "nvlinks_active": null,
                "persistence_mode": null
            }))
            .expect("gpu"),
        ],
    };
    InventorySnapshot {
        hostname: host.into(),
        kernel: "6.17.0-5-generic".into(),
        cpu_model: "cpu".into(),
        logical_cores: 16,
        numa_nodes: 1,
        mem_total_bytes: 1 << 36,
        cpu_governor: None,
        clock_offset_ms: None,
        nvidia_driver: driver.map(str::to_string),
        cuda_version: None,
        cuda_visible_gpus: driver.map(|_| 1),
        gpus,
        nics: vec![],
        ib_ports: vec![],
        ib_devices: vec![],
        xid_errors: vec![],
        gpu_libs: BTreeMap::new(),
        gpu_driver: Some(gpu_driver),
    }
}

fn rebooted_state() -> GpuDriverState {
    GpuDriverState::classify(vec![bus("0000:01:00.0")], false, smi_failed, || {
        NvidiaModule::NotLoaded {
            kernel: NEW_KERNEL.into(),
            availability: ModuleAvailability::MissingForKernel,
        }
    })
}

fn derivation() -> InventoryDerivation {
    InventoryDerivation {
        gpu_idle_max_used_mib: 1024,
        nccl_ib: None,
    }
}

fn gpu_driver_outcome(events: &[AgentEvent]) -> Option<(&Scope, &TestOutcome)> {
    events.iter().find_map(|event| match event {
        AgentEvent::Outcome {
            test: TestId::GpuDriver,
            scope,
            outcome,
        } => Some((scope, outcome)),
        _ => None,
    })
}

#[test]
fn derive_emits_one_gpu_driver_outcome_per_state() {
    let broken = inventory("10.1.0.67", None, rebooted_state());
    let events = derive_inventory_events(&broken, &derivation());
    let (scope, outcome) = gpu_driver_outcome(&events).expect("gpu_driver outcome");
    assert_eq!(*scope, Scope::Node);
    let TestOutcome::Failed { reason } = outcome else {
        panic!("expected Failed, got {outcome:?}");
    };
    assert!(reason.contains("0000:01:00.0"), "{reason}");
    assert!(reason.contains("has no nvidia module"), "{reason}");
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                AgentEvent::Outcome {
                    test: TestId::GpuDriver,
                    ..
                }
            ))
            .count(),
        1
    );

    let ok = inventory("10.1.0.68", Some("580.65.06"), GpuDriverState::DriverOk);
    let events = derive_inventory_events(&ok, &derivation());
    assert_eq!(
        gpu_driver_outcome(&events).map(|(_, o)| o),
        Some(&TestOutcome::Passed)
    );

    let cpu = inventory("cpu", None, GpuDriverState::NoNvidiaHardware);
    let events = derive_inventory_events(&cpu, &derivation());
    assert!(matches!(
        gpu_driver_outcome(&events).map(|(_, o)| o),
        Some(TestOutcome::Skipped { .. })
    ));

    // Pre-v14 inventory: no state recorded, nothing derived.
    let mut old = inventory("old", Some("580.65.06"), GpuDriverState::DriverOk);
    old.gpu_driver = None;
    assert!(gpu_driver_outcome(&derive_inventory_events(&old, &derivation())).is_none());
}

fn config_for(hosts: &[&str]) -> FleetConfig {
    let list = hosts
        .iter()
        .map(|h| format!("\"{h}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let config: FleetConfig = toml::from_str(&format!("hosts = [{list}]")).expect("test config");
    config.validate().expect("valid");
    config
}

/// The three-node fleet as the fixed agent and orchestrator record it.
fn incident_fleet() -> BTreeMap<String, HostObservations> {
    let mut observations = BTreeMap::new();
    for (host, driver) in [
        ("10.1.0.67", None),
        ("10.1.0.68", Some("580.65.06")),
        ("10.1.0.69", Some("590.48.01")),
    ] {
        let state = match driver {
            None => rebooted_state(),
            Some(_) => GpuDriverState::DriverOk,
        };
        let snapshot = inventory(host, driver, state);
        let mut obs = HostObservations::default();
        for event in derive_inventory_events(&snapshot, &derivation()) {
            if let AgentEvent::Outcome {
                test,
                scope,
                outcome,
            } = event
            {
                obs.outcomes.push((test, scope, outcome));
            }
        }
        obs.inventory = Some(snapshot);
        match driver {
            None => {
                // The agent's GPU and overlap phases on the rebooted node.
                let reason = "no usable GPU (driver unavailable: CUDA cannot open the NVIDIA \
                              GPU on the PCI bus at 0000:01:00.0; cuda driver init: \
                              DriverError(CUDA_ERROR_NO_DEVICE))"
                    .to_string();
                for test in [
                    TestId::GpuGemmCorrectness,
                    TestId::GpuGemmPerf,
                    TestId::GpuGemmSdc,
                    TestId::GpuMemBandwidth,
                    TestId::GpuP2p,
                    TestId::OverlapGemm,
                    TestId::OverlapAllReduce,
                ] {
                    obs.outcomes.push((
                        test,
                        Scope::Node,
                        TestOutcome::Skipped {
                            reason: reason.clone(),
                        },
                    ));
                }
            }
            Some(_) => {
                for test in [TestId::GpuGemmCorrectness, TestId::GpuGemmSdc] {
                    obs.outcomes
                        .push((test, Scope::Gpu { index: 0 }, TestOutcome::Passed));
                }
            }
        }
        observations.insert(host.to_string(), obs);
    }
    observations
}

#[test]
fn one_root_cause_is_one_hard_finding() {
    let hosts = ["10.1.0.67", "10.1.0.68", "10.1.0.69"];
    let results = report::build(&config_for(&hosts), incident_fleet(), 1, 2);
    assert_eq!(report::verdict(&results), Verdict::Failures);
    let hard: Vec<(&String, TestId)> = results
        .hosts
        .iter()
        .flat_map(|(host, obs)| {
            obs.outcomes
                .iter()
                .filter(|(_, _, outcome)| matches!(outcome, TestOutcome::Failed { .. }))
                .map(move |(test, _, _)| (host, *test))
        })
        .collect();
    assert_eq!(hard, [(&"10.1.0.67".to_string(), TestId::GpuDriver)]);
    assert!(
        results.fleet.sdc_failures.is_empty(),
        "{:?}",
        results.fleet.sdc_failures
    );
}

#[test]
fn a_cpu_only_host_without_gpus_is_clean() {
    let snapshot = inventory("cpu", None, GpuDriverState::NoNvidiaHardware);
    let mut obs = HostObservations::default();
    for event in derive_inventory_events(&snapshot, &derivation()) {
        if let AgentEvent::Outcome {
            test,
            scope,
            outcome,
        } = event
        {
            obs.outcomes.push((test, scope, outcome));
        }
    }
    obs.outcomes.push((
        TestId::GpuGemmCorrectness,
        Scope::Node,
        TestOutcome::Skipped {
            reason: "no NVIDIA GPU (none on the PCI bus; cuda driver init: panicked)".into(),
        },
    ));
    obs.inventory = Some(snapshot);
    let results = report::build(
        &config_for(&["cpu"]),
        BTreeMap::from([("cpu".to_string(), obs)]),
        1,
        2,
    );
    assert_eq!(report::verdict(&results), Verdict::Clean);
}

#[test]
fn infrastructure_errors_never_reach_the_sdc_section() {
    let mut observations = incident_fleet();
    let host = observations.get_mut("10.1.0.67").expect("host");
    // What the pre-fix agent recorded, and an infrastructure failure from a
    // genuine CUDA error on a working driver: hard, but not corruption.
    host.outcomes.push((
        TestId::GpuGemmCorrectness,
        Scope::Node,
        TestOutcome::Failed {
            reason: "cuda driver init: DriverError(CUDA_ERROR_NO_DEVICE, \"no CUDA-capable \
                     device is detected\")"
                .into(),
        },
    ));
    host.outcomes.push((
        TestId::GpuGemmSdc,
        Scope::Gpu { index: 0 },
        TestOutcome::infrastructure_failure("gemm: worker panicked"),
    ));
    // A genuine mismatch elsewhere stays an SDC finding.
    let mismatch = TestOutcome::Failed {
        reason: "bf16 residual 3.1e-1 exceeds tolerance 1.0e-2".into(),
    };
    assert_eq!(mismatch.failure_kind(), Some(FailureKind::Finding));
    observations
        .get_mut("10.1.0.68")
        .expect("host")
        .outcomes
        .push((
            TestId::GpuGemmCorrectness,
            Scope::Gpu { index: 0 },
            mismatch,
        ));

    let hosts = ["10.1.0.67", "10.1.0.68", "10.1.0.69"];
    let results = report::build(&config_for(&hosts), observations, 1, 2);
    let sdc: Vec<&String> = results.fleet.sdc_failures.values().flatten().collect();
    assert_eq!(sdc.len(), 1, "{sdc:?}");
    assert!(
        sdc[0].starts_with("10.1.0.68:gpu0: bf16 residual"),
        "{}",
        sdc[0]
    );
    assert!(report::is_sdc_finding(
        TestId::GpuGemmCorrectness,
        &TestOutcome::Failed {
            reason: "f32 gemm produced a non-finite result".into()
        }
    ));

    let mut table = Vec::new();
    report::render_table(&results, &mut table).expect("render");
    let table = String::from_utf8(table).expect("utf-8");
    let sdc_section = table
        .split("silent data corruption")
        .nth(1)
        .expect("sdc section")
        .split("\n\n")
        .next()
        .expect("section body");
    assert!(!sdc_section.contains("CUDA_ERROR"), "{sdc_section}");
    assert!(!sdc_section.contains("worker panicked"), "{sdc_section}");
}

#[test]
fn the_three_driver_fleet_reports_no_majority() {
    let hosts = ["10.1.0.67", "10.1.0.68", "10.1.0.69"];
    let results = report::build(&config_for(&hosts), incident_fleet(), 1, 2);
    // No host is called a dissenter on nvidia_driver.
    assert!(
        !results.fleet.consistency.contains_key("nvidia_driver"),
        "{:?}",
        results.fleet.consistency.get("nvidia_driver")
    );
    let split = &results.fleet.consistency_splits["nvidia_driver"];
    assert_eq!(
        split.values,
        BTreeMap::from([
            ("10.1.0.67".to_string(), "(absent)".to_string()),
            ("10.1.0.68".to_string(), "580.65.06".to_string()),
            ("10.1.0.69".to_string(), "590.48.01".to_string()),
        ])
    );
    // gpu_count is 0 / 1 / 1: a real majority, the rebooted node dissents.
    let gpu_count = &results.fleet.consistency["gpu_count"];
    assert_eq!(gpu_count.majority_value, "1");
    assert_eq!(
        gpu_count.dissenters,
        BTreeMap::from([("10.1.0.67".to_string(), "0".to_string())])
    );

    let mut table = Vec::new();
    report::render_table(&results, &mut table).expect("render");
    let table = String::from_utf8(table).expect("utf-8");
    assert!(table.contains("(no majority)"), "{table}");
}

#[test]
fn absent_drivers_dissent_against_a_present_majority() {
    let hosts = ["a", "b", "c"];
    let mut observations = BTreeMap::new();
    for (host, driver) in [
        ("a", Some("580.65.06")),
        ("b", Some("580.65.06")),
        ("c", None),
    ] {
        let state = match driver {
            Some(_) => GpuDriverState::DriverOk,
            None => rebooted_state(),
        };
        observations.insert(
            host.to_string(),
            HostObservations {
                inventory: Some(inventory(host, driver, state)),
                ..HostObservations::default()
            },
        );
    }
    let results = report::build(&config_for(&hosts), observations, 1, 2);
    let finding = &results.fleet.consistency["nvidia_driver"];
    assert_eq!(finding.majority_value, "580.65.06");
    assert_eq!(
        finding.dissenters,
        BTreeMap::from([("c".to_string(), "(absent)".to_string())])
    );
    assert!(results.fleet.consistency_splits.is_empty());
    // The rendered projection keeps the sentinel.
    let fields = consistency_fields(&inventory("c", None, rebooted_state()));
    assert_eq!(fields["nvidia_driver"], "(absent)");
}

#[test]
fn old_documents_decode_without_the_new_fields() {
    let hosts = ["10.1.0.67", "10.1.0.68", "10.1.0.69"];
    let results = report::build(&config_for(&hosts), incident_fleet(), 1, 2);
    let mut value = serde_json::to_value(&results).expect("to value");
    value["fleet"]
        .as_object_mut()
        .expect("fleet")
        .remove("consistency_splits");
    for host in value["hosts"].as_object_mut().expect("hosts").values_mut() {
        host["inventory"]
            .as_object_mut()
            .expect("inventory")
            .remove("gpu_driver");
    }
    let back: report::RunResults = serde_json::from_value(value).expect("decode");
    assert!(back.fleet.consistency_splits.is_empty());
    assert!(back.hosts.values().all(|obs| {
        obs.inventory
            .as_ref()
            .is_some_and(|inv| inv.gpu_driver.is_none())
    }));
}
