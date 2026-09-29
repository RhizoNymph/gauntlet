//! gpu_idle: config knob, per-GPU outcome derivation, verdict, report
//! rendering and serde compatibility.

use std::collections::BTreeMap;

use gauntlet::config::FleetConfig;
use gauntlet::orchestrator::collect::HostObservations;
use gauntlet::proto::{
    AgentEvent, GpuInventory, GpuOccupancy, GpuProcess, InventorySnapshot, ProcessOwner, Scope,
    TestId, TestOutcome, decode_event, encode_event, gpu_idle_outcomes,
};
use gauntlet::report::{self, Verdict};

fn config(extra: &str) -> FleetConfig {
    let config: FleetConfig =
        toml::from_str(&format!("hosts = [\"node0\", \"node1\"]\n{extra}")).expect("config");
    config.validate().expect("valid");
    config
}

fn gpu(index: u32, occupancy: GpuOccupancy) -> GpuInventory {
    GpuInventory {
        index,
        name: "NVIDIA GeForce RTX 3090".into(),
        uuid: format!("GPU-{index}"),
        vbios: "94.02.42.00.A9".into(),
        mem_total_bytes: 24_576 << 20,
        ecc_volatile_errors: None,
        remapped_rows_pending: None,
        pcie_gen_current: Some(4),
        pcie_gen_max: Some(4),
        pcie_width_current: Some(16),
        pcie_width_max: Some(16),
        nvlinks_active: None,
        persistence_mode: Some(true),
        occupancy,
    }
}

fn idle() -> GpuOccupancy {
    GpuOccupancy {
        memory_used_mib: Some(37),
        memory_total_mib: Some(24_576),
        compute_processes: Some(Vec::new()),
    }
}

/// The field report: vLLM holding 23.2 GiB of node1's 24 GiB card.
fn vllm_busy() -> GpuOccupancy {
    GpuOccupancy {
        memory_used_mib: Some(23_264),
        memory_total_mib: Some(24_576),
        compute_processes: Some(vec![GpuProcess {
            pid: 2_102_873,
            name: "VLLM::EngineCore".into(),
            used_mib: Some(23_232),
            owner: ProcessOwner::Foreign,
        }]),
    }
}

fn inventory(host: &str, gpus: Vec<GpuInventory>) -> InventorySnapshot {
    InventorySnapshot {
        hostname: host.into(),
        kernel: "6.8.0".into(),
        cpu_model: "TestCPU".into(),
        logical_cores: 8,
        numa_nodes: 1,
        mem_total_bytes: 1 << 34,
        cpu_governor: Some("performance".into()),
        clock_offset_ms: Some(0.1),
        nvidia_driver: Some("570.133.07".into()),
        cuda_version: Some("12.8".into()),
        gpus,
        nics: vec![],
        ib_ports: vec![],
        xid_errors: vec![],
        gpu_libs: BTreeMap::new(),
        cuda_visible_gpus: Some(1),
    }
}

/// Observations the way the orchestrator records them: the inventory plus
/// the gpu_idle outcomes it derives from it.
fn observed(host: &str, gpus: Vec<GpuInventory>, threshold: u64) -> HostObservations {
    let inventory = inventory(host, gpus);
    let outcomes = gpu_idle_outcomes(&inventory, threshold)
        .into_iter()
        .map(|(scope, outcome)| (TestId::GpuIdle, scope, outcome))
        .collect();
    HostObservations {
        inventory: Some(inventory),
        outcomes,
        ..HostObservations::default()
    }
}

fn fleet(node1: GpuOccupancy) -> BTreeMap<String, HostObservations> {
    BTreeMap::from([
        (
            "node0".to_string(),
            observed("node0", vec![gpu(0, idle())], 1024),
        ),
        (
            "node1".to_string(),
            observed("node1", vec![gpu(0, node1)], 1024),
        ),
    ])
}

fn rendered(results: &report::RunResults) -> String {
    let mut out = Vec::new();
    report::render_table(results, &mut out).expect("render");
    String::from_utf8(out).expect("utf8")
}

#[test]
fn threshold_defaults_to_1024_mib_and_is_configurable() {
    assert_eq!(config("").thresholds.gpu_idle_max_used_mib, 1024);
    let tuned = config("[thresholds]\ngpu_idle_max_used_mib = 4096\n");
    assert_eq!(tuned.thresholds.gpu_idle_max_used_mib, 4096);
    let example = FleetConfig::load(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("gauntlet.example.toml"),
    )
    .expect("example config");
    assert_eq!(example.thresholds.gpu_idle_max_used_mib, 1024);
}

#[test]
fn test_id_is_gpu_idle_on_the_wire_and_in_reports() {
    assert_eq!(report::test_display_name(TestId::GpuIdle), "gpu_idle");
    let event = AgentEvent::Outcome {
        test: TestId::GpuIdle,
        scope: Scope::Gpu { index: 0 },
        outcome: TestOutcome::Skipped {
            reason: "occupancy unknown: occupancy not reported".into(),
        },
    };
    let line = encode_event(&event);
    assert!(line.contains(r#""test":"gpu_idle""#), "{line}");
    assert_eq!(decode_event(&line).expect("decode"), event);
}

#[test]
fn one_outcome_per_gpu_in_index_order() {
    let inv = inventory(
        "node1",
        vec![
            gpu(0, idle()),
            gpu(1, vllm_busy()),
            gpu(2, GpuOccupancy::default()),
        ],
    );
    let outcomes = gpu_idle_outcomes(&inv, 1024);
    assert_eq!(
        outcomes
            .iter()
            .map(|(scope, _)| scope.clone())
            .collect::<Vec<_>>(),
        [0, 1, 2].map(|index| Scope::Gpu { index })
    );
    assert_eq!(outcomes[0].1, TestOutcome::Passed);
    assert!(matches!(outcomes[1].1, TestOutcome::Failed { .. }));
    assert!(matches!(outcomes[2].1, TestOutcome::Skipped { .. }));
    assert!(gpu_idle_outcomes(&inventory("cpu-only", vec![]), 1024).is_empty());
}

#[test]
fn threshold_boundary_on_the_derived_outcome() {
    let mut at = idle();
    at.memory_used_mib = Some(2048);
    let inv = inventory("node0", vec![gpu(0, at.clone())]);
    assert_eq!(gpu_idle_outcomes(&inv, 2048)[0].1, TestOutcome::Passed);
    assert!(matches!(
        gpu_idle_outcomes(&inv, 2047)[0].1,
        TestOutcome::Failed { .. }
    ));
}

#[test]
fn a_busy_gpu_is_a_straggler_finding_not_a_host_failure() {
    let results = report::build(&config(""), fleet(vllm_busy()), 100, 110);
    assert!(results.fleet.failed_hosts.is_empty());
    assert_eq!(report::verdict(&results), Verdict::Stragglers);

    let idle_results = report::build(&config(""), fleet(idle()), 100, 110);
    assert_eq!(report::verdict(&idle_results), Verdict::Clean);

    // Unknown occupancy skips; a skip is not a finding.
    let unknown = report::build(&config(""), fleet(GpuOccupancy::default()), 100, 110);
    assert_eq!(report::verdict(&unknown), Verdict::Clean);
}

#[test]
fn busy_gpus_get_their_own_section_with_process_detail() {
    let results = report::build(&config(""), fleet(vllm_busy()), 100, 110);
    let busy = report::gpu_idle::gpus_in_use(&results);
    assert_eq!(busy.len(), 1);
    assert_eq!((busy[0].host.as_str(), busy[0].gpu), ("node1", 0));

    let text = rendered(&results);
    assert!(text.contains("gpus in use (gpu_idle)"), "{text}");
    assert!(text.contains("node1:gpu0"), "{text}");
    assert!(text.contains("23264 / 24576 MiB"), "{text}");
    assert!(
        text.contains("VLLM::EngineCore (pid 2102873, 23232 MiB)"),
        "{text}"
    );
    assert!(
        !text.contains("node0:gpu0"),
        "idle GPUs are not listed: {text}"
    );

    let idle_text = rendered(&report::build(&config(""), fleet(idle()), 100, 110));
    assert!(!idle_text.contains("gpus in use"), "{idle_text}");
}

#[test]
fn stale_agents_and_memory_only_findings_render_readably() {
    let stale = GpuOccupancy {
        memory_used_mib: Some(700),
        memory_total_mib: Some(24_576),
        compute_processes: Some(vec![GpuProcess {
            pid: 4242,
            name: "/home/u/.gauntlet/bin/gauntlet-agent".into(),
            used_mib: Some(650),
            owner: ProcessOwner::StaleGauntletAgent,
        }]),
    };
    let text = rendered(&report::build(&config(""), fleet(stale), 100, 110));
    assert!(
        text.contains("stale gauntlet agent /home/u/.gauntlet/bin/gauntlet-agent"),
        "{text}"
    );

    // Over the threshold with no compute process listed: the outcome
    // reason stands in for the process column.
    let mut residue = idle();
    residue.memory_used_mib = Some(4096);
    let text = rendered(&report::build(&config(""), fleet(residue), 100, 110));
    assert!(text.contains("4096/24576 MiB used"), "{text}");
    assert!(text.contains("1024 MiB threshold"), "{text}");
}

#[test]
fn results_with_occupancy_round_trip() {
    let results = report::build(&config(""), fleet(vllm_busy()), 100, 110);
    assert_eq!(results.schema_version, 11);
    let json = serde_json::to_string(&results).expect("serialize");
    let back: report::RunResults = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, results);
}

#[test]
fn pre_occupancy_documents_still_load() {
    // Strip every occupancy object, as a v10 document would lack them.
    let results = report::build(&config(""), fleet(vllm_busy()), 100, 110);
    let mut value = serde_json::to_value(&results).expect("to value");
    for host in value["hosts"].as_object_mut().expect("hosts").values_mut() {
        for gpu in host["inventory"]["gpus"].as_array_mut().expect("gpus") {
            gpu.as_object_mut().expect("gpu").remove("occupancy");
        }
    }
    let back: report::RunResults = serde_json::from_value(value).expect("old doc decodes");
    let gpu = &back.hosts["node1"].inventory.as_ref().expect("inv").gpus[0];
    assert_eq!(gpu.occupancy, GpuOccupancy::default());
    // The recorded outcome still renders; detail falls back to its reason.
    let text = rendered(&back);
    assert!(text.contains("VLLM::EngineCore (pid 2102873"), "{text}");
}

#[test]
fn inventory_events_carry_occupancy_on_the_wire() {
    let event = AgentEvent::Inventory {
        snapshot: Box::new(inventory("node1", vec![gpu(0, vllm_busy())])),
    };
    let line = encode_event(&event);
    assert!(line.contains(r#""owner":"foreign""#), "{line}");
    assert_eq!(decode_event(&line).expect("decode"), event);
}
