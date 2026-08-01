mod common;

use std::collections::BTreeMap;

use gauntlet::config::{Bound, FleetConfig};
use gauntlet::orchestrator::collect::HostObservations;
use gauntlet::proto::{
    GpuInventory, InventorySnapshot, MetricRecord, Scope, TestId, TestOutcome, Unit,
};
use gauntlet::report::{self, Verdict};

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

fn inventory(host: &str, kernel: &str) -> InventorySnapshot {
    InventorySnapshot {
        hostname: host.into(),
        kernel: kernel.into(),
        cpu_model: "TestCPU".into(),
        logical_cores: 8,
        numa_nodes: 1,
        mem_total_bytes: 1 << 34,
        cpu_governor: Some("performance".into()),
        clock_offset_ms: Some(0.1),
        nvidia_driver: Some("560.35.03".into()),
        cuda_version: Some("12.6".into()),
        gpus: vec![],
        nics: vec![],
        ib_ports: vec![],
        xid_errors: vec![],
    }
}

fn node_metric(test: TestId, name: &str, value: f64, unit: Unit) -> MetricRecord {
    MetricRecord {
        test,
        scope: Scope::Node,
        name: name.into(),
        value,
        unit,
    }
}

/// Five healthy hosts, one straggler on DRAM bandwidth, one kernel dissenter.
fn fleet() -> (FleetConfig, BTreeMap<String, HostObservations>) {
    let names = ["n1", "n2", "n3", "n4", "n5", "n6"];
    let config = config_for(&names);
    let mut observations = BTreeMap::new();
    for (i, name) in names.iter().enumerate() {
        let bandwidth = match *name {
            "n6" => 90.0,          // straggler
            _ => 200.0 + i as f64, // healthy spread
        };
        let kernel = if *name == "n3" { "6.9.0" } else { "6.8.0" };
        let mut obs = HostObservations {
            inventory: Some(inventory(name, kernel)),
            ..HostObservations::default()
        };
        obs.metrics.push(node_metric(
            TestId::MemBandwidth,
            "triad",
            bandwidth,
            Unit::GibPerSec,
        ));
        obs.outcomes.push((
            TestId::CpuCorrectness,
            Scope::Core { id: 0 },
            TestOutcome::Passed,
        ));
        observations.insert((*name).to_string(), obs);
    }
    (config, observations)
}

#[test]
fn build_flags_stragglers_and_dissenters() {
    let (config, observations) = fleet();
    let results = report::build(&config, observations, 1_700_000_000, 1_700_000_600);

    assert_eq!(results.schema_version, report::SCHEMA_VERSION);
    assert!(
        results.run_id.starts_with("1700000000-"),
        "{}",
        results.run_id
    );
    assert_eq!(results.hosts.len(), 6);

    let outliers = results
        .fleet
        .outliers
        .get("mem_bandwidth.triad")
        .expect("straggler group present");
    assert_eq!(outliers.len(), 1);
    assert!(outliers[0].key.contains("n6"), "got {:?}", outliers[0]);
    assert!(outliers[0].deviation_mads < 0.0);

    let finding = results
        .fleet
        .consistency
        .get("kernel")
        .expect("kernel skew detected");
    assert_eq!(finding.majority_value, "6.8.0");
    assert_eq!(
        finding.dissenters.get("n3").map(String::as_str),
        Some("6.9.0")
    );

    assert!(results.fleet.failed_hosts.is_empty());
    assert_eq!(report::verdict(&results), Verdict::Stragglers);
}

#[test]
fn clean_fleet_is_clean() {
    let (config, mut observations) = fleet();
    // Heal the straggler and the kernel dissenter.
    if let Some(obs) = observations.get_mut("n6") {
        obs.metrics.clear();
        obs.metrics.push(node_metric(
            TestId::MemBandwidth,
            "triad",
            201.5,
            Unit::GibPerSec,
        ));
    }
    if let Some(obs) = observations.get_mut("n3") {
        obs.inventory = Some(inventory("n3", "6.8.0"));
    }
    let results = report::build(&config, observations, 1_700_000_000, 1_700_000_600);
    assert!(results.fleet.outliers.values().all(Vec::is_empty));
    assert!(results.fleet.consistency.is_empty());
    assert_eq!(report::verdict(&results), Verdict::Clean);
    assert_eq!(Verdict::Clean.exit_code(), 0);
}

#[test]
fn host_errors_dominate_the_verdict() {
    let (config, mut observations) = fleet();
    observations
        .get_mut("n2")
        .expect("n2")
        .errors
        .push("phase cpu_mem timed out after 900s".into());
    let results = report::build(&config, observations, 1_700_000_000, 1_700_000_600);
    assert!(results.fleet.failed_hosts.contains_key("n2"));
    assert_eq!(report::verdict(&results), Verdict::HostFailures);
    assert_eq!(Verdict::HostFailures.exit_code(), 2);
}

#[test]
fn absolute_thresholds_flag_independently_of_mad() {
    let (mut config, observations) = fleet();
    // Everything below 1000 GiB/s violates — all six hosts.
    config.thresholds.absolute.insert(
        "mem_bandwidth.triad".into(),
        Bound {
            min: Some(1000.0),
            max: None,
        },
    );
    let results = report::build(&config, observations, 1_700_000_000, 1_700_000_600);
    let violators = results
        .fleet
        .threshold_violations
        .get("mem_bandwidth.triad")
        .expect("violations recorded");
    assert_eq!(violators.len(), 6);
    assert_eq!(report::verdict(&results), Verdict::Stragglers);
}

#[test]
fn rooflines_take_the_worst_gpu() {
    let names = ["n1", "n2", "n3", "n4"];
    let config = config_for(&names);
    let mut observations = BTreeMap::new();
    for name in names {
        let mut obs = HostObservations::default();
        let mut inv = inventory(name, "6.8.0");
        inv.gpus = vec![GpuInventory {
            index: 0,
            name: "H100".into(),
            uuid: format!("{name}-gpu0"),
            vbios: "96.00".into(),
            mem_total_bytes: 80 << 30,
            ecc_volatile_errors: Some(0),
            remapped_rows_pending: Some(false),
            pcie_gen_current: Some(5),
            pcie_gen_max: Some(5),
            pcie_width_current: Some(16),
            pcie_width_max: Some(16),
            nvlinks_active: Some(18),
            persistence_mode: Some(true),
        }];
        obs.inventory = Some(inv);
        for (gpu, gflops) in [(0u32, 900_000.0), (1u32, 850_000.0)] {
            obs.metrics.push(MetricRecord {
                test: TestId::GpuGemmPerf,
                scope: Scope::Gpu { index: gpu },
                name: "gflops_bf16".into(),
                value: gflops,
                unit: Unit::Gflops,
            });
        }
        observations.insert(name.to_string(), obs);
    }
    let results = report::build(&config, observations, 1_700_000_000, 1_700_000_100);
    let roofline = results
        .calibration
        .rooflines
        .get("n1")
        .expect("roofline for n1");
    assert_eq!(
        roofline.gpu_gflops.get("gflops_bf16").copied(),
        Some(850_000.0),
        "the slowest GPU defines the node"
    );
}

#[test]
fn results_survive_history_round_trip_and_render() {
    let (config, observations) = fleet();
    let results = report::build(&config, observations, 1_700_000_000, 1_700_000_600);

    let dir = common::scratch_dir("history");
    let path = gauntlet::report::history::save(&results, &dir).expect("save");
    assert!(path.starts_with(&dir));
    let loaded = gauntlet::report::history::load(&path).expect("load");
    assert_eq!(loaded, results);

    let listed = gauntlet::report::history::list(&dir).expect("list");
    assert_eq!(listed, vec![path]);

    let mut rendered = Vec::new();
    report::render_table(&results, &mut rendered).expect("render");
    let text = String::from_utf8(rendered).expect("utf8 table");
    assert!(text.contains("n6"), "straggler visible in table");
    assert!(text.contains("kernel"), "consistency finding visible");

    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn missing_history_dir_lists_empty() {
    let dir = common::scratch_dir("history-empty").join("does-not-exist");
    let listed = gauntlet::report::history::list(&dir).expect("empty ok");
    assert!(listed.is_empty());
}
