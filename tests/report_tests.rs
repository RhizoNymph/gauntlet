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
        gpu_libs: BTreeMap::new(),
    }
}

fn node_metric(test: TestId, name: &str, value: f64, unit: Unit) -> MetricRecord {
    MetricRecord {
        test,
        scope: Scope::Node,
        name: name.into(),
        value,
        unit,
        repeat: 0,
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
                repeat: 0,
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

// ---------------------------------------------------------------------------
// Stable run ids and partial (in-flight) snapshots
// ---------------------------------------------------------------------------

fn hosts(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

/// A results document with an arbitrary run id, for filesystem tests.
fn results_with_id(run_id: &str) -> gauntlet::report::RunResults {
    let (config, observations) = fleet();
    let mut results = report::build(&config, observations, 1_700_000_000, 1_700_000_600);
    results.run_id = run_id.to_string();
    results
}

#[test]
fn make_run_id_is_deterministic_and_well_formed() {
    let hosts = hosts(&["n1", "n2", "n3"]);
    let id = report::make_run_id(1_700_000_000, &hosts);
    assert_eq!(id, report::make_run_id(1_700_000_000, &hosts));

    let (stamp, suffix) = id.split_once('-').expect("<epoch>-<suffix>");
    assert_eq!(stamp, "1700000000");
    assert_eq!(suffix.len(), 6);
    assert!(
        suffix
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "{suffix}"
    );
}

#[test]
fn make_run_id_varies_with_start_time_and_host_set() {
    let base = hosts(&["n1", "n2"]);
    let id = report::make_run_id(1_700_000_000, &base);

    // A different start second changes both halves.
    assert_ne!(id, report::make_run_id(1_700_000_001, &base));
    // A different host set changes the suffix at the same start second.
    assert_ne!(
        id,
        report::make_run_id(1_700_000_000, &hosts(&["n1", "n3"]))
    );
    assert_ne!(
        id,
        report::make_run_id(1_700_000_000, &hosts(&["n1", "n2", "n3"]))
    );
    // The id is independent of the run's finish time, unlike `build`'s.
    assert_eq!(id, report::make_run_id(1_700_000_000, &base));
}

#[test]
fn save_partial_writes_the_partial_name_and_leaves_no_temporary() {
    let dir = common::scratch_dir("partial-save");
    let results = results_with_id("1700000000-abcdef");

    let path = gauntlet::report::history::save_partial(&results, &dir).expect("save partial");
    assert_eq!(path, dir.join("1700000000-abcdef.partial.json"));
    assert_eq!(
        gauntlet::report::history::load(&path).expect("load partial"),
        results
    );

    let leftovers: Vec<_> = std::fs::read_dir(&dir)
        .expect("read dir")
        .map(|entry| entry.expect("entry").file_name())
        .filter(|name| name.to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "temporary left behind: {leftovers:?}");

    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn save_partial_overwrites_the_previous_snapshot() {
    let dir = common::scratch_dir("partial-overwrite");
    let mut results = results_with_id("1700000000-abcdef");

    let first = gauntlet::report::history::save_partial(&results, &dir).expect("first snapshot");
    results.finished_epoch_secs = 1_700_000_900;
    let second = gauntlet::report::history::save_partial(&results, &dir).expect("second snapshot");

    assert_eq!(first, second);
    let loaded = gauntlet::report::history::load(&second).expect("load");
    assert_eq!(loaded.finished_epoch_secs, 1_700_000_900);
    assert_eq!(
        std::fs::read_dir(&dir).expect("read dir").count(),
        1,
        "one snapshot file, replaced in place"
    );

    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn list_ignores_partials_and_hidden_files() {
    let dir = common::scratch_dir("partial-list");
    let finished =
        gauntlet::report::history::save(&results_with_id("1700000000-aaaaaa"), &dir).expect("save");
    gauntlet::report::history::save_partial(&results_with_id("1700000900-bbbbbb"), &dir)
        .expect("save partial");
    std::fs::write(dir.join(".1700000900-bbbbbb.partial.json.tmp"), "{}").expect("write tmp");
    std::fs::write(dir.join(".hidden.json"), "{}").expect("write hidden");

    assert_eq!(
        gauntlet::report::history::list(&dir).expect("list"),
        vec![finished]
    );

    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn list_live_returns_only_partials_in_order() {
    let dir = common::scratch_dir("partial-list-live");
    gauntlet::report::history::save(&results_with_id("1700000000-aaaaaa"), &dir).expect("save");
    let second =
        gauntlet::report::history::save_partial(&results_with_id("1700000900-bbbbbb"), &dir)
            .expect("save partial");
    let first =
        gauntlet::report::history::save_partial(&results_with_id("1700000000-cccccc"), &dir)
            .expect("save partial");

    assert_eq!(
        gauntlet::report::history::list_live(&dir).expect("list live"),
        vec![first, second]
    );

    let missing = dir.join("does-not-exist");
    assert!(
        gauntlet::report::history::list_live(&missing)
            .expect("empty ok")
            .is_empty()
    );

    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn remove_partial_deletes_the_snapshot_and_tolerates_absence() {
    let dir = common::scratch_dir("partial-remove");
    let path = gauntlet::report::history::save_partial(&results_with_id("1700000000-abcdef"), &dir)
        .expect("save partial");

    gauntlet::report::history::remove_partial("1700000000-abcdef", &dir).expect("remove");
    assert!(!path.exists());
    // Removing again (and removing one that never existed) is not an error.
    gauntlet::report::history::remove_partial("1700000000-abcdef", &dir).expect("idempotent");
    gauntlet::report::history::remove_partial("1700000000-nothere", &dir).expect("absent is ok");

    std::fs::remove_dir_all(&dir).expect("cleanup");
}

// ---------------------------------------------------------------------------
// --repeat: per-subject aggregates, median-based analysis, jitter
// ---------------------------------------------------------------------------

fn metric_repeat(test: TestId, scope: Scope, name: &str, value: f64, repeat: u32) -> MetricRecord {
    MetricRecord {
        test,
        scope,
        name: name.into(),
        value,
        unit: Unit::GibPerSec,
        repeat,
    }
}

fn host_with_series(name_values: &[(&str, &[f64])]) -> HostObservations {
    let mut obs = HostObservations::default();
    for (name, values) in name_values {
        for (repeat, value) in values.iter().enumerate() {
            obs.metrics.push(metric_repeat(
                TestId::MemBandwidth,
                Scope::Node,
                name,
                *value,
                repeat as u32,
            ));
        }
    }
    obs
}

#[test]
fn aggregates_summarize_repeats_per_subject() {
    let config = config_for(&["a", "b"]);
    let hosts = BTreeMap::from([
        (
            "a".to_string(),
            host_with_series(&[("triad", &[40.0, 42.0, 41.0])]),
        ),
        (
            "b".to_string(),
            host_with_series(&[("triad", &[38.0, 39.0, 40.0])]),
        ),
    ]);
    let results = report::build(&config, hosts, 1, 2);

    let group = &results.aggregates["mem_bandwidth.triad"];
    let a = &group["a"];
    assert_eq!(a.moments.n, 3);
    assert_eq!(a.moments.median, 41.0);
    assert!(a.moments.mad > 0.0);
    assert_eq!(a.unit, Unit::GibPerSec);
    assert_eq!(group["b"].moments.median, 39.0);
}

#[test]
fn outliers_use_medians_so_centered_noise_is_not_flagged() {
    let config: FleetConfig =
        toml::from_str("hosts = [\"a\", \"b\", \"c\", \"d\", \"e\"]\n[thresholds]\nmad_k = 4.0\n")
            .expect("config");
    let mut hosts = BTreeMap::new();
    for (host, values) in [
        ("a", [100.0, 100.5, 99.5]),
        ("b", [100.2, 99.8, 100.2]),
        // Noisy but centered on the fleet median: not a straggler.
        ("c", [80.0, 100.1, 120.0]),
        ("d", [99.9, 100.1, 99.9]),
        // Genuinely slow across every repeat.
        ("e", [60.0, 61.0, 59.0]),
    ] {
        hosts.insert(host.to_string(), host_with_series(&[("triad", &values)]));
    }
    let results = report::build(&config, hosts, 1, 2);

    let flagged = &results.fleet.outliers["mem_bandwidth.triad"];
    assert_eq!(flagged.len(), 1, "{flagged:?}");
    assert_eq!(flagged[0].key, "e");
    // The centered-noise host shows up as a jitter outlier instead.
    let jitter = &results.fleet.jitter_outliers["mem_bandwidth.triad"];
    assert_eq!(jitter.len(), 1, "{jitter:?}");
    assert_eq!(jitter[0].key, "c");
    assert!(jitter[0].deviation_mads > 0.0);
}

#[test]
fn sweep_series_stay_out_of_aggregates() {
    // Two records with the same (subject, repeat) pair: a per-size series,
    // not repeated single measurements.
    let mut obs = HostObservations::default();
    obs.metrics.push(metric_repeat(
        TestId::NcclAllReduce,
        Scope::Node,
        "msg_bytes",
        1024.0,
        0,
    ));
    obs.metrics.push(metric_repeat(
        TestId::NcclAllReduce,
        Scope::Node,
        "msg_bytes",
        4096.0,
        0,
    ));
    obs.metrics.push(metric_repeat(
        TestId::MemBandwidth,
        Scope::Node,
        "triad",
        40.0,
        0,
    ));
    let config = config_for(&["a"]);
    let results = report::build(&config, BTreeMap::from([("a".to_string(), obs)]), 1, 2);

    assert!(!results.aggregates.contains_key("nccl_all_reduce.msg_bytes"));
    assert!(results.aggregates.contains_key("mem_bandwidth.triad"));
}

#[test]
fn low_jitter_is_never_flagged() {
    let config: FleetConfig =
        toml::from_str("hosts = [\"a\", \"b\", \"c\", \"d\", \"e\"]\n[thresholds]\nmad_k = 4.0\n")
            .expect("config");
    let mut hosts = BTreeMap::new();
    // One host with *unusually low* spread must not be flagged; jitter is a
    // one-sided signal.
    for (host, values) in [
        ("a", [100.0, 103.0, 97.0]),
        ("b", [100.0, 104.0, 96.0]),
        ("c", [100.0, 103.5, 96.5]),
        ("d", [100.0, 102.9, 97.1]),
        ("e", [100.0, 100.0, 100.0]),
    ] {
        hosts.insert(host.to_string(), host_with_series(&[("triad", &values)]));
    }
    let results = report::build(&config, hosts, 1, 2);
    let jitter = results.fleet.jitter_outliers.get("mem_bandwidth.triad");
    assert!(
        jitter.is_none_or(|flagged| flagged.iter().all(|o| o.key != "e")),
        "{jitter:?}"
    );
}

#[test]
fn absolute_thresholds_apply_to_the_median_of_repeats() {
    let mut config = config_for(&["a"]);
    config.thresholds.absolute.insert(
        "mem_bandwidth.triad".to_string(),
        Bound {
            min: Some(10.0),
            max: None,
        },
    );
    // One glitchy low sample, but the median clears the bound.
    let hosts = BTreeMap::from([(
        "a".to_string(),
        host_with_series(&[("triad", &[5.0, 50.0, 51.0])]),
    )]);
    let results = report::build(&config, hosts, 1, 2);
    assert!(
        results.fleet.threshold_violations.is_empty(),
        "{:?}",
        results.fleet
    );
}

#[test]
fn rooflines_reduce_over_per_subject_medians() {
    let mut obs = HostObservations::default();
    // GPU 0 d2d has one glitched repeat; the median absorbs it.
    for (repeat, value) in [(0u32, 3000.0), (1, 100.0), (2, 3010.0)] {
        obs.metrics.push(MetricRecord {
            test: TestId::GpuMemBandwidth,
            scope: Scope::Gpu { index: 0 },
            name: "d2d".into(),
            value,
            unit: Unit::GibPerSec,
            repeat,
        });
    }
    let config = config_for(&["a"]);
    let results = report::build(&config, BTreeMap::from([("a".to_string(), obs)]), 1, 2);
    assert_eq!(
        results.calibration.rooflines["a"].gpu_hbm_gib_per_sec,
        Some(3000.0)
    );
}
