//! Report derivations for the rank-per-GPU fleet NCCL world (proto v7 /
//! schema v8): per-GPU fleet all-reduce retention and per-GPU barrier
//! straggler keys.

use std::collections::BTreeMap;

use gauntlet::config::FleetConfig;
use gauntlet::orchestrator::collect::HostObservations;
use gauntlet::proto::{MetricRecord, Scope, TestId, Unit};
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

fn gpu_record(test: TestId, gpu: u32, name: &str, value: f64, unit: Unit) -> MetricRecord {
    MetricRecord {
        test,
        scope: Scope::Gpu { index: gpu },
        name: name.into(),
        value,
        unit,
        repeat: 0,
    }
}

/// One host of the rank-per-GPU fleet overlap step: every GPU is a rank
/// with its own (gpu, isolated, overlapped) bus pair.
fn per_gpu_fleet_host(bus: &[(u32, f64, f64)]) -> HostObservations {
    let mut obs = HostObservations::default();
    for (gpu, isolated, overlapped) in bus {
        obs.metrics.push(gpu_record(
            TestId::OverlapFleetAllReduce,
            *gpu,
            "isolated_bus_gib_per_sec",
            *isolated,
            Unit::GibPerSec,
        ));
        obs.metrics.push(gpu_record(
            TestId::OverlapFleetAllReduce,
            *gpu,
            "overlap_bus_gib_per_sec",
            *overlapped,
            Unit::GibPerSec,
        ));
    }
    obs
}

#[test]
fn fleet_all_reduce_retention_divides_each_gpus_own_isolated_window() {
    let config = config_for(&["a"]);
    // GPU 1's isolated window is slower than GPU 0's; each ratio must use
    // its own GPU's baseline, never a sibling's.
    let obs = per_gpu_fleet_host(&[(0, 40.0, 30.0), (1, 20.0, 18.0)]);
    let results = report::build(&config, BTreeMap::from([("a".to_string(), obs)]), 1, 2);
    let retention: BTreeMap<Scope, f64> = results.hosts["a"]
        .metrics
        .iter()
        .filter(|record| {
            record.test == TestId::OverlapRetention && record.name == "fleet_all_reduce"
        })
        .map(|record| (record.scope.clone(), record.value))
        .collect();
    assert_eq!(retention.len(), 2, "{retention:?}");
    assert!((retention[&Scope::Gpu { index: 0 }] - 0.75).abs() < 1e-12);
    assert!((retention[&Scope::Gpu { index: 1 }] - 0.9).abs() < 1e-12);
    // The group is keyed per GPU subject.
    let subjects = &results.aggregates["overlap_retention.fleet_all_reduce"];
    assert!(subjects.contains_key("a:gpu0"), "{subjects:?}");
    assert!(subjects.contains_key("a:gpu1"), "{subjects:?}");
}

#[test]
fn a_gpu_without_its_own_isolated_window_derives_no_retention() {
    let config = config_for(&["a"]);
    let mut obs = per_gpu_fleet_host(&[(0, 40.0, 30.0)]);
    // GPU 1 reported only the overlapped number: a sibling's baseline must
    // not stand in for it.
    obs.metrics.push(gpu_record(
        TestId::OverlapFleetAllReduce,
        1,
        "overlap_bus_gib_per_sec",
        25.0,
        Unit::GibPerSec,
    ));
    let results = report::build(&config, BTreeMap::from([("a".to_string(), obs)]), 1, 2);
    let scopes: Vec<Scope> = results.hosts["a"]
        .metrics
        .iter()
        .filter(|record| record.test == TestId::OverlapRetention)
        .map(|record| record.scope.clone())
        .collect();
    assert_eq!(scopes, [Scope::Gpu { index: 0 }]);
}

#[test]
fn a_bad_non_zero_gpu_path_is_a_per_gpu_fleet_retention_outlier() {
    // Five 2-GPU hosts; n3's GPU 1 (a bad NIC/riser behind a non-zero GPU,
    // invisible when only GPU 0 carried the collective) collapses to 30%.
    let names = ["n1", "n2", "n3", "n4", "n5"];
    let config = config_for(&names);
    let mut observations = BTreeMap::new();
    for (i, name) in names.iter().enumerate() {
        let healthy = 30.0 + 0.1 * i as f64;
        let gpu1 = if *name == "n3" { 12.0 } else { healthy + 0.05 };
        observations.insert(
            (*name).to_string(),
            per_gpu_fleet_host(&[(0, 40.0, healthy), (1, 40.0, gpu1)]),
        );
    }
    let results = report::build(&config, observations, 1, 2);
    let flagged = results
        .fleet
        .outliers
        .get("overlap_retention.fleet_all_reduce")
        .expect("fleet retention outlier group");
    assert_eq!(flagged.len(), 1, "{flagged:?}");
    assert_eq!(flagged[0].key, "n3:gpu1", "{flagged:?}");
    assert!(flagged[0].deviation_mads < 0.0);
    assert_eq!(report::verdict(&results), Verdict::Stragglers);
}

fn node_record(test: TestId, name: &str, value: f64, unit: Unit) -> MetricRecord {
    MetricRecord {
        test,
        scope: Scope::Node,
        name: name.into(),
        value,
        unit,
        repeat: 0,
    }
}

#[test]
fn nccl_barrier_stragglers_are_one_row_per_host_with_per_gpu_percentiles() {
    // Four 2-GPU hosts. A host's ranks share one arrival group, so the
    // tally is emitted once per host (node scope) and a late host is one
    // straggler row, not one per GPU; the percentiles stay per GPU.
    let names = ["n1", "n2", "n3", "n4"];
    let config = config_for(&names);
    let mut observations = BTreeMap::new();
    for name in names {
        let mut obs = HostObservations::default();
        let frac = if name == "n2" { 0.9 } else { 0.03 };
        obs.metrics.push(node_record(
            TestId::NcclBarrier,
            "slowest_frac",
            frac,
            Unit::Ratio,
        ));
        obs.metrics.push(node_record(
            TestId::NcclBarrier,
            "slowest_considered",
            1800.0,
            Unit::Count,
        ));
        for gpu in 0..2 {
            obs.metrics.push(gpu_record(
                TestId::NcclBarrier,
                gpu,
                "p50_us",
                100.0,
                Unit::Micros,
            ));
        }
        observations.insert(name.to_string(), obs);
    }
    let results = report::build(&config, observations, 1, 2);
    let flagged: Vec<&str> = results.fleet.barrier_stragglers["nccl_barrier"]
        .iter()
        .map(|straggler| straggler.key.as_str())
        .collect();
    assert_eq!(flagged, ["n2"]);
    assert_eq!(report::verdict(&results), Verdict::Stragglers);
    let p50 = &results.aggregates["nccl_barrier.p50_us"];
    assert!(
        p50.contains_key("n1:gpu0") && p50.contains_key("n1:gpu1"),
        "{p50:?}"
    );
    let tally = &results.aggregates["nccl_barrier.slowest_frac"];
    assert_eq!(tally.len(), 4, "one tally subject per host: {tally:?}");
}

#[test]
fn schema_version_tracks_the_rank_per_gpu_granularity_change() {
    assert_eq!(report::SCHEMA_VERSION, 8);
}
