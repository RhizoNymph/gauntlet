//! Report handling of the fleet-level NCCL sweep headlines and the
//! inter-node world shapes (proto v11 / schema v13).

use std::collections::BTreeMap;

use gauntlet::config::FleetConfig;
use gauntlet::orchestrator::collect::HostObservations;
use gauntlet::proto::{MetricRecord, Scope, TestId, Unit};
use gauntlet::report;

fn config_for(hosts: &[String]) -> FleetConfig {
    let list = hosts
        .iter()
        .map(|h| format!("\"{h}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let config: FleetConfig = toml::from_str(&format!("hosts = [{list}]")).expect("test config");
    config.validate().expect("valid");
    config
}

fn node(test: TestId, name: &str, value: f64, unit: Unit, repeat: u32) -> MetricRecord {
    MetricRecord {
        test,
        scope: Scope::Node,
        name: name.into(),
        value,
        unit,
        repeat,
    }
}

/// Per-size series (alpha 10 us + 1 us/KiB) under `test`, for one repeat.
fn sweep_series(obs: &mut HostObservations, test: TestId, repeat: u32) {
    for kib in [1u64, 4, 16, 64, 256] {
        let bytes = (kib * 1024) as f64;
        let elapsed = 10.0 + kib as f64;
        obs.metrics
            .push(node(test, "elapsed_us", elapsed, Unit::Micros, repeat));
        obs.metrics
            .push(node(test, "msg_bytes", bytes, Unit::Bytes, repeat));
        obs.metrics.push(node(
            test,
            "bus_gib_per_sec",
            bytes / elapsed,
            Unit::GibPerSec,
            repeat,
        ));
    }
}

/// An inter-node series as the lead emits it: the `ranks` opener (world
/// size), then the per-size records.
fn inter_series(obs: &mut HostObservations, test: TestId, ranks: u32, repeat: u32) {
    obs.metrics
        .push(node(test, "ranks", f64::from(ranks), Unit::Count, repeat));
    sweep_series(obs, test, repeat);
}

fn hosts(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("10.0.0.{i}")).collect()
}

#[test]
fn the_fleet_headline_aggregates_across_repeats_but_never_enters_mad() {
    let names = hosts(6);
    let config = config_for(&names);
    let mut observations: BTreeMap<String, HostObservations> = names
        .iter()
        .map(|host| (host.clone(), HostObservations::default()))
        .collect();
    // The lead carries one fleet-level value per repeat; one repeat is
    // wildly off (a jittery fabric), which must show in the moments.
    let lead = observations.get_mut(&names[0]).expect("lead");
    for (repeat, value) in [(0, 20.0), (1, 21.0), (2, 2.0), (3, 20.5)] {
        lead.metrics.push(node(
            TestId::NcclAllReduce,
            "bus_gib_per_sec_peak",
            value,
            Unit::GibPerSec,
            repeat,
        ));
    }
    let results = report::build(&config, observations, 1, 2);

    let group = "nccl_all_reduce.bus_gib_per_sec_peak";
    let subjects = &results.aggregates[group];
    assert_eq!(subjects.len(), 1, "one subject: the lead host");
    let moments = &subjects[&names[0]].moments;
    assert_eq!(moments.n, 4);
    assert!(moments.min < moments.median, "{moments:?}");
    // Never a degenerate one-subject MAD group, nor a jitter group.
    assert!(!results.fleet.outliers.contains_key(group));
    assert!(!results.fleet.jitter_outliers.contains_key(group));
}

#[test]
fn barrier_fleet_span_aggregates_but_never_enters_mad() {
    // fleet_span_* is the same shape as the sweep headlines: one series
    // per run on the lead host, so the same rule keeps it out of MAD.
    let names = hosts(6);
    let config = config_for(&names);
    let mut observations: BTreeMap<String, HostObservations> = names
        .iter()
        .map(|host| (host.clone(), HostObservations::default()))
        .collect();
    for (test, lead) in [(TestId::NcclBarrier, 0), (TestId::TcpBarrier, 1)] {
        let obs = observations.get_mut(&names[lead]).expect("lead");
        for (repeat, value) in [(0, 40.0), (1, 41.0), (2, 400.0), (3, 39.0)] {
            obs.metrics
                .push(node(test, "fleet_span_p99_us", value, Unit::Micros, repeat));
        }
    }
    let results = report::build(&config, observations, 1, 2);
    for (group, lead) in [
        ("nccl_barrier.fleet_span_p99_us", &names[0]),
        ("tcp_barrier.fleet_span_p99_us", &names[1]),
    ] {
        let subjects = &results.aggregates[group];
        assert_eq!(subjects.len(), 1, "{group}: one subject, the lead");
        assert_eq!(subjects[lead].moments.n, 4, "{group}");
        assert!(!results.fleet.outliers.contains_key(group), "{group}");
        assert!(
            !results.fleet.jitter_outliers.contains_key(group),
            "{group}"
        );
    }
}

#[test]
fn a_low_fleet_headline_still_trips_an_absolute_floor() {
    let names = hosts(2);
    let mut config = config_for(&names);
    config.thresholds.absolute.insert(
        "nccl_inter_all_reduce.bus_gib_per_sec_peak".to_string(),
        toml::from_str("min = 10.0").expect("bound"),
    );
    let mut lead = HostObservations::default();
    lead.metrics.push(node(
        TestId::NcclInterAllReduce,
        "bus_gib_per_sec_peak",
        4.0,
        Unit::GibPerSec,
        0,
    ));
    let observations = BTreeMap::from([
        (names[0].clone(), lead),
        (names[1].clone(), HostObservations::default()),
    ]);
    let results = report::build(&config, observations, 1, 2);
    assert_eq!(
        results.fleet.threshold_violations["nccl_inter_all_reduce.bus_gib_per_sec_peak"],
        [names[0].clone()]
    );
}

#[test]
fn inter_node_series_fit_their_own_link_classes() {
    let names = hosts(2);
    let config = config_for(&names);
    let mut lead = HostObservations::default();
    // Two rails of the same world size, both led by the same host: their
    // series pool into one inter-node class keyed by that size, never
    // into the rank-per-GPU one.
    inter_series(&mut lead, TestId::NcclInterAllReduce, 2, 0);
    inter_series(&mut lead, TestId::NcclInterAllReduce, 2, 0);
    inter_series(&mut lead, TestId::NcclInterAllGather, 2, 0);
    let observations = BTreeMap::from([
        (names[0].clone(), lead),
        (names[1].clone(), HostObservations::default()),
    ]);
    let results = report::build(&config, observations, 1, 2);
    let links = &results.calibration.links;
    let fit = links["nccl_allreduce_inter_node_2rank"];
    assert!((fit.alpha_us - 10.0).abs() < 1e-6, "{fit:?}");
    assert!(links.contains_key("nccl_allgather_inter_node_2rank"));
    assert!(!links.contains_key("nccl_allreduce_rank_per_gpu"));
    // Per-size series repeat their sample key: never aggregated, never MAD.
    assert!(
        !results
            .aggregates
            .contains_key("nccl_inter_all_reduce.bus_gib_per_sec")
    );
}

#[test]
fn a_heterogeneous_fleets_rails_fit_one_class_per_world_size() {
    // Hosts with 8, 8 and 4 GPUs: rails 0-3 are 3-rank worlds, rails 4-7
    // 2-rank worlds, all led by the first host in one repeat.
    let names = hosts(3);
    let config = config_for(&names);
    let mut lead = HostObservations::default();
    for rail in 0..8 {
        let ranks = if rail < 4 { 3 } else { 2 };
        inter_series(&mut lead, TestId::NcclInterAllReduce, ranks, 0);
    }
    let mut observations: BTreeMap<String, HostObservations> = names
        .iter()
        .map(|host| (host.clone(), HostObservations::default()))
        .collect();
    observations.insert(names[0].clone(), lead);
    let results = report::build(&config, observations, 1, 2);
    let inter: Vec<&str> = results
        .calibration
        .links
        .keys()
        .map(String::as_str)
        .filter(|key| key.contains("inter_node"))
        .collect();
    assert_eq!(
        inter,
        [
            "nccl_allreduce_inter_node_2rank",
            "nccl_allreduce_inter_node_3rank"
        ]
    );
}

#[test]
fn the_per_rail_and_per_node_headlines_are_different_groups() {
    // rank_per_node's GPU-0 number and per_rail's worst-rail number are
    // different quantities: never one metric group.
    let names = hosts(2);
    let config = config_for(&names);
    let mut lead = HostObservations::default();
    for (name, value) in [
        ("bus_gib_per_sec_peak", 22.0),
        ("bus_gib_per_sec_peak_min_rail", 4.0),
    ] {
        lead.metrics.push(node(
            TestId::NcclInterAllReduce,
            name,
            value,
            Unit::GibPerSec,
            0,
        ));
    }
    let observations = BTreeMap::from([
        (names[0].clone(), lead),
        (names[1].clone(), HostObservations::default()),
    ]);
    let results = report::build(&config, observations, 1, 2);
    let per_node = &results.aggregates["nccl_inter_all_reduce.bus_gib_per_sec_peak"];
    let min_rail = &results.aggregates["nccl_inter_all_reduce.bus_gib_per_sec_peak_min_rail"];
    assert_eq!(per_node[&names[0]].moments.median, 22.0);
    assert_eq!(min_rail[&names[0]].moments.median, 4.0);
}

#[test]
fn per_rail_headlines_are_fleet_level_too() {
    let names = hosts(5);
    let config = config_for(&names);
    let mut observations: BTreeMap<String, HostObservations> = names
        .iter()
        .map(|host| (host.clone(), HostObservations::default()))
        .collect();
    // Rail leads can differ (a host without GPU r cannot lead rail r);
    // even spread over several hosts these are not fleet peers.
    for (index, host) in names.iter().enumerate() {
        let obs = observations.get_mut(host).expect("host");
        obs.metrics.push(node(
            TestId::NcclInterAllGather,
            "bus_gib_per_sec_peak_rail0",
            if index == 0 { 1.0 } else { 20.0 },
            Unit::GibPerSec,
            0,
        ));
    }
    let results = report::build(&config, observations, 1, 2);
    assert!(
        !results
            .fleet
            .outliers
            .contains_key("nccl_inter_all_gather.bus_gib_per_sec_peak_rail0")
    );
    assert_eq!(
        report::test_display_name(TestId::NcclInterAllGather),
        "nccl_inter_all_gather"
    );
}
