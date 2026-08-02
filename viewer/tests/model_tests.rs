//! Contract tests for the RunResults -> ViewModel projection. Results are
//! produced by the real `gauntlet::report::build` pipeline so the viewer
//! stays in lockstep with what runs actually emit.

use std::collections::BTreeMap;

use gauntlet::config::FleetConfig;
use gauntlet::orchestrator::collect::HostObservations;
use gauntlet::proto::{InventorySnapshot, MetricRecord, Scope, TestId, TestOutcome, Unit};
use gauntlet::report::{self, RunResults, Verdict};
use gauntlet_view::model::{Severity, ViewModel, format_value};

fn config(toml: &str) -> FleetConfig {
    toml::from_str(toml).expect("test config parses")
}

fn results(config: &FleetConfig, hosts: BTreeMap<String, HostObservations>) -> RunResults {
    report::build(config, hosts, 100, 160)
}

fn metric(test: TestId, scope: Scope, name: &str, value: f64, unit: Unit) -> MetricRecord {
    MetricRecord {
        test,
        scope,
        name: name.into(),
        value,
        unit,
    }
}

fn pair_bw(obs: &mut HostObservations, peer: &str, gib_per_sec: f64) {
    obs.metrics.push(metric(
        TestId::NetBandwidth,
        Scope::HostPair { peer: peer.into() },
        "gib_per_sec",
        gib_per_sec,
        Unit::GibPerSec,
    ));
}

fn pair_rtt(obs: &mut HostObservations, peer: &str, p50: f64, p99: f64) {
    for (name, value) in [("rtt_p50", p50), ("rtt_p99", p99)] {
        obs.metrics.push(metric(
            TestId::NetLatency,
            Scope::HostPair { peer: peer.into() },
            name,
            value,
            Unit::Micros,
        ));
    }
}

fn inventory(hostname: &str, kernel: &str) -> InventorySnapshot {
    InventorySnapshot {
        hostname: hostname.into(),
        kernel: kernel.into(),
        cpu_model: "test-cpu".into(),
        logical_cores: 8,
        numa_nodes: 1,
        mem_total_bytes: 1 << 34,
        cpu_governor: Some("performance".into()),
        clock_offset_ms: Some(0.1),
        nvidia_driver: None,
        cuda_version: None,
        gpus: Vec::new(),
        nics: Vec::new(),
        ib_ports: Vec::new(),
        xid_errors: Vec::new(),
        gpu_libs: BTreeMap::new(),
    }
}

fn node<'a>(vm: &'a ViewModel, host: &str) -> &'a gauntlet_view::model::NodeView {
    vm.nodes
        .iter()
        .find(|n| n.host == host)
        .unwrap_or_else(|| panic!("node {host} present"))
}

fn edge<'a>(vm: &'a ViewModel, a: &str, b: &str) -> &'a gauntlet_view::model::EdgeView {
    vm.edges
        .iter()
        .find(|e| e.a == a && e.b == b)
        .unwrap_or_else(|| panic!("edge {a}<->{b} present"))
}

#[test]
fn clean_pair_fleet_is_all_ok() {
    let config = config(r#"hosts = ["a", "b"]"#);
    let mut a = HostObservations::default();
    pair_bw(&mut a, "b", 1.0);
    pair_rtt(&mut a, "b", 80.0, 110.0);
    let mut b = HostObservations::default();
    pair_bw(&mut b, "a", 1.1);
    pair_rtt(&mut b, "a", 82.0, 112.0);
    let hosts = BTreeMap::from([("a".to_string(), a), ("b".to_string(), b)]);
    let vm = ViewModel::new(&results(&config, hosts));

    assert_eq!(vm.verdict, Verdict::Clean);
    assert_eq!(vm.nodes.len(), 2);
    assert!(vm.nodes.iter().all(|n| n.severity == Severity::Ok));
    assert_eq!(vm.edges.len(), 1);
    let e = edge(&vm, "a", "b");
    assert_eq!(e.severity, Severity::Ok);
    assert_eq!(e.bandwidth_gib.from_a, Some(1.0));
    assert_eq!(e.bandwidth_gib.from_b, Some(1.1));
    assert_eq!(e.bandwidth_gib.min(), Some(1.0));
    assert_eq!(e.rtt_p50_us.from_a, Some(80.0));
    assert_eq!(e.rtt_p99_us.from_b, Some(112.0));
    assert!(e.issues.is_empty());
    // Pair metrics land in the table with per-direction subjects.
    assert!(
        vm.rows
            .iter()
            .any(|r| r.group == "net_bandwidth.gib_per_sec" && r.subject == "a:pair:b")
    );
}

#[test]
fn fatal_error_marks_node_bad() {
    let config = config(r#"hosts = ["a", "b"]"#);
    let mut a = HostObservations::default();
    a.errors.push("agent exploded".into());
    let hosts = BTreeMap::from([
        ("a".to_string(), a),
        ("b".to_string(), HostObservations::default()),
    ]);
    let vm = ViewModel::new(&results(&config, hosts));

    assert_eq!(vm.verdict, Verdict::HostFailures);
    let n = node(&vm, "a");
    assert_eq!(n.severity, Severity::Bad);
    assert!(
        n.issues
            .iter()
            .any(|(s, text)| { *s == Severity::Bad && text.contains("agent exploded") })
    );
    assert_eq!(node(&vm, "b").severity, Severity::Ok);
}

#[test]
fn failed_node_scope_outcome_marks_node_bad() {
    let config = config(r#"hosts = ["a", "b"]"#);
    let mut a = HostObservations::default();
    a.outcomes.push((
        TestId::GpuGemmCorrectness,
        Scope::Gpu { index: 0 },
        TestOutcome::Failed {
            reason: "residual too large".into(),
        },
    ));
    let hosts = BTreeMap::from([
        ("a".to_string(), a),
        ("b".to_string(), HostObservations::default()),
    ]);
    let vm = ViewModel::new(&results(&config, hosts));

    assert_eq!(vm.verdict, Verdict::Stragglers);
    let n = node(&vm, "a");
    assert_eq!(n.severity, Severity::Bad);
    assert!(n.issues.iter().any(|(_, text)| {
        text.contains("gpu_gemm_correctness") && text.contains("residual too large")
    }));
}

#[test]
fn failed_pair_outcome_marks_edge_not_node() {
    let config = config(r#"hosts = ["a", "b"]"#);
    let mut a = HostObservations::default();
    a.outcomes.push((
        TestId::NetBandwidth,
        Scope::HostPair { peer: "b".into() },
        TestOutcome::Failed {
            reason: "peer connection refused".into(),
        },
    ));
    let hosts = BTreeMap::from([
        ("a".to_string(), a),
        ("b".to_string(), HostObservations::default()),
    ]);
    let vm = ViewModel::new(&results(&config, hosts));

    assert_eq!(node(&vm, "a").severity, Severity::Ok);
    assert_eq!(node(&vm, "b").severity, Severity::Ok);
    let e = edge(&vm, "a", "b");
    assert_eq!(e.severity, Severity::Bad);
    assert!(
        e.issues
            .iter()
            .any(|(s, text)| { *s == Severity::Bad && text.contains("peer connection refused") })
    );
}

#[test]
fn slow_pair_flags_edge_warn_and_rows() {
    let config = config("hosts = [\"a\", \"b\", \"c\", \"d\"]\n[thresholds]\nmad_k = 3.0\n");
    let names = ["a", "b", "c", "d"];
    let mut hosts = BTreeMap::new();
    for (i, host) in names.iter().enumerate() {
        let mut obs = HostObservations::default();
        for peer in names.iter().filter(|p| *p != host) {
            // The c<->d path is degraded in both directions; every other
            // pair sits near 1.0 with a little spread so the MAD is nonzero.
            let value = if (*host == "c" && *peer == "d") || (*host == "d" && *peer == "c") {
                0.3
            } else {
                1.0 + i as f64 * 0.01
            };
            pair_bw(&mut obs, peer, value);
        }
        hosts.insert(host.to_string(), obs);
    }
    let vm = ViewModel::new(&results(&config, hosts));

    assert_eq!(vm.verdict, Verdict::Stragglers);
    assert!(vm.nodes.iter().all(|n| n.severity == Severity::Ok));
    let e = edge(&vm, "c", "d");
    assert_eq!(e.severity, Severity::Warn);
    assert!(
        e.issues.iter().any(|(s, text)| {
            *s == Severity::Warn && text.contains("net_bandwidth.gib_per_sec")
        })
    );
    // Healthy edges stay Ok.
    assert_eq!(edge(&vm, "a", "b").severity, Severity::Ok);
    // Both directed rows of the degraded pair are flagged.
    for subject in ["c:pair:d", "d:pair:c"] {
        let row = vm
            .rows
            .iter()
            .find(|r| r.group == "net_bandwidth.gib_per_sec" && r.subject == subject)
            .expect("row present");
        assert!(row.flagged, "{subject} flagged");
        assert!(row.deviation_mads.expect("deviation computed") < 0.0);
    }
}

#[test]
fn consistency_dissent_marks_node_warn() {
    let config = config(r#"hosts = ["a", "b", "c"]"#);
    let mut hosts = BTreeMap::new();
    for (host, kernel) in [("a", "6.1.0"), ("b", "6.1.0"), ("c", "5.15.0")] {
        let obs = HostObservations {
            inventory: Some(inventory(&format!("node-{host}"), kernel)),
            ..Default::default()
        };
        hosts.insert(host.to_string(), obs);
    }
    let vm = ViewModel::new(&results(&config, hosts));

    let c = node(&vm, "c");
    assert_eq!(c.severity, Severity::Warn);
    assert!(c.issues.iter().any(|(s, text)| {
        *s == Severity::Warn && text.contains("kernel") && text.contains("5.15.0")
    }));
    assert_eq!(node(&vm, "a").severity, Severity::Ok);
    assert_eq!(node(&vm, "c").hostname.as_deref(), Some("node-c"));
}

#[test]
fn threshold_violation_marks_node_warn() {
    let config = config(
        "hosts = [\"a\", \"b\"]\n[thresholds.absolute.\"mem_bandwidth.triad\"]\nmin = 10.0\n",
    );
    let mut hosts = BTreeMap::new();
    for (host, triad) in [("a", 30.0), ("b", 4.0)] {
        let mut obs = HostObservations::default();
        obs.metrics.push(metric(
            TestId::MemBandwidth,
            Scope::Node,
            "triad",
            triad,
            Unit::GibPerSec,
        ));
        hosts.insert(host.to_string(), obs);
    }
    let vm = ViewModel::new(&results(&config, hosts));

    assert_eq!(node(&vm, "b").severity, Severity::Warn);
    assert_eq!(node(&vm, "a").severity, Severity::Ok);
    let row = vm
        .rows
        .iter()
        .find(|r| r.group == "mem_bandwidth.triad" && r.subject == "b")
        .expect("row present");
    assert!(row.violated);
}

#[test]
fn sweep_series_are_excluded_from_rows_but_links_survive() {
    let config = config(r#"hosts = ["a", "b"]"#);
    let mut hosts = BTreeMap::new();
    for host in ["a", "b"] {
        let mut obs = HostObservations::default();
        for (bytes, us) in [(1024.0, 30.0), (1_048_576.0, 900.0)] {
            obs.metrics.push(metric(
                TestId::NcclAllReduce,
                Scope::Node,
                "msg_bytes",
                bytes,
                Unit::Bytes,
            ));
            obs.metrics.push(metric(
                TestId::NcclAllReduce,
                Scope::Node,
                "elapsed_us",
                us,
                Unit::Micros,
            ));
        }
        hosts.insert(host.to_string(), obs);
    }
    let vm = ViewModel::new(&results(&config, hosts));

    assert!(
        !vm.rows
            .iter()
            .any(|r| r.group.starts_with("nccl_all_reduce"))
    );
    let link = vm
        .links
        .iter()
        .find(|l| l.class == "nccl_allreduce_fleet")
        .expect("fleet fit present");
    assert!(link.gib_per_sec > 0.0);
    assert!(link.alpha_us >= 0.0);
}

#[test]
fn rooflines_surface_in_node_stats() {
    let config = config(r#"hosts = ["a"]"#);
    let mut obs = HostObservations::default();
    obs.metrics.push(metric(
        TestId::CpuGflops,
        Scope::Node,
        "gflops_allcore",
        620.0,
        Unit::Gflops,
    ));
    obs.metrics.push(metric(
        TestId::MemBandwidth,
        Scope::Node,
        "triad_allnode",
        41.2,
        Unit::GibPerSec,
    ));
    let hosts = BTreeMap::from([("a".to_string(), obs)]);
    let vm = ViewModel::new(&results(&config, hosts));

    let stats = &node(&vm, "a").stats;
    assert!(
        stats
            .iter()
            .any(|(label, value)| { label.contains("cpu") && value.contains("620") })
    );
    assert!(
        stats
            .iter()
            .any(|(label, value)| { label.contains("dram") && value.contains("41.2") })
    );
}

#[test]
fn deviation_is_computed_for_unflagged_rows() {
    // High mad_k so nothing gets flagged, but deviations still render.
    let config = config("hosts = [\"a\", \"b\", \"c\", \"d\"]\n[thresholds]\nmad_k = 50.0\n");
    let mut hosts = BTreeMap::new();
    for (host, gflops) in [("a", 100.0), ("b", 101.0), ("c", 102.0), ("d", 110.0)] {
        let mut obs = HostObservations::default();
        obs.metrics.push(metric(
            TestId::CpuGflops,
            Scope::Node,
            "gflops_allcore",
            gflops,
            Unit::Gflops,
        ));
        hosts.insert(host.to_string(), obs);
    }
    let vm = ViewModel::new(&results(&config, hosts));

    let row = vm
        .rows
        .iter()
        .find(|r| r.group == "cpu_gflops.gflops_allcore" && r.subject == "d")
        .expect("row present");
    assert!(!row.flagged);
    let deviation = row.deviation_mads.expect("deviation computed");
    assert!(deviation > 0.0, "d is above the median, got {deviation}");
}

#[test]
fn format_value_uses_unit_labels() {
    assert_eq!(format_value(Unit::GibPerSec, 1.234), "1.23 GiB/s");
    assert_eq!(format_value(Unit::Gflops, 620.4), "620 GFLOPS");
    assert_eq!(format_value(Unit::Gflops, 0.42), "0.42 GFLOPS");
    assert_eq!(format_value(Unit::Micros, 88.25), "88.2 µs");
    assert_eq!(format_value(Unit::Celsius, 71.0), "71 °C");
    assert_eq!(format_value(Unit::Mhz, 2520.0), "2520 MHz");
    assert_eq!(format_value(Unit::Bytes, 268435456.0), "256.0 MiB");
    assert_eq!(format_value(Unit::Bytes, 512.0), "512 B");
    assert_eq!(format_value(Unit::Count, 3.0), "3");
    assert_eq!(format_value(Unit::Residual, 0.0000242), "2.42e-5");
    assert_eq!(format_value(Unit::Ratio, 0.5), "0.500");
}

#[test]
fn run_metadata_is_projected() {
    let config = config(r#"hosts = ["a"]"#);
    let hosts = BTreeMap::from([("a".to_string(), HostObservations::default())]);
    let results = results(&config, hosts);
    let vm = ViewModel::new(&results);
    assert_eq!(vm.run_id, results.run_id);
    assert_eq!(vm.wall_secs, 60);
}
