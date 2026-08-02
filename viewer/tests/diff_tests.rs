//! Contract tests for run-to-run diffing: unit-aware direction of
//! goodness, regression severities, and node/edge attribution.

use gauntlet::proto::Unit;
use gauntlet::report::Verdict;
use gauntlet_view::diff::{DiffView, higher_is_better};
use gauntlet_view::model::{MetricRow, NodeView, Severity, ViewModel};

fn row(group: &str, subject: &str, value: f64, unit: Unit) -> MetricRow {
    MetricRow {
        group: group.into(),
        subject: subject.into(),
        value,
        unit,
        deviation_mads: None,
        flagged: false,
        violated: false,
    }
}

fn node(host: &str) -> NodeView {
    NodeView {
        host: host.into(),
        hostname: None,
        severity: Severity::Ok,
        issues: Vec::new(),
        stats: Vec::new(),
    }
}

fn vm(run_id: &str, hosts: &[&str], rows: Vec<MetricRow>) -> ViewModel {
    ViewModel {
        run_id: run_id.into(),
        wall_secs: 60,
        verdict: Verdict::Clean,
        debug_build: false,
        nodes: hosts.iter().map(|h| node(h)).collect(),
        edges: Vec::new(),
        rows,
        links: Vec::new(),
    }
}

#[test]
fn direction_of_goodness_follows_units() {
    assert_eq!(higher_is_better(Unit::Gflops), Some(true));
    assert_eq!(higher_is_better(Unit::GibPerSec), Some(true));
    assert_eq!(higher_is_better(Unit::Mhz), Some(true));
    assert_eq!(higher_is_better(Unit::Micros), Some(false));
    assert_eq!(higher_is_better(Unit::Millis), Some(false));
    assert_eq!(higher_is_better(Unit::Celsius), Some(false));
    assert_eq!(higher_is_better(Unit::Residual), Some(false));
    assert_eq!(higher_is_better(Unit::Bytes), None);
    assert_eq!(higher_is_better(Unit::Count), None);
    assert_eq!(higher_is_better(Unit::Ratio), None);
}

#[test]
fn throughput_drop_regresses_and_gain_improves() {
    let baseline = vm(
        "base",
        &["a"],
        vec![
            row("cpu_gflops.gflops_allcore", "a", 100.0, Unit::Gflops),
            row("mem_bandwidth.triad", "a", 40.0, Unit::GibPerSec),
        ],
    );
    let current = vm(
        "cur",
        &["a"],
        vec![
            // 10% down: warn-level regression.
            row("cpu_gflops.gflops_allcore", "a", 90.0, Unit::Gflops),
            // 10% up: improvement.
            row("mem_bandwidth.triad", "a", 44.0, Unit::GibPerSec),
        ],
    );
    let diff = DiffView::new(&current, &baseline);

    assert_eq!(diff.baseline_run_id, "base");
    let slow = &diff.rows[&("cpu_gflops.gflops_allcore".to_string(), "a".to_string())];
    assert_eq!(slow.baseline_value, 100.0);
    let delta = slow.delta_fraction.expect("delta");
    assert!((delta + 0.1).abs() < 1e-9, "got {delta}");
    assert_eq!(slow.severity, Severity::Warn);
    assert!(!slow.improved);

    let fast = &diff.rows[&("mem_bandwidth.triad".to_string(), "a".to_string())];
    assert_eq!(fast.severity, Severity::Ok);
    assert!(fast.improved);

    assert_eq!(diff.node_severity.get("a"), Some(&Severity::Warn));
    let issues = &diff.node_issues["a"];
    assert!(
        issues.iter().any(|(s, text)| {
            *s == Severity::Warn && text.contains("cpu_gflops.gflops_allcore")
        })
    );
    // Improvements are not issues.
    assert!(!issues.iter().any(|(_, text)| text.contains("triad")));
}

#[test]
fn big_regressions_are_bad() {
    let baseline = vm("base", &["a"], vec![row("g.m", "a", 100.0, Unit::Gflops)]);
    let current = vm("cur", &["a"], vec![row("g.m", "a", 80.0, Unit::Gflops)]);
    let diff = DiffView::new(&current, &baseline);
    assert_eq!(
        diff.rows[&("g.m".to_string(), "a".to_string())].severity,
        Severity::Bad
    );
    assert_eq!(diff.node_severity.get("a"), Some(&Severity::Bad));
}

#[test]
fn latency_direction_is_inverted() {
    let baseline = vm(
        "base",
        &["a"],
        vec![row("net_latency.rtt_p50", "a:pair:b", 100.0, Unit::Micros)],
    );
    let current = vm(
        "cur",
        &["a", "b"],
        vec![row("net_latency.rtt_p50", "a:pair:b", 120.0, Unit::Micros)],
    );
    let diff = DiffView::new(&current, &baseline);
    let delta = &diff.rows[&("net_latency.rtt_p50".to_string(), "a:pair:b".to_string())];
    // +20% latency is a bad regression.
    assert_eq!(delta.severity, Severity::Bad);
    assert!(delta.delta_fraction.expect("delta") > 0.0);
}

#[test]
fn pair_rows_color_edges_not_nodes() {
    let baseline = vm(
        "base",
        &["a", "b"],
        vec![row(
            "net_bandwidth.gib_per_sec",
            "b:pair:a",
            1.0,
            Unit::GibPerSec,
        )],
    );
    let current = vm(
        "cur",
        &["a", "b"],
        vec![row(
            "net_bandwidth.gib_per_sec",
            "b:pair:a",
            0.5,
            Unit::GibPerSec,
        )],
    );
    let diff = DiffView::new(&current, &baseline);

    // Edge key is ordered (a, b) regardless of measuring direction.
    assert_eq!(
        diff.edge_severity.get(&("a".to_string(), "b".to_string())),
        Some(&Severity::Bad)
    );
    assert!(
        diff.edge_issues[&("a".to_string(), "b".to_string())]
            .iter()
            .any(|(_, text)| text.contains("net_bandwidth"))
    );
    assert!(!diff.node_severity.contains_key("a"));
    assert!(!diff.node_severity.contains_key("b"));
}

#[test]
fn no_judgement_units_get_delta_but_stay_ok() {
    let baseline = vm("base", &["a"], vec![row("x.ratio", "a", 1.0, Unit::Ratio)]);
    let current = vm("cur", &["a"], vec![row("x.ratio", "a", 2.0, Unit::Ratio)]);
    let diff = DiffView::new(&current, &baseline);
    let delta = &diff.rows[&("x.ratio".to_string(), "a".to_string())];
    assert_eq!(delta.severity, Severity::Ok);
    assert!(!delta.improved);
    assert!((delta.delta_fraction.expect("delta") - 1.0).abs() < 1e-9);
    assert!(!diff.node_severity.contains_key("a"));
}

#[test]
fn zero_or_nonfinite_baseline_yields_no_delta() {
    let baseline = vm(
        "base",
        &["a"],
        vec![
            row("g.zero", "a", 0.0, Unit::Gflops),
            row("g.nan", "a", f64::NAN, Unit::Gflops),
        ],
    );
    let current = vm(
        "cur",
        &["a"],
        vec![
            row("g.zero", "a", 50.0, Unit::Gflops),
            row("g.nan", "a", 50.0, Unit::Gflops),
        ],
    );
    let diff = DiffView::new(&current, &baseline);
    assert!(
        !diff
            .rows
            .contains_key(&("g.zero".to_string(), "a".to_string()))
    );
    assert!(
        !diff
            .rows
            .contains_key(&("g.nan".to_string(), "a".to_string()))
    );
}

#[test]
fn rows_missing_from_baseline_are_ignored() {
    let baseline = vm("base", &["a"], vec![]);
    let current = vm("cur", &["a"], vec![row("g.m", "a", 100.0, Unit::Gflops)]);
    let diff = DiffView::new(&current, &baseline);
    assert!(diff.rows.is_empty());
    assert!(diff.node_severity.is_empty());
}

#[test]
fn node_issue_lists_are_capped_with_a_summary_line() {
    let mut base_rows = Vec::new();
    let mut cur_rows = Vec::new();
    for i in 0..12 {
        base_rows.push(row(
            "cpu_gflops.gflops",
            &format!("a:core{i}"),
            100.0,
            Unit::Gflops,
        ));
        cur_rows.push(row(
            "cpu_gflops.gflops",
            &format!("a:core{i}"),
            80.0,
            Unit::Gflops,
        ));
    }
    let diff = DiffView::new(&vm("cur", &["a"], cur_rows), &vm("base", &["a"], base_rows));
    let issues = &diff.node_issues["a"];
    assert!(issues.len() <= 7, "got {}", issues.len());
    assert!(
        issues
            .last()
            .expect("issues present")
            .1
            .contains("more regression")
    );
}

#[test]
fn worst_regressions_lead_the_issue_list() {
    let baseline = vm(
        "base",
        &["a"],
        vec![
            row("g.mild", "a", 100.0, Unit::Gflops),
            row("g.severe", "a", 100.0, Unit::Gflops),
        ],
    );
    let current = vm(
        "cur",
        &["a"],
        vec![
            row("g.mild", "a", 93.0, Unit::Gflops),
            row("g.severe", "a", 50.0, Unit::Gflops),
        ],
    );
    let diff = DiffView::new(&current, &baseline);
    let issues = &diff.node_issues["a"];
    assert!(issues[0].1.contains("g.severe"));
    assert_eq!(issues[0].0, Severity::Bad);
}
