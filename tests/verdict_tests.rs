//! Verdict classification and the exit-code contract: hard evidence
//! (Failures) vs statistical findings (Outliers) vs incomplete hosts
//! (HostFailures), and the typed `verdict` field in RunResults.

use std::collections::BTreeMap;

use gauntlet::analysis::stats::Outlier;
use gauntlet::config::FleetConfig;
use gauntlet::orchestrator::collect::HostObservations;
use gauntlet::proto::{CounterDomain, Scope, TestId, TestOutcome};
use gauntlet::report::{self, BarrierStraggler, CounterFinding, RunResults, Verdict};

fn clean() -> RunResults {
    let config: FleetConfig = toml::from_str(r#"hosts = ["n1", "n2"]"#).expect("config");
    let observations: BTreeMap<String, HostObservations> = ["n1", "n2"]
        .into_iter()
        .map(|host| (host.to_string(), HostObservations::default()))
        .collect();
    report::build(&config, observations, 1, 2)
}

fn outlier() -> Outlier {
    Outlier {
        key: "n2".into(),
        value: 1.0,
        fleet_median: 10.0,
        deviation_mads: -9.0,
    }
}

fn fail(results: &mut RunResults, test: TestId) {
    results.hosts.get_mut("n1").expect("n1").outcomes.push((
        test,
        Scope::Node,
        TestOutcome::Failed {
            reason: "broken".into(),
        },
    ));
}

#[test]
fn exit_codes_are_the_documented_mapping() {
    assert_eq!(Verdict::Clean.exit_code(), 0);
    assert_eq!(Verdict::Outliers.exit_code(), 1);
    assert_eq!(Verdict::HostFailures.exit_code(), 2);
    assert_eq!(Verdict::Failures.exit_code(), 3);
    for verdict in Verdict::ALL {
        assert_eq!(Verdict::from_exit_code(verdict.exit_code()), Some(verdict));
    }
    assert_eq!(Verdict::from_exit_code(4), None);
    assert_eq!(Verdict::from_exit_code(-1), None);
}

#[test]
fn a_clean_run_is_clean_and_records_it() {
    let results = clean();
    assert_eq!(report::verdict(&results), Verdict::Clean);
    assert_eq!(results.verdict, Some(Verdict::Clean));
}

#[test]
fn every_failed_outcome_is_hard_evidence() {
    for test in [
        TestId::CpuCorrectness,
        TestId::GpuGemmSdc,
        TestId::CpuSdcHot,
        TestId::NcclAllReduce,
        TestId::NcclIntraAllReduce,
    ] {
        let mut results = clean();
        fail(&mut results, test);
        assert_eq!(report::verdict(&results), Verdict::Failures, "{test:?}");
    }
}

#[test]
fn a_busy_gpu_is_a_soft_finding() {
    // Another tenant's process is environmental, not broken hardware.
    let mut results = clean();
    fail(&mut results, TestId::GpuIdle);
    assert_eq!(report::verdict(&results), Verdict::Outliers);
    // ... and stays below host failures and hard evidence.
    results
        .fleet
        .failed_hosts
        .insert("n2".into(), vec!["timed out".into()]);
    assert_eq!(report::verdict(&results), Verdict::HostFailures);
    fail(&mut results, TestId::CpuCorrectness);
    assert_eq!(report::verdict(&results), Verdict::Failures);
}

#[test]
fn skipped_outcomes_do_not_affect_the_verdict() {
    let mut results = clean();
    results.hosts.get_mut("n1").expect("n1").outcomes.push((
        TestId::NetLatency,
        Scope::Node,
        TestOutcome::Skipped {
            reason: "disabled by config".into(),
        },
    ));
    assert_eq!(report::verdict(&results), Verdict::Clean);
}

#[test]
fn counter_findings_are_hard_evidence() {
    let mut results = clean();
    results.fleet.counter_findings.insert(
        "n1".into(),
        vec![CounterFinding {
            domain: CounterDomain::PcieAer,
            device: "0000:01:00.0".into(),
            counter: "correctable".into(),
            before: 0,
            after: 3,
        }],
    );
    assert_eq!(report::verdict(&results), Verdict::Failures);
}

#[test]
fn sdc_findings_are_hard_evidence_on_their_own() {
    let mut results = clean();
    results
        .fleet
        .sdc_failures
        .insert("gpu_gemm_sdc".into(), vec!["n1:gpu0: mismatch".into()]);
    assert_eq!(report::verdict(&results), Verdict::Failures);
}

#[test]
fn statistical_findings_are_outliers() {
    let mut results = clean();
    results
        .fleet
        .outliers
        .insert("mem_bandwidth.triad".into(), vec![outlier()]);
    assert_eq!(report::verdict(&results), Verdict::Outliers);

    let mut results = clean();
    results
        .fleet
        .threshold_violations
        .insert("mem_bandwidth.triad".into(), vec!["n1".into()]);
    assert_eq!(report::verdict(&results), Verdict::Outliers);

    let mut results = clean();
    results.fleet.barrier_stragglers.insert(
        "tcp_barrier".into(),
        vec![BarrierStraggler {
            key: "n2".into(),
            slowest_frac: 0.9,
            considered_iters: 1000.0,
        }],
    );
    assert_eq!(report::verdict(&results), Verdict::Outliers);
}

#[test]
fn informational_findings_leave_the_verdict_clean() {
    let mut results = clean();
    results
        .fleet
        .jitter_outliers
        .insert("mem_bandwidth.triad".into(), vec![outlier()]);
    results.fleet.consistency.insert(
        "kernel".into(),
        report::ConsistencyFinding {
            majority_value: "6.8.0".into(),
            dissenters: [("n2".to_string(), "6.9.0".to_string())].into(),
        },
    );
    assert_eq!(report::verdict(&results), Verdict::Clean);
}

#[test]
fn precedence_follows_the_exit_code() {
    let mut results = clean();
    results
        .fleet
        .outliers
        .insert("mem_bandwidth.triad".into(), vec![outlier()]);
    assert_eq!(report::verdict(&results), Verdict::Outliers);
    results
        .fleet
        .failed_hosts
        .insert("n2".into(), vec!["timed out".into()]);
    assert_eq!(report::verdict(&results), Verdict::HostFailures);
    // Hard evidence on a reachable host is not hidden by a dead one.
    fail(&mut results, TestId::CpuCorrectness);
    assert_eq!(report::verdict(&results), Verdict::Failures);

    let mut counters_and_dead_host = clean();
    counters_and_dead_host
        .fleet
        .failed_hosts
        .insert("n2".into(), vec!["unreachable".into()]);
    counters_and_dead_host.fleet.counter_findings.insert(
        "n1".into(),
        vec![CounterFinding {
            domain: CounterDomain::GpuEcc,
            device: "gpu0".into(),
            counter: "uncorrected".into(),
            before: 0,
            after: 1,
        }],
    );
    assert_eq!(report::verdict(&counters_and_dead_host), Verdict::Failures);
}

#[test]
fn build_records_the_verdict_it_would_exit_with() {
    let config: FleetConfig = toml::from_str(r#"hosts = ["n1", "n2"]"#).expect("config");
    let mut observations: BTreeMap<String, HostObservations> = ["n1", "n2"]
        .into_iter()
        .map(|host| (host.to_string(), HostObservations::default()))
        .collect();
    observations.get_mut("n2").expect("n2").outcomes.push((
        TestId::GpuGemmCorrectness,
        Scope::Gpu { index: 0 },
        TestOutcome::Failed {
            reason: "residual too large".into(),
        },
    ));
    let results = report::build(&config, observations, 1, 2);
    assert_eq!(results.verdict, Some(Verdict::Failures));
    assert_eq!(results.verdict, Some(report::verdict(&results)));
}

#[test]
fn verdict_serializes_as_a_snake_case_string() {
    let json = serde_json::to_value(clean()).expect("serialize");
    assert_eq!(json["verdict"], "clean");
    for (verdict, text) in [
        (Verdict::Clean, "\"clean\""),
        (Verdict::Outliers, "\"outliers\""),
        (Verdict::Failures, "\"failures\""),
        (Verdict::HostFailures, "\"host_failures\""),
    ] {
        assert_eq!(serde_json::to_string(&verdict).expect("serialize"), text);
        let back: Verdict = serde_json::from_str(text).expect("parse");
        assert_eq!(back, verdict);
    }
}

#[test]
fn pre_v13_documents_decode_without_a_verdict() {
    let mut json = serde_json::to_value(clean()).expect("serialize");
    json.as_object_mut().expect("object").remove("verdict");
    let back: RunResults = serde_json::from_value(json).expect("decode");
    assert_eq!(back.verdict, None);
    // The function still classifies it.
    assert_eq!(report::verdict(&back), Verdict::Clean);
}

#[test]
fn schema_version_is_bumped_for_the_verdict_field() {
    const { assert!(report::SCHEMA_VERSION == 13) };
}

#[test]
fn table_header_names_the_verdict_and_exit_code() {
    let mut results = clean();
    fail(&mut results, TestId::CpuCorrectness);
    results.verdict = Some(report::verdict(&results));
    let mut out = Vec::new();
    report::render_table(&results, &mut out).expect("render");
    let text = String::from_utf8(out).expect("utf8");
    let header = text.lines().next().expect("header");
    assert!(
        header.contains("verdict: test failures (exit 3)"),
        "{header}"
    );
}
