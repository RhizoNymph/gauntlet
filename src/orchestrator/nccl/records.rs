//! Fleet-overlap reports -> results-document records, one host's rank
//! block at a time. Pure.
//!
//! The fleet world runs one rank per GPU, so every report is one GPU's:
//! bus bandwidths and the GEMM leg both land under `Scope::Gpu { index }`
//! with `index` the rank's local GPU in its host's block.

use std::collections::BTreeMap;

use super::attribution::Attribution;
use crate::proto::{
    GemmDtype, MetricRecord, OverlapFleetReport, OverlapGemmLeg, RankBlock, Scope, TestId,
    TestOutcome, Unit, overlap_metric,
};

/// Outcome records as `(test, scope, outcome)`.
pub(super) type OutcomeRecord = (TestId, Scope, TestOutcome);

/// Fallback reason when a rank produced no report and the driver recorded
/// no failure for its host.
const NO_REPORT: &str = "rank produced no fleet overlap report";

/// Everything the results document gets for one host of the fleet-overlap
/// world, given the reports (keyed by global rank, already checked for
/// ownership) and the host's attributed failure from the driver, if any.
///
/// - No report from any rank of the block: the whole node's participation
///   failed (grouped init, process death, timeout, or an abort caused by
///   another host) — node-scope outcomes for both fleet tests: Failed when
///   this host is to blame (or nobody identifiable is), Skipped when
///   another host's failure aborted the step.
/// - Some ranks reported: each reporting GPU gets its metrics and
///   outcomes; each silent GPU gets the same kind of outcome at GPU scope,
///   so one lost rank never hides its siblings.
pub(super) fn host_overlap_records(
    block: RankBlock,
    reports: &BTreeMap<u32, OverlapFleetReport>,
    dtype: GemmDtype,
    host_failure: Option<&Attribution>,
) -> (Vec<MetricRecord>, Vec<OutcomeRecord>) {
    let silent = |scope: Scope| match host_failure {
        Some(Attribution::Skipped { reason }) => {
            outcomes_with(scope, |r| TestOutcome::Skipped { reason: r }, reason)
        }
        Some(Attribution::Failed { reason }) => step_outcomes(scope, reason),
        None => step_outcomes(scope, NO_REPORT),
    };
    let reported = block.ranks().any(|rank| reports.contains_key(&rank));
    if !reported {
        return (Vec::new(), silent(Scope::Node));
    }
    let mut metrics = Vec::new();
    let mut outcomes = Vec::new();
    for rank in block.ranks() {
        let Some(index) = block.gpu(rank) else {
            continue;
        };
        let gpu = Scope::Gpu { index };
        match reports.get(&rank) {
            Some(report) => {
                let (m, o) = rank_records(report, gpu, dtype);
                metrics.extend(m);
                outcomes.extend(o);
            }
            None => outcomes.extend(silent(gpu)),
        }
    }
    (metrics, outcomes)
}

/// Failed outcomes for both fleet-overlap tests under one scope.
pub(super) fn step_outcomes(scope: Scope, reason: &str) -> Vec<OutcomeRecord> {
    outcomes_with(scope, |r| TestOutcome::Failed { reason: r }, reason)
}

/// Both fleet-overlap tests under one scope with the given outcome.
pub(super) fn outcomes_with(
    scope: Scope,
    outcome: impl Fn(String) -> TestOutcome,
    reason: &str,
) -> Vec<OutcomeRecord> {
    [TestId::OverlapFleetGemm, TestId::OverlapFleetAllReduce]
        .into_iter()
        .map(|test| (test, scope.clone(), outcome(reason.to_string())))
        .collect()
}

/// One GPU's records: the collective leg's bus numbers and the compute
/// leg's GFLOPS, both under that GPU's scope. A failed GEMM worker is a
/// Failed compute outcome next to a Passed collective outcome.
fn rank_records(
    report: &OverlapFleetReport,
    gpu: Scope,
    dtype: GemmDtype,
) -> (Vec<MetricRecord>, Vec<OutcomeRecord>) {
    let mut metrics: Vec<MetricRecord> = [
        (
            overlap_metric::MSG_BYTES,
            report.msg_bytes as f64,
            Unit::Bytes,
        ),
        (
            overlap_metric::ISOLATED_BUS,
            report.isolated_bus_gib_per_sec,
            Unit::GibPerSec,
        ),
        (
            overlap_metric::OVERLAP_BUS,
            report.overlap_bus_gib_per_sec,
            Unit::GibPerSec,
        ),
    ]
    .into_iter()
    .map(|(name, value, unit)| MetricRecord {
        test: TestId::OverlapFleetAllReduce,
        scope: gpu.clone(),
        name: name.to_string(),
        value,
        unit,
        repeat: 0,
    })
    .collect();
    let mut outcomes = vec![(
        TestId::OverlapFleetAllReduce,
        gpu.clone(),
        TestOutcome::Passed,
    )];
    match &report.gemm {
        OverlapGemmLeg::Ok { gflops } => {
            metrics.push(MetricRecord {
                test: TestId::OverlapFleetGemm,
                scope: gpu.clone(),
                name: format!("gflops_{}", dtype.tag()),
                value: *gflops,
                unit: Unit::Gflops,
                repeat: 0,
            });
            outcomes.push((TestId::OverlapFleetGemm, gpu, TestOutcome::Passed));
        }
        OverlapGemmLeg::Failed { reason } => outcomes.push((
            TestId::OverlapFleetGemm,
            gpu,
            TestOutcome::Failed {
                reason: reason.clone(),
            },
        )),
    }
    (metrics, outcomes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(rank: u32, gemm: OverlapGemmLeg) -> OverlapFleetReport {
        OverlapFleetReport {
            rank,
            msg_bytes: 64 << 20,
            isolated_bus_gib_per_sec: 40.0 + f64::from(rank),
            overlap_bus_gib_per_sec: 30.0,
            gemm,
        }
    }

    fn by_rank(reports: Vec<OverlapFleetReport>) -> BTreeMap<u32, OverlapFleetReport> {
        reports.into_iter().map(|r| (r.rank, r)).collect()
    }

    #[test]
    fn every_reporting_gpu_gets_bus_and_gemm_under_its_local_scope() {
        // Host block 4..6: global ranks 4 and 5 are local GPUs 0 and 1.
        let block = RankBlock::new(4, 2).expect("block");
        let reports = by_rank(vec![
            report(4, OverlapGemmLeg::Ok { gflops: 88_000.0 }),
            report(
                5,
                OverlapGemmLeg::Failed {
                    reason: "worker panicked".into(),
                },
            ),
            // Another host's rank must not leak in.
            report(0, OverlapGemmLeg::Ok { gflops: 1.0 }),
        ]);
        let (metrics, outcomes) = host_overlap_records(block, &reports, GemmDtype::Bf16, None);

        let keys: Vec<(TestId, Scope, &str)> = metrics
            .iter()
            .map(|m| (m.test, m.scope.clone(), m.name.as_str()))
            .collect();
        let gpu0 = Scope::Gpu { index: 0 };
        let gpu1 = Scope::Gpu { index: 1 };
        assert_eq!(
            keys,
            [
                (TestId::OverlapFleetAllReduce, gpu0.clone(), "msg_bytes"),
                (
                    TestId::OverlapFleetAllReduce,
                    gpu0.clone(),
                    "isolated_bus_gib_per_sec"
                ),
                (
                    TestId::OverlapFleetAllReduce,
                    gpu0.clone(),
                    "overlap_bus_gib_per_sec"
                ),
                (TestId::OverlapFleetGemm, gpu0.clone(), "gflops_bf16"),
                (TestId::OverlapFleetAllReduce, gpu1.clone(), "msg_bytes"),
                (
                    TestId::OverlapFleetAllReduce,
                    gpu1.clone(),
                    "isolated_bus_gib_per_sec"
                ),
                (
                    TestId::OverlapFleetAllReduce,
                    gpu1.clone(),
                    "overlap_bus_gib_per_sec"
                ),
            ]
        );
        // Per-GPU bus numbers come from that GPU's own report.
        assert_eq!(metrics[1].value, 44.0);
        assert_eq!(metrics[5].value, 45.0);

        assert_eq!(
            outcomes,
            [
                (
                    TestId::OverlapFleetAllReduce,
                    gpu0.clone(),
                    TestOutcome::Passed
                ),
                (TestId::OverlapFleetGemm, gpu0, TestOutcome::Passed),
                (
                    TestId::OverlapFleetAllReduce,
                    gpu1.clone(),
                    TestOutcome::Passed
                ),
                (
                    TestId::OverlapFleetGemm,
                    gpu1,
                    TestOutcome::Failed {
                        reason: "worker panicked".into()
                    }
                ),
            ]
        );
    }

    #[test]
    fn a_silent_rank_inside_a_reporting_block_fails_only_its_gpu() {
        let block = RankBlock::new(0, 3).expect("block");
        let reports = by_rank(vec![
            report(0, OverlapGemmLeg::Ok { gflops: 1.0 }),
            report(2, OverlapGemmLeg::Ok { gflops: 1.0 }),
        ]);
        let (metrics, outcomes) = host_overlap_records(block, &reports, GemmDtype::F16, None);
        assert_eq!(metrics.len(), 8, "two GPUs x (3 bus + 1 gemm)");
        let gpu1: Vec<&OutcomeRecord> = outcomes
            .iter()
            .filter(|(_, scope, _)| *scope == Scope::Gpu { index: 1 })
            .collect();
        assert_eq!(gpu1.len(), 2);
        assert!(gpu1.iter().all(|(_, _, outcome)| *outcome
            == TestOutcome::Failed {
                reason: NO_REPORT.into()
            }));
    }

    #[test]
    fn a_block_with_no_reports_is_a_node_level_failure() {
        let block = RankBlock::new(8, 8).expect("block");
        let failure = Attribution::Failed {
            reason: "nccl ranks 8..16 timed out after 600s".into(),
        };
        let (metrics, outcomes) =
            host_overlap_records(block, &BTreeMap::new(), GemmDtype::Bf16, Some(&failure));
        assert!(metrics.is_empty());
        assert_eq!(
            outcomes,
            [
                (
                    TestId::OverlapFleetGemm,
                    Scope::Node,
                    TestOutcome::Failed {
                        reason: "nccl ranks 8..16 timed out after 600s".into()
                    }
                ),
                (
                    TestId::OverlapFleetAllReduce,
                    Scope::Node,
                    TestOutcome::Failed {
                        reason: "nccl ranks 8..16 timed out after 600s".into()
                    }
                ),
            ]
        );
    }

    #[test]
    fn a_host_aborted_by_another_hosts_failure_is_skipped_not_failed() {
        let block = RankBlock::new(0, 2).expect("block");
        let aborted = Attribution::Skipped {
            reason: "fleet overlap aborted: rank failure on 10.1.1.68 (killed)".into(),
        };
        let (metrics, outcomes) =
            host_overlap_records(block, &BTreeMap::new(), GemmDtype::Bf16, Some(&aborted));
        assert!(metrics.is_empty());
        assert_eq!(outcomes.len(), 2);
        for (_, scope, outcome) in &outcomes {
            assert_eq!(*scope, Scope::Node);
            assert!(
                matches!(outcome, TestOutcome::Skipped { reason } if reason.contains("10.1.1.68")),
                "{outcome:?}"
            );
        }
    }
}
