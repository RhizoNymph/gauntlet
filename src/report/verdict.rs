//! Run verdict: the classification behind `RunResults.verdict` and the
//! `gauntlet run` exit code.
//!
//! Three evidence tiers, and exit codes that rise with severity so a
//! higher code always means a worse run:
//!
//! | tier                 | verdict         | exit |
//! |----------------------|-----------------|------|
//! | nothing              | `clean`         | 0    |
//! | soft                 | `outliers`      | 1    |
//! | incomplete host      | `host_failures` | 2    |
//! | hard                 | `failures`      | 3    |
//!
//! Precedence follows the number: hard evidence on any host outranks an
//! unreachable host elsewhere, so one dead node never hides SDC or counter
//! evidence on the others.

use serde::{Deserialize, Serialize};

use super::{FleetAnalysis, RunResults};
use crate::proto::{TestId, TestOutcome};

/// Exit code of any `gauntlet` invocation that failed with an error rather
/// than producing a verdict (bad config, CLI usage error, no usable host,
/// I/O). Kept apart from the verdict codes 0-3 so "outliers only" (1)
/// never means "crashed" and a usage error never reads as host failures.
pub const EXIT_ERROR: u8 = 4;

/// Overall outcome of a run. Variants are declared in severity order, and
/// `exit_code` is monotonic in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Exit 0: every host completed; nothing found.
    Clean,
    /// Exit 1: soft findings only — MAD outliers, absolute threshold
    /// violations, barrier stragglers, and soft-tier Failed outcomes
    /// (`gpu_idle`: another tenant on the GPU). The fleet works; some of
    /// it is slower than the rest, below a floor, or not idle.
    Outliers,
    /// Exit 2: at least one host failed to complete (unreachable, deploy
    /// failure, agent Fatal, transport error, phase timeout) and no hard
    /// evidence was found on any host.
    HostFailures,
    /// Exit 3: hard evidence of broken hardware or software on some host —
    /// a hard-tier Failed outcome (correctness and SDC screens, NCCL, ...),
    /// an SDC finding, or an error counter that incremented under load.
    Failures,
}

impl Verdict {
    /// In severity (and exit-code) order.
    pub const ALL: [Verdict; 4] = [
        Verdict::Clean,
        Verdict::Outliers,
        Verdict::HostFailures,
        Verdict::Failures,
    ];

    pub const fn exit_code(self) -> i32 {
        match self {
            Verdict::Clean => 0,
            Verdict::Outliers => 1,
            Verdict::HostFailures => 2,
            Verdict::Failures => 3,
        }
    }

    /// Inverse of `exit_code`, for callers that only see the process
    /// status (the viewer's launched run). Any other code is not a verdict
    /// (`EXIT_ERROR`, a panic's 101).
    pub fn from_exit_code(code: i32) -> Option<Verdict> {
        Self::ALL
            .into_iter()
            .find(|verdict| verdict.exit_code() == code)
    }

    /// Human label for the terminal header and the viewer.
    pub const fn label(self) -> &'static str {
        match self {
            Verdict::Clean => "clean",
            Verdict::Outliers => "outliers",
            Verdict::HostFailures => "host failures",
            Verdict::Failures => "test failures",
        }
    }
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// How much a Failed outcome of a test says about the node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceTier {
    /// The node is broken (wrong results, failed collective, ...).
    Hard,
    /// The node is fine but its environment is not ready (e.g. another
    /// tenant holds a GPU). Ranks with statistical outliers.
    Soft,
}

/// Tier of a Failed outcome, per test. Deliberately an exhaustive match
/// with no wildcard: a new `TestId` does not compile until someone decides
/// what its failure means.
pub const fn failed_outcome_tier(test: TestId) -> EvidenceTier {
    match test {
        // Environmental: a busy GPU is someone else's process, not a fault.
        TestId::GpuIdle => EvidenceTier::Soft,
        TestId::Inventory
        | TestId::CpuCorrectness
        | TestId::CpuGflops
        | TestId::CpuSdcHot
        | TestId::MemBandwidth
        | TestId::DiskIo
        | TestId::GpuGemmCorrectness
        | TestId::GpuGemmPerf
        | TestId::GpuGemmSdc
        | TestId::GpuMemBandwidth
        | TestId::GpuP2p
        | TestId::NetLatency
        | TestId::NetBandwidth
        | TestId::NcclAllReduce
        | TestId::NcclAllGather
        | TestId::NcclIntraAllReduce
        | TestId::NcclIntraAllGather
        | TestId::NcclBarrier
        | TestId::TcpBarrier
        | TestId::OverlapGemm
        | TestId::OverlapAllReduce
        | TestId::OverlapFleetGemm
        | TestId::OverlapFleetAllReduce
        | TestId::OverlapRetention => EvidenceTier::Hard,
    }
}

/// Classify a results document. Every input and its tier:
///
/// | input                                  | tier            |
/// |----------------------------------------|-----------------|
/// | Failed outcome, hard-tier test         | hard            |
/// | `fleet.sdc_failures`                   | hard            |
/// | `fleet.counter_findings`               | hard            |
/// | `fleet.failed_hosts`                   | host failure    |
/// | Failed outcome, soft-tier test (gpu_idle) | soft         |
/// | `fleet.outliers` (MAD)                 | soft            |
/// | `fleet.threshold_violations`           | soft            |
/// | `fleet.barrier_stragglers`             | soft            |
/// | `fleet.jitter_outliers`                | informational   |
/// | `fleet.consistency`                    | informational   |
/// | `Skipped` outcomes                     | informational   |
pub fn verdict(results: &RunResults) -> Verdict {
    if has_failed_outcome(results, EvidenceTier::Hard) || has_hard_findings(&results.fleet) {
        Verdict::Failures
    } else if !results.fleet.failed_hosts.is_empty() {
        Verdict::HostFailures
    } else if has_failed_outcome(results, EvidenceTier::Soft)
        || has_statistical_findings(&results.fleet)
    {
        Verdict::Outliers
    } else {
        Verdict::Clean
    }
}

fn has_failed_outcome(results: &RunResults, tier: EvidenceTier) -> bool {
    results.hosts.values().any(|obs| {
        obs.outcomes.iter().any(|(test, _, outcome)| {
            matches!(outcome, TestOutcome::Failed { .. }) && failed_outcome_tier(*test) == tier
        })
    })
}

/// SDC findings and counter increments. SDC findings are derived from hard
/// Failed outcomes and checked directly too, so a document is classified
/// the same way from either view.
fn has_hard_findings(fleet: &FleetAnalysis) -> bool {
    fleet.sdc_failures.values().any(|found| !found.is_empty())
        || fleet
            .counter_findings
            .values()
            .any(|found| !found.is_empty())
}

/// MAD outliers, threshold violations, barrier stragglers.
fn has_statistical_findings(fleet: &FleetAnalysis) -> bool {
    fleet.outliers.values().any(|flagged| !flagged.is_empty())
        || fleet
            .threshold_violations
            .values()
            .any(|violators| !violators.is_empty())
        || fleet
            .barrier_stragglers
            .values()
            .any(|flagged| !flagged.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_rise_with_severity() {
        for pair in Verdict::ALL.windows(2) {
            assert!(pair[0] < pair[1]);
            assert!(pair[0].exit_code() < pair[1].exit_code());
        }
        assert!(i32::from(EXIT_ERROR) > Verdict::Failures.exit_code());
    }

    #[test]
    fn only_gpu_idle_is_soft() {
        assert_eq!(failed_outcome_tier(TestId::GpuIdle), EvidenceTier::Soft);
        assert_eq!(failed_outcome_tier(TestId::GpuGemmSdc), EvidenceTier::Hard);
    }
}
