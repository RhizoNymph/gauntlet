//! What the network phase dispatches, derived from the resolved `NetSteps`
//! in one pure step so the step selection and main's network-phase
//! structure (intra-node coverage gate, world-shape sweeps, barrier
//! placement) cannot disagree.

use super::nccl::{BarrierSelection, IntraNodeCoverage};
use crate::net_steps::{NetStep, NetSteps};

/// The fleet NCCL step: every world shape, rail and the NCCL barrier live
/// under it, so deselecting `nccl` drops all of them at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NcclStepPlan {
    /// Whether the intra-node sweep runs in this phase — exactly when its
    /// step is selected — which decides the one-host fleet-sweep gate.
    pub(super) coverage: IntraNodeCoverage,
    /// Whether any NCCL barrier placement (riding or own world) may run.
    pub(super) barrier: BarrierSelection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NetworkPhasePlan {
    pub(super) intranode: bool,
    pub(super) pairwise: bool,
    /// `None`: the fleet sweep (all shapes and rails) and the NCCL barrier
    /// are skipped.
    pub(super) nccl: Option<NcclStepPlan>,
    pub(super) tcp_barrier: bool,
}

impl NetworkPhasePlan {
    pub(super) fn new(steps: &NetSteps) -> Self {
        let intranode = steps.contains(NetStep::Intranode);
        let barrier = if steps.contains(NetStep::Barrier) {
            BarrierSelection::Enabled
        } else {
            BarrierSelection::Disabled
        };
        Self {
            intranode,
            pairwise: steps.contains(NetStep::Pairwise),
            nccl: steps.contains(NetStep::Nccl).then_some(NcclStepPlan {
                coverage: if intranode {
                    IntraNodeCoverage::Covered
                } else {
                    IntraNodeCoverage::NotCovered
                },
                barrier,
            }),
            tcp_barrier: steps.contains(NetStep::Barrier),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FleetConfig;
    use crate::net_steps::{DISABLED_REASON, disabled_outcomes};
    use crate::proto::{TestId, TestOutcome};

    fn steps(config: &str, cli: &[&str]) -> NetSteps {
        let config =
            FleetConfig::from_toml_str(config, std::path::Path::new("t.toml")).expect("config");
        let cli: Vec<String> = cli.iter().map(|s| s.to_string()).collect();
        config.resolve_net_steps(&cli).expect("steps")
    }

    const FLEET: &str = r#"hosts = ["a", "b"]"#;

    fn skipped(steps: &NetSteps) -> Vec<TestId> {
        disabled_outcomes(steps)
            .into_iter()
            .map(|(test, outcome)| {
                assert_eq!(
                    outcome,
                    TestOutcome::Skipped {
                        reason: DISABLED_REASON.into()
                    }
                );
                test
            })
            .collect()
    }

    #[test]
    fn everything_runs_by_default_with_the_intranode_sweep_covering() {
        let plan = NetworkPhasePlan::new(&steps(FLEET, &[]));
        assert!(plan.intranode && plan.pairwise && plan.tcp_barrier);
        assert_eq!(
            plan.nccl,
            Some(NcclStepPlan {
                coverage: IntraNodeCoverage::Covered,
                barrier: BarrierSelection::Enabled,
            })
        );
    }

    #[test]
    fn intranode_off_lets_a_one_host_fleet_sweep_run() {
        // Deselected on the CLI or by `nccl_intranode = false`: either way
        // the intra-node sweep does not run, so the fleet sweep is the only
        // NCCL coverage a one-host fleet has (`shape::sweep_gate` runs it
        // for NotCovered).
        for steps in [
            steps(FLEET, &["pairwise", "nccl", "barrier"]),
            steps("hosts = [\"a\"]\n[tests]\nnccl_intranode = false\n", &[]),
        ] {
            let plan = NetworkPhasePlan::new(&steps);
            assert!(!plan.intranode);
            assert_eq!(
                plan.nccl.expect("nccl selected").coverage,
                IntraNodeCoverage::NotCovered
            );
        }
        let nccl_only = steps(FLEET, &["nccl"]);
        assert_eq!(
            NetworkPhasePlan::new(&nccl_only)
                .nccl
                .expect("nccl")
                .coverage,
            IntraNodeCoverage::NotCovered
        );
        assert!(skipped(&nccl_only).contains(&TestId::NcclIntraAllReduce));
    }

    #[test]
    fn barrier_off_suppresses_riding_own_world_and_tcp_barriers() {
        let steps = steps(FLEET, &["intranode", "pairwise", "nccl"]);
        let plan = NetworkPhasePlan::new(&steps);
        assert!(!plan.tcp_barrier);
        // Every NCCL placement is gated on this (`plan_selected`).
        assert_eq!(plan.nccl.expect("nccl").barrier, BarrierSelection::Disabled);
        let skipped = skipped(&steps);
        assert_eq!(skipped, vec![TestId::NcclBarrier, TestId::TcpBarrier]);
    }

    #[test]
    fn nccl_off_skips_every_shape_rail_and_the_nccl_barrier() {
        let steps = steps(FLEET, &["intranode", "pairwise", "barrier"]);
        let plan = NetworkPhasePlan::new(&steps);
        // No fleet sweep job of any shape is dispatched.
        assert_eq!(plan.nccl, None);
        // The TCP barrier does not need NCCL and still runs.
        assert!(plan.tcp_barrier);
        let skipped = skipped(&steps);
        for test in [
            TestId::NcclAllReduce,
            TestId::NcclAllGather,
            TestId::NcclBarrier,
        ] {
            assert!(skipped.contains(&test), "{test:?}");
        }
        assert!(!skipped.contains(&TestId::TcpBarrier));
    }
}
