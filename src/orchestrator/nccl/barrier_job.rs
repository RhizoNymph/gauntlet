//! The NCCL barrier-skew probe inside the fleet sweep: where it runs, how
//! its per-rank timings are collected, and the skew analysis afterwards.
//!
//! Placement (`BarrierPlacement`): the probe rides the fleet sweep's
//! communicator when the `barrier` and `fleet` levels' effective NCCL envs
//! are identical; otherwise it runs as its own `NcclWorkload::Barrier` job
//! on a fresh communicator (same rank layout) under the barrier env. The
//! own-world job only runs after a clean sweep: if the sweep blamed a host
//! (a primary failure), re-forming a world that includes the culprit would
//! double-count its error and risk another long block, so the probe is
//! recorded as Skipped on every member, naming the failed host(s).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use tracing::{info, warn};

use super::attribution::Attribution;
use super::ownership::accept_owned;
use super::{
    EventIntercept, FleetWorld, NcclFailureMode, NcclJob, block_owners, drive_fleet_nccl,
    emit_outcomes, gpu_subject, report_violations,
};
use crate::analysis::skew::{self, Margin, RankSeries, SkewPolarity};
use crate::config::FleetConfig;
use crate::orchestrator::ObservationSink;
use crate::orchestrator::barrier::emit_barrier_metrics;
use crate::orchestrator::session::HostSession;
use crate::proto::{AgentEvent, BarrierSpec, NcclWorkload, Scope, TestId, TestOutcome};

/// Where the NCCL barrier probe runs, decided by whether the barrier and
/// fleet levels share one effective NCCL env.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BarrierPlacement {
    /// Appended to the fleet sweep, on its communicator (same env).
    RidesSweep(BarrierSpec),
    /// Its own `agent nccl` job and communicator, under the barrier env.
    OwnWorld(BarrierSpec),
}

impl BarrierPlacement {
    pub(super) fn new(spec: BarrierSpec, shares_fleet_comm: bool) -> Self {
        if shares_fleet_comm {
            BarrierPlacement::RidesSweep(spec)
        } else {
            BarrierPlacement::OwnWorld(spec)
        }
    }

    /// The probe for this sweep, if any. Skew needs at least two
    /// independent arrivals, and the ranks of one host share its launching
    /// thread's arrival, so a one-host world has none. The env comparison
    /// comes from the session's resolved levels (validated at load, so it
    /// cannot fail here).
    pub(super) fn plan(
        config: &FleetConfig,
        world: &FleetWorld,
        lead: &HostSession,
    ) -> Option<Self> {
        (config.tests.barrier_iters > 0 && world.member_count() >= 2).then(|| {
            let spec = BarrierSpec {
                iters: config.tests.barrier_iters,
                bytes: config.tests.barrier_bytes,
            };
            Self::new(spec, lead.nccl_levels().barrier_shares_fleet_comm())
        })
    }

    pub(super) fn rides_sweep(self) -> Option<BarrierSpec> {
        match self {
            BarrierPlacement::RidesSweep(spec) => Some(spec),
            BarrierPlacement::OwnWorld(_) => None,
        }
    }
}

/// Why the own-world barrier job must not run after this sweep, if it
/// must not: the sweep attributed a failure to at least one host.
pub(super) fn own_world_blocked(sweep: &BTreeMap<String, Attribution>) -> Option<String> {
    let failed: Vec<&str> = sweep
        .iter()
        .filter(|(_, attribution)| matches!(attribution, Attribution::Failed { .. }))
        .map(|(host, _)| host.as_str())
        .collect();
    (!failed.is_empty()).then(|| {
        format!(
            "nccl barrier probe skipped: the fleet sweep failed on {}; \
             not re-forming a world that includes it",
            failed.join(", ")
        )
    })
}

/// Per-rank barrier timings, collected with their sending host by the
/// fleet jobs' intercept and analyzed once all hosts are in.
#[derive(Default)]
pub(super) struct BarrierTimings(Arc<Mutex<Vec<(String, RankSeries)>>>);

impl BarrierTimings {
    /// The fleet-job event filter: consumes `NcclBarrierTimings`, forwards
    /// everything else.
    pub(super) fn intercept(&self) -> EventIntercept {
        let timings = Arc::clone(&self.0);
        Arc::new(move |host, event| match event {
            AgentEvent::NcclBarrierTimings { rank, elapsed_us } => {
                timings
                    .lock()
                    .expect("barrier timings poisoned")
                    .push((host.to_string(), RankSeries { rank, elapsed_us }));
                None
            }
            event => Some(event),
        })
    }

    /// Ownership-check the collected series and emit the skew metrics.
    pub(super) fn analyze(self, sink: &ObservationSink, world: &FleetWorld, lead: &str) {
        let collected = std::mem::take(&mut *self.0.lock().expect("barrier timings poisoned"));
        let (accepted, violations) =
            accept_owned(&block_owners(world), collected, |series| series.rank);
        report_violations(sink, violations);
        let series: Vec<RankSeries> = accepted.into_values().collect();
        match skew::analyze_grouped(
            &series,
            SkewPolarity::LateIsMin,
            Margin::default(),
            |rank| world.arrival_group(rank),
        ) {
            Some(skew) => emit_barrier_metrics(
                sink,
                TestId::NcclBarrier,
                &skew,
                |rank| gpu_subject(world, rank),
                Some(lead),
            ),
            None => warn!(
                ranks_reporting = series.len(),
                world_size = world.world_size(),
                "NCCL barrier produced no analyzable timings"
            ),
        }
    }
}

/// Run the probe as its own job after the sweep, unless the sweep blamed a
/// host (then Skipped on every member). Returns whether it ran.
pub(super) async fn run_own_world(
    config: &FleetConfig,
    world: &FleetWorld,
    sink: &ObservationSink,
    spec: BarrierSpec,
    intercept: EventIntercept,
    sweep: &BTreeMap<String, Attribution>,
) -> bool {
    if let Some(reason) = own_world_blocked(sweep) {
        warn!(reason, "skipping the separate NCCL barrier job");
        for (session, _) in world.members() {
            let outcome = TestOutcome::Skipped {
                reason: reason.clone(),
            };
            emit_outcomes(
                sink,
                session.addr(),
                vec![(TestId::NcclBarrier, Scope::Node, outcome)],
            );
        }
        return false;
    }
    info!(
        world_size = world.world_size(),
        "NCCL barrier probe in its own world (its NCCL env differs from the fleet sweep's)"
    );
    let job = NcclJob {
        workload: NcclWorkload::Barrier(spec),
    };
    drive_fleet_nccl(
        config,
        world,
        sink,
        "nccl barrier",
        &job,
        NcclFailureMode::HostError,
        intercept,
    )
    .await;
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_barrier_rides_the_sweep_only_under_the_fleet_env() {
        let spec = BarrierSpec {
            iters: 2000,
            bytes: 8,
        };
        let shared = BarrierPlacement::new(spec, true);
        assert_eq!(shared, BarrierPlacement::RidesSweep(spec));
        assert_eq!(shared.rides_sweep(), Some(spec));
        let split = BarrierPlacement::new(spec, false);
        assert_eq!(split, BarrierPlacement::OwnWorld(spec));
        assert_eq!(split.rides_sweep(), None);
    }

    #[test]
    fn a_sweep_that_blamed_a_host_blocks_the_own_world_job() {
        assert_eq!(own_world_blocked(&BTreeMap::new()), None);
        // Hosts aborted by someone else's failure are not culprits, but the
        // culprit still blocks the job.
        let sweep = BTreeMap::from([
            (
                "n1".to_string(),
                Attribution::Skipped {
                    reason: "aborted".into(),
                },
            ),
            (
                "n2".to_string(),
                Attribution::Failed {
                    reason: "ncclCommInitRank failed".into(),
                },
            ),
        ]);
        let reason = own_world_blocked(&sweep).expect("blocked");
        assert!(reason.contains("n2"), "{reason}");
        assert!(!reason.contains("n1"), "{reason}");
        // Only skips: nobody was blamed, nothing blocks.
        let only_skips = BTreeMap::from([(
            "n1".to_string(),
            Attribution::Skipped {
                reason: "aborted".into(),
            },
        )]);
        assert_eq!(own_world_blocked(&only_skips), None);
    }
}
