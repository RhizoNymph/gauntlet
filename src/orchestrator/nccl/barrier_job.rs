//! The NCCL barrier-skew probe inside the fleet sweep: where it runs, how
//! its per-rank timings are collected, and the skew analysis afterwards.
//!
//! Placement (`BarrierPlacement`), decided once per run: the probe rides
//! the fleet sweep's communicator only when the sweep runs the
//! rank-per-GPU world shape **and** the `barrier` and `fleet` levels'
//! effective NCCL envs are identical. Otherwise — a different barrier env
//! (proto v11), a NIC-forcing world shape (`rank_per_node` / `per_rail`,
//! proto v12), or both — it runs as its own `NcclWorkload::Barrier` job on
//! a fresh rank-per-GPU communicator under the barrier env, so barrier
//! subjects stay `host:gpuN` whatever the sweep shape.
//!
//! Failure handling never re-forms a world that includes a host the
//! sweep blamed:
//! - rank-per-GPU sweep: the own-world job only runs after a clean sweep;
//!   if the sweep blamed a host (a primary failure), re-forming the same
//!   world would double-count its error and risk another long block, so
//!   the probe is recorded as Skipped on every member, naming the failed
//!   host(s) (`run_own_world`).
//! - NIC-forcing sweeps: the barrier world is the rank-per-GPU world minus
//!   the hosts the sweep's worlds blamed (`rails::Exclusions`); each
//!   excluded host records Skipped with the reason, and the rest run.
//!
//! In both cases a world short of `shape::MIN_HOSTS` independent arrivals
//! never runs the probe.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use tracing::{info, warn};

use super::attribution::Attribution;
use super::ownership::accept_owned;
use super::shape::MIN_HOSTS;
use super::{
    EventIntercept, FleetWorld, NcclFailureMode, NcclJob, block_owners, drive_fleet_nccl,
    emit_outcomes, gpu_subject, report_violations,
};
use crate::analysis::skew::{self, Margin, RankSeries, SkewPolarity};
use crate::config::{FleetConfig, NcclWorldShape};
use crate::nccl_level::NcclLevelEnvs;
use crate::orchestrator::ObservationSink;
use crate::orchestrator::barrier::emit_barrier_metrics;
use crate::proto::{AgentEvent, BarrierSpec, NcclWorkload, Scope, TestId, TestOutcome};

/// Where the NCCL barrier probe runs: riding the rank-per-GPU sweep's
/// communicator, or on its own rank-per-GPU world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BarrierPlacement {
    /// Appended to the fleet sweep, on its communicator (rank-per-GPU
    /// shape, same env).
    RidesSweep(BarrierSpec),
    /// Its own `agent nccl` job and rank-per-GPU communicator, under the
    /// barrier env.
    OwnWorld(BarrierSpec),
}

impl BarrierPlacement {
    /// The env rule alone: ride the fleet communicator iff the barrier
    /// level shares the fleet level's env.
    pub(super) fn new(spec: BarrierSpec, shares_fleet_comm: bool) -> Self {
        if shares_fleet_comm {
            BarrierPlacement::RidesSweep(spec)
        } else {
            BarrierPlacement::OwnWorld(spec)
        }
    }

    /// The shape rule on top: only a rank-per-GPU sweep has the barrier's
    /// world to ride; every other shape sends it to its own world.
    pub(super) fn in_shape(self, shape: NcclWorldShape) -> Self {
        match (self, shape) {
            (BarrierPlacement::RidesSweep(spec), NcclWorldShape::RankPerGpu) => {
                BarrierPlacement::RidesSweep(spec)
            }
            (placement, _) => BarrierPlacement::OwnWorld(placement.spec()),
        }
    }

    /// The probe for this run, if it is enabled: both rules applied. The
    /// env comparison comes from the resolved levels (validated at load,
    /// so it cannot fail here). Whether a world has the `MIN_HOSTS`
    /// independent arrivals skew needs is checked against the world the
    /// probe actually runs on.
    pub(super) fn plan(
        config: &FleetConfig,
        levels: &NcclLevelEnvs,
        shape: NcclWorldShape,
    ) -> Option<Self> {
        (config.tests.barrier_iters > 0).then(|| {
            let spec = BarrierSpec {
                iters: config.tests.barrier_iters,
                bytes: config.tests.barrier_bytes,
            };
            Self::new(spec, levels.barrier_shares_fleet_comm()).in_shape(shape)
        })
    }

    pub(super) fn spec(self) -> BarrierSpec {
        match self {
            BarrierPlacement::RidesSweep(spec) | BarrierPlacement::OwnWorld(spec) => spec,
        }
    }

    pub(super) fn rides_sweep(self) -> Option<BarrierSpec> {
        match self {
            BarrierPlacement::RidesSweep(spec) => Some(spec),
            BarrierPlacement::OwnWorld(_) => None,
        }
    }

    pub(super) fn own_world(self) -> Option<BarrierSpec> {
        match self {
            BarrierPlacement::RidesSweep(_) => None,
            BarrierPlacement::OwnWorld(spec) => Some(spec),
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

/// Run the probe as its own job after the sweep, unless the world has
/// fewer than `MIN_HOSTS` hosts (no independent arrivals: nothing runs)
/// or the sweep blamed one of its hosts (then Skipped on every member).
/// Returns whether it ran.
pub(super) async fn run_own_world(
    config: &FleetConfig,
    world: &FleetWorld,
    sink: &ObservationSink,
    spec: BarrierSpec,
    intercept: EventIntercept,
    sweep: &BTreeMap<String, Attribution>,
) -> bool {
    if world.member_count() < MIN_HOSTS {
        info!(
            nccl_hosts = world.member_count(),
            min_hosts = MIN_HOSTS,
            "too few NCCL-capable hosts; skipping the NCCL barrier probe"
        );
        return false;
    }
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
        hosts = world.member_count(),
        "NCCL barrier probe in its own rank-per-GPU world"
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
    fn the_barrier_takes_its_own_world_unless_both_rules_allow_riding() {
        let spec = BarrierSpec {
            iters: 2000,
            bytes: 8,
        };
        let placement = |shares_fleet_comm, shape| {
            BarrierPlacement::new(spec, shares_fleet_comm).in_shape(shape)
        };
        // Rides only under rank-per-GPU with the fleet env.
        let rides = placement(true, NcclWorldShape::RankPerGpu);
        assert_eq!(rides, BarrierPlacement::RidesSweep(spec));
        assert_eq!(rides.own_world(), None);
        // A NIC-forcing shape alone.
        for shape in [NcclWorldShape::RankPerNode, NcclWorldShape::PerRail] {
            assert_eq!(
                placement(true, shape),
                BarrierPlacement::OwnWorld(spec),
                "{shape:?}"
            );
        }
        // A differing barrier env alone.
        assert_eq!(
            placement(false, NcclWorldShape::RankPerGpu),
            BarrierPlacement::OwnWorld(spec)
        );
        // Both.
        for shape in [NcclWorldShape::RankPerNode, NcclWorldShape::PerRail] {
            let both = placement(false, shape);
            assert_eq!(both, BarrierPlacement::OwnWorld(spec), "{shape:?}");
            assert_eq!(both.own_world(), Some(spec));
            assert_eq!(both.rides_sweep(), None);
            assert_eq!(both.spec(), spec);
        }
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
