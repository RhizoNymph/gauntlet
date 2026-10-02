//! The phase-3 fleet NCCL sweep over the configured world shape
//! (`[tests] nccl_world`, `shape`), and the barrier-skew probe.
//!
//! - `rank_per_gpu` (default): one world, every GPU a rank; the barrier
//!   probe rides the sweep's own communicator.
//! - `rank_per_node`: one world, one rank per host on GPU 0, so every
//!   collective crosses the NIC.
//! - `per_rail`: one world per rail, driven one after another (never
//!   concurrently, so rails do not contend for PCIe/NIC bandwidth and each
//!   rail's number is attributable to its path). Each rail is planned
//!   against the earlier rails' failures (`rails::plan_rail`): a host to
//!   blame for one rail sits out every later rail (and the barrier) instead
//!   of costing one phase timeout per rail. Every host gets one outcome per
//!   test for the whole per-rail sweep (`rails::RailLedger`). After the
//!   last rail the rail headlines are rolled up into
//!   `nccl_inter_all_*.bus_gib_per_sec_peak_min_rail` — peak per rail,
//!   worst rail overall (`shape::rail_rollup`) — on the lead host.
//!
//! For the two NIC-forcing shapes the barrier probe runs afterwards as its
//! own rank-per-GPU job (`NcclWorkload::BarrierOnly`), without the hosts
//! the sweep excluded: barrier subjects stay `host:gpuN` whatever the
//! shape, so barrier results stay comparable across runs with different
//! sweep shapes.
//!
//! Every world goes through `shape::sweep_gate`; a gated world records
//! Skipped outcomes naming the failed condition. A world that cannot be
//! laid out records Skipped outcomes naming the layout error, and the
//! barrier still runs on its own world. Every job reuses the shared
//! driver (`drive_fleet_nccl`: rendezvous relay, failure attribution,
//! early abort, remote kill).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use tracing::{info, warn};

use super::attribution::Attribution;
use super::layout::{LayoutError, RankLayout};
use super::rails::{Exclusions, RailFate, RailLedger, plan_rail};
use super::records::OutcomeRecord;
use super::shape::{
    IntraNodeCoverage, MIN_HOSTS, RailPeaks, ShapedWorld, SweepSkip, rail_rollup, skip_outcomes,
    sweep_gate, world_of,
};
use super::{
    EventIntercept, FleetWorld, NcclFailureMode, NcclJob, accept_owned, block_owners,
    drive_fleet_nccl, emit_outcomes, gpu_subject, nccl_hosts, report_violations,
};
use crate::analysis::skew::{self, Margin, RankSeries, SkewPolarity};
use crate::config::{FleetConfig, NcclWorldShape};
use crate::orchestrator::ObservationSink;
use crate::orchestrator::barrier::emit_barrier_metrics;
use crate::orchestrator::session::HostSession;
use crate::proto::{
    AgentEvent, BarrierSpec, InventorySnapshot, NcclWorkload, Scope, SweepSeries, TestId,
    TestOutcome,
};

/// NCCL-capable hosts with their CUDA-visible GPU counts, fleet order.
type Hosts = [(Arc<HostSession>, u32)];

fn addr(session: &Arc<HostSession>) -> &str {
    session.addr()
}

/// Fleet-wide NCCL sweep in the configured world shape, then the
/// barrier-skew probe (riding the sweep in the rank-per-GPU shape, its own
/// rank-per-GPU job otherwise). `coverage` says whether the intra-node
/// sweep ran in this phase (it decides the one-host gate).
pub(in crate::orchestrator) async fn nccl_sweep(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
    sink: &ObservationSink,
    coverage: IntraNodeCoverage,
) {
    let hosts = nccl_hosts(sessions, inventories).await;
    let barrier = barrier_spec(config);
    let exclusions = match config.tests.nccl_world {
        NcclWorldShape::RankPerGpu => {
            rank_per_gpu_sweep(config, &hosts, sink, coverage, barrier).await;
            return;
        }
        NcclWorldShape::RankPerNode => rank_per_node_sweep(config, &hosts, sink, coverage).await,
        NcclWorldShape::PerRail => per_rail_sweep(config, &hosts, sink, coverage).await,
    };
    if let Some(spec) = barrier {
        barrier_after(config, &hosts, &exclusions, sink, spec).await;
    }
}

/// The barrier-skew probe's parameters, `None` when disabled.
fn barrier_spec(config: &FleetConfig) -> Option<BarrierSpec> {
    (config.tests.barrier_iters > 0).then_some(BarrierSpec {
        iters: config.tests.barrier_iters,
        bytes: config.tests.barrier_bytes,
    })
}

/// Job step name, as failure reasons and logs show it.
fn step_name(series: SweepSeries) -> String {
    match series {
        SweepSeries::RankPerGpu => "nccl sweep".to_string(),
        series => format!("nccl sweep ({series})"),
    }
}

/// Skipped outcomes for `tests` on node scope, naming a world that could
/// not be laid out.
fn layout_failure_outcomes(
    tests: impl IntoIterator<Item = TestId>,
    what: &str,
    error: &LayoutError,
) -> Vec<OutcomeRecord> {
    let reason = format!("cannot lay out the {what} world: {error}");
    tests
        .into_iter()
        .map(|test| {
            let outcome = TestOutcome::Skipped {
                reason: reason.clone(),
            };
            (test, Scope::Node, outcome)
        })
        .collect()
}

/// Record a layout failure on every host that would have joined.
fn report_layout_failure(
    sink: &ObservationSink,
    members: &Hosts,
    tests: impl IntoIterator<Item = TestId> + Clone,
    what: &str,
    error: &LayoutError,
) {
    warn!(%error, world = what, "cannot lay out a fleet NCCL world");
    let outcomes = layout_failure_outcomes(tests, what, error);
    for (session, gpus) in members {
        if *gpus > 0 {
            emit_outcomes(sink, session.addr(), outcomes.clone());
        }
    }
}

async fn rank_per_gpu_sweep(
    config: &FleetConfig,
    hosts: &Hosts,
    sink: &ObservationSink,
    coverage: IntraNodeCoverage,
    barrier: Option<BarrierSpec>,
) {
    let series = SweepSeries::RankPerGpu;
    let world = match world_of(series, hosts) {
        Ok(world) => world,
        Err(error) => {
            // The barrier rides this very world: it cannot run either.
            let barrier_test = barrier.map(|_| TestId::NcclBarrier);
            let tests = series.tests().into_iter().chain(barrier_test);
            report_layout_failure(sink, hosts, tests, "rank-per-gpu", &error);
            return;
        }
    };
    if let WorldRun::Gated(skip) = run_world(config, &world, sink, coverage, barrier).await {
        emit_skip(sink, &world, &skip);
    }
}

/// The rank-per-node world. Returns the hosts its failures exclude from
/// the barrier.
async fn rank_per_node_sweep(
    config: &FleetConfig,
    hosts: &Hosts,
    sink: &ObservationSink,
    coverage: IntraNodeCoverage,
) -> Exclusions {
    let series = SweepSeries::RankPerNode;
    let mut exclusions = Exclusions::default();
    let world = match world_of(series, hosts) {
        Ok(world) => world,
        Err(error) => {
            report_layout_failure(sink, hosts, series.tests(), "rank-per-node", &error);
            return exclusions;
        }
    };
    match run_world(config, &world, sink, coverage, None).await {
        WorldRun::Gated(skip) => emit_skip(sink, &world, &skip),
        WorldRun::Ran { failures, .. } => exclusions.record(series, &failures),
    }
    exclusions
}

/// Every rail in turn, each planned against the earlier rails' failures,
/// then one outcome per host per test and the worst-rail roll-up. Returns
/// the hosts the rails excluded, for the barrier.
async fn per_rail_sweep(
    config: &FleetConfig,
    hosts: &Hosts,
    sink: &ObservationSink,
    coverage: IntraNodeCoverage,
) -> Exclusions {
    let rails = hosts.iter().map(|(_, gpus)| *gpus).max().unwrap_or(0);
    let mut exclusions = Exclusions::default();
    let mut ledger = RailLedger::default();
    // One entry per rail that ran (passed the gate).
    let mut rail_peaks: Vec<RailPeaks> = Vec::new();
    for rail in 0..rails {
        let plan = plan_rail(hosts, rail, &exclusions, addr);
        for (session, reason) in &plan.excluded {
            ledger.note(
                session.addr(),
                rail,
                RailFate::Excluded {
                    reason: reason.clone(),
                },
            );
        }
        let world = match plan.world {
            Ok(world) => world,
            Err(error) => {
                warn!(%error, rail, "cannot lay out a rail; skipping it");
                let (kept, _) = exclusions.split(hosts, addr);
                for (session, gpus) in kept {
                    if gpus > rail {
                        ledger.note(
                            session.addr(),
                            rail,
                            RailFate::NoLayout {
                                reason: error.to_string(),
                            },
                        );
                    }
                }
                continue;
            }
        };
        match run_world(config, &world, sink, coverage, None).await {
            WorldRun::Gated(_) => {
                for (session, _) in world.layout.members() {
                    ledger.note(session.addr(), rail, RailFate::Gated);
                }
            }
            WorldRun::Ran {
                headlines,
                failures,
            } => {
                for (session, _) in world.layout.members() {
                    ledger.note(
                        session.addr(),
                        rail,
                        RailFate::of(&failures, session.addr()),
                    );
                }
                exclusions.record(world.series, &failures);
                rail_peaks.push(headlines);
            }
        }
    }

    for summary in ledger.summaries() {
        if let Some(not_run) = &summary.not_run {
            info!(
                host = %summary.host,
                rails = %not_run,
                "per-rail sweep passed with some rails not run for this host"
            );
        }
        emit_outcomes(sink, &summary.host, summary.outcomes);
    }
    // The fleet's first NCCL host leads rail 0: the roll-up's subject.
    if let Some((lead, _)) = hosts.iter().find(|(_, gpus)| *gpus > 0) {
        for record in rail_rollup(&rail_peaks) {
            sink.metric(lead.addr(), record);
        }
    }
    exclusions
}

/// Skipped outcomes for every member of a gated world.
fn emit_skip(sink: &ObservationSink, world: &ShapedWorld<Arc<HostSession>>, skip: &SweepSkip) {
    let outcomes = skip_outcomes(skip);
    for (session, _) in world.layout.members() {
        emit_outcomes(sink, session.addr(), outcomes.clone());
    }
}

/// How one world went.
enum WorldRun {
    /// The gate kept it from running.
    Gated(SweepSkip),
    /// It ran: the lead's headline values and the driver's attributed
    /// failures (empty when every host exited cleanly).
    Ran {
        headlines: RailPeaks,
        failures: BTreeMap<String, Attribution>,
    },
}

/// Gate one world, then sweep it. `barrier` rides the sweep's own
/// communicator when given (rank-per-GPU only). The caller records gate
/// skips — per world, or summarized per host for the rails.
async fn run_world(
    config: &FleetConfig,
    world: &ShapedWorld<Arc<HostSession>>,
    sink: &ObservationSink,
    coverage: IntraNodeCoverage,
    barrier: Option<BarrierSpec>,
) -> WorldRun {
    let series = world.series;
    let layout = &world.layout;
    if let Err(skip) = sweep_gate(world, coverage) {
        info!(
            %series,
            hosts = layout.member_count(),
            ranks = layout.world_size(),
            reason = %skip,
            "skipping a fleet sweep world"
        );
        return WorldRun::Gated(skip);
    }
    let Some((lead, _)) = layout.members().first() else {
        // Unreachable: the gate requires ranks, and ranks require members.
        return WorldRun::Ran {
            headlines: Vec::new(),
            failures: BTreeMap::new(),
        };
    };
    info!(
        %series,
        world_size = layout.world_size(),
        hosts = layout.member_count(),
        lead = %lead.addr(),
        "NCCL sweep"
    );

    // Skew needs at least two independent arrivals, and the ranks of one
    // host share its launching thread's arrival: a one-host world (run
    // when the intra-node sweep is off) carries no barrier.
    let barrier = barrier.filter(|_| layout.member_count() >= MIN_HOSTS);
    let collected = Collected::default();
    let intercept = collected.intercept(lead.addr().to_string(), Some(series.headline()));
    let job = NcclJob {
        workload: NcclWorkload::Sweep {
            sizes: config.tests.nccl_sizes.clone(),
            iters_per_size: config.tests.nccl_iters_per_size,
            barrier,
            series,
        },
    };
    let failures = drive_fleet_nccl(
        config,
        layout,
        sink,
        &step_name(series),
        &job,
        NcclFailureMode::HostError,
        intercept,
    )
    .await;
    let (timings, headlines) = collected.take();
    if barrier.is_some() {
        analyze_barrier(sink, layout, timings);
    }
    WorldRun::Ran {
        headlines,
        failures,
    }
}

/// The barrier probe after a NIC-forcing sweep: its own rank-per-GPU
/// world, without the hosts the sweep excluded (each records Skipped for
/// `nccl_barrier` with the reason).
async fn barrier_after(
    config: &FleetConfig,
    hosts: &Hosts,
    exclusions: &Exclusions,
    sink: &ObservationSink,
    spec: BarrierSpec,
) {
    let (kept, excluded) = exclusions.split(hosts, addr);
    for (session, reason) in &excluded {
        emit_outcomes(
            sink,
            session.addr(),
            vec![(
                TestId::NcclBarrier,
                Scope::Node,
                TestOutcome::Skipped {
                    reason: reason.clone(),
                },
            )],
        );
    }
    match RankLayout::new(kept.iter().cloned()) {
        Ok(world) => barrier_only(config, world, sink, spec).await,
        Err(error) => report_layout_failure(
            sink,
            &kept,
            [TestId::NcclBarrier],
            "rank-per-gpu barrier",
            &error,
        ),
    }
}

/// The barrier probe as its own rank-per-GPU job, for sweeps that ran in
/// a NIC-forcing shape. Needs `MIN_HOSTS` independent arrivals, like the
/// riding probe.
async fn barrier_only(
    config: &FleetConfig,
    world: FleetWorld,
    sink: &ObservationSink,
    spec: BarrierSpec,
) {
    if world.member_count() < MIN_HOSTS {
        info!(
            nccl_hosts = world.member_count(),
            min_hosts = MIN_HOSTS,
            "too few NCCL-capable hosts; skipping the NCCL barrier probe"
        );
        return;
    }
    let Some((lead, _)) = world.members().first() else {
        return;
    };
    info!(
        world_size = world.world_size(),
        hosts = world.member_count(),
        lead = %lead.addr(),
        "NCCL barrier probe"
    );
    let collected = Collected::default();
    // No headline rides a barrier job.
    let intercept = collected.intercept(lead.addr().to_string(), None);
    let job = NcclJob {
        workload: NcclWorkload::BarrierOnly(spec),
    };
    drive_fleet_nccl(
        config,
        &world,
        sink,
        "nccl barrier",
        &job,
        NcclFailureMode::HostError,
        intercept,
    )
    .await;
    let (timings, _) = collected.take();
    analyze_barrier(sink, &world, timings);
}

/// Barrier timing series with the host that sent each.
type SenderTimings = Vec<(String, RankSeries)>;

/// What one sweep or barrier job's intercept collects from the event
/// streams: every rank's barrier timings with their sender (consumed),
/// and the lead's headline values (forwarded as well).
#[derive(Default, Clone)]
struct Collected {
    timings: Arc<Mutex<SenderTimings>>,
    headlines: Arc<Mutex<RailPeaks>>,
}

impl Collected {
    /// `headline`: the metric name whose values to capture from the lead,
    /// `None` for a job with no headline.
    fn intercept(&self, lead: String, headline: Option<String>) -> EventIntercept {
        let collected = self.clone();
        Arc::new(move |host, event| match event {
            AgentEvent::NcclBarrierTimings { rank, elapsed_us } => {
                collected
                    .timings
                    .lock()
                    .expect("barrier timings poisoned")
                    .push((host.to_string(), RankSeries { rank, elapsed_us }));
                None
            }
            AgentEvent::Metric { record } => {
                // Only the world's lead emits sweep metrics; a headline
                // from anywhere else is forwarded but never rolled up.
                if host == lead && headline.as_deref() == Some(record.name.as_str()) {
                    collected
                        .headlines
                        .lock()
                        .expect("headlines poisoned")
                        .push((record.test, record.value));
                }
                Some(AgentEvent::Metric { record })
            }
            event => Some(event),
        })
    }

    fn take(self) -> (SenderTimings, RailPeaks) {
        let timings = std::mem::take(&mut *self.timings.lock().expect("barrier timings poisoned"));
        let headlines = std::mem::take(&mut *self.headlines.lock().expect("headlines poisoned"));
        (timings, headlines)
    }
}

/// Ownership-check the collected barrier timings against the world's
/// blocks, then run the fleet-wide skew analysis and emit its metrics
/// (per rank as `host:gpuN`, fleet span against the lead).
fn analyze_barrier(sink: &ObservationSink, world: &FleetWorld, collected: SenderTimings) {
    let Some((lead, _)) = world.members().first() else {
        return;
    };
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
            Some(lead.addr()),
        ),
        None => warn!(
            ranks_reporting = series.len(),
            world_size = world.world_size(),
            "NCCL barrier produced no analyzable timings"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_failures_are_skipped_outcomes_naming_the_error() {
        let outcomes = layout_failure_outcomes(
            SweepSeries::RankPerNode.tests(),
            "rank-per-node",
            &LayoutError::WorldTooLarge,
        );
        let tests: Vec<TestId> = outcomes.iter().map(|(test, _, _)| *test).collect();
        assert_eq!(
            tests,
            [TestId::NcclInterAllReduce, TestId::NcclInterAllGather]
        );
        for (_, scope, outcome) in &outcomes {
            assert_eq!(*scope, Scope::Node);
            let TestOutcome::Skipped { reason } = outcome else {
                panic!("expected Skipped, got {outcome:?}");
            };
            assert!(reason.contains("rank-per-node"), "{reason}");
            assert!(reason.contains("exceed"), "{reason}");
        }
    }

    #[test]
    fn step_names_carry_the_shape() {
        assert_eq!(step_name(SweepSeries::RankPerGpu), "nccl sweep");
        assert_eq!(
            step_name(SweepSeries::Rail { rail: 3 }),
            "nccl sweep (rail 3)"
        );
    }
}
