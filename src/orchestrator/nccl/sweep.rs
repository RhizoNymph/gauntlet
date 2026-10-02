//! The phase-3 fleet NCCL sweep over the configured world shape
//! (`[tests] nccl_world`, `shape`), and the barrier-skew probe.
//!
//! - `rank_per_gpu` (default): one world, every GPU a rank; the barrier
//!   probe rides the sweep's own communicator.
//! - `rank_per_node`: one world, one rank per host on GPU 0, so every
//!   collective crosses the NIC.
//! - `per_rail`: one world per rail, driven one after another (never
//!   concurrently, so rails do not contend for PCIe/NIC bandwidth and each
//!   rail's number is attributable to its path). After the last rail the
//!   best rail headline is rolled up into the overall
//!   `nccl_inter_all_*.bus_gib_per_sec_peak`, attributed to the lead
//!   host.
//!
//! For the two NIC-forcing shapes the barrier probe runs afterwards as its
//! own rank-per-GPU job (`NcclWorkload::Barrier`): barrier subjects stay
//! `host:gpuN` whatever the shape, so barrier results stay comparable
//! across runs with different sweep shapes.
//!
//! Every world goes through `shape::sweep_gate` (>= 2 hosts and >= 2
//! ranks); a gated world records Skipped outcomes on its members naming
//! the failed condition. Every job reuses the shared driver
//! (`drive_fleet_nccl`: rendezvous relay, failure attribution, early
//! abort, remote kill).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use tracing::{info, warn};

use super::shape::{ShapedWorld, rail_rollup, shaped_worlds, skip_outcomes, sweep_gate};
use super::{
    EventIntercept, FleetWorld, NcclFailureMode, NcclJob, accept_owned, block_owners,
    drive_fleet_nccl, emit_outcomes, gpu_subject, nccl_hosts, rank_per_gpu_world,
    report_violations,
};
use crate::analysis::skew::{self, Margin, RankSeries, SkewPolarity};
use crate::config::{FleetConfig, NcclWorldShape};
use crate::orchestrator::ObservationSink;
use crate::orchestrator::barrier::emit_barrier_metrics;
use crate::orchestrator::session::HostSession;
use crate::proto::{AgentEvent, BarrierSpec, InventorySnapshot, NcclWorkload, SweepSeries, TestId};

/// Fleet-wide NCCL sweep in the configured world shape, then the
/// barrier-skew probe (riding the sweep in the rank-per-GPU shape, its own
/// rank-per-GPU job otherwise).
pub(in crate::orchestrator) async fn nccl_sweep(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
    sink: &ObservationSink,
) {
    let hosts = nccl_hosts(sessions, inventories).await;
    let shape = config.tests.nccl_world;
    let worlds = match shaped_worlds(shape, &hosts) {
        Ok(worlds) => worlds,
        Err(error) => {
            warn!(%error, ?shape, "cannot lay out the fleet NCCL sweep; skipping fleet NCCL work");
            return;
        }
    };
    let barrier = barrier_spec(config);
    if shape == NcclWorldShape::RankPerGpu {
        for world in &worlds {
            run_world(config, world, sink, barrier).await;
        }
        return;
    }

    let mut rail_peaks: HeadlineValues = Vec::new();
    for world in &worlds {
        rail_peaks.extend(run_world(config, world, sink, None).await);
    }
    if shape == NcclWorldShape::PerRail
        && let Some((lead, _)) = worlds.first().and_then(|w| w.layout.members().first())
    {
        for record in rail_rollup(&rail_peaks) {
            sink.metric(lead.addr(), record);
        }
    }
    if let Some(spec) = barrier {
        barrier_only(config, rank_per_gpu_world(hosts), sink, spec).await;
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

/// Gate one world, then sweep it. `barrier` rides the sweep's own
/// communicator when given (rank-per-GPU only). Returns the headline
/// values the world's lead reported, `(test, value)`, for the per-rail
/// roll-up.
async fn run_world(
    config: &FleetConfig,
    world: &ShapedWorld<Arc<HostSession>>,
    sink: &ObservationSink,
    barrier: Option<BarrierSpec>,
) -> HeadlineValues {
    let series = world.series;
    let layout = &world.layout;
    if let Err(skip) = sweep_gate(world) {
        info!(
            %series,
            hosts = layout.member_count(),
            ranks = layout.world_size(),
            reason = %skip,
            "skipping a fleet sweep world"
        );
        let outcomes = skip_outcomes(&skip);
        for (session, _) in layout.members() {
            emit_outcomes(sink, session.addr(), outcomes.clone());
        }
        return Vec::new();
    }
    let Some((lead, _)) = layout.members().first() else {
        return Vec::new();
    };
    info!(
        %series,
        world_size = layout.world_size(),
        hosts = layout.member_count(),
        lead = %lead.addr(),
        "NCCL sweep"
    );

    // Skew needs at least two independent arrivals, and the ranks of one
    // host share its launching thread's arrival (the gate already
    // guarantees two hosts).
    let barrier = barrier.filter(|_| layout.member_count() >= 2);
    let collected = Collected::default();
    let intercept = collected.intercept(lead.addr().to_string(), series.headline());
    let job = NcclJob {
        workload: NcclWorkload::Sweep {
            sizes: config.tests.nccl_sizes.clone(),
            iters_per_size: config.tests.nccl_iters_per_size,
            barrier,
            series,
        },
    };
    drive_fleet_nccl(
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
    headlines
}

/// The barrier probe as its own rank-per-GPU job, for sweeps that ran in
/// a NIC-forcing shape.
async fn barrier_only(
    config: &FleetConfig,
    world: FleetWorld,
    sink: &ObservationSink,
    spec: BarrierSpec,
) {
    if world.member_count() < 2 {
        info!(
            nccl_hosts = world.member_count(),
            "fewer than two NCCL-capable hosts; skipping the NCCL barrier probe"
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
    // No headline rides a barrier job: nothing to capture.
    let intercept = collected.intercept(lead.addr().to_string(), String::new());
    let job = NcclJob {
        workload: NcclWorkload::Barrier(spec),
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
/// Headline values a world's lead reported, `(test, value)`.
type HeadlineValues = Vec<(TestId, f64)>;

/// What one sweep or barrier job's intercept collects from the event
/// streams: every rank's barrier timings with their sender (consumed),
/// and the lead's headline values (forwarded as well).
#[derive(Default, Clone)]
struct Collected {
    timings: Arc<Mutex<SenderTimings>>,
    headlines: Arc<Mutex<HeadlineValues>>,
}

impl Collected {
    fn intercept(&self, lead: String, headline: String) -> EventIntercept {
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
                if host == lead && !headline.is_empty() && record.name == headline {
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

    fn take(self) -> (SenderTimings, HeadlineValues) {
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
