//! Message-size sweep and barrier-skew probe over this host's local ranks
//! (gpu feature only).
//!
//! Every collective is issued once per local rank inside one NCCL group.
//! The per-size loop itself is `agent::sweep::run_plan`, shared with the
//! intra-node sweep; `RankBlockCollectives` supplies this level's launch
//! (grouped, once per local rank) and its timer (global rank 0's stream).
//! Sweep timing comes from global rank 0 only: the host holding it times
//! each size until *its* stream (local rank 0 = global rank 0) completes
//! and emits the measurements (per size as measured, then the fleet-level
//! headline, `super::headline`); every other host runs silently. The
//! workload's `SweepSeries` decides the test ids and headline name. The
//! barrier probe times every local rank separately and reports one
//! `NcclBarrierTimings` per rank; it rides the sweep, or runs alone
//! (`run_barrier`) when the sweep ran in another world shape.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use cudarc::driver::{CudaSlice, CudaStream};
use cudarc::nccl::ReduceOp;

use super::F32_BYTES;
use super::headline::fleet_headline;
use super::local::{LocalRanks, PreparedRanks};
use crate::agent::EventSink;
use crate::agent::sweep::{
    self, Collective, SweepCollectives, SweepLevel, SweepPlan, SweepStep, point_records, run_plan,
};
use crate::proto::{AgentEvent, BarrierSpec, SweepSeries};

/// One sweep workload's parameters, as the directive carried them.
pub(super) struct SweepRun<'a> {
    pub sizes: &'a [u64],
    pub iters_per_size: u32,
    pub barrier: Option<BarrierSpec>,
    pub series: SweepSeries,
}

/// Untimed iterations before the barrier probe and the fleet overlap
/// windows, so channel setup and algorithm selection stay out of the
/// measurement (the sweep's own warmup is `agent::sweep::WARMUP_ITERS`).
pub(super) const WARMUP_ITERS: usize = 5;

/// Per-rank payload buffers, one pair per local rank on that rank's stream.
pub(super) struct Buffers {
    pub send: Vec<CudaSlice<f32>>,
    pub recv: Vec<CudaSlice<f32>>,
}

impl Buffers {
    /// Allocate on each local rank's collective stream. Runs before the
    /// communicators exist, so an allocation failure fails the node
    /// before it ever enters the init group.
    pub(super) fn alloc(streams: &[Arc<CudaStream>], elements: usize) -> Result<Self> {
        let mut send = Vec::with_capacity(streams.len());
        let mut recv = Vec::with_capacity(streams.len());
        for stream in streams {
            send.push(stream.alloc_zeros::<f32>(elements)?);
            recv.push(stream.alloc_zeros::<f32>(elements)?);
        }
        Ok(Self { send, recv })
    }

    /// One grouped all-reduce of the first `elements` of every rank's
    /// buffers.
    pub(super) fn all_reduce(
        &mut self,
        ranks: &LocalRanks,
        what: &str,
        elements: usize,
        op: &ReduceOp,
    ) -> Result<()> {
        let Buffers { send, recv } = self;
        ranks.collective(what, |index, rank| {
            rank.comm.all_reduce(
                &send[index].slice(0..elements),
                &mut recv[index].slice_mut(0..elements),
                op,
            )
        })
    }
}

/// The sweep's buffers, sized for its largest message, allocated before
/// the communicators are created.
pub(super) struct SweepBuffers {
    buffers: Buffers,
    max_elements: usize,
}

impl SweepBuffers {
    pub(super) fn alloc(prepared: &PreparedRanks, sizes: &[u64]) -> Result<Self> {
        // Same sizing rule as `SweepPlan::max_elements` (the world size is
        // not needed for it, and the communicators do not exist yet).
        let max_elements = sweep::max_elements(sizes);
        Ok(Self {
            buffers: Buffers::alloc(&prepared.streams(), max_elements)?,
            max_elements,
        })
    }

    /// Buffers for the barrier probe alone: exactly its payload, so the
    /// probe never clamps below the configured barrier size.
    pub(super) fn alloc_for_barrier(prepared: &PreparedRanks, spec: BarrierSpec) -> Result<Self> {
        let max_elements = barrier_elements(spec, usize::MAX);
        Ok(Self {
            buffers: Buffers::alloc(&prepared.streams(), max_elements)?,
            max_elements,
        })
    }
}

/// f32 elements of the barrier all-reduce: the payload rounded up to whole
/// elements, at least one, at most what the buffers hold.
fn barrier_elements(spec: BarrierSpec, max_elements: usize) -> usize {
    (spec.bytes as usize)
        .div_ceil(F32_BYTES)
        .clamp(1, max_elements.max(1))
}

/// The message-size sweep (plus the optional barrier-skew probe).
pub(super) fn run_sweep(
    sink: &EventSink,
    ranks: &LocalRanks,
    buffers: SweepBuffers,
    run: SweepRun<'_>,
) -> Result<()> {
    let SweepRun {
        sizes,
        iters_per_size,
        barrier,
        series,
    } = run;
    let emit_sweep = ranks.assignment().block().holds_lead();
    let world_size = ranks.world_size();
    let world = NonZeroU32::new(world_size).context("fleet world size must be at least 1")?;
    let level = SweepLevel::fleet(series);
    let plan = SweepPlan::new(sizes, world);
    let SweepBuffers {
        mut buffers,
        max_elements,
    } = buffers;
    anyhow::ensure!(
        plan.max_elements() <= max_elements,
        "sweep buffers hold {max_elements} elements, the plan needs {}",
        plan.max_elements()
    );

    // Only the host holding global rank 0 emits; every other host runs the
    // same collectives silently.
    let mut points = Vec::with_capacity(plan.steps().len());
    run_plan(
        &mut RankBlockCollectives {
            ranks,
            buffers: &mut buffers,
        },
        &plan,
        iters_per_size,
        |point| {
            if emit_sweep {
                for record in point_records(level, point, world_size) {
                    sink.metric(record);
                }
                points.push(*point);
            }
        },
    )?;
    for record in fleet_headline(&points, series, world_size) {
        sink.metric(record);
    }

    if let Some(spec) = barrier {
        barrier_probe(sink, ranks, &mut buffers, max_elements, spec)?;
    }
    Ok(())
}

/// The barrier-skew probe alone, on its own communicator
/// (`NcclWorkload::BarrierOnly`).
pub(super) fn run_barrier(
    sink: &EventSink,
    ranks: &LocalRanks,
    buffers: SweepBuffers,
    spec: BarrierSpec,
) -> Result<()> {
    let SweepBuffers {
        mut buffers,
        max_elements,
    } = buffers;
    barrier_probe(sink, ranks, &mut buffers, max_elements, spec)
}

/// Barrier-skew microbenchmark: many iterations of a tiny all-reduce, each
/// timed per local rank from the common launch instant to that rank's
/// completion. The per-iteration drain aligns iteration boundaries across
/// the world (the collective completes everywhere at once), so a host that
/// is slow to *launch* the next iteration arrives late, waits least, and
/// records the shortest elapsed on every one of its ranks — the inversion
/// `analysis::skew::SkewPolarity::LateIsMin` decodes. The ranks of one
/// host share that launch instant, which is why the orchestrator analyzes
/// them as one arrival group.
fn barrier_probe(
    sink: &EventSink,
    ranks: &LocalRanks,
    buffers: &mut Buffers,
    max_elements: usize,
    spec: BarrierSpec,
) -> Result<()> {
    let elements = barrier_elements(spec, max_elements);
    // Algorithm/channel selection is per message size; keep setup for the
    // barrier size out of the first measured iterations.
    for _ in 0..WARMUP_ITERS {
        buffers.all_reduce(ranks, "barrier warmup all_reduce", elements, &ReduceOp::Sum)?;
    }
    ranks.sync_all()?;

    let mut elapsed_us: Vec<Vec<f64>> = ranks
        .ranks()
        .iter()
        .map(|_| Vec::with_capacity(spec.iters as usize))
        .collect();
    for _ in 0..spec.iters {
        let start = Instant::now();
        buffers.all_reduce(ranks, "barrier all_reduce", elements, &ReduceOp::Sum)?;
        let done = ranks.completion_secs(start)?;
        for (series, secs) in elapsed_us.iter_mut().zip(done) {
            series.push(secs * 1e6);
        }
    }
    ranks.sync_all()?;
    for (rank, elapsed_us) in ranks.ranks().iter().zip(elapsed_us) {
        sink.emit(&AgentEvent::NcclBarrierTimings {
            rank: rank.global,
            elapsed_us,
        });
    }
    Ok(())
}

/// This host's rank block for the shared sweep loop: every collective
/// issued once per local rank inside one NCCL group, the clock stopped by
/// local rank 0's stream (on the lead host that is global rank 0, the
/// sweep's only timer), every stream drained on both sides.
struct RankBlockCollectives<'a> {
    ranks: &'a LocalRanks,
    buffers: &'a mut Buffers,
}

impl SweepCollectives for RankBlockCollectives<'_> {
    type Error = anyhow::Error;

    fn launch(&mut self, step: &SweepStep) -> Result<()> {
        match step.collective {
            Collective::AllReduce => self.buffers.all_reduce(
                self.ranks,
                "all_reduce",
                step.send_elements,
                &ReduceOp::Sum,
            ),
            // Every rank contributes one shard and receives the whole
            // gathered message (the size the bus formula wants).
            Collective::AllGather => {
                let Buffers { send, recv } = &mut *self.buffers;
                self.ranks.collective("all_gather", |index, rank| {
                    rank.comm.all_gather(
                        &send[index].slice(0..step.send_elements),
                        &mut recv[index].slice_mut(0..step.message_elements),
                    )
                })
            }
        }
    }

    fn sync(&mut self) -> Result<()> {
        self.ranks.sync_all()
    }

    fn wait_timed(&mut self) -> Result<()> {
        if let Some(rank0) = self.ranks.ranks().first() {
            rank0.stream().synchronize()?;
        }
        Ok(())
    }
}
