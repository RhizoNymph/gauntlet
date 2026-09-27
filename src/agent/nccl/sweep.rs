//! Message-size sweep and barrier-skew probe over this host's local ranks
//! (gpu feature only).
//!
//! Every collective is issued once per local rank inside one NCCL group.
//! Sweep timing comes from global rank 0 only: the host holding it times
//! each size until *its* stream (local rank 0 = global rank 0) completes
//! and emits the measurements; every other host runs silently. The
//! barrier probe times every local rank separately and reports one
//! `NcclBarrierTimings` per rank.

use std::time::Instant;

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::nccl::ReduceOp;

use super::local::LocalRanks;
use super::{F32_BYTES, all_gather_bus_gib_per_sec, all_reduce_bus_gib_per_sec};
use crate::agent::EventSink;
use crate::proto::{AgentEvent, BarrierSpec, MetricRecord, Scope, TestId, Unit};

/// Untimed iterations before the sweep, so channel setup and algorithm
/// selection do not land in the first measured size.
pub(super) const WARMUP_ITERS: usize = 5;

/// Per-rank payload buffers, one pair per local rank on that rank's stream.
pub(super) struct Buffers {
    pub send: Vec<CudaSlice<f32>>,
    pub recv: Vec<CudaSlice<f32>>,
}

impl Buffers {
    pub(super) fn alloc(ranks: &LocalRanks, elements: usize) -> Result<Self> {
        let mut send = Vec::with_capacity(ranks.ranks().len());
        let mut recv = Vec::with_capacity(ranks.ranks().len());
        for rank in ranks.ranks() {
            send.push(rank.stream.alloc_zeros::<f32>(elements)?);
            recv.push(rank.stream.alloc_zeros::<f32>(elements)?);
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

/// The message-size sweep (plus the optional barrier-skew probe).
pub(super) fn run_sweep(
    sink: &EventSink,
    ranks: &LocalRanks,
    sizes: &[u64],
    iters_per_size: u32,
    barrier: Option<BarrierSpec>,
) -> Result<()> {
    let emit_sweep = ranks.assignment().block().holds_lead();
    let world_size = ranks.world_size();
    let max_elements = sizes
        .iter()
        .map(|size| *size as usize / F32_BYTES)
        .max()
        .unwrap_or(0)
        .max(1);
    let mut buffers = Buffers::alloc(ranks, max_elements)?;

    for _ in 0..WARMUP_ITERS {
        buffers.all_reduce(ranks, "warmup all_reduce", max_elements, &ReduceOp::Sum)?;
    }
    ranks.sync_all()?;

    let iters = iters_per_size.max(1);
    let world = world_size as usize;

    for &size in sizes {
        let elements = (size as usize / F32_BYTES).clamp(1, max_elements);

        let elapsed = timed_on_rank0(ranks, iters, || {
            buffers.all_reduce(ranks, "all_reduce", elements, &ReduceOp::Sum)
        })?;
        if emit_sweep {
            emit_collective(
                sink,
                TestId::NcclAllReduce,
                (elements * F32_BYTES) as f64,
                elapsed / f64::from(iters),
                world_size,
            );
        }

        // All-gather: every rank contributes one shard and receives the
        // whole thing, so the *gathered* result is the message size the
        // bus-bandwidth formula wants. Sizes too small to split across the
        // world have no meaningful all-gather and are skipped.
        let shard = elements / world;
        if shard == 0 {
            continue;
        }
        let gathered = shard * world;
        let elapsed = timed_on_rank0(ranks, iters, || {
            let Buffers { send, recv } = &mut buffers;
            ranks.collective("all_gather", |index, rank| {
                rank.comm.all_gather(
                    &send[index].slice(0..shard),
                    &mut recv[index].slice_mut(0..gathered),
                )
            })
        })?;
        if emit_sweep {
            emit_collective(
                sink,
                TestId::NcclAllGather,
                (gathered * F32_BYTES) as f64,
                elapsed / f64::from(iters),
                world_size,
            );
        }
    }

    if let Some(spec) = barrier {
        barrier_probe(sink, ranks, &mut buffers, max_elements, spec)?;
    }
    Ok(())
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
    let elements = (spec.bytes as usize)
        .div_ceil(F32_BYTES)
        .clamp(1, max_elements);
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

/// Total wall seconds for `iters` grouped collectives, measured until
/// local rank 0's stream completes (on the lead host that is global rank
/// 0, the sweep's only timer). Every stream is drained on both sides, so
/// the interval covers exactly the device work — collectives are enqueued
/// asynchronously and would otherwise be timed at launch cost.
fn timed_on_rank0(
    ranks: &LocalRanks,
    iters: u32,
    mut op: impl FnMut() -> Result<()>,
) -> Result<f64> {
    ranks.sync_all()?;
    let start = Instant::now();
    for _ in 0..iters {
        op()?;
    }
    let elapsed = match ranks.ranks().first() {
        Some(rank0) => {
            rank0.stream.synchronize()?;
            start.elapsed().as_secs_f64()
        }
        None => 0.0,
    };
    ranks.sync_all()?;
    Ok(elapsed)
}

fn emit_collective(
    sink: &EventSink,
    test: TestId,
    message_bytes: f64,
    per_iter_secs: f64,
    world_size: u32,
) {
    let bus = match test {
        TestId::NcclAllGather => {
            all_gather_bus_gib_per_sec(message_bytes, per_iter_secs, world_size)
        }
        _ => all_reduce_bus_gib_per_sec(message_bytes, per_iter_secs, world_size),
    };
    for (name, value, unit) in [
        ("elapsed_us", per_iter_secs * 1e6, Unit::Micros),
        ("msg_bytes", message_bytes, Unit::Bytes),
        ("bus_gib_per_sec", bus, Unit::GibPerSec),
    ] {
        sink.metric(MetricRecord {
            test,
            scope: Scope::Node,
            name: name.to_string(),
            value,
            unit,
            repeat: 0,
        });
    }
}
