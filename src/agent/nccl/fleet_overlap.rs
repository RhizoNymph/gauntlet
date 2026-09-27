//! Fleet overlap protocol over this host's local ranks (gpu feature only).
//!
//! Every local GPU carries both legs: a GEMM worker on its own stream
//! (`gpu::worker`) and a collective rank on its default stream. The
//! isolated baseline window runs with the workers parked, the overlapped
//! window with them released, both on the same communicators. Window
//! boundaries are agreed through the MIN-reduced control word
//! (`agent::window`): only global rank 0 contributes a close word, so the
//! lead's local siblings are followers like every remote rank. Each local
//! rank keeps its own payload tally (its own completion stamps), and every
//! rank reports its own `OverlapFleetReport` — one per GPU.
//!
//! Wedge protection: a rank that dies mid-window leaves every other rank
//! blocked inside a collective. The whole protocol runs under a hard
//! deadline (`window::hard_deadline_secs`, from the spec): GEMM workers
//! stop on their own at it, and a watchdog terminates the process
//! (cascade exit, `watchdog`) `WATCHDOG_GRACE_SECS` later. A follower whose
//! lead never signals close also leaves through a cascade exit.

use std::time::{Duration, Instant};

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::nccl::ReduceOp;

use super::local::{LocalRanks, PreparedRanks};
use super::sweep::{Buffers, WARMUP_ITERS};
use super::watchdog::{CascadeAbort, guarded};
use super::{F32_BYTES, all_reduce_bus_gib_per_sec, message_elements};
use crate::agent::EventSink;
use crate::agent::gpu::gemm::sustained_gflops_value;
use crate::agent::gpu::worker::GemmLoad;
use crate::agent::window::{
    self, WATCHDOG_GRACE_SECS, WindowRole, WindowTally, hard_deadline_secs, role_for_rank,
};
use crate::proto::{AgentEvent, OverlapFleetReport, OverlapGemmLeg, OverlapSpec};

/// Payload all-reduces between window-consensus steps: enough to amortize
/// the (tiny) control reduce, few enough that a window boundary lands
/// within a fraction of a second on 64 MiB messages.
const OVERLAP_CONTROL_INTERVAL: u32 = 4;

/// One-element control-word buffers per local rank.
struct Control {
    send: Vec<CudaSlice<f32>>,
    recv: Vec<CudaSlice<f32>>,
}

/// Payload and control buffers of the protocol, allocated before the
/// communicators are created.
pub(super) struct OverlapBuffers {
    payload: Buffers,
    control: Control,
    elements: usize,
}

impl OverlapBuffers {
    pub(super) fn alloc(prepared: &PreparedRanks, spec: &OverlapSpec) -> Result<Self> {
        let elements = message_elements(spec.msg_bytes);
        let streams = prepared.streams();
        let payload = Buffers::alloc(&streams, elements)?;
        let mut control = Control {
            send: Vec::with_capacity(streams.len()),
            recv: Vec::with_capacity(streams.len()),
        };
        for stream in &streams {
            control.send.push(stream.alloc_zeros::<f32>(1)?);
            control.recv.push(stream.alloc_zeros::<f32>(1)?);
        }
        Ok(Self {
            payload,
            control,
            elements,
        })
    }
}

/// Fleet overlap protocol for this host's ranks, under the hard-deadline
/// watchdog: GEMM workers stop at `hard_deadline_secs` from now, and the
/// process terminates `WATCHDOG_GRACE_SECS` after that if the protocol is
/// still wedged.
pub(super) fn overlap_fleet(
    sink: &EventSink,
    ranks: &LocalRanks,
    buffers: OverlapBuffers,
    spec: &OverlapSpec,
) -> Result<()> {
    let hard = Duration::from_secs(hard_deadline_secs(spec.baseline_secs, spec.duration_secs));
    let compute_deadline = Instant::now() + hard;
    let watchdog_budget = hard + Duration::from_secs(WATCHDOG_GRACE_SECS);
    guarded(sink, "fleet overlap", watchdog_budget, || {
        run_protocol(sink, ranks, buffers, spec, compute_deadline)
    })
}

/// The protocol proper.
///
/// Ordering matters: every fallible piece of GEMM setup (operand upload,
/// warm launch) happens *before* the communicators enter the first window.
/// A bad GEMM worker degrades to a Failed compute leg on that GPU's report
/// with zero compute load; it can never abort the ranks mid-protocol and
/// strand the rest of the world in a blocking collective. Between the two
/// windows the only action is the start-barrier release.
fn run_protocol(
    sink: &EventSink,
    ranks: &LocalRanks,
    buffers: OverlapBuffers,
    spec: &OverlapSpec,
    compute_deadline: Instant,
) -> Result<()> {
    let OverlapBuffers {
        payload: mut buffers,
        mut control,
        elements,
    } = buffers;
    let message_bytes = (elements * F32_BYTES) as f64;

    // GEMM load on every local GPU (one worker per rank's context).
    // `wait_ready` returns once every worker has set up — or failed — and
    // gone quiet, so the baseline below still measures quiet GPUs.
    let mut load = GemmLoad::spawn(
        &ranks.contexts(),
        spec.gemm_dim,
        spec.gemm_dtype,
        compute_deadline,
    );
    load.wait_ready();

    let windows = run_windows(ranks, &mut buffers, &mut control, elements, spec, &mut load);
    // Workers are stopped and joined on every path — including a failed
    // baseline, where they are still parked at the start barrier — so a
    // failed group never leaks GEMM threads.
    let throughputs = load.finish();
    let (isolated, overlapped) = windows?;

    let n = spec.gemm_dim.max(1) as usize;
    let world_size = ranks.world_size();
    for (((rank, isolated), overlapped), throughput) in ranks
        .ranks()
        .iter()
        .zip(isolated)
        .zip(overlapped)
        .zip(throughputs)
    {
        let gemm = match throughput {
            Ok(throughput) => OverlapGemmLeg::Ok {
                gflops: sustained_gflops_value(n, throughput.iters, throughput.elapsed_secs),
            },
            Err(reason) => OverlapGemmLeg::Failed { reason },
        };
        sink.emit(&AgentEvent::OverlapFleetReport {
            report: Box::new(OverlapFleetReport {
                rank: rank.global,
                msg_bytes: (elements * F32_BYTES) as u64,
                isolated_bus_gib_per_sec: all_reduce_bus_gib_per_sec(
                    message_bytes,
                    isolated,
                    world_size,
                ),
                overlap_bus_gib_per_sec: all_reduce_bus_gib_per_sec(
                    message_bytes,
                    overlapped,
                    world_size,
                ),
                gemm,
            }),
        });
    }
    Ok(())
}

/// Per-rank mean seconds per payload iteration, isolated then overlapped.
type WindowSecs = (Vec<f64>, Vec<f64>);

/// Warmup, then the two consensus windows: isolated baseline (workers
/// parked at the start barrier), the start release, and the overlapped
/// window. Split out so `overlap_fleet` can join the workers on every exit
/// path before propagating an error.
fn run_windows(
    ranks: &LocalRanks,
    buffers: &mut Buffers,
    control: &mut Control,
    elements: usize,
    spec: &OverlapSpec,
    load: &mut GemmLoad,
) -> Result<WindowSecs> {
    for _ in 0..WARMUP_ITERS {
        buffers.all_reduce(ranks, "overlap warmup all_reduce", elements, &ReduceOp::Sum)?;
    }
    ranks.sync_all()?;

    let isolated = consensus_window(ranks, buffers, control, elements, spec.baseline_secs)?;
    // The one action between windows: release the (already set-up)
    // workers into the timed loop.
    load.start();
    let overlapped = consensus_window(ranks, buffers, control, elements, spec.duration_secs)?;
    Ok((isolated, overlapped))
}

/// One consensus-coordinated measurement window: batches of payload
/// all-reduces, then one MIN-reduced control word, until the lead's clock
/// closes the window for everyone (subject to the pure module's iteration
/// floor). Returns each local rank's mean seconds per payload iteration;
/// the alignment round and the control steps stay outside the tallies.
fn consensus_window(
    ranks: &LocalRanks,
    buffers: &mut Buffers,
    control: &mut Control,
    elements: usize,
    budget_secs: u64,
) -> Result<Vec<f64>> {
    ranks.sync_all()?;
    // Alignment round, untallied: whatever skew the previous window
    // boundary (or the worker start release) left behind is absorbed by
    // one batch, so the first tallied batch begins from a synchronized
    // point on every rank.
    payload_batch(ranks, buffers, elements)?;

    let roles: Vec<WindowRole> = ranks
        .ranks()
        .iter()
        .map(|rank| role_for_rank(rank.global))
        .collect();
    // Only the host driving global rank 0 has a clock that matters; every
    // other host needs the failsafe against a dead lead.
    let follower_host = !roles.contains(&WindowRole::Lead);

    let started = Instant::now();
    let deadline = started + Duration::from_secs(budget_secs.max(1));
    let failsafe_secs = window::failsafe_secs(budget_secs);
    let mut tallies = vec![WindowTally::default(); ranks.ranks().len()];
    let mut close_seen = false;
    loop {
        let secs = payload_batch(ranks, buffers, elements)?;
        for (tally, secs) in tallies.iter_mut().zip(secs) {
            tally.record(u64::from(OVERLAP_CONTROL_INTERVAL), secs);
        }

        let expired = Instant::now() >= deadline;
        for ((rank, role), ctrl) in ranks.ranks().iter().zip(&roles).zip(&mut control.send) {
            rank.stream()
                .memcpy_htod(&[window::contribution(*role, expired)], ctrl)?;
        }
        {
            let Control { send, recv } = &mut *control;
            ranks.collective("overlap control all_reduce", |index, rank| {
                rank.comm
                    .all_reduce(&send[index], &mut recv[index], &ReduceOp::Min)
            })?;
        }
        ranks.sync_all()?;
        // The collective returns the same word on every rank; read it
        // from local rank 0.
        let word = match (ranks.ranks().first(), control.recv.first()) {
            (Some(rank0), Some(recv0)) => rank0
                .stream()
                .clone_dtoh(recv0)?
                .first()
                .copied()
                .unwrap_or(window::CONTROL_CLOSE),
            _ => window::CONTROL_CLOSE,
        };
        close_seen |= window::window_closed(word);
        // Every rank runs identical batches, so the counts agree.
        let tallied = tallies.first().map_or(0, WindowTally::iters);
        if window::should_close(word, tallied) {
            return Ok(tallies.iter().map(WindowTally::per_iter_secs).collect());
        }
        // Failsafe: a follower host whose lead has never *signaled* close
        // (a dead rank 0 whose collectives still drain, a wedged clock)
        // must end the protocol with an error instead of hammering the
        // fabric until an external kill. Once the close word has been seen
        // the lead is provably alive and the iteration floor bounds the
        // loop. The lead's host needs no failsafe — its own deadline plus
        // the floor bound the loop.
        if follower_host
            && window::failsafe_tripped(close_seen, started.elapsed().as_secs_f64(), failsafe_secs)
        {
            // Not a fault on this host: the lead (or the fleet) stopped.
            // Surfaces as a cascade exit, attributed to whoever failed.
            return Err(CascadeAbort(format!(
                "fleet overlap window failsafe: no close signal from rank 0 within {failsafe_secs}s"
            ))
            .into());
        }
    }
}

/// One batch of `OVERLAP_CONTROL_INTERVAL` grouped payload all-reduces;
/// returns each local rank's seconds from the batch launch to its own
/// completion.
fn payload_batch(ranks: &LocalRanks, buffers: &mut Buffers, elements: usize) -> Result<Vec<f64>> {
    let start = Instant::now();
    for _ in 0..OVERLAP_CONTROL_INTERVAL {
        buffers.all_reduce(ranks, "overlap all_reduce", elements, &ReduceOp::Sum)?;
    }
    ranks.completion_secs(start)
}
