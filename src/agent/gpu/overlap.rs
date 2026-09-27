//! Overlap phase: sustained GEMM concurrent with an intra-node NCCL
//! all-reduce, on the same GPUs at the same time.
//!
//! Real training overlaps compute and communication; stragglers born of PCIe
//! contention, power steering, and NIC/GPU NUMA misplacement only show under
//! the combined load. The phase therefore measures, per node:
//!   - overlapped GEMM GFLOPS per GPU (`overlap_gemm.gflops_<dtype>`), and
//!   - intra-node all-reduce bus bandwidth both isolated and overlapped
//!     (`overlap_all_reduce.{isolated,overlap}_bus_gib_per_sec`).
//!
//! The straggler signal is *retention* (overlapped/isolated), derived
//! orchestrator-side in `report::build`: GEMM retention divides by the
//! phase-2 sustained number (same dim, same dtype, same run) and all-reduce
//! retention divides by the isolated baseline measured here seconds earlier
//! on the same communicator.
//!
//! Topology: one process, one NCCL rank per visible GPU
//! (`ncclCommInitAll` via `NodeComm`, shared with the intra-node NCCL
//! sweep), collectives driven from a single thread inside
//! `ncclGroupStart`/`ncclGroupEnd`, each rank on its device's default
//! stream. The compute leg runs on a *separate* stream per GPU
//! (`gpu::worker`, one plain thread per GPU), so the collective and the
//! GEMM genuinely contend. The fleet-wide sibling of this phase — the same
//! GEMM load under the cross-node NCCL world — lives in `agent::nccl`
//! (`overlap_fleet`) and shares the worker machinery.

use std::time::{Duration, Instant};

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaSlice};
use cudarc::nccl::ReduceOp;

use super::gemm::{dtype_tag, sustained_gflops_value};
use super::guard;
use super::node_comm::{NodeComm, nccl_error};
use super::worker::GemmLoad;
use crate::agent::EventSink;
use crate::agent::nccl::{F32_BYTES, all_reduce_bus_gib_per_sec, message_elements};
use crate::agent::window::hard_deadline_secs;
use crate::proto::{
    LogLevel, MetricRecord, OverlapSpec, Scope, TestId, TestOutcome, Unit, overlap_metric,
};

/// Untimed all-reduce rounds before any timed window, so channel setup and
/// algorithm selection stay out of the isolated baseline.
const WARMUP_ROUNDS: u32 = 5;
/// All-reduce rounds queued between stream synchronizations in a timed
/// window.
const ROUND_ITERS: u32 = 4;

/// Mean seconds per iteration over a timed window.
fn per_iter_secs(elapsed_secs: f64, iters: u64) -> f64 {
    elapsed_secs / iters.max(1) as f64
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the overlap test. Node-level failures (driver init, communicator
/// init) become Failed outcomes; fewer than two GPUs is a Skipped outcome
/// (a world of one moves nothing, so there is no contention to measure).
pub fn run(sink: &EventSink, spec: &OverlapSpec) -> Result<()> {
    let device_count = match guard("cuda driver init", || Ok(CudaContext::device_count()?)) {
        Ok(count) => count.max(0) as u32,
        Err(reason) => {
            node_outcomes(sink, |r| TestOutcome::Failed { reason: r }, &reason);
            return Ok(());
        }
    };
    if device_count < 2 {
        let reason = format!("overlap needs at least 2 GPUs, found {device_count}");
        node_outcomes(sink, |r| TestOutcome::Skipped { reason: r }, &reason);
        return Ok(());
    }
    sink.log(
        LogLevel::Info,
        format!(
            "overlap phase: {device_count} GPUs, {}s combined window",
            spec.duration_secs
        ),
    );
    if let Err(reason) = guard("overlap", || execute(sink, spec, device_count)) {
        node_outcomes(sink, |r| TestOutcome::Failed { reason: r }, &reason);
    }
    Ok(())
}

fn node_outcomes(sink: &EventSink, outcome: impl Fn(String) -> TestOutcome, reason: &str) {
    for test in [TestId::OverlapGemm, TestId::OverlapAllReduce] {
        sink.outcome(test, Scope::Node, outcome(reason.to_string()));
    }
}

// ---------------------------------------------------------------------------
// Combined-load driver
// ---------------------------------------------------------------------------

fn execute(sink: &EventSink, spec: &OverlapSpec, device_count: u32) -> Result<()> {
    let protocol_start = Instant::now();
    let elements = message_elements(spec.msg_bytes);
    let message_bytes = (elements * F32_BYTES) as f64;

    // One context per GPU; the default stream carries the collective.
    let node = NodeComm::init(device_count)?;
    let sends = node.alloc_per_rank(elements)?;
    let mut recvs = node.alloc_per_rank(elements)?;

    for _ in 0..WARMUP_ROUNDS {
        all_reduce_round(&node, &sends, &mut recvs)?;
    }

    // Isolated baseline: same communicator, quiet SMs (the GEMM workers are
    // not even spawned yet).
    let isolated_per_iter = timed_rounds(&node, &sends, &mut recvs, spec.baseline_secs)?;

    // One GEMM worker per GPU (`gpu::worker`). Workers set up (upload,
    // warm launch) off the timed path; `wait_ready` + `start` releases
    // them together and the combined window begins.
    // Same hard compute deadline as the fleet step: a wedged collective
    // must never leave the GPUs saturated past the step's budget.
    let deadline = protocol_start
        + Duration::from_secs(hard_deadline_secs(spec.baseline_secs, spec.duration_secs));
    let mut load = GemmLoad::spawn(&node.contexts, spec.gemm_dim, spec.gemm_dtype, deadline);
    load.wait_ready();
    load.start();

    let overlapped = timed_rounds(&node, &sends, &mut recvs, spec.duration_secs);
    // The workers are stopped and joined before any error propagates, so a
    // failed collective never leaks GEMM threads.
    let throughputs = load.finish();
    let overlapped_per_iter = overlapped?;

    // Collective results, node scope.
    for (name, value, unit) in [
        (overlap_metric::MSG_BYTES, message_bytes, Unit::Bytes),
        (
            overlap_metric::ISOLATED_BUS,
            all_reduce_bus_gib_per_sec(message_bytes, isolated_per_iter, device_count),
            Unit::GibPerSec,
        ),
        (
            overlap_metric::OVERLAP_BUS,
            all_reduce_bus_gib_per_sec(message_bytes, overlapped_per_iter, device_count),
            Unit::GibPerSec,
        ),
    ] {
        sink.metric(MetricRecord {
            test: TestId::OverlapAllReduce,
            scope: Scope::Node,
            name: name.to_string(),
            value,
            unit,
            repeat: 0,
        });
    }
    sink.outcome(TestId::OverlapAllReduce, Scope::Node, TestOutcome::Passed);

    // Compute results, per GPU. One GPU failing must not hide the others.
    let n = spec.gemm_dim.max(1) as usize;
    let tag = dtype_tag(spec.gemm_dtype);
    for (index, throughput) in throughputs.into_iter().enumerate() {
        let scope = Scope::Gpu {
            index: index as u32,
        };
        match throughput {
            Ok(throughput) => {
                sink.metric(MetricRecord {
                    test: TestId::OverlapGemm,
                    scope: scope.clone(),
                    name: format!("gflops_{tag}"),
                    value: sustained_gflops_value(n, throughput.iters, throughput.elapsed_secs),
                    unit: Unit::Gflops,
                    repeat: 0,
                });
                sink.outcome(TestId::OverlapGemm, scope, TestOutcome::Passed);
            }
            Err(reason) => {
                sink.outcome(TestId::OverlapGemm, scope, TestOutcome::Failed { reason });
            }
        }
    }
    Ok(())
}

/// One all-reduce on every rank, grouped so a single thread can drive the
/// whole node without deadlocking (see NCCL group semantics).
fn all_reduce_round(
    node: &NodeComm,
    sends: &[CudaSlice<f32>],
    recvs: &mut [CudaSlice<f32>],
) -> Result<()> {
    node.grouped(|rank, comm| {
        comm.all_reduce(&sends[rank], &mut recvs[rank], &ReduceOp::Sum)
            .map_err(|error| nccl_error("ncclAllReduce", error))?;
        Ok(())
    })
}

/// Mean per-iteration seconds of all-reduce rounds over a wall-time budget.
/// Streams are synchronized on both sides of the timed region so the
/// interval covers completed device work only.
fn timed_rounds(
    node: &NodeComm,
    sends: &[CudaSlice<f32>],
    recvs: &mut [CudaSlice<f32>],
    budget_secs: u64,
) -> Result<f64> {
    node.sync_all()?;
    let budget = Duration::from_secs(budget_secs.max(1));
    let started = Instant::now();
    let mut iters = 0u64;
    while started.elapsed() < budget {
        for _ in 0..ROUND_ITERS {
            all_reduce_round(node, sends, recvs)?;
        }
        node.sync_all()?;
        iters += u64::from(ROUND_ITERS);
    }
    Ok(per_iter_secs(started.elapsed().as_secs_f64(), iters))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_iter_secs_divides_and_survives_zero_iters() {
        assert!((per_iter_secs(10.0, 100) - 0.1).abs() < 1e-12);
        // Cannot happen in the driver loop, but never divide by zero.
        assert_eq!(per_iter_secs(5.0, 0), 5.0);
    }
}
