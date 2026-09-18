//! Background GEMM load: one plain thread per GPU hammering cuBLAS GEMMs on
//! a dedicated stream while a collective runs elsewhere. Shared by the
//! intra-node overlap phase (`gpu::overlap`) and the fleet overlap step
//! (`agent::nccl`), which differ only in what the collective leg is.
//!
//! Lifecycle contract:
//! - `GemmLoad::spawn` starts the workers; each sets up (fill, upload, warm
//!   launch) off the timed path and then blocks on a start barrier.
//! - `start` releases every worker at once; the combined window begins.
//! - `finish` sets the stop flag and joins every worker, returning one
//!   result per GPU in device order.
//!
//! Workers reach the start barrier on *every* path, including failed setup,
//! so the driver can never deadlock; `finish` consumes the load, so workers
//! can never leak past the window that spawned them.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{Context, Result};
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use super::gemm::{
    GEMM_SEED, Operands, Xorshift64, fill_matrix, launch_gemm, upload_operands, write_ptr,
};
use super::guard;
use crate::proto::GemmDtype;

/// GEMM launches queued between synchronizations. Smaller than phase 2's
/// batch so the stop flag is observed promptly under contention.
const GEMM_BATCH: u64 = 4;

/// Iterations completed and wall time spent by one worker's combined
/// window.
pub(crate) struct GemmThroughput {
    pub iters: u64,
    pub elapsed_secs: f64,
}

/// One GEMM worker thread per GPU, plus the shared start/stop plumbing.
pub(crate) struct GemmLoad {
    stop: Arc<AtomicBool>,
    start: Arc<Barrier>,
    workers: Vec<JoinHandle<Result<GemmThroughput, String>>>,
}

impl GemmLoad {
    /// Spawn one worker per context. Workers set up immediately but do not
    /// start the timed loop until [`GemmLoad::start`].
    pub(crate) fn spawn(contexts: &[Arc<CudaContext>], dim: u32, dtype: GemmDtype) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        // +1: the driver thread joins the barrier through `start`.
        let start = Arc::new(Barrier::new(contexts.len() + 1));
        let workers = contexts
            .iter()
            .map(|ctx| {
                let ctx = Arc::clone(ctx);
                let stop = Arc::clone(&stop);
                let start = Arc::clone(&start);
                std::thread::spawn(move || gemm_worker(ctx, dim, dtype, &stop, &start))
            })
            .collect();
        Self {
            stop,
            start,
            workers,
        }
    }

    /// Meet the workers at the start barrier: returns once every worker has
    /// finished setup and the combined window can begin.
    pub(crate) fn start(&self) {
        self.start.wait();
    }

    /// Stop and join every worker. Always called before the caller
    /// propagates any collective error, so a failed collective never leaks
    /// GEMM threads.
    pub(crate) fn finish(self) -> Vec<Result<GemmThroughput, String>> {
        self.stop.store(true, Ordering::Relaxed);
        self.workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .unwrap_or_else(|_| Err("gemm load worker panicked".to_string()))
            })
            .collect()
    }
}

/// Device-side state a worker holds through the combined window. The
/// operand slices are kept alive here; raw pointers are re-derived inside
/// the loop.
struct GemmState {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    operands: Operands,
    c: CudaSlice<f32>,
    n: usize,
    dtype: GemmDtype,
}

/// Set up, meet the barrier, then hammer GEMMs until told to stop.
///
/// The barrier is reached on *every* path — including a failed setup — so
/// the driver thread can never deadlock waiting for a worker.
fn gemm_worker(
    ctx: Arc<CudaContext>,
    dim: u32,
    dtype: GemmDtype,
    stop: &AtomicBool,
    start: &Barrier,
) -> Result<GemmThroughput, String> {
    let prepared = guard("overlap gemm setup", || prepare(&ctx, dim, dtype));
    start.wait();
    let mut state = prepared?;
    guard("overlap gemm loop", || gemm_loop(&mut state, stop))
}

fn prepare(ctx: &Arc<CudaContext>, dim: u32, dtype: GemmDtype) -> Result<GemmState> {
    ctx.bind_to_thread()?;
    // The collective owns the default stream; the GEMM gets its own so the
    // two workloads genuinely run concurrently on the device.
    let stream = ctx.new_stream()?;
    let blas = CudaBlas::new(Arc::clone(&stream)).context("creating cublas handle")?;
    let n = dim.max(1) as usize;
    let mut rng = Xorshift64::new(GEMM_SEED);
    let host_a = fill_matrix(&mut rng, n * n);
    let host_b = fill_matrix(&mut rng, n * n);
    let operands = upload_operands(&stream, &host_a, &host_b, dtype)?;
    let mut c = stream.alloc_zeros::<f32>(n * n)?;
    // One warm launch so cuBLAS heuristics run outside the timed window.
    let (a_ptr, b_ptr) = operands.device_ptrs(&stream);
    let c_ptr = write_ptr(&mut c, &stream);
    // SAFETY: a_ptr/b_ptr address n*n operands of `dtype` and c_ptr n*n f32,
    // all just allocated on this stream, which is the handle's stream.
    unsafe { launch_gemm(&blas, dtype, n as i32, a_ptr, b_ptr, c_ptr) }
        .context("warmup gemm launch")?;
    stream.synchronize()?;
    Ok(GemmState {
        stream,
        blas,
        operands,
        c,
        n,
        dtype,
    })
}

fn gemm_loop(state: &mut GemmState, stop: &AtomicBool) -> Result<GemmThroughput> {
    let (a_ptr, b_ptr) = state.operands.device_ptrs(&state.stream);
    let c_ptr = write_ptr(&mut state.c, &state.stream);
    let started = Instant::now();
    let mut iters = 0u64;
    while !stop.load(Ordering::Relaxed) {
        for _ in 0..GEMM_BATCH {
            // SAFETY: the pointers come from the live allocations held in
            // `state`, sized n*n as `launch_gemm` requires, on the stream
            // the handle is bound to.
            unsafe {
                launch_gemm(
                    &state.blas,
                    state.dtype,
                    state.n as i32,
                    a_ptr,
                    b_ptr,
                    c_ptr,
                )
            }?;
        }
        state.stream.synchronize()?;
        iters += GEMM_BATCH;
    }
    Ok(GemmThroughput {
        iters,
        elapsed_secs: started.elapsed().as_secs_f64().max(1e-9),
    })
}
