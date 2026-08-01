//! GEMM correctness and sustained throughput per GPU, per dtype.
//!
//! Correctness: fixed-seed matrices (deterministic host-side PRNG), one
//! cuBLAS GEMM per dtype, result compared against an f64 CPU reference on a
//! deterministic element subsample (full f64 matmul at gemm_dim=8192 is too
//! slow; ~4096 sampled elements recomputed in f64 on the host suffice).
//! Report max relative error as `gpu_gemm_correctness.residual` per
//! GPU+dtype; outcome Failed if it exceeds the dtype tolerance:
//! f32 1e-5, tf32 5e-4, bf16 5e-2, f16 5e-3 (scaled for accumulation depth).
//! Reduced-precision results legitimately vary with tile order — tolerance
//! comparison, never bit-exactness. The residual VALUE is the straggler
//! signal fleet-wide even when it passes.
//!
//! Throughput: repeated GEMM for `gemm_secs` wall seconds; report sustained
//! (not first-iteration) GFLOPS per GPU+dtype under `gpu_gemm_perf.gflops`;
//! metric name carries the dtype: "gflops_f32", "gflops_bf16", ...

use anyhow::Result;

use crate::agent::EventSink;
use crate::proto::{GemmDtype, GpuTaskSpec};

/// Per-dtype residual tolerance for correctness classification.
pub fn residual_tolerance(dtype: GemmDtype) -> f64 {
    match dtype {
        GemmDtype::F32 => 1e-5,
        GemmDtype::Tf32 => 5e-4,
        GemmDtype::Bf16 => 5e-2,
        GemmDtype::F16 => 5e-3,
    }
}

pub fn run_on_gpu(sink: &EventSink, gpu_index: u32, spec: &GpuTaskSpec) -> Result<()> {
    let _ = (sink, gpu_index, spec);
    todo!("agent C: implement")
}
