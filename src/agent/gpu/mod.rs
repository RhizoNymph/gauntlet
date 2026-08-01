//! Phase 2: per-GPU tests via cudarc (pure `dynamic-loading`: libcuda /
//! libcublas are dlopened at runtime, so the binary builds and runs on
//! GPU-less machines and only this phase needs the driver stack).
//!
//! A node whose driver stack fails to initialize emits a Failed outcome for
//! each GPU test rather than erroring the whole agent, EXCEPT when the
//! inventory saw GPUs — then initialization failure is itself a finding
//! (broken driver) and is reported as `Failed` with the CUDA error string.

pub mod bandwidth;
pub mod gemm;
pub mod p2p;

use anyhow::Result;

use crate::agent::EventSink;
use crate::proto::GpuTaskSpec;

/// Run GEMM correctness + sustained perf, memory bandwidth, and p2p tests
/// on every visible GPU. Clock/temperature sampling runs alongside the
/// sustained GEMM (via NVML or nvidia-smi polling) to expose thermal
/// throttling: emit `gpu_gemm_perf.clock_mhz_start` / `clock_mhz_end` /
/// `temp_c_max` per GPU.
pub fn run(sink: &EventSink, spec: &GpuTaskSpec) -> Result<()> {
    let _ = (sink, spec);
    todo!("agent C: implement")
}
