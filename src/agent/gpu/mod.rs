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
pub mod sdc;

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

use anyhow::Result;
use cudarc::driver::CudaContext;

use crate::agent::EventSink;
use crate::proto::{GpuTaskSpec, LogLevel, Scope, TestId, TestOutcome};

/// Run a CUDA-touching closure, converting *any* failure into a string.
///
/// cudarc's `dynamic-loading` backend resolves symbols lazily and **panics**
/// (`Missing symbol ...` / `panic_no_lib_found`) when libcuda / libcublas is
/// not installed, instead of returning a `DriverError`. A node with a broken
/// or absent driver stack is a finding to report, never an agent crash, so
/// every entry into cudarc goes through here.
pub(crate) fn guard<T>(what: &str, body: impl FnOnce() -> Result<T>) -> Result<T, String> {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(format!("{what}: {error:#}")),
        Err(payload) => Err(format!("{what}: panicked: {}", panic_text(payload))),
    }
}

fn panic_text(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// Run GEMM correctness + sustained perf, memory bandwidth, and p2p tests
/// on every visible GPU. Clock/temperature sampling runs alongside the
/// sustained GEMM (via NVML or nvidia-smi polling) to expose thermal
/// throttling: emit `gpu_gemm_perf.clock_mhz_start` / `clock_mhz_end` /
/// `temp_c_max` per GPU.
pub fn run(sink: &EventSink, spec: &GpuTaskSpec) -> Result<()> {
    let device_count = match guard("cuda driver init", || Ok(CudaContext::device_count()?)) {
        Ok(count) => count.max(0) as u32,
        Err(reason) => {
            // A broken driver on a node we were asked to test is the finding.
            sink.outcome(
                TestId::GpuGemmCorrectness,
                Scope::Node,
                TestOutcome::Failed { reason },
            );
            return Ok(());
        }
    };

    if device_count == 0 {
        sink.outcome(
            TestId::GpuGemmCorrectness,
            Scope::Node,
            TestOutcome::Skipped {
                reason: "cuda driver reports no visible devices".into(),
            },
        );
        return Ok(());
    }

    sink.log(
        LogLevel::Info,
        format!("gpu phase: {device_count} device(s) visible"),
    );

    // One GPU failing must not stop the rest: each test is caught, reported
    // as a Failed outcome under that GPU's scope, and the loop continues.
    for index in 0..device_count {
        let scope = Scope::Gpu { index };
        if let Err(reason) = guard("gemm", || gemm::run_on_gpu(sink, index, spec)) {
            sink.outcome(
                TestId::GpuGemmCorrectness,
                scope.clone(),
                TestOutcome::Failed { reason },
            );
        }
        if let Err(reason) = guard("bandwidth", || {
            bandwidth::run_on_gpu(sink, index, spec.bandwidth_bytes)
        }) {
            sink.outcome(
                TestId::GpuMemBandwidth,
                scope,
                TestOutcome::Failed { reason },
            );
        }
    }

    if let Err(reason) = guard("p2p", || p2p::run_all_pairs(sink, spec.bandwidth_bytes)) {
        sink.outcome(TestId::GpuP2p, Scope::Node, TestOutcome::Failed { reason });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_reports_errors_without_propagating() {
        let error = guard("thing", || -> Result<()> { anyhow::bail!("boom") })
            .expect_err("error must be captured");
        assert!(error.contains("thing"), "{error}");
        assert!(error.contains("boom"), "{error}");
    }

    #[test]
    fn guard_converts_panics_into_findings() {
        let error = guard("thing", || -> Result<()> {
            panic!("missing symbol cuInit")
        })
        .expect_err("panic");
        assert!(error.contains("panicked"), "{error}");
        assert!(error.contains("cuInit"), "{error}");
    }

    #[test]
    fn guard_passes_values_through() {
        assert_eq!(guard("thing", || Ok(7u32)), Ok(7));
    }
}
