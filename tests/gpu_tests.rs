//! GPU tests: compile always (with the gpu feature), execute only where an
//! NVIDIA driver stack exists — hence #[ignore]. Run on a GPU node with:
//!   cargo test --test gpu_tests -- --ignored
#![cfg(feature = "gpu")]

mod common;

use std::collections::BTreeSet;

use gauntlet::agent::gpu;
use gauntlet::proto::{AgentEvent, GemmDtype, GpuTaskSpec, Scope, TestId, TestOutcome, Unit};

fn quick_spec() -> GpuTaskSpec {
    GpuTaskSpec {
        gemm_secs: 3,
        gemm_dtypes: vec![GemmDtype::F32, GemmDtype::Bf16],
        gemm_dim: 2048,
        bandwidth_bytes: 256 << 20,
    }
}

#[test]
fn tolerances_are_ordered_by_precision() {
    use gauntlet::agent::gpu::gemm::residual_tolerance;
    assert!(residual_tolerance(GemmDtype::F32) < residual_tolerance(GemmDtype::Tf32));
    assert!(residual_tolerance(GemmDtype::Tf32) < residual_tolerance(GemmDtype::F16));
    assert!(residual_tolerance(GemmDtype::F16) < residual_tolerance(GemmDtype::Bf16));
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn gpu_phase_reports_every_gpu() {
    let (sink, buf) = common::capturing_sink();
    gpu::run(&sink, &quick_spec()).expect("gpu phase");
    let events = common::decode_events(&buf);

    let correctness_gpus: BTreeSet<u32> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::Outcome {
                test: TestId::GpuGemmCorrectness,
                scope: Scope::Gpu { index },
                outcome,
            } => {
                assert!(
                    !matches!(outcome, TestOutcome::Skipped { .. }),
                    "GPU present must not skip"
                );
                Some(*index)
            }
            _ => None,
        })
        .collect();
    assert!(!correctness_gpus.is_empty(), "no GPUs tested");

    // Residual metric emitted even for passing dtypes.
    let residuals = events.iter().filter(|event| {
        matches!(event, AgentEvent::Metric { record }
            if record.test == TestId::GpuGemmCorrectness
                && record.name.starts_with("residual")
                && record.unit == Unit::Residual
                && record.value.is_finite()
                && record.value >= 0.0)
    });
    assert!(residuals.count() >= correctness_gpus.len());

    // Sustained perf + clock telemetry per GPU.
    for gpu in &correctness_gpus {
        let has_gflops = events.iter().any(|event| {
            matches!(event, AgentEvent::Metric { record }
                if record.test == TestId::GpuGemmPerf
                    && record.scope == (Scope::Gpu { index: *gpu })
                    && record.name.starts_with("gflops_")
                    && record.value > 0.0)
        });
        assert!(has_gflops, "gpu {gpu} missing sustained gflops");
    }

    // Bandwidth triple per GPU.
    for direction in ["d2d", "h2d_pinned", "d2h_pinned"] {
        let seen = events.iter().any(|event| {
            matches!(event, AgentEvent::Metric { record }
                if record.test == TestId::GpuMemBandwidth
                    && record.name == direction
                    && record.value > 0.0)
        });
        assert!(seen, "missing {direction} bandwidth");
    }
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn gemm_correctness_passes_on_healthy_hardware() {
    let (sink, buf) = common::capturing_sink();
    gpu::run(&sink, &quick_spec()).expect("gpu phase");
    let events = common::decode_events(&buf);
    for event in &events {
        if let AgentEvent::Outcome {
            test: TestId::GpuGemmCorrectness,
            scope,
            outcome,
        } = event
        {
            assert!(
                matches!(outcome, TestOutcome::Passed),
                "healthy GPU failed correctness at {scope:?}: {outcome:?}"
            );
        }
    }
}
