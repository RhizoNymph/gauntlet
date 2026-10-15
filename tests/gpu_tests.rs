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
        sdc_check_secs: 1,
    }
}

// ---------------------------------------------------------------------------
// Hot-SDC check scheduling and accounting (pure logic, no GPU needed)
// ---------------------------------------------------------------------------

mod sdc {
    use std::time::Duration;

    use gauntlet::agent::gpu::sdc::{CheckScheduler, SdcStats, bitwise_mismatch, summary_outcome};
    use gauntlet::proto::TestOutcome;

    #[test]
    fn scheduler_fires_once_per_crossed_interval_of_busy_time() {
        let mut scheduler = CheckScheduler::new(Duration::from_secs(5));
        assert!(scheduler.enabled());
        assert!(!scheduler.due(Duration::from_secs(3)));
        assert!(scheduler.due(Duration::from_secs(5)));
        // Same busy window must not re-fire.
        assert!(!scheduler.due(Duration::from_secs(6)));
        assert!(scheduler.due(Duration::from_secs(10)));
        // A long stall skips missed checkpoints instead of bursting.
        assert!(scheduler.due(Duration::from_secs(60)));
        assert!(!scheduler.due(Duration::from_secs(60)));
        assert!(scheduler.due(Duration::from_secs(65)));
    }

    #[test]
    fn scheduler_with_zero_interval_is_disabled() {
        let mut scheduler = CheckScheduler::new(Duration::ZERO);
        assert!(!scheduler.enabled());
        assert!(!scheduler.due(Duration::from_secs(1_000_000)));
    }

    #[test]
    fn bitwise_compare_accepts_identical_outputs_only() {
        let baseline = vec![1.0f32, -2.5, 0.0, 3.75];
        assert_eq!(bitwise_mismatch(&baseline, &baseline), None);

        // A single flipped mantissa bit is a mismatch with a tiny deviation.
        let mut flipped = baseline.clone();
        flipped[1] = f32::from_bits(flipped[1].to_bits() ^ 1);
        let deviation = bitwise_mismatch(&baseline, &flipped).expect("bit flip detected");
        assert!(deviation > 0.0 && deviation < 1e-5, "{deviation}");

        // Deviation is the worst element.
        let mut off = baseline.clone();
        off[0] = 1.5;
        off[3] = 5.75;
        let worst = bitwise_mismatch(&baseline, &off).expect("mismatch");
        assert!((worst - 2.0).abs() < 1e-12, "{worst}");

        // A NaN in the hot output can never pass or produce a NaN deviation.
        let mut poisoned = baseline.clone();
        poisoned[2] = f32::NAN;
        let dev = bitwise_mismatch(&baseline, &poisoned).expect("NaN detected");
        assert!(dev.is_infinite(), "{dev}");

        // Truncated output is corruption, not a comparison error.
        assert!(
            bitwise_mismatch(&baseline, &baseline[..3])
                .expect("length mismatch detected")
                .is_infinite()
        );
    }

    #[test]
    fn stats_accumulate_checks_mismatches_and_failure_context() {
        let mut stats = SdcStats::default();
        stats.record(None, None, None);
        stats.record(Some(0.25), Some(1410.0), Some(84.0));
        stats.record(None, None, None);
        stats.record(Some(3.5), None, None);

        assert_eq!(stats.checks, 4);
        assert_eq!(stats.mismatches, 2);
        assert_eq!(stats.max_abs_dev, 3.5);
        assert_eq!(stats.failures.len(), 2);
        assert_eq!(stats.failures[0].check_index, 1);
        assert_eq!(stats.failures[0].clock_mhz, Some(1410.0));
        assert_eq!(stats.failures[0].temp_c, Some(84.0));
        assert_eq!(stats.failures[1].check_index, 3);
        assert_eq!(stats.failures[1].temp_c, None);

        let description = stats.describe_failures();
        assert!(description.contains("1410"), "{description}");
        assert!(description.contains("84"), "{description}");
    }

    #[test]
    fn outcome_is_failed_on_any_mismatch_and_skipped_when_unchecked() {
        let clean = SdcStats {
            checks: 6,
            ..SdcStats::default()
        };
        assert!(matches!(
            summary_outcome(true, &[("f32", clean.clone())]),
            TestOutcome::Passed
        ));

        let mut corrupt = SdcStats::default();
        corrupt.record(Some(0.5), Some(1350.0), Some(88.0));
        match summary_outcome(true, &[("f32", clean.clone()), ("bf16", corrupt)]) {
            TestOutcome::Failed { reason } => {
                assert!(reason.contains("bf16"), "dtype missing: {reason}");
                assert!(reason.contains("88"), "temperature missing: {reason}");
                assert!(reason.contains("1350"), "clock missing: {reason}");
            }
            other => panic!("mismatch must fail hard, got {other:?}"),
        }

        assert!(matches!(
            summary_outcome(false, &[]),
            TestOutcome::Skipped { .. }
        ));
        // Enabled but the run was too short for a single check: not a pass.
        assert!(matches!(
            summary_outcome(true, &[("f32", SdcStats::default())]),
            TestOutcome::Skipped { .. }
        ));
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

    // Hot-SDC verification ran during the sustained window on every GPU.
    for gpu in &correctness_gpus {
        let scope = Scope::Gpu { index: *gpu };
        let checks_ran = events.iter().any(|event| {
            matches!(event, AgentEvent::Metric { record }
                if record.test == TestId::GpuGemmSdc
                    && record.scope == scope
                    && record.name.starts_with("checks_")
                    && record.value > 0.0)
        });
        assert!(checks_ran, "gpu {gpu} ran no hot-SDC checks");
        let verdict = events.iter().find_map(|event| match event {
            AgentEvent::Outcome {
                test: TestId::GpuGemmSdc,
                scope: outcome_scope,
                outcome,
            } if *outcome_scope == scope => Some(outcome),
            _ => None,
        });
        assert!(
            matches!(verdict, Some(TestOutcome::Passed)),
            "gpu {gpu} hot-SDC verdict: {verdict:?}"
        );
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
