//! What kind of failure a Failed outcome is: a finding about the node
//! (wrong results, a failed collective), or an infrastructure error that
//! stopped the test from producing a verdict (a CUDA call failed, a worker
//! panicked).
//!
//! The distinction matters for the "silent data corruption" section: a
//! correctness screen that could not run is not evidence of corruption.
//! `TestOutcome::Failed` carries a free-text reason only (it is on the
//! wire and constructed in many places), so the kind rides the reason:
//! agents build infrastructure failures with
//! [`TestOutcome::infrastructure_failure`], which prefixes the reason with
//! [`INFRASTRUCTURE_PREFIX`]; [`TestOutcome::failure_kind`] reads it back.
//! As a backstop for error paths that do not use the constructor, a reason
//! carrying a CUDA driver error code (`CUDA_ERROR_*`) is also
//! infrastructure: no mismatch report ever contains one.

use std::fmt;

use super::TestOutcome;

/// Reason prefix of an infrastructure failure.
pub const INFRASTRUCTURE_PREFIX: &str = "infrastructure error: ";

/// Marker present in every cudarc `DriverError` rendering.
const CUDA_DRIVER_ERROR_MARKER: &str = "CUDA_ERROR_";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    /// The test ran and the node failed it.
    Finding,
    /// The test could not run to a verdict.
    Infrastructure,
}

impl TestOutcome {
    /// A Failed outcome for a test that could not run to a verdict.
    pub fn infrastructure_failure(detail: impl fmt::Display) -> Self {
        TestOutcome::Failed {
            reason: format!("{INFRASTRUCTURE_PREFIX}{detail}"),
        }
    }

    /// `None` unless Failed.
    pub fn failure_kind(&self) -> Option<FailureKind> {
        match self {
            TestOutcome::Failed { reason } => Some(failure_kind_of(reason)),
            TestOutcome::Passed | TestOutcome::Skipped { .. } => None,
        }
    }
}

/// Kind of a Failed outcome's reason.
pub fn failure_kind_of(reason: &str) -> FailureKind {
    if reason.starts_with(INFRASTRUCTURE_PREFIX) || reason.contains(CUDA_DRIVER_ERROR_MARKER) {
        FailureKind::Infrastructure
    } else {
        FailureKind::Finding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infrastructure_failures_round_trip_their_kind() {
        let outcome = TestOutcome::infrastructure_failure("gemm: cublas handle");
        assert_eq!(outcome.failure_kind(), Some(FailureKind::Infrastructure));
        let TestOutcome::Failed { reason } = &outcome else {
            panic!("expected Failed");
        };
        assert!(reason.ends_with("gemm: cublas handle"), "{reason}");
    }

    #[test]
    fn mismatches_are_findings() {
        for reason in [
            "bf16 residual 3.1e-1 exceeds tolerance 1.0e-2",
            "f32 gemm produced a non-finite result",
            "bf16: 2/40 checks mismatched (...)",
            "integer checksum mismatch on core 7",
        ] {
            assert_eq!(failure_kind_of(reason), FailureKind::Finding, "{reason}");
        }
    }

    #[test]
    fn raw_cuda_driver_errors_are_infrastructure() {
        // The 2026-10-15 report's reason, from before the GPU phase
        // classified driver init failures.
        let reason = "cuda driver init: DriverError(CUDA_ERROR_NO_DEVICE, \
                      \"no CUDA-capable device is detected\")";
        assert_eq!(failure_kind_of(reason), FailureKind::Infrastructure);
    }

    #[test]
    fn only_failed_outcomes_have_a_kind() {
        assert_eq!(TestOutcome::Passed.failure_kind(), None);
        assert_eq!(
            TestOutcome::Skipped { reason: "x".into() }.failure_kind(),
            None
        );
    }
}
