//! Hot silent-data-corruption checks for the sustained GEMM.
//!
//! SDC is strongly temperature- and voltage-dependent, so verifying GEMM
//! output only at start-of-phase temperatures (the fixed-seed correctness
//! pass) misses the failures that matter. This module supplies the pure
//! scheduling and accounting logic for verifying the *hot* output
//! periodically during the sustained run; the CUDA plumbing stays in
//! `gemm.rs` so everything here is unit-testable on GPU-less machines.
//!
//! ## Why bitwise comparison is the signal
//!
//! cuBLAS guarantees that a given routine from a given toolkit version
//! produces bit-wise identical results at every invocation on the same GPU
//! (same architecture, same SM count), given identical parameters. The
//! sustained loop launches the *same* GEMM on the same handle, stream and
//! buffers every iteration, so a baseline output captured after the first
//! (warm) iteration must be reproduced exactly by every later iteration.
//! Any deviation — a single flipped bit — is either silent data corruption
//! or compute-pipeline instability under load, which is exactly what this
//! screen exists to catch. No tolerance is involved: tolerance-based
//! correctness against an f64 reference remains the job of the fixed-seed
//! GEMM test.
//!
//! ## Why scheduling runs on busy time
//!
//! The reported sustained GFLOPS must not be polluted by verification
//! overhead (a device-to-host download of C plus a host-side compare). The
//! timed loop therefore accumulates *busy* time — wall time spent inside
//! launch/synchronize windows only — and the scheduler decides when a check
//! is due in that busy-time domain. Verification happens between windows and
//! its cost is invisible to the throughput figure.

use std::time::Duration;

use crate::proto::TestOutcome;

/// Maximum failures spelled out in an outcome reason; the counts still
/// cover everything.
const DESCRIBED_FAILURES: usize = 4;

/// Decides when a hot-output check is due, in accumulated-busy-time terms.
///
/// Fires once per crossed interval boundary; a batch that overshoots
/// several boundaries at once fires a single check and re-arms past the
/// current position, so a stall never causes a burst of back-to-back
/// checks. A zero interval disables scheduling entirely.
#[derive(Debug, Clone)]
pub struct CheckScheduler {
    interval: Duration,
    next_at: Duration,
}

impl CheckScheduler {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            next_at: interval,
        }
    }

    pub fn enabled(&self) -> bool {
        !self.interval.is_zero()
    }

    /// True when `busy` has reached the next checkpoint; advances the
    /// schedule so each crossing fires exactly once.
    pub fn due(&mut self, busy: Duration) -> bool {
        if self.interval.is_zero() || busy < self.next_at {
            return false;
        }
        while self.next_at <= busy {
            self.next_at += self.interval;
        }
        true
    }
}

/// Compare a hot output against the baseline, bit for bit.
///
/// Returns `None` when every element is bitwise identical, otherwise the
/// maximum absolute deviation across mismatching elements. A NaN anywhere in
/// the comparison, or a length mismatch, reports `f64::INFINITY`: corruption
/// that cannot be quantified must still never look small (and the deviation
/// itself must never be NaN, which the wire format cannot carry).
pub fn bitwise_mismatch(baseline: &[f32], current: &[f32]) -> Option<f64> {
    if baseline.len() != current.len() {
        return Some(f64::INFINITY);
    }
    let mut worst = 0.0_f64;
    let mut mismatched = false;
    for (want, got) in baseline.iter().zip(current) {
        if want.to_bits() == got.to_bits() {
            continue;
        }
        mismatched = true;
        let deviation = (f64::from(*got) - f64::from(*want)).abs();
        if deviation.is_nan() {
            return Some(f64::INFINITY);
        }
        if deviation > worst {
            worst = deviation;
        }
    }
    mismatched.then_some(worst)
}

/// One failed check, with the thermal/clock context captured at that moment
/// (absent when telemetry is unavailable on the node).
#[derive(Debug, Clone, PartialEq)]
pub struct SdcFailure {
    /// 0-based index of the check within this dtype's sustained run.
    pub check_index: u64,
    pub max_abs_dev: f64,
    pub clock_mhz: Option<f64>,
    pub temp_c: Option<f64>,
}

impl SdcFailure {
    fn describe(&self) -> String {
        let context = match (self.clock_mhz, self.temp_c) {
            (Some(clock), Some(temp)) => format!(" at {clock:.0} MHz / {temp:.0} C"),
            (Some(clock), None) => format!(" at {clock:.0} MHz"),
            (None, Some(temp)) => format!(" at {temp:.0} C"),
            (None, None) => String::new(),
        };
        format!(
            "check #{} deviated by {:.1e}{context}",
            self.check_index, self.max_abs_dev
        )
    }
}

/// Accounting for one dtype's hot checks.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SdcStats {
    pub checks: u64,
    pub mismatches: u64,
    /// Worst absolute deviation seen across all mismatching checks; 0.0
    /// when clean.
    pub max_abs_dev: f64,
    pub failures: Vec<SdcFailure>,
}

impl SdcStats {
    /// Record one check: `mismatch` is `bitwise_mismatch`'s result, the
    /// telemetry fields are the clock/temperature at the failure (queried
    /// only when a mismatch occurred).
    pub fn record(&mut self, mismatch: Option<f64>, clock_mhz: Option<f64>, temp_c: Option<f64>) {
        let check_index = self.checks;
        self.checks += 1;
        if let Some(max_abs_dev) = mismatch {
            self.mismatches += 1;
            if max_abs_dev > self.max_abs_dev {
                self.max_abs_dev = max_abs_dev;
            }
            self.failures.push(SdcFailure {
                check_index,
                max_abs_dev,
                clock_mhz,
                temp_c,
            });
        }
    }

    /// Human summary of the failures, capped at `DESCRIBED_FAILURES`.
    pub fn describe_failures(&self) -> String {
        let mut parts: Vec<String> = self
            .failures
            .iter()
            .take(DESCRIBED_FAILURES)
            .map(SdcFailure::describe)
            .collect();
        if self.failures.len() > DESCRIBED_FAILURES {
            parts.push(format!(
                "and {} more",
                self.failures.len() - DESCRIBED_FAILURES
            ));
        }
        parts.join("; ")
    }

    /// Metric-safe deviation: the wire format cannot carry non-finite
    /// values, so unquantifiable corruption saturates.
    pub fn reportable_max_abs_dev(&self) -> f64 {
        if self.max_abs_dev.is_finite() {
            self.max_abs_dev
        } else {
            f64::MAX
        }
    }
}

/// Per-GPU verdict over all dtypes' hot checks. Any mismatch anywhere is a
/// hard failure; a run that completed no checks (disabled, or shorter than
/// one interval) is Skipped, never silently Passed.
pub fn summary_outcome(enabled: bool, summary: &[(&'static str, SdcStats)]) -> TestOutcome {
    if !enabled {
        return TestOutcome::Skipped {
            reason: "hot SDC checks disabled (sdc_check_secs = 0)".into(),
        };
    }
    let total_checks: u64 = summary.iter().map(|(_, stats)| stats.checks).sum();
    let failing: Vec<String> = summary
        .iter()
        .filter(|(_, stats)| stats.mismatches > 0)
        .map(|(tag, stats)| {
            format!(
                "{tag}: {}/{} checks mismatched ({})",
                stats.mismatches,
                stats.checks,
                stats.describe_failures()
            )
        })
        .collect();
    if !failing.is_empty() {
        TestOutcome::Failed {
            reason: failing.join(" | "),
        }
    } else if total_checks == 0 {
        TestOutcome::Skipped {
            reason: "sustained run completed no hot SDC checks (window shorter than the interval)"
                .into(),
        }
    } else {
        TestOutcome::Passed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_advances_past_the_current_position() {
        let mut scheduler = CheckScheduler::new(Duration::from_millis(100));
        assert!(!scheduler.due(Duration::from_millis(99)));
        assert!(scheduler.due(Duration::from_millis(100)));
        assert!(!scheduler.due(Duration::from_millis(150)));
        assert!(scheduler.due(Duration::from_millis(250)));
        assert!(!scheduler.due(Duration::from_millis(299)));
        assert!(scheduler.due(Duration::from_millis(300)));
    }

    #[test]
    fn negative_zero_counts_as_a_bit_mismatch() {
        // -0.0 == 0.0 numerically but differs bitwise: with deterministic
        // kernels a sign-bit flip is corruption like any other.
        let mismatch = bitwise_mismatch(&[0.0f32], &[-0.0f32]).expect("sign flip detected");
        assert_eq!(mismatch, 0.0);
    }

    #[test]
    fn describe_failures_caps_the_list() {
        let mut stats = SdcStats::default();
        for _ in 0..(DESCRIBED_FAILURES + 3) {
            stats.record(Some(1.0), None, None);
        }
        let description = stats.describe_failures();
        assert!(description.contains("and 3 more"), "{description}");
    }

    #[test]
    fn infinite_deviation_is_saturated_for_metrics() {
        let mut stats = SdcStats::default();
        stats.record(Some(f64::INFINITY), None, None);
        assert_eq!(stats.reportable_max_abs_dev(), f64::MAX);
        assert!(stats.max_abs_dev.is_infinite());
    }
}
