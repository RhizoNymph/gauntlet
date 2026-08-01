//! Alpha-beta network model fit: t(size) = alpha + beta * size, least
//! squares over a message-size sweep. Alpha is startup latency, beta the
//! inverse bandwidth. These are the calibration constants simulators want.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FitError {
    #[error("need at least 2 distinct sizes, got {got}")]
    TooFewPoints { got: usize },
    #[error("non-finite timing in input")]
    NonFinite,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AlphaBetaFit {
    /// Startup latency in microseconds.
    pub alpha_us: f64,
    /// Microseconds per byte (1/beta is bandwidth).
    pub beta_us_per_byte: f64,
    /// Coefficient of determination of the fit.
    pub r_squared: f64,
}

impl AlphaBetaFit {
    /// Effective bandwidth implied by beta, in GiB/s.
    pub fn bandwidth_gib_per_sec(&self) -> f64 {
        if self.beta_us_per_byte <= 0.0 {
            return f64::INFINITY;
        }
        1.0 / self.beta_us_per_byte / 1024.0 / 1024.0 / 1024.0 * 1_000_000.0
    }

    /// Predicted transfer time for `bytes`, in microseconds.
    pub fn predict_us(&self, bytes: u64) -> f64 {
        self.alpha_us + self.beta_us_per_byte * bytes as f64
    }
}

/// Fit alpha/beta from `(message_bytes, elapsed_us)` samples. Multiple
/// samples per size are fine (ordinary least squares over all points).
pub fn fit_alpha_beta(points: &[(u64, f64)]) -> Result<AlphaBetaFit, FitError> {
    if points.iter().any(|(_, elapsed_us)| !elapsed_us.is_finite()) {
        return Err(FitError::NonFinite);
    }
    let distinct: BTreeSet<u64> = points.iter().map(|(bytes, _)| *bytes).collect();
    if distinct.len() < 2 {
        return Err(FitError::TooFewPoints {
            got: distinct.len(),
        });
    }

    let n = points.len() as f64;
    let mean_x = points.iter().map(|(bytes, _)| *bytes as f64).sum::<f64>() / n;
    let mean_y = points.iter().map(|(_, us)| *us).sum::<f64>() / n;

    // Centered (rather than raw-moment) accumulation: message sizes span
    // several orders of magnitude, and the raw normal equations lose most of
    // their significant digits at the top of the sweep.
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (bytes, elapsed_us) in points {
        let dx = *bytes as f64 - mean_x;
        sxx += dx * dx;
        sxy += dx * (elapsed_us - mean_y);
    }
    // >= 2 distinct sizes guarantees a non-zero spread in x.
    let beta_us_per_byte = sxy / sxx;
    let alpha_us = mean_y - beta_us_per_byte * mean_x;

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (bytes, elapsed_us) in points {
        let predicted = alpha_us + beta_us_per_byte * *bytes as f64;
        ss_res += (elapsed_us - predicted).powi(2);
        ss_tot += (elapsed_us - mean_y).powi(2);
    }
    let r_squared = if ss_tot <= 0.0 {
        // Every timing identical: the horizontal line is an exact fit.
        1.0
    } else {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    };

    Ok(AlphaBetaFit {
        alpha_us,
        beta_us_per_byte,
        r_squared,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_timings_are_a_perfect_horizontal_fit() {
        let fit = fit_alpha_beta(&[(1024, 7.0), (2048, 7.0), (4096, 7.0)]).expect("fit");
        assert!((fit.alpha_us - 7.0).abs() < 1e-9, "{fit:?}");
        assert!(fit.beta_us_per_byte.abs() < 1e-12, "{fit:?}");
        assert_eq!(fit.r_squared, 1.0);
    }

    #[test]
    fn r_squared_is_clamped_for_uncorrelated_data() {
        // Deliberately noisy: the fit must still report a value in [0, 1].
        let points = [
            (1_024u64, 90.0),
            (2_048, 10.0),
            (4_096, 80.0),
            (8_192, 20.0),
            (16_384, 70.0),
        ];
        let fit = fit_alpha_beta(&points).expect("fit");
        assert!((0.0..=1.0).contains(&fit.r_squared), "{fit:?}");
    }

    #[test]
    fn distinct_size_count_is_reported() {
        match fit_alpha_beta(&[(512, 1.0), (512, 2.0)]) {
            Err(FitError::TooFewPoints { got }) => assert_eq!(got, 1),
            other => panic!("expected TooFewPoints, got {other:?}"),
        }
    }

    #[test]
    fn zero_beta_reports_infinite_bandwidth() {
        let fit = AlphaBetaFit {
            alpha_us: 1.0,
            beta_us_per_byte: 0.0,
            r_squared: 1.0,
        };
        assert!(fit.bandwidth_gib_per_sec().is_infinite());
    }
}
