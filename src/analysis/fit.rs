//! Alpha-beta network model fit: t(size) = alpha + beta * size, least
//! squares over a message-size sweep. Alpha is startup latency, beta the
//! inverse bandwidth. These are the calibration constants simulators want.

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
    let _ = points;
    todo!("agent D: implement")
}
