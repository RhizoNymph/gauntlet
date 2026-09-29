//! Alpha-beta network model fit: t(size) = alpha + beta * size, least
//! squares over a message-size sweep subject to alpha >= 0 and beta >= 0.
//! Alpha is startup latency, beta the inverse bandwidth. These are the
//! calibration constants simulators want, and a negative value of either
//! would have a simulator predict negative time.

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

/// Which non-negativity constraint was active at the constrained optimum.
/// A bound fit is still the best feasible line, but the data were not
/// linear with a non-negative intercept and slope (typically a sweep whose
/// latency-bound small sizes sit above a line dominated by the
/// bandwidth-bound large ones), so it deserves a second look.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitBound {
    /// Unconstrained alpha was negative; the fit goes through the origin.
    AlphaZero,
    /// Unconstrained beta was negative; the fit is a horizontal line.
    BetaZero,
    /// Both rays collapsed to the origin (only for non-positive timings).
    BothZero,
}

impl FitBound {
    /// Short human label for tables.
    pub fn label(self) -> &'static str {
        match self {
            FitBound::AlphaZero => "alpha=0",
            FitBound::BetaZero => "beta=0",
            FitBound::BothZero => "alpha=beta=0",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AlphaBetaFit {
    /// Startup latency in microseconds.
    pub alpha_us: f64,
    /// Microseconds per byte (1/beta is bandwidth).
    pub beta_us_per_byte: f64,
    /// Coefficient of determination of the chosen parameters, in [0, 1].
    pub r_squared: f64,
    /// The constraint that bound, or `None` for a plain OLS fit. Absent in
    /// documents before schema v11, which decode as `None`.
    #[serde(default)]
    pub bound: Option<FitBound>,
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

/// A feasible (or, for OLS, candidate) parameter pair and the constraint
/// that produced it.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    alpha_us: f64,
    beta_us_per_byte: f64,
    bound: Option<FitBound>,
}

impl Candidate {
    const ORIGIN: Candidate = Candidate {
        alpha_us: 0.0,
        beta_us_per_byte: 0.0,
        bound: Some(FitBound::BothZero),
    };

    fn rss(&self, points: &[(u64, f64)]) -> f64 {
        points
            .iter()
            .map(|(bytes, elapsed_us)| {
                (elapsed_us - (self.alpha_us + self.beta_us_per_byte * *bytes as f64)).powi(2)
            })
            .sum()
    }
}

/// Fit alpha/beta from `(message_bytes, elapsed_us)` samples by least
/// squares constrained to `alpha >= 0`, `beta >= 0`. Multiple samples per
/// size are fine.
///
/// # Why the closed form is the exact constrained optimum
///
/// The residual sum of squares `RSS(a, b) = Σ (y - a - b·x)²` is a convex
/// quadratic, and *strictly* convex whenever there are ≥ 2 distinct sizes
/// (its Hessian `2·[[n, Σx], [Σx, Σx²]]` has determinant `4·n·Sxx > 0`).
/// So over the closed convex quadrant `a, b ≥ 0` it has exactly one
/// minimiser, and:
///
/// 1. If the unconstrained (OLS) minimiser is feasible, it is the answer.
/// 2. Otherwise the constrained minimiser lies on the boundary: were it
///    interior, it would be a local, hence (by convexity) the global,
///    unconstrained minimum, which is infeasible — a contradiction. The
///    boundary is the union of two rays, `{a = 0, b ≥ 0}` and
///    `{b = 0, a ≥ 0}`.
/// 3. On each ray RSS is a one-variable convex quadratic, so its minimum
///    over the ray is the unconstrained 1-D minimiser clamped at 0:
///    `b = max(0, Σxy / Σx²)` on the first (least squares through the
///    origin) and `a = max(0, mean(y))` on the second.
/// 4. The constrained minimum is the better of those two ray minima.
///
/// Both rays are therefore always evaluated when OLS is infeasible, rather
/// than only the one whose sign went wrong: that is what makes the result
/// the true optimum rather than a heuristic repair. `r_squared` is then
/// recomputed for the chosen parameters (it can only fall relative to OLS)
/// and clamped to [0, 1].
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

    let ols = ols_about_mean(points, mean_x, mean_y);
    let chosen = if ols.alpha_us >= 0.0 && ols.beta_us_per_byte >= 0.0 {
        ols
    } else {
        let through_origin = through_origin(points);
        let horizontal = horizontal(mean_y);
        if through_origin.rss(points) <= horizontal.rss(points) {
            through_origin
        } else {
            horizontal
        }
    };

    let ss_res = chosen.rss(points);
    let ss_tot: f64 = points
        .iter()
        .map(|(_, elapsed_us)| (elapsed_us - mean_y).powi(2))
        .sum();
    let r_squared = if ss_tot <= 0.0 {
        // Every timing identical: the horizontal line is an exact fit.
        1.0
    } else {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    };

    Ok(AlphaBetaFit {
        alpha_us: chosen.alpha_us,
        beta_us_per_byte: chosen.beta_us_per_byte,
        r_squared,
        bound: chosen.bound,
    })
}

/// Unconstrained OLS. Centered (rather than raw-moment) accumulation:
/// message sizes span several orders of magnitude, and the raw normal
/// equations lose most of their significant digits at the top of the sweep.
fn ols_about_mean(points: &[(u64, f64)], mean_x: f64, mean_y: f64) -> Candidate {
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (bytes, elapsed_us) in points {
        let dx = *bytes as f64 - mean_x;
        sxx += dx * dx;
        sxy += dx * (elapsed_us - mean_y);
    }
    // >= 2 distinct sizes guarantees a non-zero spread in x.
    let beta_us_per_byte = sxy / sxx;
    Candidate {
        alpha_us: mean_y - beta_us_per_byte * mean_x,
        beta_us_per_byte,
        bound: None,
    }
}

/// Minimum of RSS on the ray `alpha = 0, beta >= 0`. There is no intercept
/// to subtract, so raw moments lose nothing; Σx² > 0 because at least one of
/// the >= 2 distinct sizes is non-zero.
fn through_origin(points: &[(u64, f64)]) -> Candidate {
    let (sxy, sxx) = points
        .iter()
        .fold((0.0, 0.0), |(sxy, sxx), (bytes, elapsed_us)| {
            let x = *bytes as f64;
            (sxy + x * elapsed_us, sxx + x * x)
        });
    let slope = sxy / sxx;
    if slope >= 0.0 {
        Candidate {
            alpha_us: 0.0,
            beta_us_per_byte: slope,
            bound: Some(FitBound::AlphaZero),
        }
    } else {
        Candidate::ORIGIN
    }
}

/// Minimum of RSS on the ray `beta = 0, alpha >= 0`: the mean, clamped.
fn horizontal(mean_y: f64) -> Candidate {
    if mean_y >= 0.0 {
        Candidate {
            alpha_us: mean_y,
            beta_us_per_byte: 0.0,
            bound: Some(FitBound::BetaZero),
        }
    } else {
        Candidate::ORIGIN
    }
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
            bound: None,
        };
        assert!(fit.bandwidth_gib_per_sec().is_infinite());
    }
}
