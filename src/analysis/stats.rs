//! Fleet-relative outlier detection: median + MAD (median absolute
//! deviation). This is the primary straggler signal; absolute thresholds are
//! a secondary overlay applied in `report`.

use serde::{Deserialize, Serialize};

/// Consistency constant that makes the MAD an estimator of sigma for
/// normally distributed data.
pub const MAD_SCALE: f64 = 1.4826;

/// Below this many usable samples the fleet median carries no information,
/// so nothing is flagged.
pub const MIN_SAMPLES_FOR_OUTLIERS: usize = 4;

/// One observation of a metric, tagged with where it came from
/// (host, or host+scope rendered to a stable string key).
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub key: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Outlier {
    pub key: String,
    pub value: f64,
    pub fleet_median: f64,
    /// Signed distance from the median in MAD units (scaled by the usual
    /// 1.4826 consistency constant). Negative = below median.
    pub deviation_mads: f64,
}

/// Distribution summary of one subject's repeated measurements. Median and
/// MAD are the robust headline; mean and (sample) stddev are kept for
/// simulator consumers that want Gaussian inputs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Moments {
    pub n: usize,
    pub median: f64,
    /// Scaled median absolute deviation; 0.0 when n == 1.
    pub mad: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    /// Sample standard deviation (n - 1); 0.0 when n == 1.
    pub stddev: f64,
}

/// Summarize `values` into `Moments`, ignoring non-finite entries.
/// Returns `None` when nothing finite remains.
pub fn moments(values: &[f64]) -> Option<Moments> {
    let sorted = finite_sorted(values);
    if sorted.is_empty() {
        return None;
    }
    let n = sorted.len();
    let median = median_of_sorted(&sorted)?;
    let mad = if n >= 2 { mad(&sorted)? } else { 0.0 };
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let stddev = if n >= 2 {
        let variance = sorted
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };
    Some(Moments {
        n,
        median,
        mad,
        min: sorted[0],
        max: sorted[n - 1],
        mean,
        stddev,
    })
}

/// Median of `values`. Returns `None` for an empty slice. Non-finite inputs
/// are ignored.
pub fn median(values: &[f64]) -> Option<f64> {
    median_of_sorted(&finite_sorted(values))
}

/// Scaled median absolute deviation (multiplied by 1.4826 so it estimates
/// sigma for normal data). Returns `None` for fewer than 2 finite values.
pub fn mad(values: &[f64]) -> Option<f64> {
    let sorted = finite_sorted(values);
    if sorted.len() < 2 {
        return None;
    }
    let center = median_of_sorted(&sorted)?;
    let mut deviations: Vec<f64> = sorted.iter().map(|value| (value - center).abs()).collect();
    sort_finite(&mut deviations);
    median_of_sorted(&deviations).map(|raw| raw * MAD_SCALE)
}

/// Flag samples whose deviation from the fleet median exceeds `k` MADs.
///
/// Degenerate spread (MAD == 0, e.g. all values identical) flags nothing.
/// Fewer than 4 samples flags nothing (median is meaningless at that size).
pub fn flag_outliers(samples: &[Sample], k: f64) -> Vec<Outlier> {
    let values: Vec<f64> = samples
        .iter()
        .map(|sample| sample.value)
        .filter(|value| value.is_finite())
        .collect();
    if values.len() < MIN_SAMPLES_FOR_OUTLIERS {
        return Vec::new();
    }
    let (Some(center), Some(spread)) = (median(&values), mad(&values)) else {
        return Vec::new();
    };
    if spread <= 0.0 {
        return Vec::new();
    }
    samples
        .iter()
        .filter(|sample| sample.value.is_finite())
        .filter_map(|sample| {
            let deviation_mads = (sample.value - center) / spread;
            (deviation_mads.abs() > k).then(|| Outlier {
                key: sample.key.clone(),
                value: sample.value,
                fleet_median: center,
                deviation_mads,
            })
        })
        .collect()
}

fn finite_sorted(values: &[f64]) -> Vec<f64> {
    let mut finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    sort_finite(&mut finite);
    finite
}

/// Sorts a slice already known to hold only finite values, so the partial
/// comparison can never be `None`.
fn sort_finite(values: &mut [f64]) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

fn median_of_sorted(sorted: &[f64]) -> Option<f64> {
    let len = sorted.len();
    if len == 0 {
        return None;
    }
    if len % 2 == 1 {
        Some(sorted[len / 2])
    } else {
        Some((sorted[len / 2 - 1] + sorted[len / 2]) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(key: &str, value: f64) -> Sample {
        Sample {
            key: key.into(),
            value,
        }
    }

    #[test]
    fn median_ignores_non_finite_without_shifting_parity() {
        // Four finite values plus junk: still an even-length median.
        assert_eq!(
            median(&[1.0, f64::NAN, 2.0, 3.0, f64::NEG_INFINITY, 4.0]),
            Some(2.5)
        );
    }

    #[test]
    fn mad_needs_two_finite_values() {
        assert_eq!(mad(&[f64::NAN, f64::INFINITY]), None);
        assert_eq!(mad(&[1.0, 3.0]), Some(1.0 * MAD_SCALE));
    }

    #[test]
    fn non_finite_samples_never_become_outliers() {
        let samples = vec![
            sample("a", 10.0),
            sample("b", 10.5),
            sample("c", 9.5),
            sample("d", 10.0),
            sample("junk", f64::NAN),
            sample("bad", 100.0),
        ];
        let flagged = flag_outliers(&samples, 3.0);
        assert!(flagged.iter().all(|outlier| outlier.key != "junk"));
        assert_eq!(flagged.len(), 1);
        assert_eq!(flagged[0].key, "bad");
        assert!(flagged[0].deviation_mads > 0.0, "above-median is positive");
    }

    #[test]
    fn outlier_order_follows_sample_order() {
        let samples = vec![
            sample("z", 1000.0),
            sample("a", 10.0),
            sample("b", 10.5),
            sample("c", 9.5),
            sample("d", 10.0),
            sample("y", -1000.0),
        ];
        let flagged = flag_outliers(&samples, 3.0);
        let keys: Vec<&str> = flagged.iter().map(|o| o.key.as_str()).collect();
        assert_eq!(keys, vec!["z", "y"]);
    }
}
