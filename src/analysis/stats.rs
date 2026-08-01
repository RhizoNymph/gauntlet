//! Fleet-relative outlier detection: median + MAD (median absolute
//! deviation). This is the primary straggler signal; absolute thresholds are
//! a secondary overlay applied in `report`.

use serde::{Deserialize, Serialize};

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

/// Median of `values`. Returns `None` for an empty slice. Non-finite inputs
/// are ignored.
pub fn median(values: &[f64]) -> Option<f64> {
    let _ = values;
    todo!("agent D: implement")
}

/// Scaled median absolute deviation (multiplied by 1.4826 so it estimates
/// sigma for normal data). Returns `None` for fewer than 2 finite values.
pub fn mad(values: &[f64]) -> Option<f64> {
    let _ = values;
    todo!("agent D: implement")
}

/// Flag samples whose deviation from the fleet median exceeds `k` MADs.
///
/// Degenerate spread (MAD == 0, e.g. all values identical) flags nothing.
/// Fewer than 4 samples flags nothing (median is meaningless at that size).
pub fn flag_outliers(samples: &[Sample], k: f64) -> Vec<Outlier> {
    let _ = (samples, k);
    todo!("agent D: implement")
}
