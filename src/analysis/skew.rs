//! Barrier-skew statistics: which rank consistently arrives late at a tiny
//! collective.
//!
//! Measurement model. Each of many iterations of a minimal barrier-like
//! operation yields one value per rank, all in microseconds, but the two
//! implemented probes produce values with *opposite* polarity:
//!
//! - **NCCL tiny all-reduce** (`SkewPolarity::LateIsMin`): every rank
//!   records its local time-to-completion. A collective completes at
//!   (nearly) the same instant on every rank, but each rank *starts* its
//!   timer when it arrives — so an early arriver waits out the stragglers
//!   and records a long time, while the straggler itself arrives last and
//!   completes almost immediately. The rank with the *minimum* local
//!   elapsed in an iteration is the one that arrived late.
//! - **TCP star barrier** (`SkewPolarity::LateIsMax`): a single coordinator
//!   releases every rank at once and timestamps each rank's response on one
//!   clock. The rank with the *maximum* release-to-response time is the
//!   late one.
//!
//! Cross-rank wall-clock comparison was rejected: phase 0's
//! `clock_offset_ms` is NTP/chrony-grade (milliseconds), three orders of
//! magnitude coarser than barrier iterations, so per-rank signals must be
//! derived either from local durations (NCCL) or from a single observer's
//! clock (TCP).
//!
//! Everything here is deterministic and side-effect free; the agents and
//! the orchestrator only collect raw per-iteration vectors and feed them
//! through [`analyze`].

use serde::{Deserialize, Serialize};

/// An iteration only feeds the slowest-rank tally when the late rank's
/// deviation from the iteration median exceeds
/// `max(frac * |median|, floor_us)`; otherwise the "straggler" is noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Margin {
    pub frac: f64,
    pub floor_us: f64,
}

impl Default for Margin {
    fn default() -> Self {
        Self {
            frac: DEFAULT_MARGIN_FRAC,
            floor_us: DEFAULT_MARGIN_FLOOR_US,
        }
    }
}

/// A quarter of the iteration median: a rank must lag the pack by a
/// nontrivial fraction of the barrier time itself.
pub const DEFAULT_MARGIN_FRAC: f64 = 0.25;
/// Absolute floor so that on very fast fabrics (median of a few µs) plain
/// scheduler noise cannot win tallies.
pub const DEFAULT_MARGIN_FLOOR_US: f64 = 10.0;
/// Below this many margin-passing iterations the tally carries no signal
/// and no straggler flag may be raised from it.
pub const MIN_TALLY_ITERS: u64 = 100;

/// Which extreme of an iteration's per-rank values marks the late arriver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkewPolarity {
    /// Local completion times of a collective: the straggler waits least.
    LateIsMin,
    /// Coordinator-observed response times: the straggler responds last.
    LateIsMax,
}

/// One rank's raw per-iteration timings, as collected by the agents.
#[derive(Debug, Clone, PartialEq)]
pub struct RankSeries {
    pub rank: u32,
    pub elapsed_us: Vec<f64>,
}

/// Distribution summary of one rank plus its share of late arrivals.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RankSkew {
    pub rank: u32,
    pub p50_us: f64,
    pub p90_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
    /// Iterations in which this rank was the (unique, beyond-margin) late
    /// arriver.
    pub slowest_iters: u64,
    /// `slowest_iters / considered_iters`; 0.0 when nothing was considered.
    pub slowest_frac: f64,
}

/// Distribution of the per-iteration total barrier time (the max across
/// ranks — what every rank but the straggler experiences).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FleetBarrier {
    pub p50_us: f64,
    pub p90_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
}

/// Full analysis of one barrier-skew run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BarrierSkew {
    /// Iterations analyzed: the shortest series length across ranks, minus
    /// iterations containing a non-finite value.
    pub iters: u64,
    /// Iterations whose late arriver cleared the margin; the denominator of
    /// every `slowest_frac`.
    pub considered_iters: u64,
    /// One entry per rank, ascending by rank.
    pub per_rank: Vec<RankSkew>,
    pub fleet: FleetBarrier,
}

/// Analyze per-rank barrier timings. Returns `None` when the input cannot
/// carry a skew signal: fewer than two ranks, duplicate rank ids, or no
/// iteration in which every rank has a finite value.
pub fn analyze(
    series: &[RankSeries],
    polarity: SkewPolarity,
    margin: Margin,
) -> Option<BarrierSkew> {
    if series.len() < 2 {
        return None;
    }
    let mut ranks: Vec<u32> = series.iter().map(|s| s.rank).collect();
    ranks.sort_unstable();
    ranks.dedup();
    if ranks.len() != series.len() {
        return None;
    }

    let mut ordered: Vec<&RankSeries> = series.iter().collect();
    ordered.sort_by_key(|s| s.rank);

    let aligned = ordered
        .iter()
        .map(|s| s.elapsed_us.len())
        .min()
        .unwrap_or(0);

    let mut spans: Vec<f64> = Vec::with_capacity(aligned);
    let mut tallies: Vec<u64> = vec![0; ordered.len()];
    let mut considered: u64 = 0;

    for iter in 0..aligned {
        let values: Vec<f64> = ordered.iter().map(|s| s.elapsed_us[iter]).collect();
        if values.iter().any(|value| !value.is_finite()) {
            continue;
        }
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        spans.push(max);

        let late = match polarity {
            SkewPolarity::LateIsMin => unique_extreme(&values, |a, b| a < b),
            SkewPolarity::LateIsMax => unique_extreme(&values, |a, b| a > b),
        };
        let Some((late_index, late_value)) = late else {
            continue;
        };
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        let center = median_of_sorted(&sorted);
        let cutoff = (margin.frac * center.abs()).max(margin.floor_us);
        if (late_value - center).abs() > cutoff {
            considered += 1;
            tallies[late_index] += 1;
        }
    }

    if spans.is_empty() {
        return None;
    }
    let iters = spans.len() as u64;

    let fleet = {
        let mut sorted = spans;
        sorted.sort_by(f64::total_cmp);
        FleetBarrier {
            p50_us: percentile(&sorted, 0.50),
            p90_us: percentile(&sorted, 0.90),
            p99_us: percentile(&sorted, 0.99),
            max_us: sorted[sorted.len() - 1],
        }
    };

    let per_rank = ordered
        .iter()
        .zip(&tallies)
        .map(|(s, &slowest_iters)| {
            let mut sorted: Vec<f64> = s
                .elapsed_us
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect();
            sorted.sort_by(f64::total_cmp);
            RankSkew {
                rank: s.rank,
                p50_us: percentile(&sorted, 0.50),
                p90_us: percentile(&sorted, 0.90),
                p99_us: percentile(&sorted, 0.99),
                max_us: sorted.last().copied().unwrap_or(f64::NAN),
                slowest_iters,
                slowest_frac: if considered == 0 {
                    0.0
                } else {
                    slowest_iters as f64 / considered as f64
                },
            }
        })
        .collect();

    Some(BarrierSkew {
        iters,
        considered_iters: considered,
        per_rank,
        fleet,
    })
}

/// Index and value of the strict extreme under `beats`, or `None` when the
/// extreme is shared (an ambiguous iteration blames nobody).
fn unique_extreme(values: &[f64], beats: fn(f64, f64) -> bool) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    let mut tied = false;
    for (index, &value) in values.iter().enumerate() {
        match best {
            None => best = Some((index, value)),
            Some((_, current)) if beats(value, current) => {
                best = Some((index, value));
                tied = false;
            }
            Some((_, current)) if value == current => tied = true,
            Some(_) => {}
        }
    }
    match best {
        Some(found) if !tied => Some(found),
        _ => None,
    }
}

/// Nearest-rank percentile over an ascending slice (same convention as the
/// peer latency probe): monotone in `fraction`, so p50 <= p90 <= p99 <= max.
fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let last = sorted.len() - 1;
    let index = (fraction * last as f64).ceil() as usize;
    sorted[index.min(last)]
}

fn median_of_sorted(sorted: &[f64]) -> f64 {
    let len = sorted.len();
    if len % 2 == 1 {
        sorted[len / 2]
    } else {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series(rank: u32, values: &[f64]) -> RankSeries {
        RankSeries {
            rank,
            elapsed_us: values.to_vec(),
        }
    }

    /// A fleet where rank 2 always arrives ~200us late: every other rank
    /// waits ~200us, rank 2 completes almost immediately.
    fn straggler_fleet() -> Vec<RankSeries> {
        let iters = 200;
        let mut fleet = Vec::new();
        for rank in 0..4u32 {
            let values: Vec<f64> = (0..iters)
                .map(|i| {
                    let noise = (i % 7) as f64; // deterministic jitter, < margin
                    if rank == 2 {
                        5.0 + noise
                    } else {
                        200.0 + noise
                    }
                })
                .collect();
            fleet.push(series(rank, &values));
        }
        fleet
    }

    #[test]
    fn a_consistent_straggler_wins_nearly_every_tally_under_late_is_min() {
        let skew = analyze(
            &straggler_fleet(),
            SkewPolarity::LateIsMin,
            Margin::default(),
        )
        .expect("analyzable");
        assert_eq!(skew.iters, 200);
        assert_eq!(skew.considered_iters, 200, "every gap clears the margin");
        let rank2 = &skew.per_rank[2];
        assert_eq!(rank2.rank, 2);
        assert_eq!(rank2.slowest_iters, 200);
        assert!((rank2.slowest_frac - 1.0).abs() < 1e-12);
        for other in [0usize, 1, 3] {
            assert_eq!(skew.per_rank[other].slowest_iters, 0);
            assert_eq!(skew.per_rank[other].slowest_frac, 0.0);
        }
    }

    #[test]
    fn late_is_max_blames_the_largest_value_instead() {
        // TCP polarity: offsets, straggler has the biggest.
        let fleet = vec![
            series(0, &[10.0, 11.0, 10.0]),
            series(1, &[12.0, 10.0, 11.0]),
            series(2, &[300.0, 310.0, 305.0]),
        ];
        let skew = analyze(&fleet, SkewPolarity::LateIsMax, Margin::default()).expect("analyzable");
        assert_eq!(skew.per_rank[2].slowest_iters, 3);
        assert_eq!(skew.per_rank[0].slowest_iters, 0);
        assert_eq!(skew.per_rank[1].slowest_iters, 0);
    }

    #[test]
    fn per_rank_percentiles_are_ordered_and_max_is_exact() {
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let fleet = vec![series(0, &values), series(1, &vec![50.0; 100])];
        let skew = analyze(&fleet, SkewPolarity::LateIsMin, Margin::default()).expect("analyzable");
        let rank0 = &skew.per_rank[0];
        assert!(rank0.p50_us <= rank0.p90_us);
        assert!(rank0.p90_us <= rank0.p99_us);
        assert!(rank0.p99_us <= rank0.max_us);
        assert_eq!(rank0.max_us, 100.0);
    }

    #[test]
    fn fleet_distribution_summarizes_the_per_iteration_max() {
        let fleet = vec![
            series(0, &[100.0, 200.0, 300.0]),
            series(1, &[150.0, 120.0, 130.0]),
        ];
        let skew = analyze(&fleet, SkewPolarity::LateIsMin, Margin::default()).expect("analyzable");
        // Per-iteration spans: 150, 200, 300.
        assert_eq!(skew.fleet.p50_us, 200.0);
        assert_eq!(skew.fleet.max_us, 300.0);
        assert!(skew.fleet.p50_us <= skew.fleet.p90_us);
        assert!(skew.fleet.p90_us <= skew.fleet.p99_us);
        assert!(skew.fleet.p99_us <= skew.fleet.max_us);
    }

    #[test]
    fn noise_iterations_fail_the_margin_and_feed_no_tally() {
        // Spread of 2us on a 100us barrier: within both the fractional
        // margin (25us) and the floor (10us).
        let fleet = vec![
            series(0, &[100.0, 101.0, 100.0, 102.0]),
            series(1, &[101.0, 100.0, 102.0, 100.0]),
            series(2, &[102.0, 102.0, 101.0, 101.0]),
        ];
        let skew = analyze(&fleet, SkewPolarity::LateIsMin, Margin::default()).expect("analyzable");
        assert_eq!(skew.iters, 4);
        assert_eq!(skew.considered_iters, 0);
        for rank in &skew.per_rank {
            assert_eq!(rank.slowest_iters, 0);
            assert_eq!(rank.slowest_frac, 0.0);
        }
    }

    #[test]
    fn the_absolute_floor_guards_fast_fabrics() {
        // Median 4us, straggler at 1us: the 25% fractional margin (1us)
        // would tally this, but the 10us floor must not.
        let fleet = vec![
            series(0, &[4.0]),
            series(1, &[4.0]),
            series(2, &[1.0]),
            series(3, &[5.0]),
        ];
        let skew = analyze(&fleet, SkewPolarity::LateIsMin, Margin::default()).expect("analyzable");
        assert_eq!(skew.considered_iters, 0);

        // Loosening the floor tallies it.
        let loose = Margin {
            frac: 0.25,
            floor_us: 0.5,
        };
        let skew = analyze(&fleet, SkewPolarity::LateIsMin, loose).expect("analyzable");
        assert_eq!(skew.considered_iters, 1);
        assert_eq!(skew.per_rank[2].slowest_iters, 1);
    }

    #[test]
    fn tied_extremes_blame_nobody() {
        let fleet = vec![series(0, &[5.0]), series(1, &[5.0]), series(2, &[500.0])];
        let skew = analyze(&fleet, SkewPolarity::LateIsMin, Margin::default()).expect("analyzable");
        assert_eq!(skew.considered_iters, 0);
        assert!(skew.per_rank.iter().all(|rank| rank.slowest_iters == 0));
    }

    #[test]
    fn unequal_series_lengths_truncate_to_the_shortest() {
        let fleet = vec![
            series(0, &[10.0, 10.0, 10.0, 10.0]),
            series(1, &[500.0, 500.0]),
        ];
        let skew = analyze(&fleet, SkewPolarity::LateIsMin, Margin::default()).expect("analyzable");
        assert_eq!(skew.iters, 2);
        assert_eq!(skew.per_rank[0].slowest_iters, 2);
        // Per-rank percentiles still use the rank's full series.
        assert_eq!(skew.per_rank[0].max_us, 10.0);
    }

    #[test]
    fn non_finite_iterations_are_dropped_from_cross_rank_analysis() {
        let fleet = vec![
            series(0, &[10.0, f64::NAN, 10.0]),
            series(1, &[500.0, 500.0, 500.0]),
        ];
        let skew = analyze(&fleet, SkewPolarity::LateIsMin, Margin::default()).expect("analyzable");
        assert_eq!(skew.iters, 2);
        assert_eq!(skew.per_rank[0].slowest_iters, 2);
    }

    #[test]
    fn degenerate_inputs_yield_none() {
        assert!(analyze(&[], SkewPolarity::LateIsMin, Margin::default()).is_none());
        assert!(
            analyze(
                &[series(0, &[1.0])],
                SkewPolarity::LateIsMin,
                Margin::default()
            )
            .is_none(),
            "one rank has no skew"
        );
        assert!(
            analyze(
                &[series(0, &[1.0]), series(0, &[2.0])],
                SkewPolarity::LateIsMin,
                Margin::default()
            )
            .is_none(),
            "duplicate rank ids are a protocol error"
        );
        assert!(
            analyze(
                &[series(0, &[]), series(1, &[1.0])],
                SkewPolarity::LateIsMin,
                Margin::default()
            )
            .is_none(),
            "no aligned iterations"
        );
        assert!(
            analyze(
                &[series(0, &[f64::NAN]), series(1, &[1.0])],
                SkewPolarity::LateIsMin,
                Margin::default()
            )
            .is_none(),
            "no finite aligned iterations"
        );
    }

    #[test]
    fn rank_order_in_the_output_is_ascending_regardless_of_input_order() {
        let fleet = vec![
            series(3, &[10.0, 10.0]),
            series(1, &[500.0, 500.0]),
            series(2, &[490.0, 490.0]),
        ];
        let skew = analyze(&fleet, SkewPolarity::LateIsMin, Margin::default()).expect("analyzable");
        let ranks: Vec<u32> = skew.per_rank.iter().map(|r| r.rank).collect();
        assert_eq!(ranks, vec![1, 2, 3]);
        assert_eq!(skew.per_rank[2].slowest_iters, 2, "rank 3 is the straggler");
    }

    #[test]
    fn barrier_skew_serializes_and_round_trips() {
        let skew = analyze(
            &straggler_fleet(),
            SkewPolarity::LateIsMin,
            Margin::default(),
        )
        .expect("analyzable");
        let json = serde_json::to_string(&skew).expect("serialize");
        let back: BarrierSkew = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, skew);
    }
}
