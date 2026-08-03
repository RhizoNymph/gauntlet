use gauntlet::analysis::stats::{Sample, flag_outliers, mad, median};

fn samples(values: &[(&str, f64)]) -> Vec<Sample> {
    values
        .iter()
        .map(|(key, value)| Sample {
            key: (*key).into(),
            value: *value,
        })
        .collect()
}

#[test]
fn median_basics() {
    assert_eq!(median(&[]), None);
    assert_eq!(median(&[f64::NAN]), None);
    assert_eq!(median(&[3.0]), Some(3.0));
    assert_eq!(median(&[5.0, 1.0, 3.0]), Some(3.0));
    assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
    // Non-finite values are ignored, not propagated.
    assert_eq!(median(&[1.0, f64::INFINITY, 3.0, f64::NAN, 2.0]), Some(2.0));
}

#[test]
fn mad_is_scaled_and_needs_spread_data() {
    assert_eq!(mad(&[]), None);
    assert_eq!(mad(&[1.0]), None);
    assert_eq!(mad(&[1.0, f64::NAN]), None);
    // [1..5]: median 3, abs devs [2,1,0,1,2], raw MAD 1 -> scaled 1.4826.
    let value = mad(&[1.0, 2.0, 3.0, 4.0, 5.0]).expect("mad");
    assert!((value - 1.4826).abs() < 1e-9, "got {value}");
    // Identical values: MAD 0 (Some, but zero — degeneracy handled by caller).
    assert_eq!(mad(&[7.0, 7.0, 7.0]), Some(0.0));
}

#[test]
fn straggler_is_flagged_with_signed_deviation() {
    let fleet = samples(&[
        ("n1", 98.0),
        ("n2", 99.0),
        ("n3", 100.0),
        ("n4", 100.0),
        ("n5", 101.0),
        ("n6", 102.0),
        ("n7", 97.0),
        ("n8", 99.5),
        ("n9", 100.5),
        ("bad", 50.0),
    ]);
    let outliers = flag_outliers(&fleet, 4.0);
    assert_eq!(outliers.len(), 1, "only the straggler: {outliers:?}");
    let bad = &outliers[0];
    assert_eq!(bad.key, "bad");
    assert_eq!(bad.value, 50.0);
    assert!(
        (bad.fleet_median - 99.75).abs() < 1.0,
        "median ~99.75, got {}",
        bad.fleet_median
    );
    assert!(
        bad.deviation_mads < -4.0,
        "below-median straggler must have negative deviation, got {}",
        bad.deviation_mads
    );
}

#[test]
fn healthy_fleet_flags_nothing() {
    let fleet = samples(&[
        ("n1", 99.0),
        ("n2", 100.0),
        ("n3", 101.0),
        ("n4", 100.5),
        ("n5", 99.5),
    ]);
    assert!(flag_outliers(&fleet, 4.0).is_empty());
}

#[test]
fn degenerate_spread_flags_nothing() {
    // MAD == 0: identical values must not divide-by-zero into flagging.
    let fleet = samples(&[
        ("n1", 5.0),
        ("n2", 5.0),
        ("n3", 5.0),
        ("n4", 5.0),
        ("n5", 5.1),
    ]);
    assert!(flag_outliers(&fleet, 4.0).is_empty());
}

#[test]
fn tiny_fleets_flag_nothing() {
    let fleet = samples(&[("n1", 1.0), ("n2", 100.0), ("n3", 1.0)]);
    assert!(
        flag_outliers(&fleet, 4.0).is_empty(),
        "fewer than 4 samples has no meaningful median"
    );
}

#[test]
fn k_controls_sensitivity() {
    let fleet = samples(&[
        ("n1", 100.0),
        ("n2", 101.0),
        ("n3", 99.0),
        ("n4", 100.5),
        ("n5", 99.5),
        ("meh", 90.0),
    ]);
    let strict = flag_outliers(&fleet, 2.0);
    let lax = flag_outliers(&fleet, 50.0);
    assert!(strict.iter().any(|o| o.key == "meh"));
    assert!(lax.is_empty());
}

// ---------------------------------------------------------------------------
// Moments (per-subject distribution summaries for --repeat runs)
// ---------------------------------------------------------------------------

use gauntlet::analysis::stats::moments;

#[test]
fn moments_summarize_a_small_sample() {
    let m = moments(&[10.0, 12.0, 11.0, 13.0, 9.0]).expect("moments");
    assert_eq!(m.n, 5);
    assert_eq!(m.median, 11.0);
    assert_eq!(m.min, 9.0);
    assert_eq!(m.max, 13.0);
    assert!((m.mean - 11.0).abs() < 1e-9);
    // MAD: deviations from 11 are [1,1,0,2,2] -> median 1 -> * 1.4826.
    assert!((m.mad - 1.4826).abs() < 1e-9);
    // Sample stddev of [9..13] around 11: sqrt(10/4).
    assert!((m.stddev - (10.0f64 / 4.0).sqrt()).abs() < 1e-9);
}

#[test]
fn single_sample_moments_have_zero_spread() {
    let m = moments(&[42.0]).expect("moments");
    assert_eq!(m.n, 1);
    assert_eq!(m.median, 42.0);
    assert_eq!(m.mean, 42.0);
    assert_eq!(m.min, 42.0);
    assert_eq!(m.max, 42.0);
    assert_eq!(m.mad, 0.0);
    assert_eq!(m.stddev, 0.0);
}

#[test]
fn moments_ignore_non_finite_values_and_need_at_least_one() {
    let m = moments(&[f64::NAN, 5.0, f64::INFINITY, 7.0]).expect("moments");
    assert_eq!(m.n, 2);
    assert_eq!(m.median, 6.0);
    assert!(moments(&[]).is_none());
    assert!(moments(&[f64::NAN]).is_none());
}
