//! Non-negativity constraints on the alpha-beta fit (alpha >= 0, beta >= 0).

use gauntlet::analysis::fit::{AlphaBetaFit, FitBound, FitError, fit_alpha_beta};

/// Real fleet run (3 nodes, rank-per-GPU world): host 10.1.0.67's
/// `nccl_all_reduce` sweep, `msg_bytes` joined with `elapsed_us` by emission
/// order. Plain OLS over it gave alpha = -46.59 us: the large sizes dominate
/// the line and the latency-bound small sizes sit above it.
const REAL_ALLREDUCE_SWEEP: [(u64, f64); 10] = [
    (1_024, 4335.8636),
    (4_096, 215.2416),
    (16_384, 318.984_799_999_999_95),
    (65_536, 434.0656),
    (262_144, 635.3548),
    (1_048_576, 1364.9492),
    (4_194_304, 6854.3758),
    (16_777_216, 26436.7038),
    (67_108_864, 119_698.324),
    (268_435_456, 484_280.647_2),
];

fn rss(points: &[(u64, f64)], alpha: f64, beta: f64) -> f64 {
    points
        .iter()
        .map(|&(x, y)| (y - alpha - beta * x as f64).powi(2))
        .sum()
}

/// Unconstrained OLS, computed independently of the library.
fn ols(points: &[(u64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.0 as f64).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = points.iter().map(|p| (p.0 as f64 - mx).powi(2)).sum();
    let sxy: f64 = points.iter().map(|p| (p.0 as f64 - mx) * (p.1 - my)).sum();
    let beta = sxy / sxx;
    (my - beta * mx, beta)
}

/// Every feasible candidate the constrained problem can land on: the OLS
/// point (when feasible), the best fit through the origin, the best
/// horizontal line, and the origin itself.
fn feasible_candidates(points: &[(u64, f64)]) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0, 0.0)];
    let (a, b) = ols(points);
    if a >= 0.0 && b >= 0.0 {
        out.push((a, b));
    }
    let sxy0: f64 = points.iter().map(|p| p.0 as f64 * p.1).sum();
    let sxx0: f64 = points.iter().map(|p| (p.0 as f64).powi(2)).sum();
    out.push((0.0, (sxy0 / sxx0).max(0.0)));
    let my = points.iter().map(|p| p.1).sum::<f64>() / points.len() as f64;
    out.push((my.max(0.0), 0.0));
    out
}

fn assert_feasible(fit: &AlphaBetaFit) {
    assert!(fit.alpha_us >= 0.0, "{fit:?}");
    assert!(fit.beta_us_per_byte >= 0.0, "{fit:?}");
    assert!((0.0..=1.0).contains(&fit.r_squared), "{fit:?}");
}

#[test]
fn real_sweep_is_clamped_to_non_negative_alpha() {
    let (ols_alpha, _) = ols(&REAL_ALLREDUCE_SWEEP);
    assert!(ols_alpha < 0.0, "fixture no longer exercises the bound");

    let fit = fit_alpha_beta(&REAL_ALLREDUCE_SWEEP).expect("fit");
    assert_feasible(&fit);
    assert_eq!(fit.alpha_us, 0.0, "{fit:?}");
    assert_eq!(fit.bound, Some(FitBound::AlphaZero), "{fit:?}");
    // Through-origin slope: sum(xy) / sum(x^2).
    assert!(
        (fit.beta_us_per_byte - 1.802_006_920_472_471e-3).abs() < 1e-12,
        "{fit:?}"
    );
    // Still an excellent line, just not quite the OLS r^2.
    assert!(fit.r_squared > 0.9998 && fit.r_squared < 0.999_826_846_754_292_1);
}

#[test]
fn clean_linear_data_matches_ols_and_is_unbound() {
    let alpha = 12.5;
    let beta = 2.5e-4;
    let points: Vec<(u64, f64)> = [512u64, 8_192, 131_072, 2_097_152, 33_554_432]
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            // Small deterministic wobble so this is a real regression.
            let wobble = [0.3, -0.2, 0.1, -0.4, 0.2][i];
            (x, alpha + beta * x as f64 + wobble)
        })
        .collect();
    let (ols_alpha, ols_beta) = ols(&points);
    assert!(ols_alpha > 0.0 && ols_beta > 0.0);

    let fit = fit_alpha_beta(&points).expect("fit");
    assert_eq!(fit.bound, None, "{fit:?}");
    assert!((fit.alpha_us - ols_alpha).abs() < 1e-9, "{fit:?}");
    assert!(
        (fit.beta_us_per_byte - ols_beta).abs() / ols_beta < 1e-12,
        "{fit:?}"
    );
    assert!(fit.r_squared > 0.999_999, "{fit:?}");
}

#[test]
fn decreasing_timings_flag_beta_zero() {
    // Larger messages finishing faster: OLS slope is negative.
    let points = [(1_000u64, 100.0), (2_000, 90.0), (3_000, 80.0)];
    assert!(ols(&points).1 < 0.0);

    let fit = fit_alpha_beta(&points).expect("fit");
    assert_feasible(&fit);
    assert_eq!(fit.bound, Some(FitBound::BetaZero), "{fit:?}");
    assert_eq!(fit.beta_us_per_byte, 0.0);
    assert!((fit.alpha_us - 90.0).abs() < 1e-9, "{fit:?}");
    // A horizontal line at the mean explains none of the variance.
    assert_eq!(fit.r_squared, 0.0);
    assert!(fit.bandwidth_gib_per_sec().is_infinite());
}

#[test]
fn non_positive_timings_pin_both_parameters() {
    // Garbage in (negative "times"): both faces clamp to the origin.
    let points = [(1_000u64, -5.0), (2_000, -6.0), (4_000, -7.0)];
    let fit = fit_alpha_beta(&points).expect("fit");
    assert_feasible(&fit);
    assert_eq!(fit.alpha_us, 0.0);
    assert_eq!(fit.beta_us_per_byte, 0.0);
    assert_eq!(fit.bound, Some(FitBound::BothZero), "{fit:?}");
}

#[test]
fn constrained_fit_has_minimal_rss_among_feasible_candidates() {
    let datasets: Vec<Vec<(u64, f64)>> = vec![
        REAL_ALLREDUCE_SWEEP.to_vec(),
        vec![(1_000, 100.0), (2_000, 90.0), (3_000, 80.0)],
        vec![(1_024, 90.0), (2_048, 10.0), (4_096, 80.0), (8_192, 20.0)],
        vec![(10, 1.0), (20, 50.0), (30, 3.0), (40, 80.0)],
        vec![(100, 5.0), (200, 7.0), (400, 11.0)],
        vec![(1_000, -5.0), (2_000, -6.0), (4_000, -7.0)],
        vec![(1_000, -50.0), (2_000, 10.0), (4_000, 70.0)],
    ];
    for points in &datasets {
        let fit = fit_alpha_beta(points).expect("fit");
        assert_feasible(&fit);
        let chosen = rss(points, fit.alpha_us, fit.beta_us_per_byte);
        for (a, b) in feasible_candidates(points) {
            let other = rss(points, a, b);
            assert!(
                chosen <= other * (1.0 + 1e-12) + 1e-9,
                "{points:?}: chosen {fit:?} rss {chosen} > candidate ({a}, {b}) rss {other}"
            );
        }
    }
}

#[test]
fn constrained_fit_beats_a_feasible_grid() {
    // Brute-force check that the closed form is the true constrained
    // optimum, not just the best of its own candidates.
    let datasets: Vec<Vec<(u64, f64)>> = vec![
        vec![(1_000, 100.0), (2_000, 90.0), (3_000, 80.0)],
        vec![(10, 1.0), (20, 50.0), (30, 3.0), (40, 80.0)],
        vec![(1_000, -50.0), (2_000, 10.0), (4_000, 70.0)],
        vec![(100, 30.0), (200, 5.0), (300, 6.0), (400, 7.0)],
    ];
    for points in &datasets {
        let fit = fit_alpha_beta(points).expect("fit");
        let chosen = rss(points, fit.alpha_us, fit.beta_us_per_byte);
        let max_x = points.iter().map(|p| p.0).max().unwrap() as f64;
        let max_y = points.iter().map(|p| p.1.abs()).fold(0.0, f64::max);
        let steps = 200;
        for i in 0..=steps {
            let a = max_y * 2.0 * f64::from(i) / f64::from(steps);
            for j in 0..=steps {
                let b = max_y * 2.0 / max_x * f64::from(j) / f64::from(steps);
                let other = rss(points, a, b);
                assert!(
                    chosen <= other * (1.0 + 1e-12) + 1e-9,
                    "{points:?}: chosen {fit:?} rss {chosen} > grid ({a}, {b}) rss {other}"
                );
            }
        }
    }
}

#[test]
fn errors_are_unchanged_by_the_constraint() {
    assert!(matches!(
        fit_alpha_beta(&[(1024, -10.0), (2048, f64::NAN)]),
        Err(FitError::NonFinite)
    ));
    assert!(matches!(
        fit_alpha_beta(&[(1024, f64::NEG_INFINITY), (2048, -20.0)]),
        Err(FitError::NonFinite)
    ));
    match fit_alpha_beta(&[(4096, -1.0), (4096, -2.0)]) {
        Err(FitError::TooFewPoints { got }) => assert_eq!(got, 1),
        other => panic!("expected TooFewPoints, got {other:?}"),
    }
    assert!(matches!(
        fit_alpha_beta(&[]),
        Err(FitError::TooFewPoints { got: 0 })
    ));
}

#[test]
fn bound_round_trips_and_defaults_when_absent() {
    let fit = fit_alpha_beta(&REAL_ALLREDUCE_SWEEP).expect("fit");
    let json = serde_json::to_string(&fit).expect("encode");
    assert!(json.contains(r#""bound":"alpha_zero""#), "{json}");
    let back: AlphaBetaFit = serde_json::from_str(&json).expect("decode");
    // serde_json's default float parser may be off by an ulp.
    assert_eq!(back.bound, fit.bound);
    assert_eq!(back.alpha_us, fit.alpha_us);
    assert!((back.beta_us_per_byte - fit.beta_us_per_byte).abs() < 1e-18);
    assert!((back.r_squared - fit.r_squared).abs() < 1e-15);

    // A pre-v11 document has no `bound` field: it decodes as unbound.
    let old = r#"{"alpha_us":-46.5,"beta_us_per_byte":0.0018,"r_squared":0.99}"#;
    let back: AlphaBetaFit = serde_json::from_str(old).expect("decode old");
    assert_eq!(back.bound, None);
    assert_eq!(back.alpha_us, -46.5);
}
