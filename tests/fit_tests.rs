use gauntlet::analysis::fit::{AlphaBetaFit, FitError, fit_alpha_beta};

#[test]
fn recovers_exact_line() {
    // alpha = 100us, beta = 0.001 us/byte (~0.93 GiB/s).
    let alpha = 100.0;
    let beta = 0.001;
    let points: Vec<(u64, f64)> = [1_000u64, 10_000, 100_000, 1_000_000, 10_000_000]
        .iter()
        .map(|&bytes| (bytes, alpha + beta * bytes as f64))
        .collect();
    let fit = fit_alpha_beta(&points).expect("fit");
    assert!((fit.alpha_us - alpha).abs() / alpha < 1e-6, "{fit:?}");
    assert!((fit.beta_us_per_byte - beta).abs() / beta < 1e-6, "{fit:?}");
    assert!(fit.r_squared > 0.999999, "{fit:?}");
}

#[test]
fn tolerates_noise_and_repeated_sizes() {
    let alpha = 50.0;
    let beta = 0.0005;
    // 3 noisy samples per size, deterministic +/-2% pattern.
    let noise = [0.98, 1.0, 1.02];
    let mut points = Vec::new();
    for &bytes in &[4_096u64, 65_536, 1_048_576, 16_777_216] {
        for (i, factor) in noise.iter().enumerate() {
            let _ = i;
            points.push((bytes, (alpha + beta * bytes as f64) * factor));
        }
    }
    let fit = fit_alpha_beta(&points).expect("fit");
    assert!((fit.alpha_us - alpha).abs() / alpha < 0.15, "{fit:?}");
    assert!((fit.beta_us_per_byte - beta).abs() / beta < 0.1, "{fit:?}");
    assert!(fit.r_squared > 0.99, "{fit:?}");
    assert!((0.0..=1.0).contains(&fit.r_squared), "{fit:?}");
}

#[test]
fn needs_two_distinct_sizes() {
    assert!(matches!(
        fit_alpha_beta(&[]),
        Err(FitError::TooFewPoints { .. })
    ));
    assert!(matches!(
        fit_alpha_beta(&[(1024, 10.0)]),
        Err(FitError::TooFewPoints { .. })
    ));
    // Repeated single size is still one distinct size.
    assert!(matches!(
        fit_alpha_beta(&[(1024, 10.0), (1024, 11.0), (1024, 10.5)]),
        Err(FitError::TooFewPoints { .. })
    ));
}

#[test]
fn rejects_non_finite_timings() {
    assert!(matches!(
        fit_alpha_beta(&[(1024, 10.0), (2048, f64::NAN)]),
        Err(FitError::NonFinite)
    ));
    assert!(matches!(
        fit_alpha_beta(&[(1024, f64::INFINITY), (2048, 20.0)]),
        Err(FitError::NonFinite)
    ));
}

#[test]
fn derived_quantities() {
    let fit = AlphaBetaFit {
        alpha_us: 10.0,
        beta_us_per_byte: 0.001,
        r_squared: 1.0,
    };
    assert!((fit.predict_us(1_000_000) - 1010.0).abs() < 1e-9);
    // 0.001 us/B == 1e9 B/s ~= 0.9313 GiB/s.
    let gib = fit.bandwidth_gib_per_sec();
    assert!((gib - 0.9313).abs() < 0.001, "got {gib}");
}
