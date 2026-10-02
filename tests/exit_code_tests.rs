//! Process exit codes outside the verdict: usage errors and pre-verdict
//! errors must never collide with a verdict code (0-3).

use std::process::Command;

use clap::Parser;
use gauntlet::cli::{Cli, parse_failure_exit_code};
use gauntlet::report::{EXIT_ERROR, Verdict};

fn gauntlet(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_gauntlet"))
        .args(args)
        .output()
        .expect("run gauntlet")
}

#[test]
fn usage_errors_map_to_exit_error_not_a_verdict() {
    for args in [
        vec!["gauntlet", "--bogus"],
        vec!["gauntlet", "run", "--repeat", "0"],
        vec!["gauntlet"],
    ] {
        let error = Cli::try_parse_from(&args).expect_err("usage error");
        assert_eq!(parse_failure_exit_code(&error), EXIT_ERROR, "{args:?}");
    }
    assert_eq!(Verdict::from_exit_code(i32::from(EXIT_ERROR)), None);
}

#[test]
fn help_and_version_exit_zero() {
    for args in [vec!["gauntlet", "--help"], vec!["gauntlet", "--version"]] {
        let error = Cli::try_parse_from(&args).expect_err("display");
        assert_eq!(parse_failure_exit_code(&error), 0, "{args:?}");
    }
}

#[test]
fn the_binary_uses_the_mapping() {
    let usage = gauntlet(&["--bogus"]);
    assert_eq!(usage.status.code(), Some(i32::from(EXIT_ERROR)));
    assert!(!usage.stderr.is_empty());

    let version = gauntlet(&["--version"]);
    assert_eq!(version.status.code(), Some(0));
    assert!(
        String::from_utf8_lossy(&version.stdout).contains(env!("CARGO_PKG_VERSION")),
        "{version:?}"
    );

    assert_eq!(gauntlet(&["--help"]).status.code(), Some(0));

    // A config that cannot be loaded is an error before any verdict.
    let missing = gauntlet(&["run", "--config", "/nonexistent/gauntlet.toml"]);
    assert_eq!(missing.status.code(), Some(i32::from(EXIT_ERROR)));
}
