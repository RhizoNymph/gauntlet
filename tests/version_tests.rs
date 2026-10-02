//! `gauntlet --version` and the build info recorded in documents.

use std::collections::BTreeMap;

use clap::Parser;
use gauntlet::build_info::{BuildInfo, GitRevision, VERSION_LINE};
use gauntlet::cli::Cli;
use gauntlet::config::FleetConfig;
use gauntlet::orchestrator::bootstrap::{BOOTSTRAP_SCHEMA_VERSION, BootstrapReport};
use gauntlet::orchestrator::collect::HostObservations;
use gauntlet::report::{self, RunResults};

#[test]
fn version_flag_prints_crate_version_and_revision() {
    let error = Cli::try_parse_from(["gauntlet", "--version"]).expect_err("version exits");
    assert_eq!(error.kind(), clap::error::ErrorKind::DisplayVersion);
    let text = error.to_string();
    assert!(text.contains(env!("CARGO_PKG_VERSION")), "{text}");
    assert!(text.contains(VERSION_LINE), "{text}");
    let revision = BuildInfo::current().git.to_string();
    assert!(text.contains(&format!("({revision})")), "{text}");
}

#[test]
fn this_checkout_builds_with_a_known_revision() {
    // The test suite runs from a git checkout (`.git` is a directory, or a
    // file in a worktree); a tarball build says "unknown", which
    // GitRevision::parse's unit tests cover.
    if std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".git")
        .exists()
    {
        assert!(
            matches!(BuildInfo::current().git, GitRevision::Known { .. }),
            "{:?}",
            BuildInfo::current()
        );
    }
}

#[test]
fn run_results_record_the_producing_binary() {
    let config: FleetConfig = toml::from_str(r#"hosts = ["n1"]"#).expect("config");
    let observations: BTreeMap<String, HostObservations> =
        [("n1".to_string(), HostObservations::default())].into();
    let results = report::build(&config, observations, 1, 2);
    assert_eq!(results.gauntlet_version, Some(BuildInfo::current()));

    let mut json = serde_json::to_value(&results).expect("serialize");
    assert_eq!(
        json["gauntlet_version"]["version"],
        env!("CARGO_PKG_VERSION")
    );
    // Older documents decode with it absent.
    json.as_object_mut()
        .expect("object")
        .remove("gauntlet_version");
    let back: RunResults = serde_json::from_value(json).expect("decode");
    assert_eq!(back.gauntlet_version, None);
}

#[test]
fn bootstrap_report_records_the_producing_binary() {
    const { assert!(BOOTSTRAP_SCHEMA_VERSION == 2) };
    let report = BootstrapReport {
        schema_version: BOOTSTRAP_SCHEMA_VERSION,
        gauntlet_version: Some(BuildInfo::current()),
        finished_epoch_secs: 1,
        hosts: Vec::new(),
    };
    let json = serde_json::to_value(&report).expect("serialize");
    assert_eq!(
        json["gauntlet_version"]["version"],
        env!("CARGO_PKG_VERSION")
    );
    let mut old = json.clone();
    old.as_object_mut()
        .expect("object")
        .remove("gauntlet_version");
    let back: BootstrapReport = serde_json::from_value(old).expect("decode v1");
    assert_eq!(back.gauntlet_version, None);
}
