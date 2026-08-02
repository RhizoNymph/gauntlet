//! Tests for the bootstrap-matrix projection and stalled-run detection.

use std::time::{Duration, SystemTime};

use gauntlet::orchestrator::bootstrap::{CheckStatus, HostReadiness, ReadinessCheck};
use gauntlet_view::bootstrap::{check_columns, status_of, worst_of};
use gauntlet_view::runs::is_stalled;

fn check(name: &str, status: CheckStatus) -> ReadinessCheck {
    ReadinessCheck {
        name: name.into(),
        status,
        detail: format!("{name} detail"),
    }
}

fn host(addr: &str, checks: Vec<ReadinessCheck>) -> HostReadiness {
    HostReadiness {
        host: addr.into(),
        checks,
        inventory: None,
    }
}

#[test]
fn columns_are_the_union_in_first_seen_order() {
    let hosts = vec![
        // A host that failed early only has the first checks.
        host("a", vec![check("connectivity", CheckStatus::Fail)]),
        host(
            "b",
            vec![
                check("connectivity", CheckStatus::Ok),
                check("arch", CheckStatus::Ok),
                check("deploy", CheckStatus::Ok),
                check("governor", CheckStatus::Warn),
            ],
        ),
        host(
            "c",
            vec![
                check("connectivity", CheckStatus::Ok),
                check("arch", CheckStatus::Ok),
                // GPU-less host has a check the others lack.
                check("clock_sync", CheckStatus::Ok),
            ],
        ),
    ];
    assert_eq!(
        check_columns(&hosts),
        vec!["connectivity", "arch", "deploy", "governor", "clock_sync"]
    );
}

#[test]
fn status_lookup_and_worst_aggregate() {
    let row = host(
        "a",
        vec![
            check("connectivity", CheckStatus::Ok),
            check("governor", CheckStatus::Warn),
        ],
    );
    assert_eq!(status_of(&row, "connectivity"), Some(CheckStatus::Ok));
    assert_eq!(status_of(&row, "governor"), Some(CheckStatus::Warn));
    assert_eq!(status_of(&row, "deploy"), None);
    assert_eq!(worst_of(&row), CheckStatus::Warn);

    let failed = host("b", vec![check("connectivity", CheckStatus::Fail)]);
    assert_eq!(worst_of(&failed), CheckStatus::Fail);
    assert_eq!(worst_of(&host("c", vec![])), CheckStatus::Ok);
}

#[test]
fn stalled_means_a_live_file_stopped_updating() {
    let now = SystemTime::now();
    let fresh = now - Duration::from_secs(3);
    let old = now - Duration::from_secs(60);
    assert!(!is_stalled(true, Some(fresh), now));
    assert!(is_stalled(true, Some(old), now));
    // Finished runs and unknown mtimes are never stalled.
    assert!(!is_stalled(false, Some(old), now));
    assert!(!is_stalled(true, None, now));
}
