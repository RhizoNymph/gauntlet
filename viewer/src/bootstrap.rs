//! Pure projection of a `BootstrapReport` into the readiness matrix the UI
//! draws: column ordering, cell lookup, and per-host aggregation.

use gauntlet::orchestrator::bootstrap::{CheckStatus, HostReadiness};

/// Check names as matrix columns: the union across hosts, in first-seen
/// order (hosts run the same sequence, but a host that fails early or has
/// no GPU is missing later checks).
pub fn check_columns(hosts: &[HostReadiness]) -> Vec<String> {
    let mut columns: Vec<String> = Vec::new();
    for host in hosts {
        for check in &host.checks {
            if !columns.contains(&check.name) {
                columns.push(check.name.clone());
            }
        }
    }
    columns
}

/// The status of one cell, `None` when the host never ran that check.
pub fn status_of(host: &HostReadiness, check_name: &str) -> Option<CheckStatus> {
    host.checks
        .iter()
        .find(|check| check.name == check_name)
        .map(|check| check.status)
}

/// Worst status across a host's checks; an empty row is `Ok`.
pub fn worst_of(host: &HostReadiness) -> CheckStatus {
    let mut worst = CheckStatus::Ok;
    for check in &host.checks {
        worst = match (worst, check.status) {
            (_, CheckStatus::Fail) | (CheckStatus::Fail, _) => CheckStatus::Fail,
            (_, CheckStatus::Warn) | (CheckStatus::Warn, _) => CheckStatus::Warn,
            _ => CheckStatus::Ok,
        };
    }
    worst
}
