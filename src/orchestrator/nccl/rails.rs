//! Per-rail bookkeeping for the NIC-forcing world shapes. Pure: generic
//! over the member type and keyed by host address, so every rule is
//! unit-tested without a session.
//!
//! - **Exclusion** (`Exclusions`): rails run one after another, so a host
//!   that broke one rail would otherwise be dialed into every later rail
//!   — and cost one phase timeout and one more failure report per rail.
//!   A host attributed `Failed` in a rail (a primary failure, or a timeout
//!   nobody else is to blame for) is excluded from every later rail and
//!   from the barrier-only job; each later rail is planned (`plan_rail`)
//!   and re-gated with the reduced membership.
//! - **Summary** (`RailLedger`): every host gets at most one outcome per
//!   test for the whole per-rail sweep, not one per rail: Passed when any
//!   rail ran cleanly for it, otherwise Failed (it was to blame in some
//!   rail) or Skipped, with every rail that did not run for it named in
//!   the reason, consecutive rails with the same fate collapsed
//!   ("rails 4-7: skipped: fewer than 2 eligible hosts"). A Passed host's
//!   skipped rails are returned for the driver to log (`TestOutcome::Passed`
//!   carries no reason); the per-rail `_rail<r>` metrics show which rails
//!   produced numbers.

use std::collections::BTreeMap;

use super::attribution::Attribution;
use super::layout::LayoutError;
use super::records::OutcomeRecord;
use super::shape::{ShapedWorld, world_of};
use crate::proto::{Scope, SweepSeries, TestOutcome};

/// `(member, gpus)` still in, and the excluded members with their reason.
pub(super) type Split<M> = (Vec<(M, u32)>, Vec<(M, String)>);

/// Hosts excluded from the rest of a multi-world sweep, with why.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Exclusions {
    by_host: BTreeMap<String, String>,
}

impl Exclusions {
    /// Record one world's attributed failures: every host `Failed` there
    /// is excluded from here on (the first reason wins). Hosts merely
    /// aborted by someone else's failure stay in.
    pub(super) fn record(&mut self, series: SweepSeries, failures: &BTreeMap<String, Attribution>) {
        for (host, attribution) in failures {
            if let Attribution::Failed { reason } = attribution {
                self.by_host
                    .entry(host.clone())
                    .or_insert_with(|| format!("excluded after {series} failed: {reason}"));
            }
        }
    }

    pub(super) fn reason(&self, host: &str) -> Option<&str> {
        self.by_host.get(host).map(String::as_str)
    }

    /// Split `(member, gpus)` into the members still in and the excluded
    /// ones with their reason, order kept.
    pub(super) fn split<M: Clone>(&self, hosts: &[(M, u32)], key: impl Fn(&M) -> &str) -> Split<M> {
        let mut kept = Vec::new();
        let mut excluded = Vec::new();
        for (member, gpus) in hosts {
            match self.reason(key(member)) {
                Some(reason) => excluded.push((member.clone(), reason.to_string())),
                None => kept.push((member.clone(), *gpus)),
            }
        }
        (kept, excluded)
    }
}

/// Rail `rail` of a per-rail sweep, planned against the exclusions so far:
/// the hosts that have a GPU `rail` and are not excluded, one rank each on
/// that GPU, plus the eligible hosts left out and why.
pub(super) struct RailPlan<M> {
    pub world: Result<ShapedWorld<M>, LayoutError>,
    pub excluded: Vec<(M, String)>,
}

pub(super) fn plan_rail<M: Clone>(
    hosts: &[(M, u32)],
    rail: u32,
    exclusions: &Exclusions,
    key: impl Fn(&M) -> &str,
) -> RailPlan<M> {
    let eligible: Vec<(M, u32)> = hosts
        .iter()
        .filter(|(_, gpus)| *gpus > rail)
        .cloned()
        .collect();
    let (kept, excluded) = exclusions.split(&eligible, key);
    RailPlan {
        world: world_of(SweepSeries::Rail { rail }, &kept),
        excluded,
    }
}

/// What one rail was for one of its eligible hosts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RailFate {
    /// The rail ran and nothing was attributed to this host.
    Ran,
    /// This host was to blame (or nobody identifiable was).
    Failed { reason: String },
    /// Aborted by another host's failure.
    Aborted { reason: String },
    /// The gate kept the rail from running (fewer than 2 hosts left).
    Gated,
    /// Excluded after an earlier rail's failure.
    Excluded { reason: String },
    /// The rail could not be laid out.
    NoLayout { reason: String },
}

impl RailFate {
    /// The fate of a host in a rail that ran, from the driver's
    /// attribution map.
    pub(super) fn of(failures: &BTreeMap<String, Attribution>, host: &str) -> Self {
        match failures.get(host) {
            None => RailFate::Ran,
            Some(Attribution::Failed { reason }) => RailFate::Failed {
                reason: reason.clone(),
            },
            Some(Attribution::Skipped { reason }) => RailFate::Aborted {
                reason: reason.clone(),
            },
        }
    }

    fn label(&self) -> String {
        match self {
            RailFate::Ran => "ran".to_string(),
            RailFate::Failed { reason } => format!("failed: {reason}"),
            RailFate::Aborted { reason } => format!("aborted: {reason}"),
            RailFate::Gated => "skipped: fewer than 2 eligible hosts".to_string(),
            RailFate::Excluded { reason } => format!("skipped: {reason}"),
            RailFate::NoLayout { reason } => format!("skipped: cannot lay out the rail: {reason}"),
        }
    }
}

/// One host's summary of a per-rail sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HostRailSummary {
    pub host: String,
    /// One per test of the inter-node series.
    pub outcomes: Vec<OutcomeRecord>,
    /// The rails that did not run cleanly for a Passed host, for the
    /// driver's log (`None` when every rail ran, or when the outcome
    /// itself carries them as its reason).
    pub not_run: Option<String>,
}

/// Every eligible host's fate in every rail of a per-rail sweep.
#[derive(Debug, Clone, Default)]
pub(super) struct RailLedger {
    per_host: BTreeMap<String, Vec<(u32, RailFate)>>,
}

impl RailLedger {
    pub(super) fn note(&mut self, host: &str, rail: u32, fate: RailFate) {
        self.per_host
            .entry(host.to_string())
            .or_default()
            .push((rail, fate));
    }

    /// At most one outcome per host per test.
    pub(super) fn summaries(&self) -> Vec<HostRailSummary> {
        let tests = SweepSeries::inter_node_tests();
        self.per_host
            .iter()
            .map(|(host, fates)| {
                let not_run = describe_not_run(fates);
                let ran = fates.iter().any(|(_, fate)| *fate == RailFate::Ran);
                let failed = fates
                    .iter()
                    .any(|(_, fate)| matches!(fate, RailFate::Failed { .. }));
                let reason = not_run.clone().unwrap_or_default();
                let outcome = if ran {
                    TestOutcome::Passed
                } else if failed {
                    TestOutcome::Failed { reason }
                } else {
                    TestOutcome::Skipped { reason }
                };
                HostRailSummary {
                    host: host.clone(),
                    outcomes: tests
                        .into_iter()
                        .map(|test| (test, Scope::Node, outcome.clone()))
                        .collect(),
                    not_run: if ran { not_run } else { None },
                }
            })
            .collect()
    }
}

/// "rails 4-7: skipped: …; rail 9: aborted: …" over the rails that did
/// not run cleanly, consecutive rails with the same fate collapsed.
/// `None` when every rail ran.
fn describe_not_run(fates: &[(u32, RailFate)]) -> Option<String> {
    let mut sorted: Vec<&(u32, RailFate)> = fates
        .iter()
        .filter(|(_, fate)| *fate != RailFate::Ran)
        .collect();
    sorted.sort_by_key(|(rail, _)| *rail);
    let mut groups: Vec<(u32, u32, String)> = Vec::new();
    for (rail, fate) in sorted {
        let label = fate.label();
        match groups.last_mut() {
            Some((_, last, previous)) if *last + 1 == *rail && *previous == label => {
                *last = *rail;
            }
            _ => groups.push((*rail, *rail, label)),
        }
    }
    (!groups.is_empty()).then(|| {
        groups
            .into_iter()
            .map(|(first, last, label)| {
                if first == last {
                    format!("rail {first}: {label}")
                } else {
                    format!("rails {first}-{last}: {label}")
                }
            })
            .collect::<Vec<_>>()
            .join("; ")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::TestId;

    fn key<'a>(host: &'a &'static str) -> &'a str {
        host
    }

    fn failed(reason: &str) -> Attribution {
        Attribution::Failed {
            reason: reason.to_string(),
        }
    }

    fn members(plan: &RailPlan<&'static str>) -> Vec<(&'static str, u32)> {
        plan.world
            .as_ref()
            .expect("layout")
            .layout
            .members()
            .iter()
            .map(|(host, assignment)| (*host, assignment.block().first_gpu()))
            .collect()
    }

    #[test]
    fn a_host_that_fails_rail_0_is_excluded_from_later_rails_and_the_barrier() {
        let hosts = [("a", 4), ("b", 4), ("c", 4)];
        let mut exclusions = Exclusions::default();
        let rail0 = plan_rail(&hosts, 0, &exclusions, key);
        assert_eq!(members(&rail0), [("a", 0), ("b", 0), ("c", 0)]);

        // b broke rail 0; a and c were aborted by it.
        let failures = BTreeMap::from([
            ("b".to_string(), failed("nccl ranks 1..2 exited with 1")),
            (
                "a".to_string(),
                Attribution::Skipped {
                    reason: "aborted by b".into(),
                },
            ),
        ]);
        exclusions.record(SweepSeries::Rail { rail: 0 }, &failures);

        for rail in 1..4 {
            let plan = plan_rail(&hosts, rail, &exclusions, key);
            assert_eq!(members(&plan), [("a", rail), ("c", rail)], "rail {rail}");
            assert_eq!(plan.excluded.len(), 1);
            let (host, reason) = &plan.excluded[0];
            assert_eq!(*host, "b");
            assert!(reason.contains("rail 0 failed"), "{reason}");
            assert!(reason.contains("exited with 1"), "{reason}");
        }
        // The barrier-only job uses the same split.
        let (kept, excluded) = exclusions.split(&hosts, key);
        assert_eq!(kept, [("a", 4), ("c", 4)]);
        assert_eq!(excluded[0].0, "b");
        // Aborted hosts are never excluded.
        assert_eq!(exclusions.reason("a"), None);
    }

    #[test]
    fn the_first_failure_names_the_exclusion() {
        let mut exclusions = Exclusions::default();
        assert_eq!(exclusions, Exclusions::default());
        exclusions.record(
            SweepSeries::Rail { rail: 0 },
            &BTreeMap::from([("b".to_string(), failed("first"))]),
        );
        exclusions.record(
            SweepSeries::Rail { rail: 1 },
            &BTreeMap::from([("b".to_string(), failed("second"))]),
        );
        let reason = exclusions.reason("b").expect("excluded");
        assert!(
            reason.contains("rail 0") && reason.contains("first"),
            "{reason}"
        );
    }

    #[test]
    fn an_exclusion_can_leave_a_rail_with_one_host() {
        let hosts = [("a", 2), ("b", 2)];
        let mut exclusions = Exclusions::default();
        exclusions.record(
            SweepSeries::Rail { rail: 0 },
            &BTreeMap::from([("a".to_string(), failed("boom"))]),
        );
        let plan = plan_rail(&hosts, 1, &exclusions, key);
        let layout = plan.world.expect("layout").layout;
        assert_eq!(layout.member_count(), 1, "re-gated by the caller");
    }

    fn outcome(summary: &HostRailSummary) -> &TestOutcome {
        let tests: Vec<TestId> = summary.outcomes.iter().map(|(t, _, _)| *t).collect();
        assert_eq!(
            tests,
            [TestId::NcclInterAllReduce, TestId::NcclInterAllGather]
        );
        assert!(summary.outcomes.iter().all(|(_, s, _)| *s == Scope::Node));
        &summary.outcomes[0].2
    }

    #[test]
    fn a_host_that_ran_some_rails_passes_once_with_the_rest_named() {
        let mut ledger = RailLedger::default();
        for rail in 0..4 {
            ledger.note("big", rail, RailFate::Ran);
        }
        for rail in 4..8 {
            ledger.note("big", rail, RailFate::Gated);
        }
        let summaries = ledger.summaries();
        assert_eq!(summaries.len(), 1, "one summary per host");
        assert_eq!(*outcome(&summaries[0]), TestOutcome::Passed);
        assert_eq!(
            summaries[0].not_run.as_deref(),
            Some("rails 4-7: skipped: fewer than 2 eligible hosts")
        );
    }

    #[test]
    fn a_host_no_rail_ran_for_is_skipped_or_failed_with_every_rail_named() {
        let mut ledger = RailLedger::default();
        ledger.note(
            "b",
            0,
            RailFate::Failed {
                reason: "exited with 1".into(),
            },
        );
        for rail in 1..3 {
            ledger.note(
                "b",
                rail,
                RailFate::Excluded {
                    reason: "excluded after rail 0 failed: exited with 1".into(),
                },
            );
        }
        ledger.note(
            "c",
            0,
            RailFate::Aborted {
                reason: "aborted by b".into(),
            },
        );
        ledger.note("c", 1, RailFate::Gated);
        let summaries = ledger.summaries();
        let b = &summaries[0];
        let TestOutcome::Failed { reason } = outcome(b) else {
            panic!("b was to blame: {b:?}");
        };
        assert_eq!(
            reason,
            "rail 0: failed: exited with 1; rails 1-2: skipped: excluded after rail 0 failed: exited with 1"
        );
        assert_eq!(b.not_run, None, "the reason carries it");
        let TestOutcome::Skipped { reason } = outcome(&summaries[1]) else {
            panic!("c only aborted: {:?}", summaries[1]);
        };
        assert_eq!(
            reason,
            "rail 0: aborted: aborted by b; rail 1: skipped: fewer than 2 eligible hosts"
        );
    }

    #[test]
    fn every_rail_ran_means_nothing_to_report() {
        let mut ledger = RailLedger::default();
        ledger.note("a", 1, RailFate::Ran);
        ledger.note("a", 0, RailFate::Ran);
        let summaries = ledger.summaries();
        assert_eq!(*outcome(&summaries[0]), TestOutcome::Passed);
        assert_eq!(summaries[0].not_run, None);
    }

    #[test]
    fn fates_follow_the_attribution() {
        let failures = BTreeMap::from([
            ("a".to_string(), failed("boom")),
            (
                "b".to_string(),
                Attribution::Skipped {
                    reason: "aborted".into(),
                },
            ),
        ]);
        assert_eq!(
            RailFate::of(&failures, "a"),
            RailFate::Failed {
                reason: "boom".into()
            }
        );
        assert_eq!(
            RailFate::of(&failures, "b"),
            RailFate::Aborted {
                reason: "aborted".into()
            }
        );
        assert_eq!(RailFate::of(&failures, "c"), RailFate::Ran);
    }
}
