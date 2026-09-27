//! Who caused a fleet NCCL job to fail. Pure.
//!
//! One dead rank blocks every other rank inside the collective, so a
//! single fault used to surface as a Failed timeout on *every* host (or,
//! when the lead minted no rendezvous id, as "never started" on every
//! follower) — the whole fleet flagged, nothing pointing at the culprit.
//! Each host's failure is therefore classified:
//!
//! - **Primary** — something went wrong on this host: an explicit error,
//!   a nonzero exit, or the lead failing before the id relay.
//! - **Secondary** — a timeout, a "never started", or a cascade exit
//!   (`AGENT_EXIT_CASCADE`: the agent stopped itself because the fleet
//!   stopped) while some other host had a primary failure.
//!
//! Primaries stay Failed; secondaries become Skipped with a reason naming
//! the primary host(s). With no primary at all (everyone timed out) there
//! is nobody to blame, and every failure stays Failed.
//!
//! Early abort ([`AbortTracker`]): the first primary failure means the job
//! is over — every other rank is (or soon will be) blocked in a collective
//! that can never complete — so the driver kills every still-running host
//! right away instead of letting them hang to the phase timeout. Whatever
//! those hosts then report is the driver's doing, classified
//! [`FailureKind::Aborted`] (secondary).

use std::collections::{BTreeMap, BTreeSet};

/// How one host's rank block failed, before attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FailureKind {
    /// An explicit error or nonzero exit on this host, or the lead failing
    /// before the rendezvous id was relayed.
    Primary,
    /// The phase timeout reaped the host.
    TimedOut,
    /// Never launched: the lead produced no rendezvous id.
    NeverStarted,
    /// The agent stopped itself because the fleet stopped (hard-deadline
    /// watchdog or follower failsafe).
    Cascade,
    /// Killed by the driver's early abort after another host's primary
    /// failure.
    Aborted,
}

impl FailureKind {
    fn is_primary(self) -> bool {
        self == FailureKind::Primary
    }
}

/// One host's failure as the driver observed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HostFailure {
    pub kind: FailureKind,
    pub message: String,
}

impl HostFailure {
    pub(super) fn new(kind: FailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

/// What the driver must do after observing a finished host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AbortAction {
    Continue,
    /// A primary failure: kill every host still running.
    AbortRest,
}

/// The "primary seen → abort the rest" decision, as hosts finish one by
/// one. Pure: the driver feeds it each host's outcome and carries out the
/// kills it asks for.
#[derive(Debug, Default)]
pub(super) struct AbortTracker {
    triggered: bool,
    aborted: BTreeSet<String>,
}

impl AbortTracker {
    /// A host finished with `failure` (None = clean exit). Returns the
    /// failure as it must be recorded — reclassified as
    /// [`FailureKind::Aborted`] if the driver killed this host — and
    /// whether the rest of the world must be aborted now (only the first
    /// primary failure triggers that).
    pub(super) fn observe(
        &mut self,
        host: &str,
        failure: Option<HostFailure>,
    ) -> (Option<HostFailure>, AbortAction) {
        let failure = failure.map(|failure| {
            if self.aborted.contains(host) {
                HostFailure::new(
                    FailureKind::Aborted,
                    format!("aborted after another host's failure ({})", failure.message),
                )
            } else {
                failure
            }
        });
        let action = match &failure {
            Some(failure) if failure.kind.is_primary() => self.primary_seen(),
            _ => AbortAction::Continue,
        };
        (failure, action)
    }

    /// A primary failure observed outside any host's completion (e.g. a
    /// directive that could not even be sent). Only the first primary
    /// triggers the abort.
    pub(super) fn primary_seen(&mut self) -> AbortAction {
        if self.triggered {
            AbortAction::Continue
        } else {
            self.triggered = true;
            AbortAction::AbortRest
        }
    }

    /// Record the hosts the driver is killing in response to
    /// [`AbortAction::AbortRest`].
    pub(super) fn mark_aborted(&mut self, hosts: impl IntoIterator<Item = String>) {
        self.aborted.extend(hosts);
    }
}

/// What the results document records for a failed host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Attribution {
    /// This host caused (or, with no identifiable cause, suffered) the
    /// failure.
    Failed { reason: String },
    /// This host was healthy but the job was aborted by another host's
    /// failure.
    Skipped { reason: String },
}

/// Attribute every failure of one job. `step` names the job in secondary
/// reasons ("fleet overlap", "nccl sweep").
pub(super) fn attribute(
    step: &str,
    failures: &BTreeMap<String, HostFailure>,
) -> BTreeMap<String, Attribution> {
    let culprits: Vec<&str> = failures
        .iter()
        .filter(|(_, failure)| failure.kind.is_primary())
        .map(|(host, _)| host.as_str())
        .collect();
    failures
        .iter()
        .map(|(host, failure)| {
            let attribution = if failure.kind.is_primary() || culprits.is_empty() {
                Attribution::Failed {
                    reason: failure.message.clone(),
                }
            } else {
                Attribution::Skipped {
                    reason: format!(
                        "{step} aborted: rank failure on {} ({})",
                        culprits.join(", "),
                        failure.message
                    ),
                }
            };
            (host.clone(), attribution)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failures(entries: &[(&str, FailureKind, &str)]) -> BTreeMap<String, HostFailure> {
        entries
            .iter()
            .map(|(host, kind, message)| (host.to_string(), HostFailure::new(*kind, *message)))
            .collect()
    }

    fn failed(reason: &str) -> Attribution {
        Attribution::Failed {
            reason: reason.to_string(),
        }
    }

    #[test]
    fn a_single_primary_takes_the_blame_for_the_cascade() {
        let attributed = attribute(
            "fleet overlap",
            &failures(&[
                (
                    "10.1.1.68",
                    FailureKind::Primary,
                    "nccl ranks 8..16 exited with 1",
                ),
                ("10.1.1.67", FailureKind::Cascade, "watchdog"),
                ("10.1.1.69", FailureKind::TimedOut, "timed out after 900s"),
            ]),
        );
        assert_eq!(
            attributed["10.1.1.68"],
            failed("nccl ranks 8..16 exited with 1")
        );
        for host in ["10.1.1.67", "10.1.1.69"] {
            let Attribution::Skipped { reason } = &attributed[host] else {
                panic!("{host} should be skipped: {:?}", attributed[host]);
            };
            assert!(
                reason.starts_with("fleet overlap aborted: rank failure on 10.1.1.68"),
                "{reason}"
            );
        }
    }

    #[test]
    fn a_lead_without_an_id_is_blamed_for_every_unstarted_follower() {
        let attributed = attribute(
            "nccl sweep",
            &failures(&[
                (
                    "lead",
                    FailureKind::Primary,
                    "nccl ranks 0..8 exited with 1",
                ),
                ("f1", FailureKind::NeverStarted, "never started"),
                ("f2", FailureKind::NeverStarted, "never started"),
            ]),
        );
        assert_eq!(attributed["lead"], failed("nccl ranks 0..8 exited with 1"));
        for host in ["f1", "f2"] {
            assert!(
                matches!(
                    &attributed[host],
                    Attribution::Skipped { reason }
                        if reason.starts_with("nccl sweep aborted: rank failure on lead")
                ),
                "{:?}",
                attributed[host]
            );
        }
    }

    #[test]
    fn with_no_primary_everyone_stays_failed() {
        let input = failures(&[
            ("a", FailureKind::TimedOut, "a timed out"),
            ("b", FailureKind::TimedOut, "b timed out"),
            ("c", FailureKind::Cascade, "c watchdog"),
        ]);
        let attributed = attribute("fleet overlap", &input);
        assert_eq!(attributed["a"], failed("a timed out"));
        assert_eq!(attributed["b"], failed("b timed out"));
        assert_eq!(attributed["c"], failed("c watchdog"));
    }

    #[test]
    fn multiple_primaries_are_all_failed_and_all_named() {
        let attributed = attribute(
            "fleet overlap",
            &failures(&[
                ("a", FailureKind::Primary, "a crashed"),
                ("b", FailureKind::Primary, "b crashed"),
                ("c", FailureKind::TimedOut, "c timed out"),
            ]),
        );
        assert_eq!(attributed["a"], failed("a crashed"));
        assert_eq!(attributed["b"], failed("b crashed"));
        assert_eq!(
            attributed["c"],
            Attribution::Skipped {
                reason: "fleet overlap aborted: rank failure on a, b (c timed out)".into()
            }
        );
    }

    #[test]
    fn the_first_primary_aborts_the_rest_and_their_fallout_is_secondary() {
        let mut tracker = AbortTracker::default();
        // A clean host changes nothing.
        assert_eq!(tracker.observe("a", None), (None, AbortAction::Continue));
        // The first primary triggers the abort.
        let (failure, action) = tracker.observe(
            "b",
            Some(HostFailure::new(FailureKind::Primary, "b exited with 1")),
        );
        assert_eq!(action, AbortAction::AbortRest);
        assert_eq!(failure.expect("recorded").kind, FailureKind::Primary);
        tracker.mark_aborted(["c".to_string(), "d".to_string()]);
        // A killed host's nonzero exit is the driver's doing, not a fault.
        let (failure, action) = tracker.observe(
            "c",
            Some(HostFailure::new(
                FailureKind::Primary,
                "c exited with signal 15",
            )),
        );
        assert_eq!(action, AbortAction::Continue);
        let failure = failure.expect("recorded");
        assert_eq!(failure.kind, FailureKind::Aborted);
        assert!(failure.message.contains("c exited with signal 15"));
        // A killed host that had already finished cleanly stays clean.
        assert_eq!(tracker.observe("d", None), (None, AbortAction::Continue));
    }

    #[test]
    fn only_the_first_primary_triggers_an_abort() {
        let mut tracker = AbortTracker::default();
        let primary = |host: &str| Some(HostFailure::new(FailureKind::Primary, host));
        assert_eq!(tracker.observe("a", primary("a")).1, AbortAction::AbortRest);
        // A second genuine primary (not killed by us) is still primary,
        // but there is nothing left to abort.
        let (failure, action) = tracker.observe("b", primary("b"));
        assert_eq!(action, AbortAction::Continue);
        assert_eq!(failure.expect("recorded").kind, FailureKind::Primary);
    }

    #[test]
    fn a_primary_seen_before_any_host_finished_triggers_once() {
        let mut tracker = AbortTracker::default();
        assert_eq!(tracker.primary_seen(), AbortAction::AbortRest);
        assert_eq!(tracker.primary_seen(), AbortAction::Continue);
        let (_, action) = tracker.observe(
            "a",
            Some(HostFailure::new(FailureKind::Primary, "a crashed")),
        );
        assert_eq!(action, AbortAction::Continue);
    }

    #[test]
    fn secondary_failures_never_trigger_an_abort() {
        let mut tracker = AbortTracker::default();
        for kind in [
            FailureKind::TimedOut,
            FailureKind::NeverStarted,
            FailureKind::Cascade,
        ] {
            let (_, action) = tracker.observe("x", Some(HostFailure::new(kind, "x")));
            assert_eq!(action, AbortAction::Continue, "{kind:?}");
        }
    }

    #[test]
    fn an_early_abort_attributes_everyone_to_the_culprit() {
        // End-to-end: the culprit is Failed, every host the driver killed
        // is Skipped naming it.
        let mut tracker = AbortTracker::default();
        let mut recorded = BTreeMap::new();
        let (failure, action) = tracker.observe(
            "10.1.1.68",
            Some(HostFailure::new(FailureKind::Primary, "rank died")),
        );
        assert_eq!(action, AbortAction::AbortRest);
        recorded.insert("10.1.1.68".to_string(), failure.expect("recorded"));
        tracker.mark_aborted(["10.1.1.67".to_string()]);
        let (failure, _) = tracker.observe(
            "10.1.1.67",
            Some(HostFailure::new(FailureKind::Primary, "killed")),
        );
        recorded.insert("10.1.1.67".to_string(), failure.expect("recorded"));
        let attributed = attribute("fleet overlap", &recorded);
        assert_eq!(attributed["10.1.1.68"], failed("rank died"));
        assert!(matches!(
            &attributed["10.1.1.67"],
            Attribution::Skipped { reason }
                if reason.starts_with("fleet overlap aborted: rank failure on 10.1.1.68")
        ));
    }

    #[test]
    fn no_failures_attribute_nothing() {
        assert!(attribute("fleet overlap", &BTreeMap::new()).is_empty());
    }
}
