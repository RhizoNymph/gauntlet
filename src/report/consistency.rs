//! Fleet consistency findings: a majority vote per field over every host's
//! inventory fields (`proto::consistency_values`) and its recorded NCCL NIC
//! fields (`NcclNicSummary::consistency_fields`, from the orchestrator's
//! derivation).
//!
//! The vote is over *present* values only: a host without the field at all
//! (`FieldValue::Missing`, e.g. no NVIDIA driver) can never be the
//! majority, and dissents whenever any host has a value. When the present
//! values have no strict plurality (two hosts on two different drivers),
//! nobody is the dissenter: the field is reported as a split with every
//! host's value. Only fields with a dissenter or a split are reported.

use std::collections::BTreeMap;

use super::{ConsistencyFinding, ConsistencySplit};
use crate::orchestrator::collect::HostObservations;
use crate::proto::{FieldValue, NcclNicSummary, consistency_values};

/// The outcome of one field's vote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Vote {
    /// Every host has the same value (or every host lacks it): nothing to
    /// report.
    Agreed,
    /// A strict plurality of present values; at least one host differs.
    Majority(ConsistencyFindingParts),
    /// Present values without a strict plurality.
    Split,
}

/// `ConsistencyFinding` before rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ConsistencyFindingParts {
    pub majority: String,
    pub dissenters: BTreeMap<String, FieldValue>,
}

/// Both kinds of finding, keyed by field.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct Findings {
    pub majorities: BTreeMap<String, ConsistencyFinding>,
    pub splits: BTreeMap<String, ConsistencySplit>,
}

pub(super) fn findings(observations: &BTreeMap<String, HostObservations>) -> Findings {
    let mut by_field: BTreeMap<String, BTreeMap<String, FieldValue>> = BTreeMap::new();
    for (host, obs) in observations {
        let Some(inventory) = &obs.inventory else {
            continue;
        };
        let nic_fields = obs
            .nccl_nics
            .as_ref()
            .map(NcclNicSummary::consistency_fields)
            .unwrap_or_default()
            .into_iter()
            .map(|(field, value)| (field, FieldValue::Present(value)));
        for (field, value) in consistency_values(inventory).into_iter().chain(nic_fields) {
            by_field
                .entry(field)
                .or_default()
                .insert(host.clone(), value);
        }
    }

    let mut findings = Findings::default();
    for (field, values) in by_field {
        match vote(&values) {
            Vote::Agreed => {}
            Vote::Majority(parts) => {
                findings.majorities.insert(
                    field,
                    ConsistencyFinding {
                        majority_value: parts.majority,
                        dissenters: render(parts.dissenters),
                    },
                );
            }
            Vote::Split => {
                findings.splits.insert(
                    field,
                    ConsistencySplit {
                        values: render(values),
                    },
                );
            }
        }
    }
    findings
}

fn render(values: BTreeMap<String, FieldValue>) -> BTreeMap<String, String> {
    values
        .into_iter()
        .map(|(host, value)| (host, value.render()))
        .collect()
}

/// One field's vote over host -> value.
pub(super) fn vote(values: &BTreeMap<String, FieldValue>) -> Vote {
    let Some(majority) = majority(values) else {
        // No majority: either nobody has a value (all agree on its
        // absence), or the present values tie.
        let any_present = values.values().any(|value| value.present().is_some());
        return if any_present {
            Vote::Split
        } else {
            Vote::Agreed
        };
    };
    let dissenters: BTreeMap<String, FieldValue> = values
        .iter()
        .filter(|(_, value)| value.present() != Some(majority.as_str()))
        .map(|(host, value)| (host.clone(), value.clone()))
        .collect();
    if dissenters.is_empty() {
        Vote::Agreed
    } else {
        Vote::Majority(ConsistencyFindingParts {
            majority,
            dissenters,
        })
    }
}

/// A host's entry in a vote: a present value, or none.
pub(super) trait Ballot {
    fn present(&self) -> Option<&str>;
}

impl Ballot for FieldValue {
    fn present(&self) -> Option<&str> {
        FieldValue::present(self)
    }
}

/// A plain string is always a present value.
impl Ballot for String {
    fn present(&self) -> Option<&str> {
        Some(self)
    }
}

/// The present value held by strictly more hosts than any other present
/// value. `None` when no host has a value, or when the top present values
/// tie: there is then no meaningful majority, and the result never depends
/// on value spelling or map order.
pub(super) fn majority<V: Ballot>(values: &BTreeMap<String, V>) -> Option<String> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for value in values.values().filter_map(Ballot::present) {
        *counts.entry(value).or_default() += 1;
    }
    let mut ranked: Vec<(&str, usize)> = counts.into_iter().collect();
    ranked.sort_by(|(_, a), (_, b)| b.cmp(a));
    match ranked.as_slice() {
        [] => None,
        [(top, _)] => Some((*top).to_string()),
        [(top, top_count), (_, runner_up), ..] => {
            (top_count > runner_up).then(|| (*top).to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn present(value: &str) -> FieldValue {
        FieldValue::Present(value.to_string())
    }

    fn field(entries: &[(&str, FieldValue)]) -> BTreeMap<String, FieldValue> {
        entries
            .iter()
            .map(|(host, value)| (host.to_string(), value.clone()))
            .collect()
    }

    #[test]
    fn the_three_driver_fleet_has_no_majority() {
        // 2026-10-15: two hosts on two drivers, the rebooted one without.
        let values = field(&[
            ("10.1.0.67", FieldValue::Missing),
            ("10.1.0.68", present("580.65.06")),
            ("10.1.0.69", present("590.48.01")),
        ]);
        assert_eq!(majority(&values), None);
        assert_eq!(vote(&values), Vote::Split);
    }

    #[test]
    fn a_missing_value_never_wins() {
        // Two hosts without, one with: the present value is the majority
        // and both hosts without it dissent.
        let values = field(&[
            ("a", FieldValue::Missing),
            ("b", FieldValue::Missing),
            ("c", present("580.65.06")),
        ]);
        let Vote::Majority(parts) = vote(&values) else {
            panic!("expected a majority");
        };
        assert_eq!(parts.majority, "580.65.06");
        assert_eq!(
            parts.dissenters.keys().collect::<Vec<_>>(),
            ["a", "b"],
            "hosts without the value dissent"
        );
    }

    #[test]
    fn two_against_one() {
        let values = field(&[
            ("a", present("580.65.06")),
            ("b", present("580.65.06")),
            ("c", present("590.48.01")),
        ]);
        let Vote::Majority(parts) = vote(&values) else {
            panic!("expected a majority");
        };
        assert_eq!(parts.majority, "580.65.06");
        assert_eq!(
            parts.dissenters,
            BTreeMap::from([("c".to_string(), present("590.48.01"))])
        );

        let with_missing = field(&[
            ("a", present("580.65.06")),
            ("b", present("580.65.06")),
            ("c", FieldValue::Missing),
        ]);
        let Vote::Majority(parts) = vote(&with_missing) else {
            panic!("expected a majority");
        };
        assert_eq!(parts.majority, "580.65.06");
        assert_eq!(
            parts.dissenters,
            BTreeMap::from([("c".to_string(), FieldValue::Missing)])
        );
    }

    #[test]
    fn agreement_is_not_reported() {
        assert_eq!(
            vote(&field(&[("a", present("x")), ("b", present("x"))])),
            Vote::Agreed
        );
        // Nobody has it: the fleet agrees on its absence.
        assert_eq!(
            vote(&field(&[
                ("a", FieldValue::Missing),
                ("b", FieldValue::Missing)
            ])),
            Vote::Agreed
        );
        assert_eq!(vote(&BTreeMap::new()), Vote::Agreed);
    }

    #[test]
    fn a_tie_with_a_missing_host_is_a_split_not_a_dissent() {
        // 2 vs 2 present plus one missing: no strict plurality.
        let values = field(&[
            ("a", present("x")),
            ("b", present("x")),
            ("c", present("y")),
            ("d", present("y")),
            ("e", FieldValue::Missing),
        ]);
        assert_eq!(vote(&values), Vote::Split);
    }
}
