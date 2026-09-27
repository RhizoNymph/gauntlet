//! Rank ownership of per-rank reports (barrier timings, fleet-overlap
//! reports). Pure.
//!
//! Every per-rank event arrives on some host's event stream. A host may
//! only report ranks of its own `RankBlock`: anything else would silently
//! attribute one GPU's numbers to another. The first report of a rank from
//! its owner is kept; out-of-block reports and later duplicates are
//! dropped as structured per-host violations — so one misbehaving host can
//! neither misattribute data nor (as a duplicate used to) void the whole
//! fleet's barrier analysis.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::proto::RankBlock;

/// A per-rank report a host was not entitled to send.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(super) enum OwnershipViolation {
    #[error("host {host} reported rank {rank} but is not part of the NCCL world")]
    UnknownHost { host: String, rank: u32 },
    #[error("host {host} reported rank {rank} outside its rank block {base}..{end}")]
    OutsideBlock {
        host: String,
        rank: u32,
        base: u32,
        end: u32,
    },
    #[error("host {host} reported rank {rank} more than once; keeping the first report")]
    Duplicate { host: String, rank: u32 },
}

impl OwnershipViolation {
    pub(super) fn host(&self) -> &str {
        match self {
            OwnershipViolation::UnknownHost { host, .. }
            | OwnershipViolation::OutsideBlock { host, .. }
            | OwnershipViolation::Duplicate { host, .. } => host,
        }
    }
}

/// Keep each rank's first report from its owning host, in arrival order.
/// Returns the accepted reports keyed by rank and every violation.
pub(super) fn accept_owned<T>(
    owners: &BTreeMap<String, RankBlock>,
    reports: Vec<(String, T)>,
    rank_of: impl Fn(&T) -> u32,
) -> (BTreeMap<u32, T>, Vec<OwnershipViolation>) {
    let mut accepted = BTreeMap::new();
    let mut violations = Vec::new();
    for (host, report) in reports {
        let rank = rank_of(&report);
        let Some(block) = owners.get(&host) else {
            violations.push(OwnershipViolation::UnknownHost { host, rank });
            continue;
        };
        if !block.contains(rank) {
            violations.push(OwnershipViolation::OutsideBlock {
                host,
                rank,
                base: block.base(),
                end: block.end(),
            });
            continue;
        }
        if accepted.contains_key(&rank) {
            violations.push(OwnershipViolation::Duplicate { host, rank });
            continue;
        }
        accepted.insert(rank, report);
    }
    (accepted, violations)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owners() -> BTreeMap<String, RankBlock> {
        BTreeMap::from([
            ("n1".to_string(), RankBlock::new(0, 2).expect("block")),
            ("n2".to_string(), RankBlock::new(2, 2).expect("block")),
        ])
    }

    fn report(host: &str, rank: u32, value: &'static str) -> (String, (u32, &'static str)) {
        (host.to_string(), (rank, value))
    }

    #[test]
    fn owned_reports_are_accepted_by_rank() {
        let (accepted, violations) = accept_owned(
            &owners(),
            vec![
                report("n1", 0, "a"),
                report("n2", 3, "d"),
                report("n1", 1, "b"),
                report("n2", 2, "c"),
            ],
            |(rank, _)| *rank,
        );
        assert!(violations.is_empty(), "{violations:?}");
        let values: Vec<&str> = accepted.values().map(|(_, value)| *value).collect();
        assert_eq!(values, ["a", "b", "c", "d"]);
    }

    #[test]
    fn a_rank_outside_the_senders_block_is_rejected_not_misattributed() {
        let (accepted, violations) = accept_owned(
            &owners(),
            vec![report("n1", 2, "stolen"), report("n2", 2, "real")],
            |(rank, _)| *rank,
        );
        assert_eq!(accepted[&2].1, "real");
        assert_eq!(
            violations,
            [OwnershipViolation::OutsideBlock {
                host: "n1".into(),
                rank: 2,
                base: 0,
                end: 2
            }]
        );
        assert_eq!(violations[0].host(), "n1");
    }

    #[test]
    fn duplicates_keep_the_first_report_and_flag_the_sender() {
        let (accepted, violations) = accept_owned(
            &owners(),
            vec![
                report("n1", 0, "first"),
                report("n1", 0, "second"),
                report("n1", 1, "other"),
            ],
            |(rank, _)| *rank,
        );
        // Both ranks survive: a duplicate does not void the rest.
        assert_eq!(accepted.len(), 2);
        assert_eq!(accepted[&0].1, "first");
        assert_eq!(
            violations,
            [OwnershipViolation::Duplicate {
                host: "n1".into(),
                rank: 0
            }]
        );
    }

    #[test]
    fn reports_from_outside_the_world_are_rejected() {
        let (accepted, violations) =
            accept_owned(&owners(), vec![report("n9", 0, "x")], |(rank, _)| *rank);
        assert!(accepted.is_empty());
        assert_eq!(
            violations,
            [OwnershipViolation::UnknownHost {
                host: "n9".into(),
                rank: 0
            }]
        );
    }
}
