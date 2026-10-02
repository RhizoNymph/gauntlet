//! Network-phase step selection (`[tests] net_steps`, `--net-steps`).
//!
//! `Phase::Network` is a hierarchy of independent steps; a bandwidth-only
//! check wants the NCCL levels without the full pairwise TCP mesh or the
//! barrier probe. The selection is a typed set: `NetSteps` can only be built
//! by `FleetConfig::resolve_net_steps`, which guarantees it is non-empty and
//! already folds in `tests.nccl_intranode`.

use std::collections::BTreeSet;

use crate::names::NameTable;
use crate::proto::{TestId, TestOutcome};

/// Reason recorded on every test a deselected step would have produced.
pub const DISABLED_REASON: &str = "disabled by config";

/// One step of the network phase, in run order (innermost level first).
/// Serialized by canonical `name`; deserialized from any spelling in
/// `PARSE_TABLE`, so `[tests] net_steps` accepts exactly what
/// `--net-steps` accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NetStep {
    /// Intra-node NCCL sweep (all local GPUs, NVLink/PCIe).
    Intranode,
    /// Pairwise TCP latency + bandwidth mesh (tournament rounds).
    Pairwise,
    /// Fleet-wide NCCL sweep (one rank per GPU across hosts).
    Nccl,
    /// Barrier-skew probes: the NCCL barrier (rides the fleet NCCL sweep's
    /// communicator, so it also needs `Nccl`) and the TCP star barrier.
    Barrier,
}

impl NetStep {
    pub const ALL: [NetStep; 4] = [
        NetStep::Intranode,
        NetStep::Pairwise,
        NetStep::Nccl,
        NetStep::Barrier,
    ];

    /// Every accepted spelling; `parse` and the CLI help read this table.
    pub const PARSE_TABLE: &'static [(&'static str, NetStep)] = &[
        ("intranode", NetStep::Intranode),
        ("pairwise", NetStep::Pairwise),
        ("tcp", NetStep::Pairwise),
        ("nccl", NetStep::Nccl),
        ("barrier", NetStep::Barrier),
    ];

    /// Canonical name: the serde, config and CLI spelling.
    pub const fn name(self) -> &'static str {
        match self {
            NetStep::Intranode => "intranode",
            NetStep::Pairwise => "pairwise",
            NetStep::Nccl => "nccl",
            NetStep::Barrier => "barrier",
        }
    }

    pub fn parse(s: &str) -> Option<NetStep> {
        crate::names::parse(s)
    }

    /// Run-order names with aliases, for help text.
    pub fn help_list() -> String {
        crate::names::help_list::<NetStep>()
    }

    /// The tests this step produces, i.e. the ones that record a Skipped
    /// outcome when it is deselected.
    pub const fn tests(self) -> &'static [TestId] {
        match self {
            NetStep::Intranode => &[TestId::NcclIntraAllReduce, TestId::NcclIntraAllGather],
            NetStep::Pairwise => &[TestId::NetLatency, TestId::NetBandwidth],
            NetStep::Nccl => &[TestId::NcclAllReduce, TestId::NcclAllGather],
            NetStep::Barrier => &[TestId::NcclBarrier, TestId::TcpBarrier],
        }
    }
}

impl NameTable for NetStep {
    const KIND: &'static str = "network step";
    const VARIANTS: &'static [NetStep] = &NetStep::ALL;
    const SPELLINGS: &'static [(&'static str, NetStep)] = NetStep::PARSE_TABLE;

    fn name(self) -> &'static str {
        NetStep::name(self)
    }
}

crate::serde_via_name_table!(NetStep);

/// A resolved, non-empty selection of network steps. Construct through
/// `FleetConfig::resolve_net_steps`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetSteps(BTreeSet<NetStep>);

impl NetSteps {
    /// `None` for an empty selection: a network phase that runs nothing is
    /// a config mistake, not a quiet no-op.
    pub(crate) fn new(steps: BTreeSet<NetStep>) -> Option<NetSteps> {
        (!steps.is_empty()).then_some(NetSteps(steps))
    }

    pub fn contains(&self, step: NetStep) -> bool {
        self.0.contains(&step)
    }

    /// Selected steps in run order.
    pub fn iter(&self) -> impl Iterator<Item = NetStep> + '_ {
        self.0.iter().copied()
    }

    /// Whether the NCCL barrier runs: it needs both the barrier step and
    /// the fleet sweep whose communicator it rides.
    pub fn nccl_barrier(&self) -> bool {
        self.contains(NetStep::Barrier) && self.contains(NetStep::Nccl)
    }
}

/// Skipped outcomes (node scope, `DISABLED_REASON`) for every test the
/// deselected steps would have produced, in step order. The NCCL barrier
/// counts as deselected when the fleet sweep it rides is.
pub fn disabled_outcomes(steps: &NetSteps) -> Vec<(TestId, TestOutcome)> {
    let mut tests: Vec<TestId> = NetStep::ALL
        .iter()
        .filter(|step| !steps.contains(**step))
        .flat_map(|step| step.tests().iter().copied())
        .collect();
    // Barrier is the last step, so appending keeps step order.
    if steps.contains(NetStep::Barrier) && !steps.nccl_barrier() {
        tests.push(TestId::NcclBarrier);
    }
    tests
        .into_iter()
        .map(|test| {
            (
                test,
                TestOutcome::Skipped {
                    reason: DISABLED_REASON.to_string(),
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_step_has_tests_and_a_canonical_parse_entry() {
        for step in NetStep::ALL {
            assert!(!step.tests().is_empty());
            assert!(
                NetStep::PARSE_TABLE
                    .iter()
                    .any(|(alias, target)| *alias == step.name() && *target == step)
            );
        }
    }

    #[test]
    fn empty_selection_is_unrepresentable() {
        assert!(NetSteps::new(BTreeSet::new()).is_none());
    }

    #[test]
    fn help_lists_aliases() {
        assert_eq!(
            NetStep::help_list(),
            "intranode, pairwise (tcp), nccl, barrier"
        );
    }
}
