//! Which fleet sweep a `NcclWorkload::Sweep` is (proto v11): the world
//! shape it was laid out in (`[tests] nccl_world`). The one source of the
//! fleet sweeps' test ids and headline names — the agent's emitters, the
//! orchestrator's gating/roll-up and the report's fleet-level rule and
//! link classes all derive from it.

use serde::{Deserialize, Serialize};

use super::{TestId, nccl_metric};

/// Which fleet sweep a `NcclWorkload::Sweep` is. Decides the test ids of
/// the lead's per-size series and the name of its headline:
///
/// - `RankPerGpu` (default): `nccl_all_*`, headline
///   `bus_gib_per_sec_peak`.
/// - `RankPerNode`: one rank per host on its GPU 0, every peer on another
///   node — `nccl_inter_all_*`, headline `bus_gib_per_sec_peak`.
/// - `Rail { rail }`: one rank per host on its GPU `rail` —
///   `nccl_inter_all_*`, headline `bus_gib_per_sec_peak_rail<r>`. The
///   per-rail world's overall headline (worst rail) is
///   `bus_gib_per_sec_peak_min_rail`, a distinct name so it never shares a
///   metric group with rank-per-node's single-NIC number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case", deny_unknown_fields)]
pub enum SweepSeries {
    #[default]
    RankPerGpu,
    RankPerNode,
    Rail {
        rail: u32,
    },
}

impl SweepSeries {
    /// Whether every peer of the world is on another node, so the sweep
    /// measures the NIC path alone.
    pub fn is_inter_node(self) -> bool {
        !matches!(self, SweepSeries::RankPerGpu)
    }

    /// Name of the headline the world's lead emits for this sweep.
    pub fn headline(self) -> String {
        match self {
            SweepSeries::RankPerGpu | SweepSeries::RankPerNode => nccl_metric::BUS_PEAK.to_string(),
            SweepSeries::Rail { rail } => nccl_metric::bus_peak_rail(rail),
        }
    }

    /// The two test ids this sweep's records land under, in collective
    /// order (all-reduce, all-gather).
    pub fn tests(self) -> [TestId; 2] {
        if self.is_inter_node() {
            [TestId::NcclInterAllReduce, TestId::NcclInterAllGather]
        } else {
            [TestId::NcclAllReduce, TestId::NcclAllGather]
        }
    }

    /// The inter-node shapes' test ids (all-reduce, all-gather).
    pub fn inter_node_tests() -> [TestId; 2] {
        SweepSeries::RankPerNode.tests()
    }

    /// Every fleet-wide sweep test id: rank-per-GPU then inter-node.
    pub fn all_fleet_tests() -> [TestId; 4] {
        let [reduce, gather] = SweepSeries::RankPerGpu.tests();
        let [inter_reduce, inter_gather] = SweepSeries::inter_node_tests();
        [reduce, gather, inter_reduce, inter_gather]
    }
}

impl std::fmt::Display for SweepSeries {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SweepSeries::RankPerGpu => f.write_str("rank-per-gpu"),
            SweepSeries::RankPerNode => f.write_str("rank-per-node"),
            SweepSeries::Rail { rail } => write!(f, "rail {rail}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_series_round_trip_and_name_their_headline() {
        for (series, json, headline, tests) in [
            (
                SweepSeries::RankPerGpu,
                r#"{"shape":"rank_per_gpu"}"#,
                "bus_gib_per_sec_peak",
                [TestId::NcclAllReduce, TestId::NcclAllGather],
            ),
            (
                SweepSeries::RankPerNode,
                r#"{"shape":"rank_per_node"}"#,
                "bus_gib_per_sec_peak",
                [TestId::NcclInterAllReduce, TestId::NcclInterAllGather],
            ),
            (
                SweepSeries::Rail { rail: 7 },
                r#"{"shape":"rail","rail":7}"#,
                "bus_gib_per_sec_peak_rail7",
                [TestId::NcclInterAllReduce, TestId::NcclInterAllGather],
            ),
        ] {
            assert_eq!(serde_json::to_string(&series).expect("serialize"), json);
            let back: SweepSeries = serde_json::from_str(json).expect("deserialize");
            assert_eq!(back, series);
            assert_eq!(series.headline(), headline);
            assert_eq!(series.tests(), tests);
            assert_eq!(series.is_inter_node(), series != SweepSeries::RankPerGpu);
        }
        assert!(serde_json::from_str::<SweepSeries>(r#"{"shape":"rail"}"#).is_err());
        for (test, wire) in [
            (TestId::NcclInterAllReduce, "\"nccl_inter_all_reduce\""),
            (TestId::NcclInterAllGather, "\"nccl_inter_all_gather\""),
        ] {
            assert_eq!(serde_json::to_string(&test).expect("serialize"), wire);
        }
    }

    #[test]
    fn the_fleet_test_lists_derive_from_the_series() {
        assert_eq!(
            SweepSeries::all_fleet_tests(),
            [
                TestId::NcclAllReduce,
                TestId::NcclAllGather,
                TestId::NcclInterAllReduce,
                TestId::NcclInterAllGather,
            ]
        );
        assert_eq!(
            SweepSeries::inter_node_tests(),
            SweepSeries::Rail { rail: 3 }.tests()
        );
    }

    #[test]
    fn rail_and_roll_up_names_never_collide_with_the_bare_headline() {
        assert_eq!(nccl_metric::bus_peak_rail(0), "bus_gib_per_sec_peak_rail0");
        assert_eq!(nccl_metric::rail_suffix(3), "rail3");
        assert!(nccl_metric::bus_peak_rail(2).starts_with(nccl_metric::BUS_PEAK_PREFIX));
        assert_eq!(nccl_metric::BUS_PEAK, "bus_gib_per_sec_peak");
        assert_eq!(
            nccl_metric::BUS_PEAK_MIN_RAIL,
            "bus_gib_per_sec_peak_min_rail"
        );
        assert_ne!(nccl_metric::BUS_PEAK_MIN_RAIL, nccl_metric::BUS_PEAK);
        assert!(nccl_metric::BUS_PEAK_MIN_RAIL.starts_with(nccl_metric::BUS_PEAK_PREFIX));
        assert_eq!(nccl_metric::rank_class_suffix(4), "4rank");
    }
}
