//! World shapes of the fleet NCCL sweep (`[tests] nccl_world`), the sweep
//! gate, and the per-rail roll-up. Pure: generic over the member type like
//! `layout`, so every rule here is unit-tested with host names.
//!
//! A shape turns the NCCL-capable hosts (with their CUDA-visible GPU
//! counts) into one or more `ShapedWorld`s, each a `RankLayout` plus the
//! `SweepSeries` its lead emits under:
//!
//! - `rank_per_gpu`: one world, every GPU of every host a rank.
//! - `rank_per_node`: one world, one rank per host on its GPU 0 — every
//!   peer on another node.
//! - `per_rail`: one world per local GPU index `r` < the largest GPU
//!   count, holding one rank per host *that has a GPU `r`*, on that GPU.
//!   The driver runs them one after another so rails never contend.
//!
//! Every world then passes `sweep_gate`: at least 2 ranks, and at least 2
//! hosts *when the intra-node sweep covers a one-host world* — or it is
//! Skipped with the failed condition named. (Per-rail worlds are planned
//! one at a time against earlier rails' exclusions: `rails::plan_rail`.)

use thiserror::Error;

use super::layout::{GpuSpan, LayoutError, RankLayout};
use super::records::OutcomeRecord;
#[cfg(test)]
use crate::config::NcclWorldShape;
use crate::proto::{MetricRecord, Scope, SweepSeries, TestId, TestOutcome, Unit, nccl_metric};

/// Fewest hosts a fleet sweep (when the intra-node level covers one host)
/// and the barrier probe (skew needs two independent arrivals) need. The
/// one source of that number.
pub(crate) const MIN_HOSTS: usize = 2;
/// Fewest ranks any fleet sweep needs: one rank has no peer.
const MIN_RANKS: u32 = 2;

/// Whether the intra-node NCCL sweep runs in this network phase, and so
/// already measures a one-host world's communicator. Decided where the
/// intra-node step is dispatched (`orchestrator::network_phase`), so the
/// gate follows whatever actually selects that step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntraNodeCoverage {
    /// The intra-node sweep runs: a one-host fleet sweep would repeat it.
    Covered,
    /// It does not run: a one-host world with 2+ GPUs is the only NCCL
    /// coverage there is, so the fleet sweep runs it.
    NotCovered,
}

/// One world the fleet sweep runs: its layout and the series it emits.
#[derive(Debug, Clone)]
pub(crate) struct ShapedWorld<M> {
    pub series: SweepSeries,
    pub layout: RankLayout<M>,
}

/// The world of one series over `(host, gpus)` in fleet order (hosts
/// with no GPU never join): every GPU for rank-per-GPU, GPU 0 of every
/// host for rank-per-node, GPU `r` of every host that has one for rail
/// `r`. The orchestrator lays out each world through this, one at a time
/// (rails against the earlier rails' exclusions, `rails::plan_rail`).
pub(crate) fn world_of<M: Clone>(
    series: SweepSeries,
    hosts: &[(M, u32)],
) -> Result<ShapedWorld<M>, LayoutError> {
    let layout = match series {
        SweepSeries::RankPerGpu => RankLayout::new(hosts.iter().cloned())?,
        SweepSeries::RankPerNode => rail_layout(hosts, 0)?,
        SweepSeries::Rail { rail } => rail_layout(hosts, rail)?,
    };
    Ok(ShapedWorld { series, layout })
}

/// Every world of `shape` with nothing excluded, in run order — the full
/// enumeration the unit tests check membership against (the driver walks
/// the same series one world at a time).
#[cfg(test)]
fn shaped_worlds<M: Clone>(
    shape: NcclWorldShape,
    hosts: &[(M, u32)],
) -> Result<Vec<ShapedWorld<M>>, LayoutError> {
    let max_gpus = hosts.iter().map(|(_, gpus)| *gpus).max().unwrap_or(0);
    let series = match shape {
        NcclWorldShape::RankPerGpu => vec![SweepSeries::RankPerGpu],
        NcclWorldShape::RankPerNode => vec![SweepSeries::RankPerNode],
        NcclWorldShape::PerRail => (0..max_gpus)
            .map(|rail| SweepSeries::Rail { rail })
            .collect(),
    };
    series
        .into_iter()
        .map(|series| world_of(series, hosts))
        .collect()
}

/// One rank per host that has a GPU `gpu`, on that GPU, in fleet order.
pub(crate) fn rail_layout<M: Clone>(
    hosts: &[(M, u32)],
    gpu: u32,
) -> Result<RankLayout<M>, LayoutError> {
    RankLayout::with_spans(
        hosts
            .iter()
            .filter(|(_, gpus)| *gpus > gpu)
            .map(|(member, _)| (member.clone(), GpuSpan::one(gpu))),
    )
}

/// Why a world is too small for a fleet sweep. Every member holds at
/// least one rank, so a world short of ranks is also short of hosts; the
/// two variants are the only possible combinations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum SweepSkip {
    /// Enough ranks, all on one host, and the intra-node sweep runs: it
    /// already measures that communicator.
    #[error(
        "fleet nccl sweep ({series}) needs at least {MIN_HOSTS} hosts, found {hosts} \
         ({ranks} ranks on one node are covered by the intra-node sweep)"
    )]
    TooFewHosts {
        series: SweepSeries,
        hosts: usize,
        ranks: u32,
    },
    /// Fewer than two ranks (and so fewer than two hosts): an "all-reduce"
    /// would be a local copy.
    #[error(
        "fleet nccl sweep ({series}) needs at least {MIN_HOSTS} hosts and {MIN_RANKS} ranks, \
         found {hosts} hosts and {ranks} ranks"
    )]
    TooFewRanks {
        series: SweepSeries,
        hosts: usize,
        ranks: u32,
    },
}

/// The fleet sweep needs a peer. With fewer than two ranks an "all-reduce"
/// is a local copy whose timings would calibrate a link that does not
/// exist; with every rank on one host the sweep only repeats the
/// intra-node level — when that level runs (`coverage`). Without it, a
/// one-host world of 2+ GPUs still runs, so the fleet keeps NCCL
/// coverage. `Ok` means run.
pub(crate) fn sweep_gate<M>(
    world: &ShapedWorld<M>,
    coverage: IntraNodeCoverage,
) -> Result<(), SweepSkip> {
    let hosts = world.layout.member_count();
    let ranks = world.layout.world_size();
    let series = world.series;
    if ranks < MIN_RANKS {
        return Err(SweepSkip::TooFewRanks {
            series,
            hosts,
            ranks,
        });
    }
    if hosts < MIN_HOSTS && coverage == IntraNodeCoverage::Covered {
        return Err(SweepSkip::TooFewHosts {
            series,
            hosts,
            ranks,
        });
    }
    Ok(())
}

/// The Skipped outcomes a gated world records on each of its members:
/// both of its series' tests, node scope, the skip as the reason.
pub(crate) fn skip_outcomes(skip: &SweepSkip) -> Vec<OutcomeRecord> {
    let series = match skip {
        SweepSkip::TooFewHosts { series, .. } | SweepSkip::TooFewRanks { series, .. } => *series,
    };
    let reason = skip.to_string();
    series
        .tests()
        .into_iter()
        .map(|test| {
            let outcome = TestOutcome::Skipped {
                reason: reason.clone(),
            };
            (test, Scope::Node, outcome)
        })
        .collect()
}

/// The headline values one rail's lead reported, `(test, value)`.
pub(crate) type RailPeaks = Vec<(TestId, f64)>;

/// The per-rail world's overall headline — **peak per rail, worst rail
/// overall**: per test, the minimum across rails of each rail's peak
/// (`bus_gib_per_sec_peak_rail<r>`), emitted as
/// `bus_gib_per_sec_peak_min_rail` (its own name: rank-per-node's bare
/// `bus_gib_per_sec_peak` is GPU 0's NIC alone, a different quantity).
///
/// Worst, not best: this is the number absolute thresholds gate on and
/// downstream tooling compares against a NIC ceiling, so one healthy rail
/// must never hide a degraded one (rails at 22/22/22/4 GiB/s must fail a
/// 20 GiB/s floor).
///
/// `rails` holds one entry per rail that *ran* (passed the gate). A rail
/// that ran but reported no finite value for a test (it failed part-way)
/// makes that test's overall headline absent rather than optimistic: the
/// worst rail is unknown, and the failure is already recorded against
/// its host. Rails gated out (fewer than 2 hosts) never ran, carry no
/// NIC number, and are not part of the roll-up.
pub(crate) fn rail_rollup(rails: &[RailPeaks]) -> Vec<MetricRecord> {
    SweepSeries::inter_node_tests()
        .into_iter()
        .filter_map(|test| {
            rails
                .iter()
                .map(|peaks| {
                    peaks
                        .iter()
                        .find(|(rail_test, value)| *rail_test == test && value.is_finite())
                        .map(|(_, value)| *value)
                })
                .collect::<Option<Vec<f64>>>()?
                .into_iter()
                .reduce(f64::min)
                .map(|worst| MetricRecord {
                    test,
                    scope: Scope::Node,
                    name: nccl_metric::BUS_PEAK_MIN_RAIL.to_string(),
                    value: worst,
                    unit: Unit::GibPerSec,
                    repeat: 0,
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    type World = ShapedWorld<&'static str>;

    fn worlds(shape: NcclWorldShape, hosts: &[(&'static str, u32)]) -> Vec<World> {
        shaped_worlds(shape, hosts).expect("worlds")
    }

    /// (host, base, count, first_gpu) per member.
    fn blocks(world: &World) -> Vec<(&'static str, u32, u32, u32)> {
        world
            .layout
            .members()
            .iter()
            .map(|(host, assignment)| {
                let block = assignment.block();
                (*host, block.base(), block.count(), block.first_gpu())
            })
            .collect()
    }

    const FLEET: [(&str, u32); 4] = [("n1", 8), ("cpu", 0), ("n2", 4), ("n3", 8)];

    #[test]
    fn rank_per_gpu_is_one_world_of_every_gpu() {
        let worlds = worlds(NcclWorldShape::RankPerGpu, &FLEET);
        assert_eq!(worlds.len(), 1);
        assert_eq!(worlds[0].series, SweepSeries::RankPerGpu);
        assert_eq!(
            blocks(&worlds[0]),
            [("n1", 0, 8, 0), ("n2", 8, 4, 0), ("n3", 12, 8, 0)]
        );
        assert_eq!(worlds[0].layout.world_size(), 20);
    }

    #[test]
    fn rank_per_node_is_one_rank_per_host_on_gpu_zero() {
        let worlds = worlds(NcclWorldShape::RankPerNode, &FLEET);
        assert_eq!(worlds.len(), 1);
        assert_eq!(worlds[0].series, SweepSeries::RankPerNode);
        assert_eq!(
            blocks(&worlds[0]),
            [("n1", 0, 1, 0), ("n2", 1, 1, 0), ("n3", 2, 1, 0)]
        );
        // Every peer is on another node: one rank per member.
        assert_eq!(
            worlds[0].layout.world_size() as usize,
            worlds[0].layout.member_count()
        );
    }

    #[test]
    fn rails_hold_the_hosts_that_have_that_gpu() {
        let worlds = worlds(NcclWorldShape::PerRail, &FLEET);
        // One rail per local GPU index up to the largest host.
        assert_eq!(worlds.len(), 8);
        for (rail, world) in worlds.iter().enumerate() {
            let rail = rail as u32;
            assert_eq!(world.series, SweepSeries::Rail { rail });
            let expected: Vec<(&str, u32, u32, u32)> = if rail < 4 {
                vec![("n1", 0, 1, rail), ("n2", 1, 1, rail), ("n3", 2, 1, rail)]
            } else {
                // The 4-GPU host has no GPU 4..7.
                vec![("n1", 0, 1, rail), ("n3", 1, 1, rail)]
            };
            assert_eq!(blocks(world), expected, "rail {rail}");
            // Each rail is its own world with its own rank 0.
            assert!(world.layout.members()[0].1.block().holds_lead());
        }
    }

    #[test]
    fn every_rail_rank_maps_back_to_its_host_and_gpu() {
        let worlds = worlds(NcclWorldShape::PerRail, &[("a", 2), ("b", 1), ("c", 2)]);
        assert_eq!(worlds.len(), 2);
        let rail1 = &worlds[1];
        let located: Vec<(&str, usize, u32)> = (0..rail1.layout.world_size())
            .map(|rank| {
                let location = rail1.layout.locate(rank).expect("rank inside the rail");
                (*location.member, location.member_index, location.gpu)
            })
            .collect();
        assert_eq!(located, [("a", 0, 1), ("c", 1, 1)]);
    }

    #[test]
    fn a_fleet_without_gpus_has_no_rails_and_empty_worlds() {
        let cpu_only = [("a", 0), ("b", 0)];
        assert!(worlds(NcclWorldShape::PerRail, &cpu_only).is_empty());
        assert!(worlds(NcclWorldShape::PerRail, &[]).is_empty());
        for shape in [NcclWorldShape::RankPerGpu, NcclWorldShape::RankPerNode] {
            let worlds = worlds(shape, &cpu_only);
            assert_eq!(worlds.len(), 1);
            assert_eq!(worlds[0].layout.member_count(), 0);
        }
    }

    fn gate_with(
        shape: NcclWorldShape,
        hosts: &[(&'static str, u32)],
        coverage: IntraNodeCoverage,
    ) -> Vec<Result<(), SweepSkip>> {
        worlds(shape, hosts)
            .iter()
            .map(|world| sweep_gate(world, coverage))
            .collect()
    }

    /// The default network phase: the intra-node sweep runs.
    fn gate(shape: NcclWorldShape, hosts: &[(&'static str, u32)]) -> Vec<Result<(), SweepSkip>> {
        gate_with(shape, hosts, IntraNodeCoverage::Covered)
    }

    /// The single world's gate result.
    fn only(results: Vec<Result<(), SweepSkip>>) -> Result<(), SweepSkip> {
        assert_eq!(results.len(), 1, "one world");
        results.into_iter().next().expect("one world")
    }

    #[test]
    fn one_host_never_runs_the_fleet_sweep_whatever_its_gpu_count() {
        // Gap 8: 8 ranks on one node repeat the intra-node sweep.
        let result = only(gate(NcclWorldShape::RankPerGpu, &[("n1", 8)]));
        assert_eq!(
            result,
            Err(SweepSkip::TooFewHosts {
                series: SweepSeries::RankPerGpu,
                hosts: 1,
                ranks: 8
            })
        );
        let reason = result.expect_err("skip").to_string();
        assert!(reason.contains("at least 2 hosts, found 1"), "{reason}");
        assert!(
            reason.contains("covered by the intra-node sweep"),
            "{reason}"
        );
        assert!(!reason.contains("ranks, found"), "{reason}");
    }

    #[test]
    fn one_host_runs_the_fleet_sweep_when_the_intra_node_sweep_does_not() {
        // tests.nccl_intranode = false: the fleet sweep is the only NCCL
        // coverage a one-host fleet has, so it runs as before.
        let covered = IntraNodeCoverage::NotCovered;
        assert_eq!(
            only(gate_with(NcclWorldShape::RankPerGpu, &[("n1", 8)], covered)),
            Ok(())
        );
        // A world with no peer at all still never runs.
        let single = only(gate_with(NcclWorldShape::RankPerGpu, &[("n1", 1)], covered));
        assert!(matches!(single, Err(SweepSkip::TooFewRanks { .. })));
        // The NIC-forcing shapes put one rank per host: still short.
        let per_node = only(gate_with(
            NcclWorldShape::RankPerNode,
            &[("n1", 8)],
            covered,
        ));
        assert!(matches!(per_node, Err(SweepSkip::TooFewRanks { .. })));
    }

    #[test]
    fn a_single_rank_world_names_both_conditions() {
        for hosts in [&[("n1", 1)][..], &[][..], &[("cpu", 0)][..]] {
            let result = only(gate(NcclWorldShape::RankPerGpu, hosts));
            let skip = result.expect_err("must skip");
            assert!(matches!(skip, SweepSkip::TooFewRanks { .. }), "{skip:?}");
            let reason = skip.to_string();
            assert!(reason.contains("at least 2 hosts and 2 ranks"), "{reason}");
        }
    }

    #[test]
    fn two_hosts_run_in_every_shape() {
        for shape in [
            NcclWorldShape::RankPerGpu,
            NcclWorldShape::RankPerNode,
            NcclWorldShape::PerRail,
        ] {
            for result in gate(shape, &[("n1", 2), ("n2", 2)]) {
                assert_eq!(result, Ok(()), "{shape:?}");
            }
        }
        // Two one-GPU hosts are a real two-rank, two-host world.
        assert_eq!(
            gate(NcclWorldShape::RankPerGpu, &[("a", 1), ("b", 1)]),
            [Ok(())]
        );
    }

    #[test]
    fn rank_per_node_on_one_host_is_short_of_both() {
        let result = only(gate(NcclWorldShape::RankPerNode, &[("n1", 8)]));
        let reason = result.expect_err("skip").to_string();
        assert!(reason.contains("rank-per-node"), "{reason}");
        assert!(reason.contains("found 1 hosts and 1 ranks"), "{reason}");
    }

    #[test]
    fn a_rail_with_one_host_is_skipped_by_name() {
        let results = gate(NcclWorldShape::PerRail, &[("n1", 8), ("n2", 4)]);
        assert_eq!(results.len(), 8);
        assert!(results[..4].iter().all(Result::is_ok));
        for (rail, result) in results.iter().enumerate().skip(4) {
            let reason = result.clone().expect_err("one-host rail").to_string();
            assert!(reason.contains(&format!("rail {rail}")), "{reason}");
            assert!(reason.contains("found 1 hosts"), "{reason}");
        }
    }

    #[test]
    fn skips_record_both_tests_of_the_series() {
        let rank_per_gpu = skip_outcomes(&SweepSkip::TooFewHosts {
            series: SweepSeries::RankPerGpu,
            hosts: 1,
            ranks: 8,
        });
        let tests: Vec<TestId> = rank_per_gpu.iter().map(|(test, _, _)| *test).collect();
        assert_eq!(tests, [TestId::NcclAllReduce, TestId::NcclAllGather]);
        for (_, scope, outcome) in &rank_per_gpu {
            assert_eq!(*scope, Scope::Node);
            let TestOutcome::Skipped { reason } = outcome else {
                panic!("expected Skipped, got {outcome:?}");
            };
            assert!(reason.contains("at least 2 hosts"), "{reason}");
        }
        let rail = skip_outcomes(&SweepSkip::TooFewRanks {
            series: SweepSeries::Rail { rail: 5 },
            hosts: 1,
            ranks: 1,
        });
        let tests: Vec<TestId> = rail.iter().map(|(test, _, _)| *test).collect();
        assert_eq!(
            tests,
            [TestId::NcclInterAllReduce, TestId::NcclInterAllGather]
        );
    }

    fn rail(reduce: f64, gather: f64) -> RailPeaks {
        vec![
            (TestId::NcclInterAllReduce, reduce),
            (TestId::NcclInterAllGather, gather),
        ]
    }

    fn rollup_shape(records: &[MetricRecord]) -> Vec<(TestId, &str, f64)> {
        records
            .iter()
            .map(|r| (r.test, r.name.as_str(), r.value))
            .collect()
    }

    #[test]
    fn the_rail_rollup_is_the_worst_rail_per_collective() {
        let records = rail_rollup(&[rail(20.0, 12.0), rail(23.5, 11.0), rail(21.0, 13.0)]);
        assert_eq!(
            rollup_shape(&records),
            [
                (
                    TestId::NcclInterAllReduce,
                    "bus_gib_per_sec_peak_min_rail",
                    20.0
                ),
                (
                    TestId::NcclInterAllGather,
                    "bus_gib_per_sec_peak_min_rail",
                    11.0
                ),
            ]
        );
        assert!(records.iter().all(|r| r.scope == Scope::Node));
        assert!(records.iter().all(|r| r.unit == Unit::GibPerSec));
        assert!(rail_rollup(&[]).is_empty(), "no rail ran, no headline");
    }

    #[test]
    fn one_degraded_rail_drags_the_headline_down() {
        // Three healthy rails cannot hide the fourth: a 20 GiB/s floor on
        // the headline must trip.
        let records = rail_rollup(&[
            rail(22.0, 11.0),
            rail(22.0, 11.0),
            rail(22.0, 11.0),
            rail(4.0, 2.0),
        ]);
        assert_eq!(
            rollup_shape(&records),
            [
                (
                    TestId::NcclInterAllReduce,
                    "bus_gib_per_sec_peak_min_rail",
                    4.0
                ),
                (
                    TestId::NcclInterAllGather,
                    "bus_gib_per_sec_peak_min_rail",
                    2.0
                ),
            ]
        );
    }

    #[test]
    fn a_rail_that_ran_without_a_number_voids_that_headline() {
        // Rail 1 measured all-reduce but its all-gather leg failed (or
        // came back non-finite): the worst all-gather rail is unknown.
        let records = rail_rollup(&[
            rail(22.0, 11.0),
            vec![
                (TestId::NcclInterAllReduce, 21.0),
                (TestId::NcclInterAllGather, f64::NAN),
            ],
        ]);
        assert_eq!(
            rollup_shape(&records),
            [(
                TestId::NcclInterAllReduce,
                "bus_gib_per_sec_peak_min_rail",
                21.0
            )]
        );
        // A rail that ran but reported nothing voids both.
        assert!(rail_rollup(&[rail(22.0, 11.0), Vec::new()]).is_empty());
    }
}
