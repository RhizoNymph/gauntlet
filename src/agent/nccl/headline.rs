//! Fleet-level sweep headline (proto v11). Pure, so it is tested without a
//! GPU; its only caller (`sweep::run_sweep`) is gpu-gated.
//!
//! After a fleet sweep the host holding global rank 0 emits one node-scope
//! headline per collective: the best bus bandwidth across the sweep's
//! sizes, chosen exactly like the intra-node headline
//! (`agent::intranode::peak_bus_gib_per_sec`, max over sizes, non-finite
//! values ignored). The world shape decides the test id and the name
//! (`SweepSeries::tests`, `SweepSeries::headline`):
//!
//! - rank-per-GPU: `nccl_all_*.bus_gib_per_sec_peak`
//! - rank-per-node: `nccl_inter_all_*.bus_gib_per_sec_peak`
//! - rail `r`: `nccl_inter_all_*.bus_gib_per_sec_peak_rail<r>` (the
//!   orchestrator derives the per-rail world's overall
//!   `bus_gib_per_sec_peak_min_rail` — worst rail — from these once every
//!   rail ran).
//!
//! A world of one rank has no bus bandwidth to speak of, and a collective
//! no size of which could run (all-gather sizes too small to shard) has no
//! peak: neither emits a headline.
//!
//! Inter-node sweeps also open their per-size series with a `ranks`
//! record per test (`series_opener`): one lead can lead several rails of
//! different world sizes in one repeat, so the report keys each series by
//! the `ranks` record that precedes it.

use crate::agent::intranode::{MultiGpuWorld, peak_bus_gib_per_sec};
use crate::agent::sweep::{Collective, SweepPoint};
use crate::proto::{MetricRecord, Scope, SweepSeries, Unit, nccl_metric};

/// Records the lead emits before an inter-node sweep's per-size series:
/// `ranks` = the world size, once per test. Nothing for rank-per-GPU (its
/// link class is not keyed by size).
pub(crate) fn series_opener(series: SweepSeries, world_size: u32) -> Vec<MetricRecord> {
    if !series.is_inter_node() {
        return Vec::new();
    }
    series
        .tests()
        .into_iter()
        .map(|test| MetricRecord {
            test,
            scope: Scope::Node,
            name: nccl_metric::RANKS.to_string(),
            value: f64::from(world_size),
            unit: Unit::Count,
            repeat: 0,
        })
        .collect()
}

/// The headline records for a completed fleet sweep of `world_size` ranks.
pub(crate) fn fleet_headline(
    points: &[SweepPoint],
    series: SweepSeries,
    world_size: u32,
) -> Vec<MetricRecord> {
    let Some(world) = MultiGpuWorld::new(world_size) else {
        return Vec::new();
    };
    Collective::ALL
        .into_iter()
        .zip(series.tests())
        .filter_map(|(collective, test)| {
            peak_bus_gib_per_sec(points, collective, world).map(|peak| MetricRecord {
                test,
                scope: Scope::Node,
                name: series.headline(),
                value: peak,
                unit: Unit::GibPerSec,
                repeat: 0,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::sweep::SweepStep;
    use crate::proto::TestId;

    const GIB_ELEMS: usize = (1 << 30) / 4;

    fn point(collective: Collective, message_elements: usize, secs: f64) -> SweepPoint {
        SweepPoint {
            step: SweepStep {
                collective,
                send_elements: message_elements,
                message_elements,
            },
            per_iter_secs: secs,
        }
    }

    fn points() -> Vec<SweepPoint> {
        vec![
            point(Collective::AllReduce, GIB_ELEMS / 4, 1.0),
            point(Collective::AllReduce, GIB_ELEMS, 0.5), // the peak
            point(Collective::AllReduce, GIB_ELEMS, 1.0), // top size, degraded
            point(Collective::AllGather, GIB_ELEMS, 0.25),
            point(Collective::AllGather, GIB_ELEMS, f64::NAN),
        ]
    }

    fn shape(records: &[MetricRecord]) -> Vec<(TestId, &str)> {
        records
            .iter()
            .map(|record| (record.test, record.name.as_str()))
            .collect()
    }

    #[test]
    fn the_rank_per_gpu_headline_is_the_peak_across_sizes() {
        let records = fleet_headline(&points(), SweepSeries::RankPerGpu, 8);
        assert_eq!(
            shape(&records),
            [
                (TestId::NcclAllReduce, "bus_gib_per_sec_peak"),
                (TestId::NcclAllGather, "bus_gib_per_sec_peak"),
            ]
        );
        // 2 GiB/s algBW x 2*(7/8); not the largest-size value.
        assert!(
            (records[0].value - 3.5).abs() < 1e-12,
            "{}",
            records[0].value
        );
        // 4 GiB/s algBW x 7/8; the NaN point is ignored.
        assert!(
            (records[1].value - 3.5).abs() < 1e-12,
            "{}",
            records[1].value
        );
        assert!(records.iter().all(|r| r.scope == Scope::Node));
        assert!(records.iter().all(|r| r.unit == Unit::GibPerSec));
    }

    #[test]
    fn inter_node_shapes_land_under_the_inter_test_ids() {
        let per_node = fleet_headline(&points(), SweepSeries::RankPerNode, 2);
        assert_eq!(
            shape(&per_node),
            [
                (TestId::NcclInterAllReduce, "bus_gib_per_sec_peak"),
                (TestId::NcclInterAllGather, "bus_gib_per_sec_peak"),
            ]
        );
        // Bus factors use the world's own size: 2 GiB/s x 2*(1/2).
        assert!((per_node[0].value - 2.0).abs() < 1e-12);

        let rail = fleet_headline(&points(), SweepSeries::Rail { rail: 3 }, 4);
        assert_eq!(
            shape(&rail),
            [
                (TestId::NcclInterAllReduce, "bus_gib_per_sec_peak_rail3"),
                (TestId::NcclInterAllGather, "bus_gib_per_sec_peak_rail3"),
            ]
        );
    }

    #[test]
    fn inter_node_series_open_with_their_world_size() {
        let opener = series_opener(SweepSeries::Rail { rail: 2 }, 3);
        let shape: Vec<(TestId, &str, f64, Unit)> = opener
            .iter()
            .map(|r| (r.test, r.name.as_str(), r.value, r.unit))
            .collect();
        assert_eq!(
            shape,
            [
                (TestId::NcclInterAllReduce, "ranks", 3.0, Unit::Count),
                (TestId::NcclInterAllGather, "ranks", 3.0, Unit::Count),
            ]
        );
        assert_eq!(series_opener(SweepSeries::RankPerNode, 2).len(), 2);
        assert!(series_opener(SweepSeries::RankPerGpu, 16).is_empty());
    }

    #[test]
    fn no_headline_without_a_peak_or_a_peer() {
        assert!(fleet_headline(&[], SweepSeries::RankPerGpu, 8).is_empty());
        // A world of one has no bus bandwidth.
        assert!(fleet_headline(&points(), SweepSeries::RankPerGpu, 1).is_empty());
        // Only the collective that measured something gets a headline.
        let reduce_only = [point(Collective::AllReduce, 1024, 0.001)];
        let records = fleet_headline(&reduce_only, SweepSeries::RankPerNode, 4);
        assert_eq!(
            shape(&records),
            [(TestId::NcclInterAllReduce, "bus_gib_per_sec_peak")]
        );
    }
}
