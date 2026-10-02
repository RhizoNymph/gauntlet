//! Fleet-level results (schema v13): which metric groups are fleet-level
//! (one value per run), and the calibration link classes of the fleet and
//! inter-node sweeps.
//!
//! **Fleet-level groups stay out of MAD.** Two kinds of metric are one
//! value per *run*, attributed to a lead host at node scope:
//! - the sweep headlines — the fleet sweep's `bus_gib_per_sec_peak`, the
//!   inter-node `bus_gib_per_sec_peak` (rank-per-node),
//!   `bus_gib_per_sec_peak_rail<r>` and `bus_gib_per_sec_peak_min_rail`
//!   (per-rail), plus the inter-node `ranks` (world size);
//! - the barrier probes' fleet span — `nccl_barrier.fleet_span_*` and
//!   `tcp_barrier.fleet_span_*`, the per-iteration spread of the whole
//!   world's arrivals.
//!
//! A group with one subject has no fleet peers: a median/MAD over it is
//! degenerate (MAD 0, the subject is its own median), and a spread
//! comparison across subjects (jitter) is meaningless. So these groups
//! skip the outlier and jitter passes by rule, instead of relying on
//! `flag_outliers`' minimum sample count to say nothing. They still
//! aggregate like any metric — per-subject moments across `--repeat`
//! (median, MAD, min/max, stddev: the run-to-run jitter of the fleet
//! number itself) — and `[thresholds.absolute]` bounds check their
//! median, which is how a run fails on a low fleet bandwidth.

use std::collections::BTreeMap;

use super::{metric_key, sweep_points};
use crate::analysis::fit::{AlphaBetaFit, fit_alpha_beta};
use crate::orchestrator::collect::HostObservations;
use crate::proto::{SweepSeries, TestId, barrier_metric, nccl_metric};

/// Every fleet-level metric family: (test, metric-name prefix). One rule,
/// one place — a new fleet-level metric is a new family here. The sweep
/// test ids come from `SweepSeries`, never a hand-copied list.
fn fleet_level_families() -> impl Iterator<Item = (TestId, &'static str)> {
    let headlines = SweepSeries::all_fleet_tests()
        .into_iter()
        .map(|test| (test, nccl_metric::BUS_PEAK_PREFIX));
    // The inter-node world size: one value per run on the lead.
    let ranks = SweepSeries::inter_node_tests()
        .into_iter()
        .map(|test| (test, nccl_metric::RANKS));
    let spans = [TestId::NcclBarrier, TestId::TcpBarrier]
        .into_iter()
        .map(|test| (test, barrier_metric::FLEET_SPAN_PREFIX));
    headlines.chain(ranks).chain(spans)
}

/// Whether a metric group ("<test>.<metric>") is fleet-level: one value
/// per run on a lead host, never MAD- or jitter-compared.
pub(crate) fn is_fleet_level(group: &str) -> bool {
    fleet_level_families().any(|(test, prefix)| {
        group
            .strip_prefix(&metric_key(test, ""))
            .is_some_and(|name| name.starts_with(prefix))
    })
}

/// Link-class slugs of the two collectives, in `SweepSeries::tests` order
/// (all-reduce, all-gather).
const COLLECTIVE_SLUGS: [&str; 2] = ["allreduce", "allgather"];

/// Calibration link classes fitted from the fleet-wide sweeps.
///
/// - `nccl_{allreduce,allgather}_rank_per_gpu`: the rank-per-GPU world
///   (n = total GPUs, the ring mixes NVLink with the fabric). One world
///   per run, so one class.
/// - `nccl_{allreduce,allgather}_inter_node_<n>rank`: the NIC-forcing
///   shapes, keyed by world size like the intra-node `_<n>gpu` classes.
///   Rails of a heterogeneous fleet have different world sizes (rail 7 of
///   a fleet with some 4-GPU hosts has fewer members than rail 0), and one
///   fit pooled across sizes describes no real link; rails of equal size
///   are the same nominal link and pool.
pub(crate) fn link_fits(
    observations: &BTreeMap<String, HostObservations>,
) -> BTreeMap<String, AlphaBetaFit> {
    let mut links = BTreeMap::new();
    for (test, slug) in SweepSeries::RankPerGpu
        .tests()
        .into_iter()
        .zip(COLLECTIVE_SLUGS)
    {
        if let Ok(fit) = fit_alpha_beta(&sweep_points(observations, test)) {
            links.insert(format!("nccl_{slug}_rank_per_gpu"), fit);
        }
    }
    for (test, slug) in SweepSeries::inter_node_tests()
        .into_iter()
        .zip(COLLECTIVE_SLUGS)
    {
        for (ranks, points) in points_by_world_size(observations, test) {
            if let Ok(fit) = fit_alpha_beta(&points) {
                links.insert(inter_node_class(slug, ranks), fit);
            }
        }
    }
    links
}

/// `nccl_<collective>_inter_node_<n>rank`.
fn inter_node_class(slug: &str, ranks: u32) -> String {
    format!(
        "nccl_{slug}_inter_node_{}",
        nccl_metric::rank_class_suffix(ranks)
    )
}

/// `(msg_bytes, elapsed_us)` points of one inter-node test across the
/// fleet, bucketed by world size. The lead opens every series with a
/// `ranks` record (`agent::nccl::headline::series_opener`), so walking a
/// host's records in emission order, each `ranks` record starts a new
/// series of that world size — several rails led by one host in one
/// repeat stay apart. `msg_bytes` and `elapsed_us` pair by order within a
/// series. Points before any opener (no world size) are dropped.
fn points_by_world_size(
    observations: &BTreeMap<String, HostObservations>,
    test: TestId,
) -> BTreeMap<u32, Vec<(u64, f64)>> {
    let mut classes: BTreeMap<u32, Vec<(u64, f64)>> = BTreeMap::new();
    let mut flush = |ranks: Option<u32>, sizes: &mut Vec<f64>, timings: &mut Vec<f64>| {
        if let Some(ranks) = ranks {
            let points = classes.entry(ranks).or_default();
            for (bytes, elapsed_us) in sizes.iter().zip(timings.iter()) {
                if bytes.is_finite() && *bytes >= 0.0 {
                    points.push((*bytes as u64, *elapsed_us));
                }
            }
        }
        sizes.clear();
        timings.clear();
    };
    for obs in observations.values() {
        let mut ranks: Option<u32> = None;
        let mut sizes = Vec::new();
        let mut timings = Vec::new();
        for record in obs.metrics.iter().filter(|record| record.test == test) {
            match record.name.as_str() {
                nccl_metric::RANKS => {
                    flush(ranks, &mut sizes, &mut timings);
                    ranks = world_size(record.value);
                }
                nccl_metric::MSG_BYTES => sizes.push(record.value),
                nccl_metric::ELAPSED_US => timings.push(record.value),
                _ => {}
            }
        }
        flush(ranks, &mut sizes, &mut timings);
    }
    classes.retain(|_, points| !points.is_empty());
    classes
}

/// A `ranks` value as a world size: a finite whole number of at least two
/// ranks (the sweep never runs on fewer).
fn world_size(value: f64) -> Option<u32> {
    (value.is_finite() && value >= 2.0 && value.fract() == 0.0 && value <= f64::from(u32::MAX))
        .then_some(value as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fleet_headlines_and_spans_are_fleet_level_per_node_metrics_are_not() {
        for group in [
            "nccl_all_reduce.bus_gib_per_sec_peak",
            "nccl_all_gather.bus_gib_per_sec_peak",
            "nccl_inter_all_reduce.bus_gib_per_sec_peak",
            "nccl_inter_all_gather.bus_gib_per_sec_peak_rail3",
            "nccl_inter_all_reduce.bus_gib_per_sec_peak_min_rail",
            "nccl_inter_all_reduce.ranks",
            "nccl_barrier.fleet_span_p50_us",
            "nccl_barrier.fleet_span_max_us",
            "tcp_barrier.fleet_span_p99_us",
        ] {
            assert!(is_fleet_level(group), "{group}");
        }
        for group in [
            // Per-node, MAD-compared across nodes of one topology.
            "nccl_intra_all_reduce.bus_gib_per_sec_peak_8gpu",
            // Per-size series (excluded from MAD by their repeated keys).
            "nccl_all_reduce.bus_gib_per_sec",
            "nccl_inter_all_reduce.elapsed_us",
            "gpu_gemm_perf.gflops_bf16",
            // Intra-node ranks are per node, MAD-compared.
            "nccl_intra_all_reduce.ranks",
            // Per-rank barrier metrics are fleet comparisons.
            "nccl_barrier.p99_us",
            "tcp_barrier.slowest_frac",
            // A fleet_span name under another test is not this family.
            "nccl_all_reduce.fleet_span_p50_us",
        ] {
            assert!(!is_fleet_level(group), "{group}");
        }
    }

    use crate::proto::{MetricRecord, Scope, Unit};

    fn record(test: TestId, name: &str, value: f64) -> MetricRecord {
        MetricRecord {
            test,
            scope: Scope::Node,
            name: name.into(),
            value,
            unit: Unit::Count,
            repeat: 0,
        }
    }

    /// One inter-node series on `obs`: opener, then (bytes, us) points.
    fn series(obs: &mut HostObservations, test: TestId, ranks: u32, points: &[(f64, f64)]) {
        obs.metrics
            .push(record(test, nccl_metric::RANKS, f64::from(ranks)));
        for (bytes, us) in points {
            obs.metrics.push(record(test, nccl_metric::ELAPSED_US, *us));
            obs.metrics
                .push(record(test, nccl_metric::MSG_BYTES, *bytes));
        }
    }

    #[test]
    fn rails_of_different_world_sizes_fit_separate_classes() {
        // A heterogeneous fleet: rails 0-3 span 3 hosts, rails 4-7 span 2
        // (one host has 4 GPUs). Both kinds of rail are led by one host
        // in one repeat; their series must not pool.
        let mut lead = HostObservations::default();
        let test = TestId::NcclInterAllReduce;
        let three = [(1024.0, 11.0), (4096.0, 14.0), (16384.0, 26.0)];
        let two = [(1024.0, 6.0), (4096.0, 12.0), (16384.0, 36.0)];
        series(&mut lead, test, 3, &three);
        series(&mut lead, test, 2, &two);
        series(&mut lead, test, 3, &three);
        let observations = BTreeMap::from([("lead".to_string(), lead)]);

        let buckets = points_by_world_size(&observations, test);
        assert_eq!(buckets.keys().copied().collect::<Vec<_>>(), [2, 3]);
        assert_eq!(buckets[&3].len(), 6);
        assert_eq!(buckets[&2].len(), 3);
        assert_eq!(buckets[&2][0], (1024, 6.0));

        let links = link_fits(&observations);
        let keys: Vec<&str> = links.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "nccl_allreduce_inter_node_2rank",
                "nccl_allreduce_inter_node_3rank"
            ]
        );
        // 3-rank series: 10 us + 1 us/KiB.
        let three_rank = links["nccl_allreduce_inter_node_3rank"];
        assert!((three_rank.alpha_us - 10.0).abs() < 1e-6, "{three_rank:?}");
    }

    #[test]
    fn points_without_a_world_size_are_dropped() {
        let mut obs = HostObservations::default();
        let test = TestId::NcclInterAllGather;
        // Points before any opener, then an invalid opener.
        obs.metrics.push(record(test, nccl_metric::ELAPSED_US, 5.0));
        obs.metrics
            .push(record(test, nccl_metric::MSG_BYTES, 1024.0));
        series(&mut obs, test, 1, &[(1024.0, 5.0)]);
        let observations = BTreeMap::from([("h".to_string(), obs)]);
        assert!(points_by_world_size(&observations, test).is_empty());
        assert_eq!(world_size(2.5), None);
        assert_eq!(world_size(f64::NAN), None);
        assert_eq!(world_size(8.0), Some(8));
    }
}
