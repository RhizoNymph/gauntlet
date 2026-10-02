//! Fleet-level NCCL sweep results (schema v13): which metric groups are
//! fleet-level headlines, and the calibration link classes of the fleet
//! and inter-node sweeps.
//!
//! **Fleet-level headlines stay out of MAD.** The fleet sweep's
//! `bus_gib_per_sec_peak` (and the inter-node `bus_gib_per_sec_peak`,
//! `bus_gib_per_sec_peak_rail<r>`) is one value per *run*, attributed to
//! the world's lead host at node scope (the `fleet_span` convention). A
//! group with one subject has no fleet peers: a median/MAD over it is
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
use crate::proto::{TestId, nccl_metric};

/// The sweep test ids whose headline is fleet-level (one per run).
const FLEET_LEVEL_TESTS: [TestId; 4] = [
    TestId::NcclAllReduce,
    TestId::NcclAllGather,
    TestId::NcclInterAllReduce,
    TestId::NcclInterAllGather,
];

/// Whether a metric group ("<test>.<metric>") is a fleet-level sweep
/// headline: one value per run, never MAD-compared.
pub(crate) fn is_fleet_level(group: &str) -> bool {
    FLEET_LEVEL_TESTS.iter().any(|test| {
        group
            .strip_prefix(&metric_key(*test, ""))
            .is_some_and(|name| name.starts_with(nccl_metric::BUS_PEAK_PREFIX))
    })
}

/// Calibration link classes fitted from the fleet-wide sweeps.
///
/// - `nccl_{allreduce,allgather}_rank_per_gpu`: the rank-per-GPU world
///   (n = total GPUs, the ring mixes NVLink with the fabric).
/// - `nccl_{allreduce,allgather}_inter_node`: the NIC-forcing shapes
///   (one rank per host; every rail pooled into one class — each rail is
///   the same nominal link, and the per-rail headlines already single out
///   a degraded one).
const LINK_CLASSES: [(TestId, &str); 4] = [
    (TestId::NcclAllReduce, "nccl_allreduce_rank_per_gpu"),
    (TestId::NcclAllGather, "nccl_allgather_rank_per_gpu"),
    (TestId::NcclInterAllReduce, "nccl_allreduce_inter_node"),
    (TestId::NcclInterAllGather, "nccl_allgather_inter_node"),
];

/// Alpha/beta fits for every fleet-wide link class with data.
pub(crate) fn link_fits(
    observations: &BTreeMap<String, HostObservations>,
) -> BTreeMap<String, AlphaBetaFit> {
    LINK_CLASSES
        .into_iter()
        .filter_map(|(test, class)| {
            fit_alpha_beta(&sweep_points(observations, test))
                .ok()
                .map(|fit| (class.to_string(), fit))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fleet_headlines_are_fleet_level_and_intra_node_ones_are_not() {
        for group in [
            "nccl_all_reduce.bus_gib_per_sec_peak",
            "nccl_all_gather.bus_gib_per_sec_peak",
            "nccl_inter_all_reduce.bus_gib_per_sec_peak",
            "nccl_inter_all_gather.bus_gib_per_sec_peak_rail3",
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
        ] {
            assert!(!is_fleet_level(group), "{group}");
        }
    }
}
