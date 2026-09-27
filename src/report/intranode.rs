//! Intra-node calibration link classes.
//!
//! The intra-node sweep's `msg_bytes` / `elapsed_us` series are fitted to
//! `t = alpha + beta * size` across the fleet, like the fleet sweep — but a
//! node-local collective over 8 NVLink-connected GPUs and one over 4
//! PCIe-attached GPUs are different links, so the class is keyed by the
//! communicator size: `nccl_allreduce_intranode_<n>gpu`,
//! `nccl_allgather_intranode_<n>gpu`. Keying (rather than restricting the
//! fit to the majority GPU count) keeps every node's data and gives a
//! heterogeneous fleet one calibration per topology the simulator may
//! need; a homogeneous fleet simply gets one class per collective.
//!
//! The GPU count comes from the node's own `ranks` record for that test and
//! repeat (emitted alongside the headline once the sweep completed), so the
//! extraction never depends on an inventory being present. A (host,
//! repeat) without a usable `ranks` value — a sweep that failed part-way —
//! contributes no points: a partial sweep is not calibration data.

use std::collections::BTreeMap;

use crate::analysis::fit::{AlphaBetaFit, fit_alpha_beta};
use crate::orchestrator::collect::HostObservations;
use crate::proto::{TestId, nccl_metric};

/// The intra-node sweep tests and their link-class prefixes.
pub const LINK_PREFIXES: [(TestId, &str); 2] = [
    (TestId::NcclIntraAllReduce, "nccl_allreduce_intranode"),
    (TestId::NcclIntraAllGather, "nccl_allgather_intranode"),
];

/// Link-class key for one intra-node collective at one communicator size.
pub fn link_class(prefix: &str, gpus: u32) -> String {
    format!("{prefix}_{}", nccl_metric::gpu_class_suffix(gpus))
}

/// `(msg_bytes, elapsed_us)` points of one intra-node test across the
/// fleet, bucketed by communicator size. Within one (host, repeat) the two
/// series are joined by emission order, exactly like the fleet sweep.
pub fn points_by_gpu_count(
    observations: &BTreeMap<String, HostObservations>,
    test: TestId,
) -> BTreeMap<u32, Vec<(u64, f64)>> {
    #[derive(Default)]
    struct Run {
        sizes: Vec<f64>,
        timings: Vec<f64>,
        ranks: Option<f64>,
    }

    let mut classes: BTreeMap<u32, Vec<(u64, f64)>> = BTreeMap::new();
    for obs in observations.values() {
        let mut runs: BTreeMap<u32, Run> = BTreeMap::new();
        for record in obs.metrics.iter().filter(|record| record.test == test) {
            let run = runs.entry(record.repeat).or_default();
            match record.name.as_str() {
                nccl_metric::MSG_BYTES => run.sizes.push(record.value),
                nccl_metric::ELAPSED_US => run.timings.push(record.value),
                nccl_metric::RANKS => run.ranks = Some(record.value),
                _ => {}
            }
        }
        for run in runs.into_values() {
            let Some(gpus) = run.ranks.and_then(gpu_count) else {
                continue;
            };
            let points = classes.entry(gpus).or_default();
            for (bytes, elapsed_us) in run.sizes.into_iter().zip(run.timings) {
                if bytes.is_finite() && bytes >= 0.0 {
                    points.push((bytes as u64, elapsed_us));
                }
            }
        }
    }
    classes.retain(|_, points| !points.is_empty());
    classes
}

/// A `ranks` value as a communicator size: a finite whole number of at
/// least two GPUs (the sweep never runs on fewer).
fn gpu_count(value: f64) -> Option<u32> {
    let whole =
        value.is_finite() && value.fract() == 0.0 && value >= 2.0 && value <= f64::from(u32::MAX);
    whole.then_some(value as u32)
}

/// Every intra-node link class with a successful fit (at least two distinct
/// sizes, finite timings).
pub fn link_fits(
    observations: &BTreeMap<String, HostObservations>,
) -> BTreeMap<String, AlphaBetaFit> {
    let mut links = BTreeMap::new();
    for (test, prefix) in LINK_PREFIXES {
        for (gpus, points) in points_by_gpu_count(observations, test) {
            if let Ok(fit) = fit_alpha_beta(&points) {
                links.insert(link_class(prefix, gpus), fit);
            }
        }
    }
    links
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{MetricRecord, Scope, Unit};

    fn record(test: TestId, name: &str, value: f64, repeat: u32) -> MetricRecord {
        MetricRecord {
            test,
            scope: Scope::Node,
            name: name.into(),
            value,
            unit: Unit::Count,
            repeat,
        }
    }

    /// A completed intra-node sweep as the agent emits it: per size
    /// elapsed/msg/bus, then the headline and `ranks`.
    fn sweep(obs: &mut HostObservations, test: TestId, gpus: u32, repeat: u32, us_per_kib: f64) {
        for kib in [1.0, 64.0, 1024.0] {
            obs.metrics
                .push(record(test, "elapsed_us", 5.0 + us_per_kib * kib, repeat));
            obs.metrics
                .push(record(test, "msg_bytes", kib * 1024.0, repeat));
            obs.metrics
                .push(record(test, "bus_gib_per_sec", 100.0, repeat));
        }
        obs.metrics
            .push(record(test, &nccl_metric::bus_peak(gpus), 100.0, repeat));
        obs.metrics
            .push(record(test, "ranks", f64::from(gpus), repeat));
    }

    #[test]
    fn classes_are_keyed_by_gpu_count() {
        assert_eq!(
            link_class("nccl_allreduce_intranode", 8),
            "nccl_allreduce_intranode_8gpu"
        );
        assert_eq!(
            link_class("nccl_allgather_intranode", 4),
            "nccl_allgather_intranode_4gpu"
        );
    }

    #[test]
    fn nodes_with_different_gpu_counts_are_different_link_classes() {
        let mut observations = BTreeMap::new();
        for (host, gpus) in [("n1", 8), ("n2", 8), ("n3", 4)] {
            let mut obs = HostObservations::default();
            sweep(&mut obs, TestId::NcclIntraAllReduce, gpus, 0, 0.1);
            observations.insert(host.to_string(), obs);
        }
        let points = points_by_gpu_count(&observations, TestId::NcclIntraAllReduce);
        assert_eq!(points.keys().copied().collect::<Vec<_>>(), [4, 8]);
        assert_eq!(points[&8].len(), 6, "two 8-GPU hosts x three sizes");
        assert_eq!(points[&4].len(), 3);
        assert_eq!(points[&4][0], (1024, 5.1));

        let links = link_fits(&observations);
        assert_eq!(
            links.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "nccl_allreduce_intranode_4gpu",
                "nccl_allreduce_intranode_8gpu"
            ]
        );
        let fit = links["nccl_allreduce_intranode_8gpu"];
        assert!((fit.alpha_us - 5.0).abs() < 1e-6, "{fit:?}");
        assert!(
            (fit.beta_us_per_byte - 0.1 / 1024.0).abs() < 1e-12,
            "{fit:?}"
        );
    }

    #[test]
    fn repeats_join_within_themselves() {
        let mut obs = HostObservations::default();
        sweep(&mut obs, TestId::NcclIntraAllGather, 8, 0, 0.1);
        sweep(&mut obs, TestId::NcclIntraAllGather, 8, 1, 0.1);
        let observations = BTreeMap::from([("n1".to_string(), obs)]);
        let points = points_by_gpu_count(&observations, TestId::NcclIntraAllGather);
        assert_eq!(points[&8].len(), 6);
        assert!(link_fits(&observations).contains_key("nccl_allgather_intranode_8gpu"));
    }

    #[test]
    fn a_sweep_without_ranks_is_not_calibration_data() {
        // A sweep that failed part-way: per-size points, no summary.
        let mut obs = HostObservations::default();
        for (bytes, us) in [(1024.0, 5.0), (65_536.0, 9.0)] {
            obs.metrics
                .push(record(TestId::NcclIntraAllReduce, "elapsed_us", us, 0));
            obs.metrics
                .push(record(TestId::NcclIntraAllReduce, "msg_bytes", bytes, 0));
        }
        let observations = BTreeMap::from([("n1".to_string(), obs)]);
        assert!(points_by_gpu_count(&observations, TestId::NcclIntraAllReduce).is_empty());
        assert!(link_fits(&observations).is_empty());
    }

    #[test]
    fn nonsense_rank_counts_are_rejected() {
        assert_eq!(gpu_count(8.0), Some(8));
        assert_eq!(gpu_count(2.0), Some(2));
        for bad in [1.0, 0.0, -4.0, 2.5, f64::NAN, f64::INFINITY] {
            assert_eq!(gpu_count(bad), None, "{bad}");
        }
    }

    #[test]
    fn fleet_sweep_series_never_leak_into_intranode_classes() {
        let mut obs = HostObservations::default();
        sweep(&mut obs, TestId::NcclAllReduce, 8, 0, 0.1);
        let observations = BTreeMap::from([("n1".to_string(), obs)]);
        assert!(link_fits(&observations).is_empty());
    }
}
