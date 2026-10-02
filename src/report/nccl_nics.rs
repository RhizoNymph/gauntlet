//! The NCCL NIC picture per host: `calibration.nccl_nics` (one
//! `NcclNicSummary` per host with an inventory) and its "nccl nics" table
//! section.
//!
//! Derived from each host's inventory under the run's resolved
//! `NCCL_IB_HCA` (`crate::nccl_ib`), the same function the orchestrator
//! uses for the `nccl_nics` outcome and ceiling metric, so the section, the
//! outcome and the metric cannot disagree.

use std::collections::BTreeMap;
use std::io::Write;

use anyhow::Result;

use super::{RunResults, new_table, section};
use crate::nccl_ib::{NcclIbConfig, NcclNicSummary, SelectedPort, summarize};
use crate::orchestrator::collect::HostObservations;
use crate::proto::{PciLocality, TestId, TestOutcome};

/// One summary per host that reported an inventory.
pub fn summaries(
    observations: &BTreeMap<String, HostObservations>,
    config: &NcclIbConfig,
) -> BTreeMap<String, NcclNicSummary> {
    observations
        .iter()
        .filter_map(|(host, obs)| {
            obs.inventory
                .as_ref()
                .map(|inventory| (host.clone(), summarize(inventory, config)))
        })
        .collect()
}

/// "mlx5_0:1 ib 200G (ibp12s0, gpu0-3 pcie_switch)".
fn port_cell(port: &SelectedPort) -> String {
    let rate = port
        .rate_gbps
        .map_or_else(|| "?G".to_string(), |gbps| format!("{gbps}G"));
    let mut detail: Vec<String> = Vec::new();
    if !port.netdevs.is_empty() {
        detail.push(port.netdevs.join("/"));
    }
    let nearest = port
        .gpu_locality
        .values()
        .copied()
        .min()
        .filter(|locality| *locality != PciLocality::Unknown);
    if let Some(nearest) = nearest {
        let gpus: Vec<String> = port
            .gpu_locality
            .iter()
            .filter(|(_, locality)| **locality == nearest)
            .map(|(gpu, _)| format!("gpu{gpu}"))
            .collect();
        detail.push(format!("{} {}", gpus.join(","), nearest.label()));
    }
    let detail = if detail.is_empty() {
        String::new()
    } else {
        format!(" ({})", detail.join(", "))
    };
    format!(
        "{}:{} {} {rate}{detail}",
        port.device,
        port.port,
        port.link_layer.label()
    )
}

/// The host's `nccl_nics` outcome reason, when it failed or was skipped.
fn outcome_note(results: &RunResults, host: &str) -> String {
    results
        .hosts
        .get(host)
        .and_then(|obs| {
            obs.outcomes
                .iter()
                .find(|(test, _, _)| *test == TestId::NcclNics)
        })
        .map(|(_, _, outcome)| match outcome {
            TestOutcome::Passed => String::new(),
            TestOutcome::Failed { reason } => format!("FAIL: {reason}"),
            TestOutcome::Skipped { reason } => reason.clone(),
        })
        .unwrap_or_default()
}

/// Rendered when any host has an IB/RoCE port; a fleet without RDMA has
/// nothing to say here.
pub fn render(results: &RunResults, out: &mut dyn Write) -> Result<()> {
    let summaries = &results.calibration.nccl_nics;
    if summaries.values().all(NcclNicSummary::has_no_ports) {
        return Ok(());
    }
    let hca = results
        .nccl_env
        .as_ref()
        .and_then(|env| env.get(crate::nccl_ib::IB_HCA))
        .map_or_else(
            || "NCCL_IB_HCA unset".to_string(),
            |value| format!("NCCL_IB_HCA={value}"),
        );
    let mut table = new_table(&[
        "host",
        "ports nccl would use",
        "link layer",
        "ceiling gib/s",
        "excluded",
        "note",
    ]);
    for (host, summary) in summaries {
        let selected: Vec<String> = summary.selected.iter().map(port_cell).collect();
        let excluded: Vec<String> = summary
            .excluded
            .iter()
            .map(|port| format!("{}:{} {}", port.device, port.port, port.reason.describe()))
            .collect();
        table.add_row(vec![
            host.clone(),
            if selected.is_empty() {
                "-".to_string()
            } else {
                selected.join("\n")
            },
            summary.link_layer().label().to_string(),
            summary
                .ceiling_gib_per_sec
                .map_or_else(|| "-".to_string(), |gib| format!("{gib:.2}")),
            if excluded.is_empty() {
                "-".to_string()
            } else {
                excluded.join("\n")
            },
            outcome_note(results, host),
        ]);
    }
    section(out, &format!("nccl nics ({hca})"), &table)
}
