//! The "nccl nics" table section: per host, the IB/RoCE ports NCCL would
//! use (`hosts.*.nccl_nics`, recorded by the orchestrator's inventory
//! derivation), their link layer, the ceiling, the excluded ports and the
//! `nccl_nics` outcome note.
//!
//! A projection only: every string comes from the recorded summary's own
//! describers (`describe_hca`, `ExcludedPort::describe`,
//! `ceiling_unknown_reason`), the same ones the outcome reason uses.

use std::io::Write;

use anyhow::Result;

use super::{RunResults, new_table, section};
use crate::proto::nccl_nics::{ExcludedPort, IB_HCA, SelectedPort, describe_hca};
use crate::proto::{NcclNicSummary, TestId, TestOutcome};

/// "mlx5_0:1 infiniband 200G (ibp12s0, gpu0,gpu1 pcie_switch)".
fn port_cell(port: &SelectedPort) -> String {
    let mut detail: Vec<String> = Vec::new();
    if !port.netdevs.is_empty() {
        detail.push(port.netdevs.join("/"));
    }
    if let Some((nearest, gpus)) = port.nearest_gpus() {
        let gpus: Vec<String> = gpus.iter().map(|gpu| format!("gpu{gpu}")).collect();
        detail.push(format!("{} {}", gpus.join(","), nearest.label()));
    }
    let detail = if detail.is_empty() {
        String::new()
    } else {
        format!(" ({})", detail.join(", "))
    };
    format!(
        "{}:{} {} {}{detail}",
        port.device,
        port.port,
        port.link_layer.label(),
        port.rate.describe()
    )
}

/// The outcome reason (FAIL / skip), the plugin caveat and the reason a
/// ceiling is unknown, one per line.
fn note(summary: &NcclNicSummary, outcome: Option<&TestOutcome>) -> String {
    let mut lines = Vec::new();
    match outcome {
        Some(TestOutcome::Failed { reason }) => lines.push(format!("FAIL: {reason}")),
        Some(TestOutcome::Skipped { reason }) => lines.push(reason.clone()),
        Some(TestOutcome::Passed) | None => {}
    }
    lines.extend(summary.net_caveat());
    lines.extend(summary.ceiling_unknown_reason());
    lines.join("\n")
}

fn or_dash(lines: Vec<String>) -> String {
    if lines.is_empty() {
        "-".to_string()
    } else {
        lines.join("\n")
    }
}

/// Rendered when any host recorded an IB/RoCE port; a fleet without RDMA
/// has nothing to say here.
pub fn render(results: &RunResults, out: &mut dyn Write) -> Result<()> {
    let summaries: Vec<(&String, &NcclNicSummary, Option<&TestOutcome>)> = results
        .hosts
        .iter()
        .filter_map(|(host, obs)| {
            let summary = obs.nccl_nics.as_ref()?;
            let outcome = obs
                .outcomes
                .iter()
                .find(|(test, _, _)| *test == TestId::NcclNics)
                .map(|(_, _, outcome)| outcome);
            Some((host, summary, outcome))
        })
        .collect();
    if summaries
        .iter()
        .all(|(_, summary, _)| summary.has_no_ports())
    {
        return Ok(());
    }
    let hca = results
        .nccl_env
        .as_ref()
        .and_then(|env| env.get(IB_HCA))
        .map(String::as_str);
    let mut table = new_table(&[
        "host",
        "ports nccl would use",
        "link layer",
        "ceiling gib/s",
        "excluded",
        "note",
    ]);
    for (host, summary, outcome) in summaries {
        table.add_row(vec![
            host.clone(),
            or_dash(summary.selected.iter().map(port_cell).collect()),
            summary.link_layer().label().to_string(),
            summary
                .ceiling_gib_per_sec()
                .map_or_else(|| "-".to_string(), |gib| format!("{gib:.2}")),
            or_dash(
                summary
                    .excluded
                    .iter()
                    .map(ExcludedPort::describe)
                    .collect(),
            ),
            note(summary, outcome),
        ]);
    }
    section(out, &format!("nccl nics ({})", describe_hca(hca)), &table)
}
