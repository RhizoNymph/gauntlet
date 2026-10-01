//! The "gpus in use" report section: every Failed `gpu_idle` outcome with
//! its process detail.
//!
//! A projection of the results document, never a second source of truth:
//! which GPUs failed comes from `hosts.*.outcomes`, the per-process detail
//! from the same host's inventory occupancy. The host table only counts
//! failed outcomes, so without this section a busy GPU would read as an
//! anonymous "1 fail".

use std::io::Write;

use anyhow::Result;

use super::{RunResults, new_table, section};
use crate::proto::{GpuOccupancy, GpuProcess, Scope, TestId, TestOutcome};

/// One busy GPU, ready to render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInUse {
    pub host: String,
    pub gpu: u32,
    /// The GPU's occupancy as inventoried, when the host's snapshot has it.
    pub occupancy: Option<GpuOccupancy>,
    /// The outcome's reason (the fallback when no occupancy is at hand).
    pub reason: String,
}

/// Failed `gpu_idle` outcomes in host, then GPU order.
pub fn gpus_in_use(results: &RunResults) -> Vec<GpuInUse> {
    let mut busy = Vec::new();
    for (host, obs) in &results.hosts {
        for (test, scope, outcome) in &obs.outcomes {
            let (TestId::GpuIdle, Scope::Gpu { index }, TestOutcome::Failed { reason }) =
                (test, scope, outcome)
            else {
                continue;
            };
            let occupancy = obs
                .inventory
                .as_ref()
                .and_then(|inventory| inventory.gpus.iter().find(|gpu| gpu.index == *index))
                .map(|gpu| gpu.occupancy.clone());
            busy.push(GpuInUse {
                host: host.clone(),
                gpu: *index,
                occupancy,
                reason: reason.clone(),
            });
        }
    }
    busy.sort_by(|a, b| (&a.host, a.gpu).cmp(&(&b.host, b.gpu)));
    busy
}

/// "23264 / 24576 MiB", "23264 MiB", or "-".
fn memory_cell(occupancy: Option<&GpuOccupancy>) -> String {
    match occupancy.map(|occ| (occ.memory_used_mib, occ.memory_total_mib)) {
        Some((Some(used), Some(total))) => format!("{used} / {total} MiB"),
        Some((Some(used), None)) => format!("{used} MiB"),
        _ => "-".to_string(),
    }
}

/// One process per line; the outcome reason when there is no process to
/// name (memory over the threshold with an empty or unknown list).
fn processes_cell(entry: &GpuInUse) -> String {
    let processes: Vec<String> = entry
        .occupancy
        .as_ref()
        .and_then(|occ| occ.compute_processes.as_ref())
        .map(|list| list.iter().map(GpuProcess::describe).collect())
        .unwrap_or_default();
    if processes.is_empty() {
        entry.reason.clone()
    } else {
        processes.join("\n")
    }
}

pub(super) fn render(results: &RunResults, out: &mut dyn Write) -> Result<()> {
    let busy = gpus_in_use(results);
    if busy.is_empty() {
        return Ok(());
    }
    let mut table = new_table(&["gpu", "memory used", "compute processes"]);
    for entry in &busy {
        table.add_row(vec![
            format!("{}:gpu{}", entry.host, entry.gpu),
            memory_cell(entry.occupancy.as_ref()),
            processes_cell(entry),
        ]);
    }
    section(out, "gpus in use (gpu_idle)", &table)
}
