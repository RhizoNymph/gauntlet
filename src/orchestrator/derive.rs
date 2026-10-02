//! Orchestrator-side tests derived from a host's inventory as it arrives.
//!
//! The agent reports facts; judging them needs orchestrator config (the
//! gpu_idle threshold, the run's NCCL env). Every such derivation goes
//! through [`derive_inventory_events`], whose events follow the inventory
//! through the normal sink, so the collector records them (and partial
//! snapshots show them) exactly like agent events. The report only ever
//! reads these recorded results; it never re-derives.

use crate::config::FleetConfig;
use crate::nccl_ib::{NcclIbConfig, summarize};
use crate::proto::{AgentEvent, InventorySnapshot, TestId, gpu_idle_outcomes};

/// The config slice the derivations need, resolved once per phase.
#[derive(Debug, Clone)]
pub struct InventoryDerivation {
    pub gpu_idle_max_used_mib: u64,
    /// `None` only when `[nccl]` failed to validate, which
    /// `FleetConfig::load` rules out for a run; then no nccl_nics result is
    /// derived rather than a guessed one.
    pub nccl_ib: Option<NcclIbConfig>,
}

impl InventoryDerivation {
    pub fn from_config(config: &FleetConfig) -> Self {
        Self {
            gpu_idle_max_used_mib: config.thresholds.gpu_idle_max_used_mib,
            nccl_ib: config.nccl_env().ok().map(NcclIbConfig::from_env),
        }
    }
}

/// Everything derived from one inventory, in emission order:
/// - one `gpu_idle` outcome per GPU (index order);
/// - when the NCCL env is known: the node-scope `nccl_nics` outcome, the
///   `nccl_nics.ceiling_gib_per_sec` metric when it applies, and the
///   host's `NcclNics` summary.
pub fn derive_inventory_events(
    snapshot: &InventorySnapshot,
    derivation: &InventoryDerivation,
) -> Vec<AgentEvent> {
    let mut events: Vec<AgentEvent> = gpu_idle_outcomes(snapshot, derivation.gpu_idle_max_used_mib)
        .into_iter()
        .map(|(scope, outcome)| AgentEvent::Outcome {
            test: TestId::GpuIdle,
            scope,
            outcome,
        })
        .collect();
    if let Some(nccl_ib) = &derivation.nccl_ib {
        let summary = summarize(snapshot, nccl_ib);
        events.push(AgentEvent::Outcome {
            test: TestId::NcclNics,
            scope: crate::proto::Scope::Node,
            outcome: summary.outcome(),
        });
        if let Some(record) = summary.ceiling_metric() {
            events.push(AgentEvent::Metric { record });
        }
        events.push(AgentEvent::NcclNics {
            summary: Box::new(summary),
        });
    }
    events
}
