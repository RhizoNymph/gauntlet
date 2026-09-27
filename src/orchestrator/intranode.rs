//! Intra-node NCCL sweep, orchestrator side: the first level of the
//! phase-3 hierarchy, run before the pairwise and fleet levels.
//!
//! The sweep is node-local and single-process (`ncclCommInitAll`), so it
//! needs no rendezvous and no relay: it rides the ordinary per-node fan-out
//! (`node_phase`) as the node-local part of `Phase::Network`, with the
//! network-phase task spec carrying `nccl_intranode`. Only GPU-bearing
//! hosts are sent it — a CPU-only host has no intra-node level, and
//! dispatching to it would turn a missing driver into a Failed outcome.
//! GPU hosts whose inventory says libnccl cannot load get explicit Skipped
//! outcomes from here instead of a loader failure on the node.

use std::collections::BTreeMap;
use std::sync::Arc;

use tracing::{info, warn};

use super::nccl::nccl_loadable;
use super::session::HostSession;
use super::{ObservationSink, gpu_bearing_hosts, node_phase};
use crate::agent::intranode::node_outcomes;
use crate::config::FleetConfig;
use crate::proto::{AgentEvent, InventorySnapshot, Phase, Scope, TestOutcome};

/// Reason recorded against GPU hosts that cannot load libnccl.
const NO_NCCL: &str = "libnccl is not loadable on this host (inventory probe)";

pub(super) async fn intranode_sweep(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
    sink: &ObservationSink,
) {
    let gpu_hosts = gpu_bearing_hosts(sessions, inventories).await;
    let (eligible, without_nccl): (Vec<_>, Vec<_>) = gpu_hosts
        .into_iter()
        .partition(|session| nccl_loadable(inventories.get(session.addr())));

    for session in &without_nccl {
        for (test, outcome) in node_outcomes(|r| TestOutcome::Skipped { reason: r }, NO_NCCL) {
            sink.event(
                session.addr(),
                AgentEvent::Outcome {
                    test,
                    scope: Scope::Node,
                    outcome,
                },
            );
        }
    }
    if !without_nccl.is_empty() {
        warn!(
            hosts = without_nccl.len(),
            "GPU hosts without a loadable libnccl skip the intra-node sweep"
        );
    }
    if eligible.is_empty() {
        info!("no NCCL-capable GPU hosts; skipping the intra-node NCCL sweep");
        return;
    }
    info!(hosts = eligible.len(), "intra-node NCCL sweep");
    // Network-phase task spec: the agent's network arm runs exactly the
    // node-local sweep (`nccl_intranode` is set whenever this step runs,
    // since both follow `tests.nccl_intranode`).
    node_phase(config, &eligible, Phase::Network, sink).await;
}
