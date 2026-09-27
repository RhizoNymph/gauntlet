//! Intra-node NCCL sweep, orchestrator side: the first level of the
//! phase-3 hierarchy, run before the pairwise and fleet levels.
//!
//! The sweep is node-local and single-process (`ncclCommInitAll`), so it
//! needs no rendezvous and no relay: it rides the ordinary per-node fan-out
//! (`node_phase`) as the node-local part of `Phase::Network`, with the
//! network-phase task spec carrying `nccl_intranode`. Only GPU-bearing
//! hosts are considered — a CPU-only host has no intra-node level, and
//! dispatching to it would turn a missing driver into a Failed outcome.
//! Among those, `eligibility` decides from the inventory: a host whose
//! libnccl cannot load, or where CUDA can open fewer than two GPUs
//! (`cuda_visible_gpus`, the same count the fleet world sizes rank blocks
//! from), gets explicit Skipped outcomes from here with the reason instead
//! of being dispatched.

use std::collections::BTreeMap;
use std::sync::Arc;

use tracing::{info, warn};

use super::nccl::nccl_loadable;
use super::session::HostSession;
use super::{ObservationSink, gpu_bearing_hosts, node_phase};
use crate::agent::intranode::{MIN_GPUS, node_outcomes};
use crate::config::FleetConfig;
use crate::proto::{AgentEvent, InventorySnapshot, Phase, Scope, TestOutcome};

/// Whether a GPU-bearing host runs the intra-node sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Eligibility {
    /// Dispatch; the agent gates again on the device count it actually
    /// sees (its answer is authoritative).
    Run,
    /// Record Skipped outcomes on both intra-node tests with this reason.
    Skip(String),
}

/// Decide from the inventory. Without a CUDA-visible count (`None`: the
/// agent could not ask CUDA) the host is still dispatched — on a
/// GPU-bearing host that is a driver problem, which the agent reports as a
/// Failed outcome, a real finding.
pub(super) fn eligibility(inventory: &InventorySnapshot) -> Eligibility {
    if !nccl_loadable(inventory) {
        return Eligibility::Skip("libnccl is not loadable on this host (inventory probe)".into());
    }
    match inventory.cuda_visible_gpus {
        Some(visible) if visible < MIN_GPUS => Eligibility::Skip(format!(
            "intra-node nccl sweep needs at least {MIN_GPUS} GPUs, CUDA can open {visible}"
        )),
        _ => Eligibility::Run,
    }
}

pub(super) async fn intranode_sweep(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
    sink: &ObservationSink,
) {
    let gpu_hosts = gpu_bearing_hosts(sessions, inventories).await;
    let mut eligible = Vec::with_capacity(gpu_hosts.len());
    let mut skipped = 0usize;
    for session in gpu_hosts {
        // `gpu_bearing_hosts` only returns hosts with an inventory.
        let decision = inventories
            .get(session.addr())
            .map_or(Eligibility::Run, eligibility);
        match decision {
            Eligibility::Run => eligible.push(session),
            Eligibility::Skip(reason) => {
                skipped += 1;
                for (test, outcome) in
                    node_outcomes(|r| TestOutcome::Skipped { reason: r }, &reason)
                {
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
        }
    }
    if skipped > 0 {
        warn!(
            hosts = skipped,
            "GPU hosts skip the intra-node sweep (no libnccl or fewer than 2 CUDA-visible GPUs)"
        );
    }
    if eligible.is_empty() {
        info!("no eligible GPU hosts; skipping the intra-node NCCL sweep");
        return;
    }
    info!(hosts = eligible.len(), "intra-node NCCL sweep");
    // Network-phase task spec: the agent's network arm runs exactly the
    // node-local sweep (`nccl_intranode` is set whenever this step runs,
    // since both follow `tests.nccl_intranode`).
    node_phase(config, &eligible, Phase::Network, sink).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inventory(libs: &[(&str, bool)], cuda: Option<u32>) -> InventorySnapshot {
        InventorySnapshot {
            hostname: "n1".into(),
            kernel: "6.8.0".into(),
            cpu_model: "cpu".into(),
            logical_cores: 8,
            numa_nodes: 1,
            mem_total_bytes: 1 << 34,
            cpu_governor: None,
            clock_offset_ms: None,
            nvidia_driver: None,
            cuda_version: None,
            gpus: vec![],
            nics: vec![],
            ib_ports: vec![],
            xid_errors: vec![],
            gpu_libs: libs
                .iter()
                .map(|(lib, ok)| (lib.to_string(), *ok))
                .collect(),
            cuda_visible_gpus: cuda,
        }
    }

    #[test]
    fn multi_gpu_hosts_with_nccl_run() {
        assert_eq!(eligibility(&inventory(&[], Some(8))), Eligibility::Run);
        assert_eq!(
            eligibility(&inventory(&[("nccl", true)], Some(2))),
            Eligibility::Run
        );
        // No CUDA count: the agent's own gate is authoritative.
        assert_eq!(eligibility(&inventory(&[], None)), Eligibility::Run);
    }

    #[test]
    fn fewer_than_two_cuda_visible_gpus_skip_with_the_count() {
        for visible in [0, 1] {
            let Eligibility::Skip(reason) = eligibility(&inventory(&[], Some(visible))) else {
                panic!("{visible} CUDA-visible GPUs must skip");
            };
            assert!(reason.contains("at least 2"), "{reason}");
            assert!(reason.contains(&format!("can open {visible}")), "{reason}");
        }
    }

    #[test]
    fn an_unloadable_libnccl_skips_regardless_of_gpus() {
        let Eligibility::Skip(reason) =
            eligibility(&inventory(&[("cuda", true), ("nccl", false)], Some(8)))
        else {
            panic!("no libnccl must skip");
        };
        assert!(reason.contains("libnccl"), "{reason}");
    }
}
