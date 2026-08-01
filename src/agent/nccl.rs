//! Phase 3b: NCCL collective sweeps (`gauntlet agent nccl`).
//!
//! Rendezvous: the orchestrator sends rank 0 a `GenerateId` directive; the
//! agent prints the `NcclUniqueId` JSON on stdout. The orchestrator then
//! sends every rank a `Participate` directive carrying that id. No
//! filesystem or TCP-store dependency.
//!
//! Sweep: for each size in `sizes`, `iters_per_size` all-reduce (and
//! all-gather) iterations after a warmup; emit per-size elapsed micros as
//! `nccl_all_reduce.elapsed_us` metrics with the size recorded in a
//! companion `msg_bytes` metric under the same scope, plus computed bus
//! bandwidth `bus_gib_per_sec` (the alpha-beta fit itself happens
//! orchestrator-side in `analysis::fit`). Rank 0 emits the events; other
//! ranks emit only Fatal on error.
//!
//! One process per node, one GPU per rank for v1 (world_size == node count;
//! intra-node NVLink is covered by the p2p test). `socket_ifname`, when
//! set, is exported as NCCL_SOCKET_IFNAME before init.

use anyhow::Result;

/// Read an `NcclDirective` JSON document from stdin and execute it.
pub fn run_from_stdin() -> Result<()> {
    todo!("agent C: implement")
}

#[cfg(feature = "gpu")]
pub mod imp {
    use anyhow::Result;

    use crate::proto::{NcclDirective, NcclUniqueId};

    pub fn generate_id() -> Result<NcclUniqueId> {
        todo!("agent C: implement")
    }

    pub fn participate(directive: &NcclDirective) -> Result<()> {
        let _ = directive;
        todo!("agent C: implement")
    }
}
