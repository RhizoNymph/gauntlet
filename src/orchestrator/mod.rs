//! Operator-side driver for `gauntlet run` and `gauntlet bootstrap`.
//!
//! Control flow of a run:
//!   load config -> open sessions (bounded concurrency, then held open) ->
//!   ensure agent deployed (hash check) -> per-phase:
//!     phases 0-2: spawn `agent run` on every host simultaneously, decode
//!       event streams into the collector;
//!     phase 3: tournament rounds of `agent peer` pairs, then the
//!       hierarchical NCCL sweeps (per-node, pairs, full fleet) with the
//!       uniqueId relayed from rank 0 by this process;
//!   -> collector -> analysis -> report to disk + terminal, exit code.
//!
//! Per-host failures (unreachable, agent Fatal, phase timeout from
//! `tests.phase_timeout_secs`) mark that host failed and the run continues;
//! the failure lands in the report instead of aborting the fleet.

pub mod bootstrap;
pub mod collect;
pub mod deploy;
pub mod session;

use anyhow::Result;

use crate::cli::RunArgs;

pub async fn run(args: RunArgs) -> Result<()> {
    let _ = args;
    todo!("agent A: implement")
}
