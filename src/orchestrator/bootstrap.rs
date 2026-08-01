//! `gauntlet bootstrap`: make every node ready without manual setup.
//!
//! Steps per host (bounded-concurrency fan-out):
//!   1. connectivity — open the ssh session, fail fast with a clear reason;
//!   2. arch check — `uname -m` must match the orchestrator's;
//!   3. deploy — `deploy::ensure_agent`;
//!   4. probe — run `agent probe`, parse the InventorySnapshot: GPU count,
//!      driver present, CUDA libs resolvable, IB ports up, clock sync;
//!   5. tuning (only with --tune; each step is sudo-gated and a refusal is
//!      reported, not fatal): `nvidia-smi -pm 1` (persistence mode),
//!      cpu governor -> performance.
//!
//! Output: a readiness matrix (host x check: ok / warn / fail + detail) to
//! the terminal, non-zero exit if any host hard-failed. The matrix is the
//! contract: after a clean bootstrap, `gauntlet run` has no setup left to
//! discover.

use anyhow::Result;

use serde::{Deserialize, Serialize};

use crate::cli::BootstrapArgs;
use crate::proto::InventorySnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Ok,
    Warn,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostReadiness {
    pub host: String,
    pub checks: Vec<ReadinessCheck>,
    pub inventory: Option<InventorySnapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadinessCheck {
    /// "connectivity", "arch", "deploy", "gpu_driver", "clock_sync",
    /// "persistence_mode", "governor", ...
    pub name: String,
    pub status: CheckStatus,
    pub detail: String,
}

pub async fn run(args: BootstrapArgs) -> Result<()> {
    let _ = args;
    todo!("agent A: implement")
}
