//! Self-deploy: put the running binary on each node as
//! `<remote_dir>/bin/gauntlet-agent`.
//!
//! Idempotent via content hash: compute sha256 of the local binary
//! (std::env::current_exe), compare with `sha256sum` of the remote path,
//! upload only on mismatch. A cross-architecture fleet is out of scope for
//! v1: bootstrap fails loudly if a node's `uname -m` differs from the
//! orchestrator's.

use anyhow::Result;

use super::session::HostSession;

/// Remote path of the agent binary relative to `remote_dir`.
pub const AGENT_RELPATH: &str = "bin/gauntlet-agent";

/// Ensure the agent binary on `session`'s host matches the local one.
/// Returns true if an upload happened.
pub async fn ensure_agent(session: &HostSession) -> Result<bool> {
    let _ = session;
    todo!("agent A: implement")
}

/// sha256 of a local file, hex-encoded.
pub fn local_sha256(path: &std::path::Path) -> Result<String> {
    let _ = path;
    todo!("agent A: implement")
}
