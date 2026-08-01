//! Self-deploy: put the running binary on each node as
//! `<remote_dir>/bin/gauntlet-agent`.
//!
//! Idempotent via content hash: compute sha256 of the local binary
//! (std::env::current_exe), compare with `sha256sum` of the remote path,
//! upload only on mismatch. A cross-architecture fleet is out of scope for
//! v1: bootstrap fails loudly if a node's `uname -m` differs from the
//! orchestrator's.

use std::io::Read;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use tracing::debug;

use super::session::{HostSession, single_quote};

/// Remote path of the agent binary relative to `remote_dir`.
pub const AGENT_RELPATH: &str = "bin/gauntlet-agent";

/// Ensure the agent binary on `session`'s host matches the local one.
/// Returns true if an upload happened.
pub async fn ensure_agent(session: &HostSession) -> Result<bool> {
    let local_exe = std::env::current_exe().context("locating the running gauntlet binary")?;
    let local = local_sha256(&local_exe)?;

    // The pipeline swallows a missing file: no remote agent yields an empty
    // digest, which never matches and therefore triggers an upload.
    let remote = session
        .exec(&format!(
            "sha256sum {path} 2>/dev/null | cut -d' ' -f1",
            path = single_quote(session.agent_path())
        ))
        .await
        .with_context(|| format!("hashing remote agent on {}", session.addr()))?;
    let remote = remote.trim();

    if !needs_upload(&local, remote) {
        debug!(host = %session.addr(), sha256 = %local, "agent already up to date");
        return Ok(false);
    }

    debug!(
        host = %session.addr(),
        local_sha256 = %local,
        remote_sha256 = %remote,
        "deploying agent"
    );
    session
        .upload(&local_exe, session.agent_path(), true)
        .await
        .with_context(|| format!("deploying agent to {}", session.addr()))?;
    Ok(true)
}

/// Upload decision: anything other than an exact digest match (including a
/// missing remote file, which reports an empty digest) means upload.
fn needs_upload(local: &str, remote: &str) -> bool {
    remote.is_empty() || !remote.eq_ignore_ascii_case(local)
}

/// sha256 of a local file, hex-encoded.
pub fn local_sha256(path: &std::path::Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("opening {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("reading {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        // Writing into a String is infallible.
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_digests_skip_the_upload() {
        assert!(!needs_upload("abc123", "abc123"));
        assert!(!needs_upload("abc123", "ABC123"));
    }

    #[test]
    fn missing_or_different_remote_triggers_upload() {
        assert!(needs_upload("abc123", ""));
        assert!(needs_upload("abc123", "abc124"));
    }

    #[test]
    fn sha256_matches_the_known_vector() {
        let dir = std::env::temp_dir().join(format!(
            "gauntlet-deploy-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock before epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("abc.bin");
        std::fs::write(&path, b"abc").expect("write");
        assert_eq!(
            local_sha256(&path).expect("hash"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}
