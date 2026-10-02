//! Self-deploy: put the running binary on each node as
//! `<remote_dir>/bin/gauntlet-agent`.
//!
//! Idempotent via content hash: compute sha256 of the local binary
//! (std::env::current_exe), compare with `sha256sum` of the remote path,
//! install only on mismatch. A cross-architecture fleet is out of scope for
//! v1: bootstrap fails loudly if a node's `uname -m` differs from the
//! orchestrator's.
//!
//! How the bytes travel depends on the launcher (`ensure_fleet`):
//!
//! - ssh: per host, sftp to a staging file, then `chmod` + `mv -f`.
//! - srun + `sbcast` (default): one `sbcast` of the binary to a staging
//!   file next to the agent dir, scoped by a carrier step to exactly the
//!   stale, connected nodes (Slurm's tree broadcast, no per-node upload
//!   from the orchestrator); if that fails, one sbcast per node so a bad
//!   node fails only its own deploy. Then a per-node install step: copy to
//!   a per-process temp name, `chmod`, `mv -f` into place, re-hash. The
//!   staging file is removed from every targeted node afterwards, whatever
//!   the outcome.
//! - srun + `shared`: the agent dir is one shared filesystem path; the
//!   orchestrator installs the binary there once, locally (temp name +
//!   rename), and every node only verifies the hash.

use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use super::fanout::fan_out;
use super::session::{HostSession, single_quote};
use super::transport::{Launcher, SrunLauncher, SrunTransportError};
use crate::launch::srun::{SrunDeploy, sbcast_staging_path};

/// Remote path of the agent binary relative to `remote_dir`.
pub const AGENT_RELPATH: &str = "bin/gauntlet-agent";

/// How an install reached a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeployMethod {
    Sftp,
    Sbcast,
    SharedPath,
}

/// Result of making one node's agent current.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeployOutcome {
    UpToDate,
    Installed(DeployMethod),
}

impl fmt::Display for DeployOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DeployOutcome::UpToDate => "up to date",
            DeployOutcome::Installed(DeployMethod::Sftp) => "uploaded",
            DeployOutcome::Installed(DeployMethod::Sbcast) => "installed via sbcast",
            DeployOutcome::Installed(DeployMethod::SharedPath) => "installed on the shared path",
        })
    }
}

/// Make every session's agent match the local binary. Results are in
/// `sessions` order; one host's failure never fails another's deploy.
/// `max_concurrent` bounds per-host operations; `timeout` bounds each
/// host's operations and each broadcast.
pub async fn ensure_fleet(
    launcher: &Launcher,
    sessions: &[Arc<HostSession>],
    max_concurrent: usize,
    timeout: Duration,
) -> Vec<Result<DeployOutcome>> {
    let local_exe = match std::env::current_exe().context("locating the running gauntlet binary") {
        Ok(path) => path,
        Err(error) => return sessions.iter().map(|_| Err(anyhow!("{error:#}"))).collect(),
    };
    let local = match local_sha256(&local_exe) {
        Ok(digest) => digest,
        Err(error) => return sessions.iter().map(|_| Err(anyhow!("{error:#}"))).collect(),
    };
    match launcher {
        Launcher::Ssh(_) => {
            let local_exe = Arc::new(local_exe);
            let local = Arc::new(local);
            per_host(sessions, max_concurrent, timeout, move |session| {
                let local_exe = Arc::clone(&local_exe);
                let local = Arc::clone(&local);
                async move { ensure_agent_sftp(&session, &local_exe, &local).await }
            })
            .await
        }
        Launcher::Srun(srun) => {
            ensure_fleet_srun(srun, sessions, &local_exe, &local, max_concurrent, timeout).await
        }
    }
}

/// Run `op` on every session with bounded concurrency and a per-host
/// timeout, collecting results in session order.
async fn per_host<T, F, Fut>(
    sessions: &[Arc<HostSession>],
    max_concurrent: usize,
    timeout: Duration,
    op: F,
) -> Vec<Result<T>>
where
    T: Send + 'static,
    F: Fn(Arc<HostSession>) -> Fut,
    Fut: std::future::Future<Output = Result<T>> + Send + 'static,
{
    fan_out(sessions.to_vec(), max_concurrent, |session| {
        let future = op(session);
        async move {
            match tokio::time::timeout(timeout, future).await {
                Ok(result) => result,
                Err(_) => Err(anyhow!("timed out after {}s", timeout.as_secs())),
            }
        }
    })
    .await
    .into_iter()
    .map(|slot| slot.unwrap_or_else(|| Err(anyhow!("deploy task aborted"))))
    .collect()
}

/// ssh: make `session`'s agent match the local binary, sftp-uploading on
/// a digest mismatch.
async fn ensure_agent_sftp(
    session: &HostSession,
    local_exe: &Path,
    local: &str,
) -> Result<DeployOutcome> {
    let remote = remote_sha256(session).await?;
    if !needs_upload(local, &remote) {
        debug!(host = %session.addr(), sha256 = %local, "agent already up to date");
        return Ok(DeployOutcome::UpToDate);
    }
    debug!(
        host = %session.addr(),
        local_sha256 = %local,
        remote_sha256 = %remote,
        "deploying agent"
    );
    session
        .upload(local_exe, session.agent_path(), true)
        .await
        .with_context(|| format!("deploying agent to {}", session.addr()))?;
    Ok(DeployOutcome::Installed(DeployMethod::Sftp))
}

/// The node's agent digest; empty when there is no agent yet.
async fn remote_sha256(session: &HostSession) -> Result<String> {
    // The pipeline swallows a missing file: no remote agent yields an empty
    // digest, which never matches and therefore triggers an upload.
    let remote = session
        .exec(&hash_script(session.agent_path()))
        .await
        .with_context(|| format!("hashing remote agent on {}", session.addr()))?;
    Ok(remote.trim().to_string())
}

fn hash_script(agent_path: &str) -> String {
    format!(
        "sha256sum {path} 2>/dev/null | cut -d' ' -f1",
        path = single_quote(agent_path)
    )
}

/// Per-node install from the sbcast staging file: copy to a temp name
/// unique to this shell (`$$`, so nodes sharing a filesystem never write
/// the same temp file), make it executable, rename over the agent (atomic;
/// a running old agent keeps its inode, no `ETXTBSY`), print the new
/// digest.
fn sbcast_install_script(staging: &str, agent_path: &str) -> String {
    let bin_dir = agent_path
        .rsplit_once('/')
        .map(|(parent, _)| parent)
        .filter(|parent| !parent.is_empty())
        .unwrap_or("/");
    let agent = single_quote(agent_path);
    format!(
        "set -e; mkdir -p {bin}; tmp={agent}.staging.$$; cp -f {staging} \"$tmp\"; \
         chmod 755 \"$tmp\"; mv -f \"$tmp\" {agent}; {hash}",
        bin = single_quote(bin_dir),
        staging = single_quote(staging),
        hash = hash_script(agent_path),
    )
}

async fn ensure_fleet_srun(
    srun: &Arc<SrunLauncher>,
    sessions: &[Arc<HostSession>],
    local_exe: &Path,
    local: &str,
    max_concurrent: usize,
    timeout: Duration,
) -> Vec<Result<DeployOutcome>> {
    // 1. Which nodes are stale?
    let digests = per_host(sessions, max_concurrent, timeout, |session| async move {
        remote_sha256(&session).await
    })
    .await;
    let mut results: Vec<Option<Result<DeployOutcome>>> = Vec::with_capacity(sessions.len());
    let mut stale: Vec<Arc<HostSession>> = Vec::new();
    let mut stale_slots: Vec<usize> = Vec::new();
    for (index, (session, digest)) in sessions.iter().zip(digests).enumerate() {
        match digest {
            Ok(remote) if !needs_upload(local, &remote) => {
                results.push(Some(Ok(DeployOutcome::UpToDate)))
            }
            Ok(_) => {
                results.push(None);
                stale.push(Arc::clone(session));
                stale_slots.push(index);
            }
            Err(error) => results.push(Some(Err(error))),
        }
    }
    if stale.is_empty() {
        return results.into_iter().flatten().collect();
    }

    // 2. Move the bytes once, then 3. install/verify per stale node.
    let installed = match srun.deploy() {
        SrunDeploy::Sbcast => {
            sbcast_and_install(srun, &stale, local_exe, local, max_concurrent, timeout).await
        }
        SrunDeploy::Shared => {
            shared_install(srun, &stale, local_exe, local, max_concurrent, timeout).await
        }
    };
    for (slot, outcome) in stale_slots.into_iter().zip(installed) {
        results[slot] = Some(outcome);
    }
    results
        .into_iter()
        .map(|slot| slot.unwrap_or_else(|| Err(anyhow!("deploy result missing"))))
        .collect()
}

async fn sbcast_and_install(
    srun: &Arc<SrunLauncher>,
    stale: &[Arc<HostSession>],
    local_exe: &Path,
    local: &str,
    max_concurrent: usize,
    timeout: Duration,
) -> Vec<Result<DeployOutcome>> {
    let staging = sbcast_staging_path(srun.dir(), srun.job_id(), local);
    info!(hosts = stale.len(), staging = %staging, "broadcasting agent with sbcast");
    let reached = scoped_broadcast(srun, stale, local_exe, &staging, max_concurrent, timeout).await;

    let staging = Arc::new(staging);
    let local = Arc::new(local.to_string());
    let installs: Vec<(Arc<HostSession>, Option<SrunTransportError>)> = stale
        .iter()
        .cloned()
        .zip(reached.into_iter().map(Result::err))
        .collect();
    let outcomes = fan_out(installs, max_concurrent, {
        let staging = Arc::clone(&staging);
        let local = Arc::clone(&local);
        move |(session, broadcast_error)| {
            let staging = Arc::clone(&staging);
            let local = Arc::clone(&local);
            async move {
                if let Some(error) = broadcast_error {
                    return Err(anyhow::Error::new(error)
                        .context(format!("broadcasting the agent to {}", session.addr())));
                }
                let script = sbcast_install_script(&staging, session.agent_path());
                let install = session.exec(&script);
                let digest = match tokio::time::timeout(timeout, install).await {
                    Ok(result) => {
                        result.with_context(|| format!("installing agent on {}", session.addr()))?
                    }
                    Err(_) => bail!(
                        "install on {} timed out after {}s",
                        session.addr(),
                        timeout.as_secs()
                    ),
                };
                verify(&session, &local, digest.trim(), DeployMethod::Sbcast)
            }
        }
    })
    .await
    .into_iter()
    .map(|slot| slot.unwrap_or_else(|| Err(anyhow!("deploy task aborted"))))
    .collect();
    // The staging copies are only a vehicle; leaving them would pile one
    // binary per job and version into the node's scratch space. Every node
    // the broadcast targeted is cleaned, whatever the broadcast or install
    // outcome (`rm -f` of a file that never arrived is harmless).
    let cleanup = per_host(stale, max_concurrent, timeout, {
        let staging = Arc::clone(&staging);
        move |session| {
            let staging = Arc::clone(&staging);
            async move {
                session
                    .exec(&format!("rm -f {}", single_quote(&staging)))
                    .await
                    .map(|_| ())
            }
        }
    })
    .await;
    for (session, result) in stale.iter().zip(cleanup) {
        if let Err(error) = result {
            debug!(host = %session.addr(), %error, "sbcast staging file not removed");
        }
    }
    outcomes
}

/// Broadcast to exactly the stale, connected hosts. First one sbcast over a
/// carrier step spanning all of them (Slurm's tree broadcast); if that
/// fails, retry per node so one bad node fails only its own deploy.
/// Returns one result per host, in order.
async fn scoped_broadcast(
    srun: &Arc<SrunLauncher>,
    stale: &[Arc<HostSession>],
    local_exe: &Path,
    staging: &str,
    max_concurrent: usize,
    timeout: Duration,
) -> Vec<Result<(), SrunTransportError>> {
    let hosts: Vec<&str> = stale.iter().map(|session| session.addr()).collect();
    let group = srun.broadcast(&hosts, local_exe, staging, timeout).await;
    let error = match group {
        Ok(()) => return stale.iter().map(|_| Ok(())).collect(),
        Err(error) => error,
    };
    if stale.len() == 1 || !isolates_per_node(&error) {
        let error = Arc::new(error);
        return stale
            .iter()
            .map(|_| {
                Err(SrunTransportError::Batched {
                    source: Arc::clone(&error),
                })
            })
            .collect();
    }
    warn!(
        hosts = stale.len(),
        %error,
        "fleet sbcast failed; retrying per node to isolate the failure"
    );
    let jobs: Vec<String> = hosts.iter().map(|host| host.to_string()).collect();
    let local_exe = Arc::new(local_exe.to_path_buf());
    let staging = Arc::new(staging.to_string());
    fan_out(jobs, max_concurrent, {
        let srun = Arc::clone(srun);
        move |host| {
            let srun = Arc::clone(&srun);
            let local_exe = Arc::clone(&local_exe);
            let staging = Arc::clone(&staging);
            async move {
                srun.broadcast(&[&host], &local_exe, &staging, timeout)
                    .await
            }
        }
    })
    .await
    .into_iter()
    .map(|slot| {
        slot.unwrap_or_else(|| {
            Err(SrunTransportError::Carrier {
                nodes: "?".into(),
                detail: "broadcast task aborted".into(),
            })
        })
    })
    .collect()
}

/// Whether a failed fleet broadcast can be narrowed down per node: a node
/// that rejected the transfer or its carrier step, or a slow one. A
/// missing Slurm client or an unusable source path would fail every node
/// the same way.
fn isolates_per_node(error: &SrunTransportError) -> bool {
    matches!(
        error,
        SrunTransportError::Sbcast { .. }
            | SrunTransportError::Carrier { .. }
            | SrunTransportError::Timeout { .. }
            | SrunTransportError::Squeue { .. }
            | SrunTransportError::Scancel { .. }
    )
}

async fn shared_install(
    srun: &Arc<SrunLauncher>,
    stale: &[Arc<HostSession>],
    local_exe: &Path,
    local: &str,
    max_concurrent: usize,
    timeout: Duration,
) -> Vec<Result<DeployOutcome>> {
    let dest = PathBuf::from(srun.dir().as_str()).join(AGENT_RELPATH);
    info!(dest = %dest.display(), "installing agent on the shared path");
    if let Err(error) = install_local(local_exe, &dest) {
        let message = format!("{error:#}");
        return stale.iter().map(|_| Err(anyhow!("{message}"))).collect();
    }
    let local = Arc::new(local.to_string());
    per_host(stale, max_concurrent, timeout, move |session| {
        let local = Arc::clone(&local);
        async move {
            let digest = remote_sha256(&session).await?;
            verify(&session, &local, &digest, DeployMethod::SharedPath).map_err(|error| {
                error.context(
                    "the node does not see the orchestrator's copy: is [launch.srun] dir \
                     really on a shared filesystem?",
                )
            })
        }
    })
    .await
}

fn verify(
    session: &HostSession,
    local: &str,
    remote: &str,
    method: DeployMethod,
) -> Result<DeployOutcome> {
    if needs_upload(local, remote) {
        bail!(
            "agent on {} has sha256 {:?} after install, expected {local}",
            session.addr(),
            remote
        );
    }
    Ok(DeployOutcome::Installed(method))
}

/// Install `source` at `dest` on this machine: copy to a temp sibling,
/// make it executable, rename into place.
fn install_local(source: &Path, dest: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let parent = dest
        .parent()
        .with_context(|| format!("{} has no parent directory", dest.display()))?;
    std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let staging = parent.join(format!(".gauntlet-agent.staging.{}", std::process::id()));
    std::fs::copy(source, &staging)
        .with_context(|| format!("copying {} to {}", source.display(), staging.display()))?;
    std::fs::set_permissions(&staging, std::fs::Permissions::from_mode(0o755))
        .with_context(|| format!("chmod {}", staging.display()))?;
    std::fs::rename(&staging, dest)
        .with_context(|| format!("renaming {} to {}", staging.display(), dest.display()))?;
    Ok(())
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

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gauntlet-deploy-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock before epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// The per-node install script, run by a real `sh` against a fake
    /// staging file, installs an executable copy and prints its digest —
    /// even with a path a shell would otherwise split or expand.
    #[test]
    fn sbcast_install_script_installs_atomically_and_reports_the_digest() {
        let dir = scratch("install");
        let staging = dir.join("stage $x 'q'.bin");
        std::fs::write(&staging, b"abc").expect("write staging");
        let agent = dir.join("agent dir/bin/gauntlet-agent");
        let script = sbcast_install_script(
            staging.to_str().expect("utf8"),
            agent.to_str().expect("utf8"),
        );
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .output()
            .expect("run sh");
        assert!(
            output.status.success(),
            "{script}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&agent)
            .expect("installed")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o755);
        // No temp file left behind next to the agent.
        let leftovers: Vec<_> = std::fs::read_dir(agent.parent().expect("bin"))
            .expect("bin dir")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name())
            .collect();
        assert_eq!(leftovers, vec![std::ffi::OsString::from("gauntlet-agent")]);
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn local_install_replaces_the_destination_executable() {
        let dir = scratch("local");
        let source = dir.join("src.bin");
        std::fs::write(&source, b"new").expect("write");
        let dest = dir.join("shared/bin/gauntlet-agent");
        std::fs::create_dir_all(dest.parent().expect("parent")).expect("mkdir");
        std::fs::write(&dest, b"old").expect("old");
        install_local(&source, &dest).expect("install");
        assert_eq!(std::fs::read(&dest).expect("read"), b"new");
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&dest).expect("meta").permissions().mode() & 0o777,
            0o755
        );
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn outcomes_describe_how_the_agent_arrived() {
        assert_eq!(DeployOutcome::UpToDate.to_string(), "up to date");
        assert_eq!(
            DeployOutcome::Installed(DeployMethod::Sftp).to_string(),
            "uploaded"
        );
        assert_eq!(
            DeployOutcome::Installed(DeployMethod::Sbcast).to_string(),
            "installed via sbcast"
        );
    }
}
