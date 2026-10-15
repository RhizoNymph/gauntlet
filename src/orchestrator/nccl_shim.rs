//! The `<remote_dir>/lib/libnccl.so -> libnccl.so.2` shim, shared by
//! `bootstrap` (which reports it as a readiness check) and `run` (which
//! re-creates it when it went missing).
//!
//! cudarc's loader does not search libnccl.so.2 (NCCL's actual runtime
//! soname), so a node with only the runtime package installed cannot run
//! the NCCL sweeps. Every agent invocation carries
//! `LD_LIBRARY_PATH=<remote_dir>/lib`, so a symlink there makes
//! dlopen("libnccl.so") resolve. With the node-local default remote_dir
//! (`/tmp/gauntlet-$USER`) the shim disappears on reboot or tmpfiles aging
//! while the agent binary is simply re-uploaded, so `run` must be able to
//! rebuild it without a prior bootstrap.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{debug, info, warn};

use super::session::{HostSession, single_quote};
use crate::proto::{InventorySnapshot, Phase};

/// Remote script that symlinks the NCCL runtime soname under a name cudarc
/// searches, inside the agent's own directory (no sudo, no system change).
/// `{lib_dir}` is substituted with `<remote_dir>/lib`. Prints the target.
pub(super) const NCCL_SHIM_SCRIPT: &str = r#"mkdir -p {lib_dir} && target=""; for p in $(ldconfig -p 2>/dev/null | awk '/libnccl\.so\.2 /{print $NF}') /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib64/libnccl.so.2 /usr/lib/libnccl.so.2; do if [ -e "$p" ]; then target="$p"; break; fi; done; if [ -n "$target" ]; then ln -sf "$target" {lib_dir}/libnccl.so && echo "$target"; else exit 3; fi"#;

/// Does this inventory describe the runtime-only NCCL install (libnccl.so.2
/// present, but no name cudarc's loader searches)? The probe runs under the
/// shim's LD_LIBRARY_PATH, so an existing working shim reads as loadable.
pub(super) fn nccl_shim_needed(inventory: &InventorySnapshot) -> bool {
    !inventory.gpus.is_empty()
        && inventory.gpu_libs.get("nccl") == Some(&false)
        && inventory.gpu_libs.get("nccl_runtime") == Some(&true)
}

/// Whether a run's phases use NCCL at all (network sweeps, overlap); runs
/// that do not never touch the shim.
pub(super) fn phases_use_nccl(phases: &[Phase]) -> bool {
    phases
        .iter()
        .any(|phase| matches!(phase, Phase::Network | Phase::Overlap))
}

/// Succeeds iff `<lib_dir>/libnccl.so` resolves to an existing file (`-e`
/// follows the link, so a dangling shim counts as missing).
pub(super) fn shim_present_command(lib_dir: &str) -> String {
    format!("[ -e {} ]", single_quote(&format!("{lib_dir}/libnccl.so")))
}

/// What `ensure_shim` did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ShimOutcome {
    /// The node does not need a shim (NCCL loadable, no NCCL runtime, or
    /// no GPUs).
    NotNeeded,
    /// Shim created; the re-probe finds libnccl loadable.
    Built { target: String },
    /// Shim created, but libnccl is still not loadable.
    StillUnloadable { target: String },
    /// The shim script failed (no libnccl.so.2 found, unwritable lib dir).
    CreateFailed { error: String },
    /// The shim was created but the confirming re-probe failed.
    ReprobeFailed { target: String, error: String },
}

/// Run `agent probe` and parse its `InventorySnapshot`.
pub(super) async fn probe(session: &HostSession) -> Result<InventorySnapshot> {
    let output = session.run_agent_capture(&["probe"], None).await?;
    if !output.success() {
        bail!("agent probe failed: {}", output.detail());
    }
    serde_json::from_str(output.stdout.trim()).context("parsing InventorySnapshot from agent probe")
}

/// Build the shim when `inventory` says the node needs one, then re-probe
/// so the returned inventory reflects reality (the original one when the
/// shim could not be built or confirmed).
pub(super) async fn ensure_shim(
    session: &HostSession,
    inventory: InventorySnapshot,
) -> (InventorySnapshot, ShimOutcome) {
    if !nccl_shim_needed(&inventory) {
        return (inventory, ShimOutcome::NotNeeded);
    }
    let script = NCCL_SHIM_SCRIPT.replace("{lib_dir}", &format!("{}/lib", session.remote_dir()));
    let target = match session.exec(&script).await {
        Ok(stdout) => stdout.trim().to_string(),
        Err(error) => {
            let error = format!("{error:#}");
            return (inventory, ShimOutcome::CreateFailed { error });
        }
    };
    match probe(session).await {
        Ok(reprobed) if reprobed.gpu_libs.get("nccl") == Some(&true) => {
            (reprobed, ShimOutcome::Built { target })
        }
        Ok(reprobed) => (reprobed, ShimOutcome::StillUnloadable { target }),
        Err(error) => (
            inventory,
            ShimOutcome::ReprobeFailed {
                target,
                error: format!("{error:#}"),
            },
        ),
    }
}

/// `run`'s check: a present shim costs one `[ -e ]`; otherwise probe the
/// node and rebuild the shim if it needs one (`ensure_shim`). Returns the
/// probed inventory when a probe happened.
pub(super) async fn ensure_shim_for_run(
    session: &HostSession,
) -> Result<(Option<InventorySnapshot>, ShimOutcome)> {
    let lib_dir = format!("{}/lib", session.remote_dir());
    let present = session
        .exec_capture(&shim_present_command(&lib_dir))
        .await
        .context("checking for the NCCL shim")?;
    if present.success() {
        return Ok((None, ShimOutcome::NotNeeded));
    }
    let inventory = probe(session).await?;
    let (inventory, outcome) = ensure_shim(session, inventory).await;
    Ok((Some(inventory), outcome))
}

/// `run`'s deploy-step pass: on every host, make sure the shim exists when
/// the node needs one (`ensure_shim_for_run`), `max_concurrent` hosts at a
/// time. Advisory like bootstrap's `nccl_shim` column: a failure is logged
/// and the NCCL steps report the unloadable library themselves; it never
/// fails a host. Returns the inventories probed on the way, which spare
/// the network phase its own probe.
pub(super) async fn ensure_fleet_shims(
    sessions: &[Arc<HostSession>],
    max_concurrent: usize,
) -> BTreeMap<String, InventorySnapshot> {
    let permits = Arc::new(Semaphore::new(max_concurrent.max(1)));
    let mut tasks = JoinSet::new();
    for session in sessions {
        let session = Arc::clone(session);
        let permits = Arc::clone(&permits);
        tasks.spawn(async move {
            let _permit = permits.acquire_owned().await.ok();
            let addr = session.addr().to_string();
            let result = ensure_shim_for_run(&session).await;
            (addr, result)
        });
    }
    let mut inventories = BTreeMap::new();
    while let Some(joined) = tasks.join_next().await {
        let (host, result) = match joined {
            Ok(done) => done,
            Err(error) => {
                warn!(%error, "nccl shim task did not complete");
                continue;
            }
        };
        match result {
            Ok((inventory, outcome)) => {
                log_outcome(&host, &outcome);
                if let Some(inventory) = inventory {
                    inventories.insert(host, inventory);
                }
            }
            Err(error) => {
                warn!(host = %host, error = %format!("{error:#}"), "nccl shim check failed")
            }
        }
    }
    inventories
}

fn log_outcome(host: &str, outcome: &ShimOutcome) {
    match outcome {
        ShimOutcome::NotNeeded => debug!(host, "nccl shim not needed or present"),
        ShimOutcome::Built { target } => info!(host, target, "nccl shim rebuilt"),
        ShimOutcome::StillUnloadable { target } => {
            warn!(
                host,
                target, "nccl shim created but libnccl still not loadable"
            )
        }
        ShimOutcome::CreateFailed { error } => warn!(host, error, "nccl shim creation failed"),
        ShimOutcome::ReprobeFailed { target, error } => {
            warn!(host, target, error, "re-probe after nccl shim failed")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gauntlet-shim-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn present(lib_dir: &std::path::Path) -> bool {
        std::process::Command::new("sh")
            .arg("-c")
            .arg(shim_present_command(lib_dir.to_str().expect("utf8")))
            .status()
            .expect("run sh")
            .success()
    }

    #[test]
    fn a_missing_or_dangling_shim_reads_as_absent() {
        let base = scratch("present");
        let lib_dir = base.join("lib dir");
        assert!(!present(&lib_dir), "no lib dir yet");
        std::fs::create_dir_all(&lib_dir).expect("lib dir");
        assert!(!present(&lib_dir), "empty lib dir");

        // After a reboot the target may be gone while the link survives
        // (or the reverse); only a resolving link counts.
        let target = base.join("libnccl.so.2");
        std::os::unix::fs::symlink(&target, lib_dir.join("libnccl.so")).expect("link");
        assert!(!present(&lib_dir), "dangling link");
        std::fs::write(&target, b"elf").expect("target");
        assert!(present(&lib_dir), "resolving link");
        std::fs::remove_dir_all(&base).expect("cleanup");
    }

    #[test]
    fn only_nccl_phases_need_the_shim() {
        assert!(phases_use_nccl(&[Phase::Network]));
        assert!(phases_use_nccl(&[Phase::Gpu, Phase::Overlap]));
        assert!(!phases_use_nccl(&[
            Phase::Inventory,
            Phase::CpuMem,
            Phase::Gpu
        ]));
    }
}
