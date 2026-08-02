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

use std::io::Write;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use comfy_table::{Cell, Color, ContentArrangement, Table, presets};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{debug, info};

use super::deploy::ensure_agent;
use super::session::HostSession;
use crate::cli::BootstrapArgs;
use crate::config::{FleetConfig, HostConfig, SshConfig};
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
    /// "connectivity", "arch", "deploy", "gpu_driver", "gpu_libs", "clock_sync",
    /// "persistence_mode", "governor", ...
    pub name: String,
    pub status: CheckStatus,
    pub detail: String,
}

impl ReadinessCheck {
    fn new(name: &str, status: CheckStatus, detail: impl AsRef<str>) -> Self {
        Self {
            name: name.to_string(),
            status,
            detail: one_line(detail.as_ref()),
        }
    }

    fn ok(name: &str, detail: impl AsRef<str>) -> Self {
        Self::new(name, CheckStatus::Ok, detail)
    }

    fn warn(name: &str, detail: impl AsRef<str>) -> Self {
        Self::new(name, CheckStatus::Warn, detail)
    }

    fn fail(name: &str, detail: impl AsRef<str>) -> Self {
        Self::new(name, CheckStatus::Fail, detail)
    }
}

/// Clock offsets beyond this are a warning: cross-host timestamps drift.
const CLOCK_WARN_MS: f64 = 100.0;
/// Beyond this the fleet's timestamps cannot be correlated at all.
const CLOCK_FAIL_MS: f64 = 1_000.0;

pub async fn run(args: BootstrapArgs) -> Result<()> {
    let config = FleetConfig::load(&args.config)
        .with_context(|| format!("loading {}", args.config.display()))?;
    let hosts: Vec<HostConfig> = config.hosts().collect();
    let ssh = Arc::new(config.ssh.clone());
    let permits = Arc::new(Semaphore::new(config.ssh.max_concurrent.max(1)));
    info!(
        hosts = hosts.len(),
        max_concurrent = config.ssh.max_concurrent,
        tune = args.tune,
        "bootstrapping fleet"
    );

    let mut tasks = JoinSet::new();
    for (index, host) in hosts.iter().cloned().enumerate() {
        let ssh = Arc::clone(&ssh);
        let permits = Arc::clone(&permits);
        let tune = args.tune;
        tasks.spawn(async move {
            let _permit = permits.acquire_owned().await.ok();
            (index, prepare_host(host, &ssh, tune).await)
        });
    }

    let mut slots: Vec<Option<HostReadiness>> = vec![None; hosts.len()];
    while let Some(joined) = tasks.join_next().await {
        match joined {
            Ok((index, readiness)) => slots[index] = Some(readiness),
            Err(error) => debug!(%error, "bootstrap task did not complete"),
        }
    }

    // Invariant: every configured host appears exactly once, in config order.
    let rows: Vec<HostReadiness> = slots
        .into_iter()
        .zip(hosts.iter())
        .map(|(slot, host)| {
            slot.unwrap_or_else(|| HostReadiness {
                host: host.addr.clone(),
                checks: vec![ReadinessCheck::fail(
                    "connectivity",
                    "bootstrap task aborted",
                )],
                inventory: None,
            })
        })
        .collect();

    let mut stdout = std::io::stdout();
    render_matrix(&rows, &mut stdout)?;
    stdout.flush().context("flushing stdout")?;

    let failed: Vec<&str> = rows
        .iter()
        .filter(|row| worst_status(row) == CheckStatus::Fail)
        .map(|row| row.host.as_str())
        .collect();
    if !failed.is_empty() {
        bail!("{} host(s) not ready: {}", failed.len(), failed.join(", "));
    }
    Ok(())
}

/// Full per-host sequence. Never returns an error: a failure is a matrix
/// cell, and one bad node must not abort the rest of the fleet.
async fn prepare_host(host: HostConfig, ssh: &SshConfig, tune: bool) -> HostReadiness {
    let addr = host.addr.clone();
    let mut checks = Vec::new();

    let session = match HostSession::connect(host, ssh).await {
        Ok(session) => {
            checks.push(ReadinessCheck::ok("connectivity", session.remote_dir()));
            session
        }
        Err(error) => {
            checks.push(ReadinessCheck::fail("connectivity", format!("{error:#}")));
            return HostReadiness {
                host: addr,
                checks,
                inventory: None,
            };
        }
    };

    match session.exec("uname -m").await {
        Ok(output) => {
            let remote = output.trim().to_string();
            let check = arch_check(&remote, std::env::consts::ARCH);
            let fatal = check.status == CheckStatus::Fail;
            checks.push(check);
            if fatal {
                // Deploying a binary the node cannot execute helps nobody.
                return HostReadiness {
                    host: addr,
                    checks,
                    inventory: None,
                };
            }
        }
        Err(error) => {
            checks.push(ReadinessCheck::fail("arch", format!("{error:#}")));
            return HostReadiness {
                host: addr,
                checks,
                inventory: None,
            };
        }
    }

    match ensure_agent(&session).await {
        Ok(true) => checks.push(ReadinessCheck::ok("deploy", "uploaded")),
        Ok(false) => checks.push(ReadinessCheck::ok("deploy", "up to date")),
        Err(error) => {
            checks.push(ReadinessCheck::fail("deploy", format!("{error:#}")));
            return HostReadiness {
                host: addr,
                checks,
                inventory: None,
            };
        }
    }

    let inventory = match probe(&session).await {
        Ok(inventory) => {
            checks.push(ReadinessCheck::ok(
                "probe",
                format!(
                    "{} cores, {} gpu(s)",
                    inventory.logical_cores,
                    inventory.gpus.len()
                ),
            ));
            inventory
        }
        Err(error) => {
            checks.push(ReadinessCheck::fail("probe", format!("{error:#}")));
            return HostReadiness {
                host: addr,
                checks,
                inventory: None,
            };
        }
    };

    checks.extend(readiness_checks(&inventory));
    if tune {
        checks.extend(apply_tuning(&session, &inventory).await);
    }

    HostReadiness {
        host: addr,
        checks,
        inventory: Some(inventory),
    }
}

async fn probe(session: &HostSession) -> Result<InventorySnapshot> {
    let output = session.run_agent_capture(&["probe"], None).await?;
    if !output.success() {
        bail!("agent probe failed: {}", output.detail());
    }
    serde_json::from_str(output.stdout.trim()).context("parsing InventorySnapshot from agent probe")
}

/// Node arch as reported by `uname -m`, mapped onto `std::env::consts::ARCH`
/// spelling. Unknown machines return `None` and are reported as a warning
/// rather than a hard failure: the deploy step will fail loudly anyway if the
/// binary really cannot run.
fn normalize_arch(uname_machine: &str) -> Option<&'static str> {
    match uname_machine.trim() {
        "x86_64" | "amd64" => Some("x86_64"),
        "aarch64" | "arm64" | "aarch64_be" => Some("aarch64"),
        "armv6l" | "armv7l" | "armv8l" | "arm" => Some("arm"),
        "ppc64le" | "powerpc64le" | "ppc64" | "powerpc64" => Some("powerpc64"),
        "riscv64" => Some("riscv64"),
        "s390x" => Some("s390x"),
        "i386" | "i486" | "i586" | "i686" => Some("x86"),
        "loongarch64" => Some("loongarch64"),
        _ => None,
    }
}

fn arch_check(uname_machine: &str, local_arch: &str) -> ReadinessCheck {
    match normalize_arch(uname_machine) {
        Some(remote) if remote == local_arch => ReadinessCheck::ok("arch", remote),
        Some(remote) => ReadinessCheck::fail(
            "arch",
            format!("node is {remote}, orchestrator is {local_arch}"),
        ),
        None => ReadinessCheck::warn(
            "arch",
            format!("unrecognized machine {uname_machine:?}, expected {local_arch}"),
        ),
    }
}

/// The readiness checks derivable from a probe: pure, so the policy is
/// testable without a fleet.
fn readiness_checks(inventory: &InventorySnapshot) -> Vec<ReadinessCheck> {
    vec![
        gpu_driver_check(inventory),
        gpu_libs_check(inventory),
        clock_sync_check(inventory),
        ib_ports_check(inventory),
        governor_check(inventory),
        persistence_mode_check(inventory),
    ]
}

/// Runtime loadability of the GPU library stack, from the agent's dlopen
/// probe. Only advisory: the run degrades per phase, but an operator wants
/// to know *before* the run that the NCCL sweep has nothing to work with.
fn gpu_libs_check(inventory: &InventorySnapshot) -> ReadinessCheck {
    if inventory.gpus.is_empty() {
        return ReadinessCheck::ok("gpu_libs", "no GPUs (n/a)");
    }
    let missing: Vec<&str> = inventory
        .gpu_libs
        .iter()
        .filter(|(_, available)| !**available)
        .map(|(name, _)| name.as_str())
        .collect();
    if inventory.gpu_libs.is_empty() {
        ReadinessCheck::warn("gpu_libs", "agent reported no library probe data")
    } else if missing.is_empty() {
        ReadinessCheck::ok("gpu_libs", "cuda, cublas, nccl loadable")
    } else {
        let phases: &str = if missing == ["nccl"] {
            "NCCL sweep unavailable"
        } else {
            "GPU phases will fail"
        };
        ReadinessCheck::warn(
            "gpu_libs",
            format!("not loadable: {}; {phases}", missing.join(", ")),
        )
    }
}

fn gpu_driver_check(inventory: &InventorySnapshot) -> ReadinessCheck {
    match (&inventory.nvidia_driver, inventory.gpus.len()) {
        (Some(driver), 0) => ReadinessCheck::warn(
            "gpu_driver",
            format!("driver {driver} present but no GPUs visible"),
        ),
        (Some(driver), count) => {
            let cuda = inventory.cuda_version.as_deref().unwrap_or("cuda unknown");
            let check = ReadinessCheck::ok(
                "gpu_driver",
                format!("{count} gpu(s), driver {driver}, {cuda}"),
            );
            if inventory.xid_errors.is_empty() {
                check
            } else {
                ReadinessCheck::warn(
                    "gpu_driver",
                    format!(
                        "{count} gpu(s), driver {driver}, {cuda}; xid errors {:?}",
                        inventory.xid_errors
                    ),
                )
            }
        }
        (None, _) => ReadinessCheck::warn("gpu_driver", "no NVIDIA driver; GPU phases will skip"),
    }
}

fn clock_sync_check(inventory: &InventorySnapshot) -> ReadinessCheck {
    match inventory.clock_offset_ms {
        None => ReadinessCheck::warn("clock_sync", "no chrony/ntp offset available"),
        Some(offset) => {
            let magnitude = offset.abs();
            let detail = format!("offset {offset:.1} ms");
            if !magnitude.is_finite() || magnitude >= CLOCK_FAIL_MS {
                ReadinessCheck::fail("clock_sync", detail)
            } else if magnitude >= CLOCK_WARN_MS {
                ReadinessCheck::warn("clock_sync", detail)
            } else {
                ReadinessCheck::ok("clock_sync", detail)
            }
        }
    }
}

fn ib_ports_check(inventory: &InventorySnapshot) -> ReadinessCheck {
    if inventory.ib_ports.is_empty() {
        return ReadinessCheck::warn("ib_ports", "no InfiniBand ports found");
    }
    let active: Vec<&str> = inventory
        .ib_ports
        .iter()
        .filter(|port| port.state.eq_ignore_ascii_case("active"))
        .map(|port| port.device.as_str())
        .collect();
    let inactive: Vec<String> = inventory
        .ib_ports
        .iter()
        .filter(|port| !port.state.eq_ignore_ascii_case("active"))
        .map(|port| format!("{}:{} {}", port.device, port.port, port.state))
        .collect();

    if inactive.is_empty() {
        ReadinessCheck::ok("ib_ports", format!("{} active", active.len()))
    } else if active.is_empty() {
        ReadinessCheck::fail("ib_ports", format!("none active ({})", inactive.join(", ")))
    } else {
        ReadinessCheck::warn(
            "ib_ports",
            format!("{} active, down: {}", active.len(), inactive.join(", ")),
        )
    }
}

fn governor_check(inventory: &InventorySnapshot) -> ReadinessCheck {
    match inventory.cpu_governor.as_deref() {
        Some("performance") => ReadinessCheck::ok("governor", "performance"),
        Some(other) => ReadinessCheck::warn(
            "governor",
            format!("{other}; run with --tune for performance"),
        ),
        None => ReadinessCheck::warn("governor", "governor unavailable (no cpufreq?)"),
    }
}

fn persistence_mode_check(inventory: &InventorySnapshot) -> ReadinessCheck {
    if inventory.gpus.is_empty() {
        return ReadinessCheck::ok("persistence_mode", "n/a (no GPUs)");
    }
    let off: Vec<String> = inventory
        .gpus
        .iter()
        .filter(|gpu| gpu.persistence_mode != Some(true))
        .map(|gpu| gpu.index.to_string())
        .collect();
    if off.is_empty() {
        ReadinessCheck::ok("persistence_mode", "enabled on all GPUs")
    } else {
        ReadinessCheck::warn(
            "persistence_mode",
            format!("off on gpu {}; run with --tune", off.join(",")),
        )
    }
}

/// `--tune` steps. Each is sudo-gated; a refusal is a warning, never fatal.
async fn apply_tuning(session: &HostSession, inventory: &InventorySnapshot) -> Vec<ReadinessCheck> {
    let mut checks = Vec::new();

    if inventory.gpus.is_empty() {
        checks.push(ReadinessCheck::ok("tune_persistence", "n/a (no GPUs)"));
    } else {
        checks.push(
            match session.exec_capture("sudo -n nvidia-smi -pm 1").await {
                Ok(output) if output.success() => {
                    ReadinessCheck::ok("tune_persistence", "persistence mode enabled")
                }
                Ok(output) => ReadinessCheck::warn("tune_persistence", output.detail()),
                Err(error) => ReadinessCheck::warn("tune_persistence", format!("{error:#}")),
            },
        );
    }

    // One `tee` per policy file: `sudo sh -c` would need a second layer of
    // quoting and many sudoers policies refuse a bare shell anyway.
    let script = "set -e; \
         for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do \
           [ -w \"$governor\" ] || [ -e \"$governor\" ] || continue; \
           echo performance | sudo -n tee \"$governor\" > /dev/null; \
         done; \
         cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true";
    checks.push(match session.exec_capture(script).await {
        Ok(output) if output.success() => {
            let now = output.stdout.trim();
            if now.is_empty() {
                ReadinessCheck::warn("tune_governor", "no cpufreq policy files")
            } else if now == "performance" {
                ReadinessCheck::ok("tune_governor", "performance")
            } else {
                ReadinessCheck::warn("tune_governor", format!("still {now}"))
            }
        }
        Ok(output) => ReadinessCheck::warn("tune_governor", output.detail()),
        Err(error) => ReadinessCheck::warn("tune_governor", format!("{error:#}")),
    });

    checks
}

fn worst_status(row: &HostReadiness) -> CheckStatus {
    if row
        .checks
        .iter()
        .any(|check| check.status == CheckStatus::Fail)
    {
        CheckStatus::Fail
    } else if row
        .checks
        .iter()
        .any(|check| check.status == CheckStatus::Warn)
    {
        CheckStatus::Warn
    } else {
        CheckStatus::Ok
    }
}

/// Column order is first-appearance order across rows, which is the order the
/// checks are performed in.
fn column_names(rows: &[HostReadiness]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for row in rows {
        for check in &row.checks {
            if !names.iter().any(|name| name == &check.name) {
                names.push(check.name.clone());
            }
        }
    }
    names
}

fn render_matrix(rows: &[HostReadiness], out: &mut dyn Write) -> Result<()> {
    let columns = column_names(rows);
    let mut table = Table::new();
    table
        .load_preset(presets::UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);
    let mut header = vec![Cell::new("host")];
    header.extend(columns.iter().map(Cell::new));
    table.set_header(header);

    for row in rows {
        let mut cells = vec![Cell::new(&row.host)];
        for column in &columns {
            cells.push(
                match row.checks.iter().find(|check| &check.name == column) {
                    Some(check) => cell_for(check),
                    None => Cell::new("-").fg(Color::DarkGrey),
                },
            );
        }
        table.add_row(cells);
    }

    writeln!(out, "{table}").context("writing readiness matrix")?;

    let (mut ok, mut warn, mut fail) = (0usize, 0usize, 0usize);
    for row in rows {
        match worst_status(row) {
            CheckStatus::Ok => ok += 1,
            CheckStatus::Warn => warn += 1,
            CheckStatus::Fail => fail += 1,
        }
    }
    writeln!(
        out,
        "{} host(s): {ok} ready, {warn} with warnings, {fail} failed",
        rows.len()
    )
    .context("writing readiness summary")?;
    Ok(())
}

fn cell_for(check: &ReadinessCheck) -> Cell {
    match check.status {
        CheckStatus::Ok => Cell::new("ok").fg(Color::Green),
        CheckStatus::Warn => Cell::new(format!("warn: {}", check.detail)).fg(Color::Yellow),
        CheckStatus::Fail => Cell::new(format!("fail: {}", check.detail)).fg(Color::Red),
    }
}

/// Matrix cells are one line; keep details short and free of newlines.
fn one_line(text: &str) -> String {
    let flat: String = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(160)
        .collect();
    flat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{GpuInventory, IbPortInventory};

    fn inventory() -> InventorySnapshot {
        InventorySnapshot {
            hostname: "node-a".into(),
            kernel: "6.1.0".into(),
            cpu_model: "test".into(),
            logical_cores: 8,
            numa_nodes: 1,
            mem_total_bytes: 1 << 34,
            cpu_governor: Some("performance".into()),
            clock_offset_ms: Some(1.5),
            nvidia_driver: Some("550.54.15".into()),
            cuda_version: Some("12.4".into()),
            gpus: vec![gpu(0, Some(true))],
            nics: Vec::new(),
            ib_ports: vec![IbPortInventory {
                device: "mlx5_0".into(),
                port: 1,
                state: "Active".into(),
                rate_gbps: Some(400.0),
                link_downed_count: Some(0),
            }],
            xid_errors: Vec::new(),
            gpu_libs: [("cuda", true), ("cublas", true), ("nccl", true)]
                .into_iter()
                .map(|(name, ok)| (name.to_string(), ok))
                .collect(),
        }
    }

    fn gpu(index: u32, persistence: Option<bool>) -> GpuInventory {
        GpuInventory {
            index,
            name: "H100".into(),
            uuid: format!("GPU-{index}"),
            vbios: "96.00".into(),
            mem_total_bytes: 80 << 30,
            ecc_volatile_errors: Some(0),
            remapped_rows_pending: Some(false),
            pcie_gen_current: Some(5),
            pcie_gen_max: Some(5),
            pcie_width_current: Some(16),
            pcie_width_max: Some(16),
            nvlinks_active: Some(18),
            persistence_mode: persistence,
        }
    }

    fn status(checks: &[ReadinessCheck], name: &str) -> CheckStatus {
        checks
            .iter()
            .find(|check| check.name == name)
            .map(|check| check.status)
            .unwrap_or_else(|| panic!("check {name} missing"))
    }

    #[test]
    fn uname_machines_map_onto_rust_arch_names() {
        assert_eq!(normalize_arch("x86_64"), Some("x86_64"));
        assert_eq!(normalize_arch("amd64"), Some("x86_64"));
        assert_eq!(normalize_arch("aarch64"), Some("aarch64"));
        assert_eq!(normalize_arch("arm64"), Some("aarch64"));
        assert_eq!(normalize_arch("ppc64le"), Some("powerpc64"));
        assert_eq!(normalize_arch("i686"), Some("x86"));
        assert_eq!(normalize_arch(" x86_64\n"), Some("x86_64"));
        assert_eq!(normalize_arch("vax"), None);
    }

    #[test]
    fn arch_mismatch_is_fatal_and_unknown_is_a_warning() {
        assert_eq!(arch_check("x86_64", "x86_64").status, CheckStatus::Ok);
        assert_eq!(arch_check("aarch64", "x86_64").status, CheckStatus::Fail);
        assert_eq!(arch_check("vax", "x86_64").status, CheckStatus::Warn);
    }

    #[test]
    fn a_healthy_node_passes_every_derived_check() {
        let checks = readiness_checks(&inventory());
        assert!(
            checks.iter().all(|check| check.status == CheckStatus::Ok),
            "{checks:?}"
        );
    }

    #[test]
    fn missing_driver_warns_but_never_fails() {
        let mut inv = inventory();
        inv.nvidia_driver = None;
        inv.gpus.clear();
        let checks = readiness_checks(&inv);
        assert_eq!(status(&checks, "gpu_driver"), CheckStatus::Warn);
        assert_eq!(status(&checks, "persistence_mode"), CheckStatus::Ok);
    }

    #[test]
    fn clock_offset_escalates_with_magnitude() {
        let mut inv = inventory();
        inv.clock_offset_ms = Some(-5.0);
        assert_eq!(clock_sync_check(&inv).status, CheckStatus::Ok);
        inv.clock_offset_ms = Some(250.0);
        assert_eq!(clock_sync_check(&inv).status, CheckStatus::Warn);
        inv.clock_offset_ms = Some(-4_000.0);
        assert_eq!(clock_sync_check(&inv).status, CheckStatus::Fail);
        inv.clock_offset_ms = None;
        assert_eq!(clock_sync_check(&inv).status, CheckStatus::Warn);
    }

    #[test]
    fn every_ib_port_down_is_fatal_but_a_partial_outage_warns() {
        let mut inv = inventory();
        inv.ib_ports[0].state = "Down".into();
        assert_eq!(ib_ports_check(&inv).status, CheckStatus::Fail);
        inv.ib_ports.push(IbPortInventory {
            device: "mlx5_1".into(),
            port: 1,
            state: "Active".into(),
            rate_gbps: Some(400.0),
            link_downed_count: Some(0),
        });
        assert_eq!(ib_ports_check(&inv).status, CheckStatus::Warn);
        inv.ib_ports.clear();
        assert_eq!(ib_ports_check(&inv).status, CheckStatus::Warn);
    }

    #[test]
    fn persistence_and_governor_are_tunable_warnings() {
        let mut inv = inventory();
        inv.gpus.push(gpu(1, Some(false)));
        inv.cpu_governor = Some("powersave".into());
        let checks = readiness_checks(&inv);
        assert_eq!(status(&checks, "persistence_mode"), CheckStatus::Warn);
        assert_eq!(status(&checks, "governor"), CheckStatus::Warn);
    }

    #[test]
    fn columns_follow_first_appearance_order() {
        let rows = vec![
            HostReadiness {
                host: "a".into(),
                checks: vec![
                    ReadinessCheck::ok("connectivity", ""),
                    ReadinessCheck::fail("arch", "aarch64"),
                ],
                inventory: None,
            },
            HostReadiness {
                host: "b".into(),
                checks: vec![
                    ReadinessCheck::ok("connectivity", ""),
                    ReadinessCheck::ok("arch", "x86_64"),
                    ReadinessCheck::ok("deploy", "up to date"),
                ],
                inventory: None,
            },
        ];
        assert_eq!(column_names(&rows), ["connectivity", "arch", "deploy"]);
        assert_eq!(worst_status(&rows[0]), CheckStatus::Fail);
        assert_eq!(worst_status(&rows[1]), CheckStatus::Ok);
    }

    #[test]
    fn details_are_flattened_to_one_line() {
        let check = ReadinessCheck::fail("deploy", "boom:\n  caused by:\n    no space left");
        assert_eq!(check.detail, "boom: caused by: no space left");
    }

    #[test]
    fn gpu_libs_na_without_gpus() {
        let mut inv = inventory();
        inv.gpus.clear();
        inv.gpu_libs.clear();
        assert_eq!(gpu_libs_check(&inv).status, CheckStatus::Ok);
    }

    #[test]
    fn gpu_libs_missing_nccl_warns_about_the_sweep() {
        let mut inv = inventory();
        inv.gpus.push(gpu(0, Some(true)));
        inv.gpu_libs = [
            ("cuda".to_string(), true),
            ("cublas".to_string(), true),
            ("nccl".to_string(), false),
        ]
        .into_iter()
        .collect();
        let check = gpu_libs_check(&inv);
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.detail.contains("nccl"), "{}", check.detail);
        assert!(check.detail.contains("NCCL sweep"), "{}", check.detail);
    }

    #[test]
    fn gpu_libs_missing_cuda_warns_about_gpu_phases() {
        let mut inv = inventory();
        inv.gpus.push(gpu(0, Some(true)));
        inv.gpu_libs = [
            ("cuda".to_string(), false),
            ("cublas".to_string(), false),
            ("nccl".to_string(), false),
        ]
        .into_iter()
        .collect();
        let check = gpu_libs_check(&inv);
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(check.detail.contains("GPU phases"), "{}", check.detail);
    }

    #[test]
    fn gpu_libs_all_loadable_is_ok() {
        let mut inv = inventory();
        inv.gpus.push(gpu(0, Some(true)));
        inv.gpu_libs = [
            ("cuda".to_string(), true),
            ("cublas".to_string(), true),
            ("nccl".to_string(), true),
        ]
        .into_iter()
        .collect();
        assert_eq!(gpu_libs_check(&inv).status, CheckStatus::Ok);
    }
}
