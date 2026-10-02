//! How the orchestrator reaches a node: the transport behind every
//! `HostSession` operation (run a short shell command, spawn an agent with
//! piped stdio, kill an agent) plus the fleet-level `Launcher` that opens
//! sessions and resolves the host list.
//!
//! Two transports, dispatched by enum (async methods, no trait objects):
//!
//! - `Ssh`: one persistent openssh ControlMaster session per host (the
//!   original behaviour, unchanged). Agents are `env <vars> <agent> agent
//!   <args>` remote commands; kill is a remote `pkill -f`.
//! - `Srun`: every operation is an `srun` job step inside the Slurm
//!   allocation the orchestrator itself runs in (`srun --nodelist=<host>
//!   --overlap ...`); the agent env rides the local srun process's
//!   environment (`--export=ALL`); kill is `scancel --signal=KILL` of the
//!   step found by name in `squeue --steps`.
//!
//! Nothing above `HostSession` knows which one is in use.

mod srun;
mod ssh;

use std::pin::Pin;
use std::process::ExitStatus;
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::io::{AsyncRead, AsyncWrite};
use tracing::{debug, info};

pub use self::srun::{SlurmTools, SrunLauncher};
pub(crate) use self::ssh::upload as ssh_upload;
use super::session::{HostSession, RemoteOutput};
use crate::config::{FleetConfig, HostConfig, SshConfig};
use crate::launch::slurm::{
    SlurmAllocation, parse_hostnames, scontrol_hostnames_args, select_hosts,
};
use crate::launch::srun::{SrunDir, env_to_strip};
use crate::launch::{LaunchMode, LaunchRecord};
use crate::nccl_env::NcclEnv;

/// Fleet-level launcher: opens `HostSession`s over one transport.
#[derive(Clone)]
pub enum Launcher {
    Ssh(Arc<SshConfig>),
    Srun(Arc<SrunLauncher>),
}

impl Launcher {
    pub fn ssh(config: SshConfig) -> Self {
        Launcher::Ssh(Arc::new(config))
    }

    pub fn mode(&self) -> LaunchMode {
        match self {
            Launcher::Ssh(_) => LaunchMode::Ssh,
            Launcher::Srun(_) => LaunchMode::Srun,
        }
    }

    /// What `RunResults.launch` records.
    pub fn record(&self) -> LaunchRecord {
        match self {
            Launcher::Ssh(_) => LaunchRecord::Ssh,
            Launcher::Srun(srun) => LaunchRecord::Srun {
                job_id: srun.job_id(),
            },
        }
    }

    /// Open a session to `host`: establish the transport, then create and
    /// resolve the agent directory on the node. `nccl_env` is set on every
    /// agent spawn of the session.
    pub async fn connect(&self, host: HostConfig, nccl_env: &NcclEnv) -> Result<HostSession> {
        match self {
            Launcher::Ssh(config) => {
                let session = ssh::connect(&host.addr, config).await?;
                let timeout = ssh::connect_timeout(config);
                HostSession::establish(
                    host,
                    HostTransport::Ssh(session),
                    &config.remote_dir,
                    nccl_env,
                    timeout,
                )
                .await
            }
            Launcher::Srun(srun) => {
                let dir = srun.dir().as_str().to_string();
                HostSession::establish(
                    host,
                    HostTransport::Srun(Arc::clone(srun)),
                    &dir,
                    nccl_env,
                    srun::STEP_SETUP_TIMEOUT,
                )
                .await
            }
        }
    }
}

/// Resolve the effective launch mode (`--launch` over `[launch] mode`) into
/// a launcher and the fleet it targets. ssh: the configured hosts. srun:
/// must run inside an allocation (typed `SlurmError::NotInAllocation`
/// otherwise); hosts are the allocation's nodes (`scontrol show hostnames
/// $SLURM_JOB_NODELIST`) or, when configured, a validated subset of them.
pub async fn resolve_launch(
    config: &FleetConfig,
    cli_override: Option<LaunchMode>,
) -> Result<(FleetConfig, Launcher)> {
    let mode = config.launch.effective_mode(cli_override);
    match mode {
        LaunchMode::Ssh => {
            config.require_hosts_for(mode)?;
            Ok((config.clone(), Launcher::ssh(config.ssh.clone())))
        }
        LaunchMode::Srun => {
            let env = |key: &str| std::env::var(key).ok();
            let allocation = SlurmAllocation::from_env(env)?;
            let nodelist = allocation.require_nodelist()?;
            let nodes = scontrol_hostnames(nodelist).await?;
            let configured: Vec<HostConfig> = config.hosts().collect();
            let hosts = select_hosts(allocation.job_id, configured, &nodes)?;
            let fleet = config.with_hosts(hosts)?;
            let dir = match &config.launch.srun.dir {
                Some(dir) => dir.clone(),
                None => SrunDir::default_for(env)?,
            };
            let inherited: Vec<String> = std::env::vars_os()
                .filter_map(|(name, _)| name.into_string().ok())
                .collect();
            let strip = env_to_strip(inherited.iter().map(String::as_str));
            info!(
                job_id = %allocation.job_id,
                nodes = nodes.len(),
                hosts = fleet.hosts.len(),
                dir = dir.as_str(),
                "srun launch inside the Slurm allocation"
            );
            let launcher = SrunLauncher::new(
                allocation.job_id,
                config.launch.srun.clone(),
                dir,
                strip,
                SlurmTools::default(),
            );
            Ok((fleet, Launcher::Srun(Arc::new(launcher))))
        }
    }
}

/// Expand a compressed Slurm node list with `scontrol show hostnames`.
async fn scontrol_hostnames(nodelist: &str) -> Result<Vec<String>> {
    let output = tokio::process::Command::new(srun::SCONTROL)
        .args(scontrol_hostnames_args(nodelist))
        .stdin(std::process::Stdio::null())
        .output()
        .await
        .with_context(|| format!("running `scontrol show hostnames {nodelist}`"))?;
    let result = RemoteOutput::from_output(output);
    if !result.success() {
        anyhow::bail!(
            "`scontrol show hostnames {nodelist}` failed: {}",
            result.detail()
        );
    }
    let hosts = parse_hostnames(&result.stdout)?;
    debug!(nodelist, hosts = ?hosts, "allocation expanded");
    Ok(hosts)
}

/// Per-session transport handle.
pub(crate) enum HostTransport {
    Ssh(Arc<openssh::Session>),
    Srun(Arc<SrunLauncher>),
}

/// Whether one stdio stream of a spawned agent is piped back or discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pipe {
    Piped,
    Null,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SpawnStdio {
    pub stdin: Pipe,
    pub stdout: Pipe,
    pub stderr: Pipe,
}

/// Everything a spawn needs besides the transport: the agent binary, the
/// environment it must start under, and `agent <args>`.
pub(crate) struct AgentCommand<'a> {
    pub host: &'a str,
    pub env: &'a [(String, String)],
    pub agent_path: &'a str,
    pub args: &'a [&'a str],
}

impl HostTransport {
    pub(crate) async fn exec_capture(&self, host: &str, script: &str) -> Result<RemoteOutput> {
        match self {
            HostTransport::Ssh(session) => ssh::exec_capture(session, host, script).await,
            HostTransport::Srun(srun) => srun.exec_capture(host, script).await,
        }
    }

    pub(crate) async fn spawn(
        &self,
        command: AgentCommand<'_>,
        stdio: SpawnStdio,
    ) -> Result<AgentChild> {
        match self {
            HostTransport::Ssh(session) => ssh::spawn(session, &command, stdio)
                .await
                .map(AgentChild::Ssh),
            HostTransport::Srun(srun) => srun.spawn(&command, stdio).map(AgentChild::Srun),
        }
    }

    /// Kill the agent running `agent <args...>` (word prefix) on `host`.
    /// Returns a short description of what was done, for logging.
    pub(crate) async fn kill_agent(&self, host: &str, args: &str) -> Result<String> {
        match self {
            HostTransport::Ssh(session) => ssh::kill_agent(session, host, args).await,
            HostTransport::Srun(srun) => srun.kill_agent(host, args).await,
        }
    }

    pub(crate) fn ssh_session(&self) -> Option<&Arc<openssh::Session>> {
        match self {
            HostTransport::Ssh(session) => Some(session),
            HostTransport::Srun(_) => None,
        }
    }
}

pub type AgentStdin = Pin<Box<dyn AsyncWrite + Send>>;
pub type AgentOutput = Pin<Box<dyn AsyncRead + Send>>;

/// A running agent process: the remote ssh command, or the local `srun`
/// process standing for the step (its stdio is the task's stdio).
pub enum AgentChild {
    Ssh(openssh::Child<Arc<openssh::Session>>),
    Srun(tokio::process::Child),
}

impl AgentChild {
    pub fn take_stdin(&mut self) -> Option<AgentStdin> {
        match self {
            AgentChild::Ssh(child) => child
                .stdin()
                .take()
                .map(|pipe| Box::pin(pipe) as AgentStdin),
            AgentChild::Srun(child) => child.stdin.take().map(|pipe| Box::pin(pipe) as AgentStdin),
        }
    }

    pub fn take_stdout(&mut self) -> Option<AgentOutput> {
        match self {
            AgentChild::Ssh(child) => child
                .stdout()
                .take()
                .map(|pipe| Box::pin(pipe) as AgentOutput),
            AgentChild::Srun(child) => child
                .stdout
                .take()
                .map(|pipe| Box::pin(pipe) as AgentOutput),
        }
    }

    pub fn take_stderr(&mut self) -> Option<AgentOutput> {
        match self {
            AgentChild::Ssh(child) => child
                .stderr()
                .take()
                .map(|pipe| Box::pin(pipe) as AgentOutput),
            AgentChild::Srun(child) => child
                .stderr
                .take()
                .map(|pipe| Box::pin(pipe) as AgentOutput),
        }
    }

    /// Wait for the process to exit. For srun this is the step's exit:
    /// srun returns the task's status (or the signal that ended it).
    pub async fn wait(self) -> Result<ExitStatus> {
        match self {
            AgentChild::Ssh(child) => child.wait().await.context("waiting for remote agent"),
            AgentChild::Srun(mut child) => child.wait().await.context("waiting for srun step"),
        }
    }
}
