//! How the orchestrator reaches a node: the transport behind every
//! `HostSession` operation (run a short shell command, spawn an agent with
//! piped stdio, kill agents) plus the fleet-level `Launcher` that opens
//! sessions and resolves the host list.
//!
//! Two transports, dispatched by enum (async methods, no trait objects):
//!
//! - `Ssh`: one persistent openssh ControlMaster session per host (the
//!   original behaviour, unchanged). Agents are `env <vars> <agent> agent
//!   <args>` remote commands; kill is a remote `pkill -f`.
//! - `Srun`: every operation is a named `srun` job step inside the Slurm
//!   allocation the orchestrator itself runs in (`srun --nodelist=<host>
//!   --overlap ...`); the NCCL env rides the local srun process's
//!   environment (`--export=ALL`); kill is one `squeue --steps` listing
//!   plus `scancel --signal=KILL` of the matching steps; an abandoned
//!   srun is sent SIGTERM so it cancels its own step.
//!
//! Nothing above `HostSession` knows which one is in use.

mod srun;
mod ssh;

use std::pin::Pin;
use std::process::ExitStatus;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::io::{AsyncRead, AsyncWrite};
use tracing::{debug, info, warn};

pub(crate) use self::srun::local as slurm_local;
pub use self::srun::{SlurmTools, SrunChild, SrunLauncher, SrunTransportError};
pub(crate) use self::ssh::upload as ssh_upload;
use super::session::{AgentEnv, HostSession, RemoteOutput};
use crate::config::{FleetConfig, HostConfig, SshConfig};
use crate::launch::slurm::{
    NodeAddrs, SlurmAllocation, apply_node_addrs, parse_hostnames, parse_node_addrs,
    scontrol_hostnames_args, scontrol_show_nodes_args, select_hosts,
};
use crate::launch::srun::{SrunDir, StepTarget, env_to_strip};
use crate::launch::{LaunchMode, LaunchRecord};
use crate::nccl_env::NcclEnv;

/// Bound on each `scontrol` call while resolving the fleet.
const SCONTROL_TIMEOUT: Duration = Duration::from_secs(30);

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
                let addr = host.addr.clone();
                let established = HostSession::establish(
                    host,
                    HostTransport::Srun(Arc::clone(srun)),
                    &dir,
                    nccl_env,
                    srun::STEP_SETUP_TIMEOUT,
                )
                .await;
                if established.is_err() {
                    // The abandoned setup step was already SIGTERMed via its
                    // srun; cancel by name as well in case srun itself was
                    // stuck before the step started.
                    if let Err(error) = srun.kill_steps(&[(&addr, StepTarget::Exec)]).await {
                        debug!(host = %addr, %error, "exec step cleanup after failed connect");
                    }
                }
                established
            }
        }
    }
}

/// Resolve the effective launch mode (`--launch` over `[launch] mode`) into
/// a launcher and the fleet it targets. ssh: the configured hosts. srun:
/// must run inside an allocation (typed `SlurmError::NotInAllocation`
/// otherwise); hosts are the allocation's nodes (`scontrol show hostnames
/// $SLURM_JOB_NODELIST`) or, when configured, a validated subset of them,
/// with each node's NodeAddr (`scontrol --oneliner show node`, one call)
/// as its data-plane address.
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
            let tools = SlurmTools::default();
            let allocation = SlurmAllocation::from_env(env)?;
            let nodelist = allocation.require_nodelist()?;
            let nodes = scontrol_hostnames(&tools, nodelist).await?;
            let configured: Vec<HostConfig> = config.hosts().collect();
            let hosts = select_hosts(allocation.job_id, configured, &nodes)?;
            let addrs = match scontrol_node_addrs(&tools, nodelist).await {
                Ok(addrs) => addrs,
                Err(error) => {
                    // Peer traffic then targets the NodeName, as before.
                    warn!(%error, "cannot resolve NodeAddr; peers will use node names");
                    NodeAddrs::new()
                }
            };
            let fleet = config.with_hosts(apply_node_addrs(hosts, &addrs))?;
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
                node_addrs = addrs.len(),
                dir = dir.as_str(),
                "srun launch inside the Slurm allocation"
            );
            let launcher = SrunLauncher::new(
                allocation.job_id,
                config.launch.srun.clone(),
                dir,
                strip,
                env("LD_LIBRARY_PATH"),
                tools,
            );
            Ok((fleet, Launcher::Srun(Arc::new(launcher))))
        }
    }
}

async fn scontrol(tools: &SlurmTools, args: Vec<String>, what: &str) -> Result<RemoteOutput> {
    let output = slurm_local(&tools.scontrol, &args, SCONTROL_TIMEOUT).await?;
    if !output.success() {
        return Err(SrunTransportError::Scontrol {
            what: what.to_string(),
            detail: output.detail(),
        }
        .into());
    }
    Ok(output)
}

/// Expand a compressed Slurm node list with `scontrol show hostnames`.
async fn scontrol_hostnames(tools: &SlurmTools, nodelist: &str) -> Result<Vec<String>> {
    let output = scontrol(
        tools,
        scontrol_hostnames_args(nodelist),
        &format!("show hostnames {nodelist}"),
    )
    .await?;
    let hosts = parse_hostnames(&output.stdout)?;
    debug!(nodelist, hosts = ?hosts, "allocation expanded");
    Ok(hosts)
}

/// NodeName -> NodeAddr for the whole allocation in one `scontrol` call.
async fn scontrol_node_addrs(tools: &SlurmTools, nodelist: &str) -> Result<NodeAddrs> {
    let output = scontrol(
        tools,
        scontrol_show_nodes_args(nodelist),
        &format!("show node {nodelist}"),
    )
    .await?;
    let addrs = parse_node_addrs(&output.stdout)?;
    debug!(nodelist, addrs = ?addrs, "node addresses resolved");
    Ok(addrs)
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
    pub env: &'a AgentEnv,
    pub agent_path: &'a str,
    pub args: &'a [&'a str],
}

impl HostTransport {
    pub(crate) async fn exec_capture(&self, host: &str, script: &str) -> Result<RemoteOutput> {
        match self {
            HostTransport::Ssh(session) => ssh::exec_capture(session, host, script).await,
            HostTransport::Srun(srun) => srun
                .exec_capture(host, script)
                .await
                .with_context(|| format!("running `{script}` on {host} via srun")),
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
            HostTransport::Srun(srun) => Ok(AgentChild::Srun(srun.spawn(&command, stdio)?)),
        }
    }

    /// Kill the agent running `agent <args...>` (word prefix) on `host`.
    /// Returns a short description of what was done, for logging.
    pub(crate) async fn kill_agent(&self, host: &str, args: &str) -> Result<String> {
        match self {
            HostTransport::Ssh(session) => ssh::kill_agent(session, host, args).await,
            HostTransport::Srun(srun) => {
                let cancelled = srun.kill_steps(&[(host, StepTarget::Agent(args))]).await?;
                Ok(if cancelled.is_empty() {
                    "no matching step".to_string()
                } else {
                    format!("scancel --signal=KILL {}", cancelled.join(" "))
                })
            }
        }
    }

    pub(crate) fn ssh_session(&self) -> Option<&Arc<openssh::Session>> {
        match self {
            HostTransport::Ssh(session) => Some(session),
            HostTransport::Srun(_) => None,
        }
    }
}

/// Kill the agent running `agent <args...>` (word prefix) on every given
/// host. srun: one squeue listing and one scancel for all hosts sharing a
/// launcher; ssh: one `pkill -f` per host, concurrently. Best effort:
/// failures are logged, not raised.
pub(crate) async fn kill_agents(sessions: &[Arc<HostSession>], args: &str) {
    let mut srun_groups: Vec<(Arc<SrunLauncher>, Vec<String>)> = Vec::new();
    let mut ssh_kills = tokio::task::JoinSet::new();
    for session in sessions {
        match &session.transport {
            HostTransport::Srun(launcher) => {
                match srun_groups
                    .iter_mut()
                    .find(|(group, _)| Arc::ptr_eq(group, launcher))
                {
                    Some((_, hosts)) => hosts.push(session.addr().to_string()),
                    None => {
                        srun_groups.push((Arc::clone(launcher), vec![session.addr().to_string()]))
                    }
                }
            }
            HostTransport::Ssh(ssh_session) => {
                let ssh_session = Arc::clone(ssh_session);
                let host = session.addr().to_string();
                let args = args.to_string();
                ssh_kills.spawn(async move {
                    match ssh::kill_agent(&ssh_session, &host, &args).await {
                        Ok(action) => debug!(host = %host, action, "remote agent cleanup sent"),
                        Err(error) => warn!(host = %host, %error, "remote agent cleanup failed"),
                    }
                });
            }
        }
    }
    for (launcher, hosts) in srun_groups {
        let requests: Vec<(&str, StepTarget<'_>)> = hosts
            .iter()
            .map(|host| (host.as_str(), StepTarget::Agent(args)))
            .collect();
        match launcher.kill_steps(&requests).await {
            Ok(cancelled) => {
                debug!(hosts = ?hosts, args, cancelled = ?cancelled, "steps cancelled")
            }
            Err(error) => warn!(hosts = ?hosts, args, %error, "step cleanup failed"),
        }
    }
    while ssh_kills.join_next().await.is_some() {}
}

pub type AgentStdin = Pin<Box<dyn AsyncWrite + Send>>;
pub type AgentOutput = Pin<Box<dyn AsyncRead + Send>>;

/// A running agent process: the remote ssh command, or the local `srun`
/// process standing for the step (its stdio is the task's stdio; dropping
/// it unfinished cancels the step).
pub enum AgentChild {
    Ssh(openssh::Child<Arc<openssh::Session>>),
    Srun(SrunChild),
}

impl AgentChild {
    pub fn take_stdin(&mut self) -> Option<AgentStdin> {
        match self {
            AgentChild::Ssh(child) => child
                .stdin()
                .take()
                .map(|pipe| Box::pin(pipe) as AgentStdin),
            AgentChild::Srun(child) => child.take_stdin().map(|pipe| Box::pin(pipe) as AgentStdin),
        }
    }

    pub fn take_stdout(&mut self) -> Option<AgentOutput> {
        match self {
            AgentChild::Ssh(child) => child
                .stdout()
                .take()
                .map(|pipe| Box::pin(pipe) as AgentOutput),
            AgentChild::Srun(child) => child
                .take_stdout()
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
                .take_stderr()
                .map(|pipe| Box::pin(pipe) as AgentOutput),
        }
    }

    /// Wait for the process to exit. For srun this is the step's exit:
    /// the task's exit code, or 128 + signal for a signalled task.
    pub async fn wait(self) -> Result<ExitStatus> {
        match self {
            AgentChild::Ssh(child) => child.wait().await.context("waiting for remote agent"),
            AgentChild::Srun(child) => child.wait().await.context("waiting for srun step"),
        }
    }
}
