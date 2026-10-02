//! srun transport: every node operation is a job step of the allocation the
//! orchestrator runs in. Command lines come from `crate::launch::srun`
//! (pure, unit-tested); this file only executes them.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use tracing::debug;

use super::{AgentCommand, Pipe, SpawnStdio};
use crate::launch::SlurmJobId;
use crate::launch::srun::{
    SrunConfig, SrunDeploy, SrunDir, StepName, parse_steps, sbcast_args, scancel_args, squeue_args,
    step_args, steps_to_kill,
};
use crate::orchestrator::session::RemoteOutput;

pub(super) const SCONTROL: &str = "scontrol";

/// The Slurm client programs the launcher runs on the orchestrator's node.
/// Bare names (resolved on PATH) by default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlurmTools {
    pub srun: PathBuf,
    pub squeue: PathBuf,
    pub scancel: PathBuf,
    pub sbcast: PathBuf,
}

impl Default for SlurmTools {
    fn default() -> Self {
        Self {
            srun: "srun".into(),
            squeue: "squeue".into(),
            scancel: "scancel".into(),
            sbcast: "sbcast".into(),
        }
    }
}

impl SlurmTools {
    /// All four programs from one directory (site installs outside PATH,
    /// and tests).
    pub fn in_dir(dir: &Path) -> Self {
        Self {
            srun: dir.join("srun"),
            squeue: dir.join("squeue"),
            scancel: dir.join("scancel"),
            sbcast: dir.join("sbcast"),
        }
    }
}

/// Bound on creating and resolving the agent directory through a first
/// step. Step launch goes through slurmctld, which is slower than an ssh
/// mux handshake on a busy controller.
pub(super) const STEP_SETUP_TIMEOUT: Duration = Duration::from_secs(60);

/// The allocation-wide srun launcher (shared by every host's session).
#[derive(Debug)]
pub struct SrunLauncher {
    job_id: SlurmJobId,
    config: SrunConfig,
    dir: SrunDir,
    /// Orchestrator env vars removed from every step (`env_to_strip`).
    strip_env: Vec<String>,
    tools: SlurmTools,
}

impl SrunLauncher {
    pub fn new(
        job_id: SlurmJobId,
        config: SrunConfig,
        dir: SrunDir,
        strip_env: Vec<String>,
        tools: SlurmTools,
    ) -> Self {
        Self {
            job_id,
            config,
            dir,
            strip_env,
            tools,
        }
    }

    pub fn job_id(&self) -> SlurmJobId {
        self.job_id
    }

    pub fn dir(&self) -> &SrunDir {
        &self.dir
    }

    pub fn deploy(&self) -> SrunDeploy {
        self.config.deploy
    }

    /// `srun <step args>` for `command` on `host`, with the orchestrator's
    /// GPU-visibility and NCCL variables removed and `env` added. The
    /// local srun process's environment is what `--export=ALL` hands the
    /// task, so `env` reaches the agent as-is: no shell, no quoting.
    fn command(
        &self,
        host: &str,
        name: &StepName,
        command: &[String],
        env: &[(String, String)],
    ) -> tokio::process::Command {
        let mut srun = tokio::process::Command::new(&self.tools.srun);
        srun.args(step_args(self.config.all_flags(), host, name, command));
        for key in &self.strip_env {
            srun.env_remove(key);
        }
        srun.envs(env.iter().map(|(key, value)| (key, value)));
        // A dropped child (timeout) must not leave a local srun behind;
        // the remote task is then killed through `kill_agent`.
        srun.kill_on_drop(true);
        srun
    }

    pub(super) async fn exec_capture(&self, host: &str, script: &str) -> Result<RemoteOutput> {
        let command = ["sh", "-c", script].map(String::from);
        let output = self
            .command(host, &StepName::exec(host), &command, &[])
            .stdin(Stdio::null())
            .output()
            .await
            .with_context(|| format!("running `{script}` on {host} via srun"))?;
        Ok(RemoteOutput::from_output(output))
    }

    pub(super) fn spawn(
        &self,
        command: &AgentCommand<'_>,
        pipes: SpawnStdio,
    ) -> Result<tokio::process::Child> {
        let argv: Vec<String> = [command.agent_path, "agent"]
            .into_iter()
            .chain(command.args.iter().copied())
            .map(String::from)
            .collect();
        let name = StepName::agent(command.host, command.args);
        self.command(command.host, &name, &argv, command.env)
            .stdin(stdio(pipes.stdin))
            .stdout(stdio(pipes.stdout))
            .stderr(stdio(pipes.stderr))
            .spawn()
            .with_context(|| {
                format!(
                    "spawning srun step {name} (is `{}` available inside the allocation?)",
                    self.tools.srun.display()
                )
            })
    }

    /// Find this job's steps running `agent <args>` on `host` by name and
    /// SIGKILL each (`scancel --signal=KILL <job>.<step>`). Returns what
    /// was cancelled; finding nothing is not an error (the step may have
    /// exited on its own).
    pub(super) async fn kill_agent(&self, host: &str, args: &str) -> Result<String> {
        let listing = local(&self.tools.squeue, &squeue_args(self.job_id)).await?;
        if !listing.success() {
            bail!("squeue failed: {}", listing.detail());
        }
        let targets = steps_to_kill(&parse_steps(&listing.stdout), self.job_id, host, args);
        if targets.is_empty() {
            return Ok("no matching step".to_string());
        }
        let mut cancelled = Vec::new();
        for step in targets {
            let output = local(&self.tools.scancel, &scancel_args(step)).await?;
            if output.success() {
                cancelled.push(step.to_string());
            } else {
                // The step may have ended between squeue and scancel.
                debug!(%step, detail = %output.detail(), "scancel did not succeed");
            }
        }
        Ok(format!("scancel --signal=KILL {}", cancelled.join(" ")))
    }

    /// `sbcast` the local file to `dest` on every node of the allocation.
    pub async fn broadcast(&self, local_file: &Path, dest: &str) -> Result<()> {
        let source = local_file
            .to_str()
            .with_context(|| format!("{} is not valid UTF-8", local_file.display()))?;
        let output = local(&self.tools.sbcast, &sbcast_args(self.job_id, source, dest)).await?;
        if !output.success() {
            bail!("sbcast to {dest} failed: {}", output.detail());
        }
        debug!(dest, job_id = %self.job_id, "sbcast complete");
        Ok(())
    }
}

fn stdio(pipe: Pipe) -> Stdio {
    match pipe {
        Pipe::Piped => Stdio::piped(),
        Pipe::Null => Stdio::null(),
    }
}

/// Run a Slurm client command on the orchestrator's node.
async fn local(program: &Path, args: &[String]) -> Result<RemoteOutput> {
    let output = tokio::process::Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .kill_on_drop(true)
        .output()
        .await
        .with_context(|| format!("running `{} {}`", program.display(), args.join(" ")))?;
    Ok(RemoteOutput::from_output(output))
}
