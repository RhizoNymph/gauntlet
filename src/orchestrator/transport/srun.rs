//! srun transport: every node operation is a job step of the allocation the
//! orchestrator runs in. Command lines come from `crate::launch::srun`
//! (pure, unit-tested); this file only executes them.
//!
//! No step is ever orphaned:
//! - every local `srun` is held by an `SrunChild`, which on drop (a timeout
//!   abandoning it) sends srun **SIGTERM** — srun's handler for SIGTERM is
//!   "forcing job termination": it forwards SIGKILL to every task of the
//!   step and exits (`src/srun/signals.c`, `_forward_signal`). A single
//!   SIGINT would only print task status; SIGKILL to srun would leave the
//!   step running with nobody to cancel it.
//! - every step is named (`StepName`), so `kill_steps` can find and
//!   `scancel` agent and exec steps alike, one squeue listing for any
//!   number of hosts.

use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use thiserror::Error;
use tokio::io::AsyncReadExt;
use tokio::sync::{mpsc, oneshot};
use tracing::debug;

use super::{AgentCommand, Pipe, SpawnStdio};
use crate::launch::SlurmJobId;
use crate::launch::srun::{
    SrunConfig, SrunDeploy, SrunDir, StepEntry, StepId, StepName, StepTarget, agent_step_command,
    carrier_step_args, find_step, parse_steps, sbcast_args, scancel_args, squeue_args, step_args,
    steps_to_kill,
};
use crate::orchestrator::session::RemoteOutput;

/// Typed failures of the srun transport, so callers (deploy in
/// particular) can tell a broadcast failure from a timeout or a missing
/// Slurm client.
#[derive(Debug, Error)]
pub enum SrunTransportError {
    #[error("cannot run `{program}` (is it available inside the allocation?): {source}")]
    Spawn {
        program: String,
        source: std::io::Error,
    },
    #[error("i/o with `{program}` failed: {source}")]
    Io {
        program: String,
        source: std::io::Error,
    },
    #[error("`{command}` timed out after {secs}s")]
    Timeout { command: String, secs: u64 },
    #[error("squeue failed: {detail}")]
    Squeue { detail: String },
    #[error("scancel {steps} failed: {detail}")]
    Scancel { steps: String, detail: String },
    #[error("sbcast to {dest} on {nodes} failed: {detail}")]
    Sbcast {
        dest: String,
        nodes: String,
        detail: String,
    },
    #[error("sbcast carrier step on {nodes} did not start: {detail}")]
    Carrier { nodes: String, detail: String },
    #[error("scontrol {what} failed: {detail}")]
    Scontrol { what: String, detail: String },
    #[error("{path} is not valid UTF-8")]
    NonUtf8Path { path: PathBuf },
    /// A batched kill failed for every request in the batch.
    #[error("{source}")]
    Batched { source: Arc<SrunTransportError> },
    #[error("the srun kill batcher is not running")]
    KillBatcherGone,
}

/// The Slurm client programs the launcher runs on the orchestrator's node.
/// Bare names (resolved on PATH) by default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlurmTools {
    pub srun: PathBuf,
    pub squeue: PathBuf,
    pub scancel: PathBuf,
    pub sbcast: PathBuf,
    pub scontrol: PathBuf,
}

impl Default for SlurmTools {
    fn default() -> Self {
        Self {
            srun: "srun".into(),
            squeue: "squeue".into(),
            scancel: "scancel".into(),
            sbcast: "sbcast".into(),
            scontrol: "scontrol".into(),
        }
    }
}

impl SlurmTools {
    /// All programs from one directory (site installs outside PATH, and
    /// tests).
    pub fn in_dir(dir: &Path) -> Self {
        Self {
            srun: dir.join("srun"),
            squeue: dir.join("squeue"),
            scancel: dir.join("scancel"),
            sbcast: dir.join("sbcast"),
            scontrol: dir.join("scontrol"),
        }
    }
}

/// Bound on creating and resolving the agent directory through a first
/// step. Step launch goes through slurmctld, which is slower than an ssh
/// mux handshake on a busy controller.
pub(super) const STEP_SETUP_TIMEOUT: Duration = Duration::from_secs(60);
/// Bound on one squeue + scancel round of a kill.
const KILL_TIMEOUT: Duration = Duration::from_secs(30);
/// How often a broadcast polls squeue for its carrier step to appear.
const CARRIER_POLL: Duration = Duration::from_millis(200);

/// Unique carrier-step tokens within this orchestrator process.
static CARRIER_SEQ: AtomicU64 = AtomicU64::new(0);

/// The allocation-wide srun launcher (shared by every host's session).
#[derive(Debug)]
pub struct SrunLauncher {
    job_id: SlurmJobId,
    config: SrunConfig,
    dir: SrunDir,
    /// Orchestrator env vars removed from every step (`env_to_strip`).
    strip_env: Vec<String>,
    /// The orchestrator's own LD_LIBRARY_PATH, which `--export=ALL` hands
    /// every task; the agent gets `<dir>/lib` prepended to it.
    inherited_ld_library_path: Option<String>,
    tools: SlurmTools,
    /// Sending end of the kill batcher's channel (`kill_batcher`).
    kills: mpsc::UnboundedSender<KillRequest>,
}

impl SrunLauncher {
    /// Must be called inside a tokio runtime: it starts the launcher's kill
    /// batcher task, which lives as long as the launcher.
    pub fn new(
        job_id: SlurmJobId,
        config: SrunConfig,
        dir: SrunDir,
        strip_env: Vec<String>,
        inherited_ld_library_path: Option<String>,
        tools: SlurmTools,
    ) -> Self {
        let (kills, requests) = mpsc::unbounded_channel();
        tokio::spawn(kill_batcher(tools.clone(), job_id, requests));
        Self {
            job_id,
            config,
            dir,
            strip_env,
            inherited_ld_library_path,
            tools,
            kills,
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

    pub fn tools(&self) -> &SlurmTools {
        &self.tools
    }

    /// `srun <args>` with the orchestrator's GPU-visibility and NCCL
    /// variables removed and `env` (the NCCL env) added. The local srun
    /// process's environment is what `--export=ALL` hands the task; srun
    /// itself keeps everything else, LD_LIBRARY_PATH included.
    fn srun(&self, args: Vec<String>, env: &[(String, String)]) -> tokio::process::Command {
        let mut srun = tokio::process::Command::new(&self.tools.srun);
        srun.args(args);
        for key in &self.strip_env {
            srun.env_remove(key);
        }
        srun.envs(env.iter().map(|(key, value)| (key, value)));
        srun
    }

    fn spawn_guarded(
        &self,
        mut command: tokio::process::Command,
        pipes: SpawnStdio,
    ) -> Result<SrunChild, SrunTransportError> {
        command
            .stdin(stdio(pipes.stdin))
            .stdout(stdio(pipes.stdout))
            .stderr(stdio(pipes.stderr));
        let child = command
            .spawn()
            .map_err(|source| SrunTransportError::Spawn {
                program: self.tools.srun.display().to_string(),
                source,
            })?;
        Ok(SrunChild { child: Some(child) })
    }

    /// `sh -c <script>` as a step named `gauntlet:<host>:exec`. Dropping
    /// the future (a timeout) cancels the step (`SrunChild`).
    pub(super) async fn exec_capture(
        &self,
        host: &str,
        script: &str,
    ) -> Result<RemoteOutput, SrunTransportError> {
        let command = ["sh", "-c", script].map(String::from);
        let args = step_args(
            self.config.all_flags(),
            host,
            &StepName::exec(host),
            &command,
        );
        let child = self.spawn_guarded(
            self.srun(args, &[]),
            SpawnStdio {
                stdin: Pipe::Null,
                stdout: Pipe::Piped,
                stderr: Pipe::Piped,
            },
        )?;
        child
            .output()
            .await
            .map_err(|source| SrunTransportError::Io {
                program: self.tools.srun.display().to_string(),
                source,
            })
    }

    pub(super) fn spawn(
        &self,
        command: &AgentCommand<'_>,
        pipes: SpawnStdio,
    ) -> Result<SrunChild, SrunTransportError> {
        let argv = agent_step_command(
            command.agent_path,
            command.args,
            &command.env.lib_dir,
            self.inherited_ld_library_path.as_deref(),
        );
        let name = StepName::agent(command.host, command.args);
        let args = step_args(self.config.all_flags(), command.host, &name, &argv);
        self.spawn_guarded(self.srun(args, &command.env.nccl), pipes)
    }

    /// Cancel every step any of `requests` (host, target) selects. Requests
    /// go to the launcher's kill batcher, which merges everything arriving
    /// within `KILL_BATCH_WINDOW` into one `squeue --steps` listing and one
    /// `scancel --signal=KILL` — so a fleet abort, or every host of a world
    /// hitting the same phase timeout, costs one RPC pair, not one per
    /// host. Finding nothing is not an error (the steps may have exited on
    /// their own). Returns the ids cancelled for these requests.
    pub(super) async fn kill_steps(
        &self,
        requests: &[(&str, StepTarget<'_>)],
    ) -> Result<Vec<String>, SrunTransportError> {
        let (reply, response) = oneshot::channel();
        let request = KillRequest {
            targets: requests
                .iter()
                .map(|(host, target)| (host.to_string(), OwnedTarget::from(*target)))
                .collect(),
            reply,
        };
        if self.kills.send(request).is_err() {
            return Err(SrunTransportError::KillBatcherGone);
        }
        match response.await {
            Ok(Ok(ids)) => Ok(ids),
            Ok(Err(source)) => Err(SrunTransportError::Batched { source }),
            Err(_) => Err(SrunTransportError::KillBatcherGone),
        }
    }

    async fn list_steps(&self) -> Result<Vec<StepEntry>, SrunTransportError> {
        list_steps(&self.tools, self.job_id).await
    }

    /// `sbcast` `local_file` to `dest` on exactly `hosts`.
    ///
    /// sbcast targets every node of the job unless given a step
    /// (`--jobid=<job>.<step>`, supported since at least Slurm 20.11;
    /// `--nodelist` only exists from 24.11). So a short-lived *carrier*
    /// step (`sleep`) is started on `hosts`, found by name in squeue, used
    /// as sbcast's target, and cancelled afterwards — whatever happened.
    pub async fn broadcast(
        &self,
        hosts: &[&str],
        local_file: &Path,
        dest: &str,
        timeout: Duration,
    ) -> Result<(), SrunTransportError> {
        let source = local_file
            .to_str()
            .ok_or_else(|| SrunTransportError::NonUtf8Path {
                path: local_file.to_path_buf(),
            })?;
        let nodes = hosts.join(",");
        let token = format!(
            "{}-{}",
            std::process::id(),
            CARRIER_SEQ.fetch_add(1, Ordering::Relaxed)
        );
        let name = StepName::carrier(&token);
        // The carrier outlives the broadcast window; it is cancelled as
        // soon as sbcast returns.
        let lifetime = timeout.as_secs().saturating_mul(2).max(60);
        let args = carrier_step_args(self.config.all_flags(), hosts, &name, lifetime);
        let mut carrier = self.spawn_guarded(
            self.srun(args, &[]),
            SpawnStdio {
                stdin: Pipe::Null,
                stdout: Pipe::Null,
                stderr: Pipe::Null,
            },
        )?;

        let deadline = tokio::time::Instant::now() + timeout;
        let result = async {
            let step = loop {
                if let Some(status) = carrier.try_status()? {
                    return Err(SrunTransportError::Carrier {
                        nodes: nodes.clone(),
                        detail: format!("srun exited with {status}"),
                    });
                }
                let listing = self.list_steps().await?;
                if let Some(step) = find_step(&listing, &name) {
                    break step.clone();
                }
                if tokio::time::Instant::now() >= deadline {
                    return Err(SrunTransportError::Timeout {
                        command: format!("srun carrier step {name}"),
                        secs: timeout.as_secs(),
                    });
                }
                tokio::time::sleep(CARRIER_POLL).await;
            };
            let jobid = step.sbcast_jobid(self.job_id);
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            let output = local(
                &self.tools.sbcast,
                &sbcast_args(&jobid, source, dest),
                remaining.max(Duration::from_secs(1)),
            )
            .await?;
            let cancelled = local(
                &self.tools.scancel,
                &scancel_args(std::slice::from_ref(&step)),
                KILL_TIMEOUT,
            )
            .await;
            if let Err(error) = cancelled {
                debug!(%error, step = %step, "carrier step not cancelled; srun teardown follows");
            }
            if !output.success() {
                return Err(SrunTransportError::Sbcast {
                    dest: dest.to_string(),
                    nodes: nodes.clone(),
                    detail: output.detail(),
                });
            }
            debug!(dest, nodes = %nodes, step = %step, "sbcast complete");
            Ok(())
        }
        .await;
        // Dropping the guard SIGTERMs a carrier that is still running.
        drop(carrier);
        result
    }
}

/// A kill target that can cross the batcher channel.
#[derive(Debug, Clone, PartialEq, Eq)]
enum OwnedTarget {
    Agent(String),
    Exec,
}

impl From<StepTarget<'_>> for OwnedTarget {
    fn from(target: StepTarget<'_>) -> Self {
        match target {
            StepTarget::Agent(args) => OwnedTarget::Agent(args.to_string()),
            StepTarget::Exec => OwnedTarget::Exec,
        }
    }
}

impl OwnedTarget {
    fn borrow(&self) -> StepTarget<'_> {
        match self {
            OwnedTarget::Agent(args) => StepTarget::Agent(args),
            OwnedTarget::Exec => StepTarget::Exec,
        }
    }
}

type KillReply = Result<Vec<String>, Arc<SrunTransportError>>;

struct KillRequest {
    targets: Vec<(String, OwnedTarget)>,
    reply: oneshot::Sender<KillReply>,
}

/// How long the batcher keeps collecting requests after the first one of a
/// batch arrives. Kills from one abort or one shared phase timeout land
/// within a few milliseconds of each other.
const KILL_BATCH_WINDOW: Duration = Duration::from_millis(100);

/// The launcher's kill batcher: owns the receiving end of the kill
/// channel; each batch is one squeue listing and one scancel. Ends when
/// the launcher (the only sender) is dropped.
async fn kill_batcher(
    tools: SlurmTools,
    job_id: SlurmJobId,
    mut requests: mpsc::UnboundedReceiver<KillRequest>,
) {
    while let Some(first) = requests.recv().await {
        let mut batch = vec![first];
        let window = tokio::time::sleep(KILL_BATCH_WINDOW);
        tokio::pin!(window);
        loop {
            tokio::select! {
                _ = &mut window => break,
                next = requests.recv() => match next {
                    Some(request) => batch.push(request),
                    None => break,
                },
            }
        }
        let outcome = kill_batch(&tools, job_id, &batch).await;
        for (request, result) in batch.into_iter().zip(outcome) {
            // A requester that gave up waiting is not an error.
            let _ = request.reply.send(result);
        }
    }
}

/// One squeue + one scancel for a whole batch; per-request results.
async fn kill_batch(
    tools: &SlurmTools,
    job_id: SlurmJobId,
    batch: &[KillRequest],
) -> Vec<KillReply> {
    let failed = |error: SrunTransportError| {
        let error = Arc::new(error);
        batch.iter().map(|_| Err(Arc::clone(&error))).collect()
    };
    let listing = match list_steps(tools, job_id).await {
        Ok(listing) => listing,
        Err(error) => return failed(error),
    };
    let per_request: Vec<Vec<StepId>> = batch
        .iter()
        .map(|request| {
            let targets: Vec<(&str, StepTarget<'_>)> = request
                .targets
                .iter()
                .map(|(host, target)| (host.as_str(), target.borrow()))
                .collect();
            steps_to_kill(&listing, &targets)
        })
        .collect();
    let mut union: Vec<StepId> = per_request.iter().flatten().cloned().collect();
    union.sort();
    union.dedup();
    if !union.is_empty() {
        match local(&tools.scancel, &scancel_args(&union), KILL_TIMEOUT).await {
            Ok(output) if output.success() => {}
            // A step may have ended between squeue and scancel; scancel
            // still signals the rest, so this is reported, not retried.
            Ok(output) => {
                return failed(SrunTransportError::Scancel {
                    steps: union
                        .iter()
                        .map(StepId::to_string)
                        .collect::<Vec<_>>()
                        .join(" "),
                    detail: output.detail(),
                });
            }
            Err(error) => return failed(error),
        }
    }
    per_request
        .into_iter()
        .map(|ids| Ok(ids.iter().map(StepId::to_string).collect()))
        .collect()
}

async fn list_steps(
    tools: &SlurmTools,
    job_id: SlurmJobId,
) -> Result<Vec<StepEntry>, SrunTransportError> {
    let listing = local(&tools.squeue, &squeue_args(job_id), KILL_TIMEOUT).await?;
    if !listing.success() {
        return Err(SrunTransportError::Squeue {
            detail: listing.detail(),
        });
    }
    Ok(parse_steps(&listing.stdout))
}

/// A local `srun` process standing for one step. Dropped while srun is
/// still running (a caller's timeout), it sends srun SIGTERM, which makes
/// srun SIGKILL the step's tasks and exit — the step is cancelled rather
/// than orphaned. (tokio's `kill_on_drop` would SIGKILL srun itself and
/// leave the step behind.)
pub struct SrunChild {
    child: Option<tokio::process::Child>,
}

impl SrunChild {
    pub(super) fn take_stdin(&mut self) -> Option<tokio::process::ChildStdin> {
        self.child.as_mut().and_then(|child| child.stdin.take())
    }

    pub(super) fn take_stdout(&mut self) -> Option<tokio::process::ChildStdout> {
        self.child.as_mut().and_then(|child| child.stdout.take())
    }

    pub(super) fn take_stderr(&mut self) -> Option<tokio::process::ChildStderr> {
        self.child.as_mut().and_then(|child| child.stderr.take())
    }

    /// srun's exit status, which is the step's: the task's exit code, or
    /// 128 + signal for a task killed by a signal.
    pub(super) async fn wait(mut self) -> std::io::Result<ExitStatus> {
        match self.child.as_mut() {
            Some(child) => child.wait().await,
            None => Err(std::io::Error::other("srun child already consumed")),
        }
    }

    fn try_status(&mut self) -> Result<Option<ExitStatus>, SrunTransportError> {
        match self.child.as_mut() {
            Some(child) => child.try_wait().map_err(|source| SrunTransportError::Io {
                program: "srun".into(),
                source,
            }),
            None => Ok(None),
        }
    }

    /// Read stdout and stderr to the end, then wait.
    async fn output(mut self) -> std::io::Result<RemoteOutput> {
        let stdout = self.take_stdout();
        let stderr = self.take_stderr();
        let (stdout, stderr) = tokio::try_join!(read_all(stdout), read_all(stderr))?;
        let status = self.wait().await?;
        Ok(RemoteOutput {
            status,
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
        })
    }
}

impl Drop for SrunChild {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        // Still running (not yet reaped, so the pid cannot have been
        // reused): ask srun to cancel its step.
        if let (Ok(None), Some(pid)) = (child.try_wait(), child.id()) {
            let sent = std::process::Command::new("kill")
                .args(["-s", "TERM", &pid.to_string()])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            match sent {
                Ok(status) if status.success() => {
                    debug!(pid, "abandoned srun sent SIGTERM; its step is cancelled")
                }
                Ok(status) => debug!(pid, %status, "SIGTERM to abandoned srun not delivered"),
                Err(error) => debug!(pid, %error, "cannot signal abandoned srun"),
            }
        }
        // tokio reaps the dropped child in the background.
    }
}

async fn read_all<R: tokio::io::AsyncRead + Unpin>(pipe: Option<R>) -> std::io::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    if let Some(mut pipe) = pipe {
        pipe.read_to_end(&mut buffer).await?;
    }
    Ok(buffer)
}

fn stdio(pipe: Pipe) -> Stdio {
    match pipe {
        Pipe::Piped => Stdio::piped(),
        Pipe::Null => Stdio::null(),
    }
}

/// Run a Slurm client command on the orchestrator's node, bounded by
/// `timeout` (the process is killed if it overruns: these are plain RPC
/// clients, not steps).
pub(crate) async fn local(
    program: &Path,
    args: &[String],
    timeout: Duration,
) -> Result<RemoteOutput, SrunTransportError> {
    let command_text = || format!("{} {}", program.display(), args.join(" "));
    let output = tokio::process::Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .kill_on_drop(true)
        .output();
    match tokio::time::timeout(timeout, output).await {
        Ok(Ok(output)) => Ok(RemoteOutput::from_output(output)),
        Ok(Err(source)) => Err(SrunTransportError::Spawn {
            program: program.display().to_string(),
            source,
        }),
        Err(_) => Err(SrunTransportError::Timeout {
            command: command_text(),
            secs: timeout.as_secs(),
        }),
    }
}
