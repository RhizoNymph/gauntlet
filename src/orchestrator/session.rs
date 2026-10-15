//! One session per host, over whichever transport the run launches with
//! (`transport::Launcher`: persistent ssh, or srun job steps). Every agent
//! interaction — run to completion streaming events, capture a JSON
//! document, start in the background, kill — goes through here, so the
//! rest of the orchestrator never knows which transport is in use.

use std::path::Path;
use std::process::ExitStatus;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, warn};

use super::deploy::AGENT_RELPATH;
use super::transport::{AgentChild, AgentCommand, HostTransport, Pipe, SpawnStdio};
use crate::config::HostConfig;
use crate::launch::LaunchMode;
use crate::nccl_env::{NcclEnv, NcclEnvError};
use crate::nccl_level::{NcclLevel, NcclLevelEnvs, PerLevel};
use crate::nccl_transport::debug::{DebugSettings, capture_env, managed_log_path};
use crate::proto::{AgentEvent, PROTO_VERSION, decode_event};
use crate::remote_dir::RemoteDir;

pub(crate) mod remote_fs;

use self::remote_fs::remote_dir_script;

/// Which environment an agent spawn is started with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentEnv {
    /// The process hosts no NCCL communicator (inventory, cpu/mem, gpu,
    /// counters, probe, peer, TCP barrier): LD_LIBRARY_PATH plus the
    /// global `[nccl]` env — harmless there, and uniform.
    Base,
    /// The process hosts the communicator of this level: LD_LIBRARY_PATH,
    /// the level's effective env (global <- `[nccl.levels.<level>]`), and
    /// the NCCL debug-log variables transport capture needs, never
    /// overriding one the config sets (`nccl_transport::debug`).
    Nccl(NcclLevel),
}

/// Captured result of a remote command that was allowed to fail.
#[derive(Debug, Clone)]
pub struct RemoteOutput {
    pub status: ExitStatus,
    pub stdout: String,
    pub stderr: String,
}

impl RemoteOutput {
    pub(crate) fn from_output(output: std::process::Output) -> Self {
        Self {
            status: output.status,
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        }
    }

    pub fn success(&self) -> bool {
        self.status.success()
    }

    /// Short one-line reason suitable for a readiness-matrix cell.
    pub fn detail(&self) -> String {
        let text = if !self.stderr.trim().is_empty() {
            self.stderr.trim()
        } else {
            self.stdout.trim()
        };
        let mut lines = text.lines().map(str::trim).filter(|line| !line.is_empty());
        let first = lines.next().unwrap_or("");
        // A Rust panic banner carries its message on the *next* line; keeping
        // only the header would drop the one part worth reading.
        let summary = if first.starts_with("thread '") && first.contains("panicked at") {
            match lines.next() {
                Some(message) => format!("panicked: {message}"),
                None => first.to_string(),
            }
        } else {
            first.to_string()
        };
        if summary.is_empty() {
            format!("exit {}", exit_code(&self.status))
        } else {
            format!("exit {}: {summary}", exit_code(&self.status))
        }
    }
}

fn exit_code(status: &ExitStatus) -> i32 {
    status.code().unwrap_or(-1)
}

pub struct HostSession {
    pub host: HostConfig,
    pub(crate) transport: HostTransport,
    /// The agent directory after node-side resolution (`remote_dir_script`):
    /// the physical, owner-checked path; always absolute.
    remote_dir: String,
    /// `<remote_dir>/bin/gauntlet-agent`.
    agent_path: String,
    /// The resolved NCCL env of every level (`nccl_levels`).
    nccl: Arc<NcclLevelEnvs>,
    /// Pre-quoted env words per `AgentEnv` (ssh: the remote `env` line),
    /// computed once at connect (`SpawnWords::build`).
    words: SpawnWords,
    /// The same environments as typed maps (srun: the local srun process's
    /// environment, exported to the step), computed once at connect.
    envs: SpawnEnvs,
}

/// The env words of every `AgentEnv` for one host: `Base` plus one list
/// per NCCL level, each already quoted (`spawn_env_words`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpawnWords {
    base: Vec<String>,
    levels: PerLevel<Vec<String>>,
}

impl SpawnWords {
    pub(crate) fn build(remote_dir: &str, nccl: &NcclLevelEnvs) -> Result<Self, NcclEnvError> {
        Ok(Self {
            base: spawn_env_words(remote_dir, nccl, AgentEnv::Base)?,
            levels: PerLevel::try_from_fn(|level| {
                spawn_env_words(remote_dir, nccl, AgentEnv::Nccl(level))
            })?,
        })
    }

    pub(crate) fn get(&self, env: AgentEnv) -> &[String] {
        match env {
            AgentEnv::Base => &self.base,
            AgentEnv::Nccl(level) => self.levels.get(level),
        }
    }
}

/// The NCCL env of every `AgentEnv` for one host (`spawn_env`), unquoted:
/// what the srun transport sets on the local srun process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpawnEnvs {
    base: NcclEnv,
    levels: PerLevel<NcclEnv>,
}

impl SpawnEnvs {
    pub(crate) fn build(remote_dir: &str, nccl: &NcclLevelEnvs) -> Result<Self, NcclEnvError> {
        Ok(Self {
            base: spawn_env(remote_dir, nccl, AgentEnv::Base)?,
            levels: PerLevel::try_from_fn(|level| {
                spawn_env(remote_dir, nccl, AgentEnv::Nccl(level))
            })?,
        })
    }

    pub(crate) fn get(&self, env: AgentEnv) -> &NcclEnv {
        match env {
            AgentEnv::Base => &self.base,
            AgentEnv::Nccl(level) => self.levels.get(level),
        }
    }
}

impl HostSession {
    /// Finish opening a session over an established `transport`: create,
    /// check and resolve the agent directory on the node
    /// (`remote_dir_script`: mode 700, owned by the login user, no foreign
    /// symlink, `nccl-logs/` inside, physical path), bounded by `timeout`.
    /// Every agent spawn of this session carries the env `nccl` resolves
    /// for its `AgentEnv`.
    pub(crate) async fn establish(
        host: HostConfig,
        transport: HostTransport,
        dir: &RemoteDir,
        nccl: Arc<NcclLevelEnvs>,
        timeout: Duration,
    ) -> Result<HostSession> {
        let remote_dir =
            tokio::time::timeout(timeout, resolve_remote_dir(&transport, &host.addr, dir))
                .await
                .map_err(|_| {
                    anyhow!(
                        "resolving remote_dir {dir} on {} timed out after {}s",
                        host.addr,
                        timeout.as_secs()
                    )
                })??;
        let agent_path = format!("{remote_dir}/{AGENT_RELPATH}");
        let unfit = || {
            format!(
                "remote_dir {remote_dir} on {} cannot hold the NCCL debug log",
                host.addr
            )
        };
        let words = SpawnWords::build(&remote_dir, &nccl).with_context(unfit)?;
        let envs = SpawnEnvs::build(&remote_dir, &nccl).with_context(unfit)?;
        debug!(host = %host.addr, remote_dir = %remote_dir, "session established");
        Ok(HostSession {
            host,
            transport,
            remote_dir,
            agent_path,
            nccl,
            words,
            envs,
        })
    }

    /// The run's resolved NCCL env, per level.
    pub fn nccl_levels(&self) -> &NcclLevelEnvs {
        &self.nccl
    }

    pub fn addr(&self) -> &str {
        &self.host.addr
    }

    pub fn launch_mode(&self) -> LaunchMode {
        match self.transport {
            HostTransport::Ssh(_) => LaunchMode::Ssh,
            HostTransport::Srun(_) => LaunchMode::Srun,
        }
    }

    /// Absolute agent directory on the node (physical, no `~`).
    pub fn remote_dir(&self) -> &str {
        &self.remote_dir
    }

    /// Absolute path of the deployed agent binary.
    pub fn agent_path(&self) -> &str {
        &self.agent_path
    }

    /// Run a short remote command, capturing stdout (used by deploy for
    /// hash checks and by bootstrap tuning). Non-zero exit is an error.
    pub async fn exec(&self, command: &str) -> Result<String> {
        let output = self.exec_capture(command).await?;
        if !output.success() {
            bail!(
                "remote command failed on {}: `{command}` ({})",
                self.host.addr,
                output.detail()
            );
        }
        Ok(output.stdout)
    }

    /// Like [`HostSession::exec`] but a non-zero exit is returned instead of
    /// raised: bootstrap tuning reports refusals as warnings.
    pub async fn exec_capture(&self, command: &str) -> Result<RemoteOutput> {
        self.transport.exec_capture(&self.host.addr, command).await
    }

    /// Upload a local file over the session (ssh/sftp only; srun deploys
    /// fleet-wide through `deploy::ensure_fleet`). Staged under a random
    /// sibling name and renamed into place (`transport::ssh::upload`).
    pub async fn upload(&self, local: &Path, remote_path: &str, executable: bool) -> Result<()> {
        match self.transport.ssh_session() {
            Some(session) => {
                super::transport::ssh_upload(
                    session,
                    &self.host.addr,
                    local,
                    remote_path,
                    executable,
                )
                .await
            }
            None => bail!(
                "{}: per-host upload is not available in srun launch mode",
                self.host.addr
            ),
        }
    }

    async fn spawn(&self, env: AgentEnv, args: &[&str], stdio: SpawnStdio) -> Result<AgentChild> {
        let lib_dir = format!("{}/lib", self.remote_dir);
        self.transport
            .spawn(
                AgentCommand {
                    host: &self.host.addr,
                    ssh_words: self.words.get(env),
                    lib_dir: &lib_dir,
                    nccl: self.envs.get(env),
                    agent_path: &self.agent_path,
                    args,
                },
                stdio,
            )
            .await
            .with_context(|| format!("spawning agent {args:?} on {}", self.host.addr))
    }

    /// Spawn with stdin piped when there is a document to send, write it
    /// (one JSON line) and close stdin so the agent sees EOF and starts.
    async fn spawn_with_document(
        &self,
        env: AgentEnv,
        args: &[&str],
        stdin_doc: Option<String>,
    ) -> Result<AgentChild> {
        let mut child = self
            .spawn(
                env,
                args,
                SpawnStdio {
                    stdin: if stdin_doc.is_some() {
                        Pipe::Piped
                    } else {
                        Pipe::Null
                    },
                    stdout: Pipe::Piped,
                    stderr: Pipe::Piped,
                },
            )
            .await?;
        if let Some(doc) = stdin_doc {
            let mut stdin = child.take_stdin().context("agent stdin unavailable")?;
            stdin
                .write_all(doc.as_bytes())
                .await
                .context("writing agent stdin document")?;
            stdin
                .write_all(b"\n")
                .await
                .context("writing agent stdin")?;
            stdin.flush().await.context("flushing agent stdin")?;
            // Dropping closes the pipe, so the agent sees EOF and starts work.
            drop(stdin);
        }
        Ok(child)
    }

    /// Spawn the deployed agent with `args` under `env`, write `stdin_doc`
    /// (a single JSON line) to its stdin, and stream decoded events to
    /// `on_event` until EOF. Returns the remote exit status.
    pub async fn run_agent(
        &self,
        env: AgentEnv,
        args: &[&str],
        stdin_doc: Option<String>,
        mut on_event: impl FnMut(AgentEvent) + Send,
    ) -> Result<ExitStatus> {
        let mut child = self.spawn_with_document(env, args, stdin_doc).await?;

        let stderr_task = child
            .take_stderr()
            .map(|stderr| tokio::spawn(drain_stderr(self.host.addr.clone(), stderr)));

        let stdout = child.take_stdout().context("agent stdout unavailable")?;
        let mut lines = BufReader::new(stdout).lines();
        let mut checked_hello = false;
        while let Some(line) = lines
            .next_line()
            .await
            .with_context(|| format!("reading agent stdout from {}", self.host.addr))?
        {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match decode_event(line) {
                Ok(event) => {
                    if !checked_hello {
                        checked_hello = true;
                        check_hello(&self.host.addr, &event)?;
                    }
                    on_event(event);
                }
                Err(error) => {
                    warn!(
                        host = %self.host.addr,
                        %error,
                        line,
                        "discarding malformed agent event line"
                    );
                }
            }
        }

        let status = child
            .wait()
            .await
            .with_context(|| format!("waiting for agent on {}", self.host.addr))?;
        if let Some(task) = stderr_task {
            let _ = task.await;
        }
        Ok(status)
    }

    /// Run the agent to completion (base env) and capture its stdout
    /// verbatim. Used for the sub-commands that print a single JSON
    /// document rather than the event stream (`probe`, `peer
    /// latency|bandwidth`, TCP barrier).
    pub async fn run_agent_capture(
        &self,
        args: &[&str],
        stdin_doc: Option<String>,
    ) -> Result<RemoteOutput> {
        let mut child = self
            .spawn_with_document(AgentEnv::Base, args, stdin_doc)
            .await?;
        let stdout = child.take_stdout().context("agent stdout unavailable")?;
        let stderr = child.take_stderr().context("agent stderr unavailable")?;
        let (stdout, stderr) = tokio::try_join!(read_all(stdout), read_all(stderr))
            .with_context(|| format!("running agent {args:?} on {}", self.host.addr))?;
        let status = child
            .wait()
            .await
            .with_context(|| format!("running agent {args:?} on {}", self.host.addr))?;
        Ok(RemoteOutput {
            status,
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
        })
    }

    /// Start the agent in the background (base env) and hand back the
    /// child. The handle owns what it needs (a session clone, or the local
    /// srun process), so it outlives this borrow (phase 3 keeps a `peer
    /// serve` running while it drives the other end of the pair).
    pub async fn spawn_agent(&self, args: &[&str]) -> Result<AgentChild> {
        let mut child = self
            .spawn(
                AgentEnv::Base,
                args,
                SpawnStdio {
                    stdin: Pipe::Null,
                    stdout: Pipe::Null,
                    stderr: Pipe::Piped,
                },
            )
            .await?;
        if let Some(stderr) = child.take_stderr() {
            tokio::spawn(drain_stderr(self.host.addr.clone(), stderr));
        }
        Ok(child)
    }

    /// Kill this host's agent running `agent <args...>` (word prefix:
    /// `barrier serve --port 29500` also matches its trailing flags).
    /// Dropping the local future on a timeout does not stop the remote
    /// process — it can sit blocked in a collective or a socket and never
    /// write to stdout again, so it never even dies of SIGPIPE — so every
    /// timeout that abandons an agent ends here. ssh: `pkill -f` on the
    /// node; srun: `scancel --signal=KILL` of the matching step (batched
    /// with concurrent kills). Best effort: a failure is logged, not raised.
    pub async fn kill_agent(&self, args: &str) {
        match self.transport.kill_agent(&self.host.addr, args).await {
            Ok(action) => debug!(host = %self.addr(), args, action, "remote agent cleanup sent"),
            Err(error) => warn!(host = %self.addr(), args, %error, "remote agent cleanup failed"),
        }
    }

    /// `kill_agent` for many hosts at once: over srun one squeue listing
    /// and one scancel cover all of them (the NCCL early-abort path kills
    /// every surviving host of a world). Best effort.
    pub async fn kill_agents(sessions: &[Arc<HostSession>], args: &str) {
        super::transport::kill_agents(sessions, args).await;
    }
}

async fn read_all(mut stream: super::transport::AgentOutput) -> std::io::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    stream.read_to_end(&mut buffer).await?;
    Ok(buffer)
}

fn check_hello(host: &str, event: &AgentEvent) -> Result<()> {
    match event {
        AgentEvent::Hello {
            proto_version,
            hostname,
        } => {
            if *proto_version != PROTO_VERSION {
                bail!(
                    "{host}: agent speaks proto v{proto_version}, orchestrator requires \
                     v{PROTO_VERSION}"
                );
            }
            debug!(host, hostname, "agent hello");
            Ok(())
        }
        other => {
            warn!(host, event = ?other, "agent stream did not start with hello");
            Ok(())
        }
    }
}

async fn drain_stderr(host: String, stderr: impl AsyncRead + Unpin) {
    let mut lines = BufReader::new(stderr).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                if !line.trim().is_empty() {
                    debug!(host = %host, line = %line, "agent stderr");
                }
            }
            Ok(None) => break,
            Err(error) => {
                debug!(host = %host, %error, "agent stderr closed");
                break;
            }
        }
    }
}

/// Create, check and resolve the agent directory on the node and report its
/// physical path. The expansion happens on the node: `~` means the *remote*
/// home directory, `$USER` the remote login name (`remote_dir_script`).
/// Same script over either transport.
async fn resolve_remote_dir(
    transport: &HostTransport,
    addr: &str,
    remote_dir: &RemoteDir,
) -> Result<String> {
    let script = remote_dir_script(remote_dir);
    let output = transport.exec_capture(addr, &script).await?;
    if !output.success() {
        bail!(
            "cannot prepare remote_dir {remote_dir} on {addr}: {}",
            output.detail()
        );
    }
    let path = output.stdout.trim().to_string();
    if path.is_empty() {
        bail!("remote_dir {remote_dir} on {addr} resolved to an empty path");
    }
    Ok(path)
}

/// `pkill -f` pattern for a deployed agent running `agent <args>`. The
/// remote command line is `<remote_dir>/bin/gauntlet-agent agent <args>`
/// (`HostSession::run_agent`); the bracket keeps the pattern from matching
/// the `pkill` invocation itself.
pub(crate) fn agent_kill_pattern(args: &str) -> String {
    format!("[g]auntlet-agent agent {args}")
}

/// Environment words for every agent invocation, in order, each already a
/// single quoted POSIX-sh word (they go through `raw_arg`, unescaped by
/// openssh):
///
/// - `LD_LIBRARY_PATH=<remote_dir>/lib`: `<remote_dir>/lib` holds shim
///   symlinks bootstrap may have created for runtime-only libraries (e.g.
///   libnccl.so -> libnccl.so.2); dlopen consults LD_LIBRARY_PATH as
///   captured at process start, so it must be set at spawn time. (srun
///   applies it inside the step instead, prepended to the inherited
///   value: `launch::srun::agent_step_command`.)
/// - one `NCCL_*=<value>` per resolved NCCL env entry, in key order. NCCL
///   reads its knobs with `getenv` at communicator init; setting them here
///   puts them in the environment before the agent starts any thread. The
///   agent itself never calls `set_var` — it runs on a multi-threaded tokio
///   runtime, where mutating the environment is unsound.
///
/// Keys are validated `^NCCL_[A-Z0-9_]+$` (literal shell words); values are
/// single-quoted, so spaces, quotes, `$`, backticks and globs stay literal.
pub(crate) fn agent_env_words(remote_dir: &str, nccl_env: &NcclEnv) -> Vec<String> {
    std::iter::once(format!(
        "LD_LIBRARY_PATH={}",
        single_quote(&format!("{remote_dir}/lib"))
    ))
    .chain(nccl_env_words(nccl_env))
    .collect()
}

/// The NCCL env of a spawn of kind `env` (see `AgentEnv`): `Base` is the
/// global env; `Nccl(level)` the level's effective env plus the NCCL
/// debug-log additions (`capture_env`, whose keys never collide with the
/// env's) overlaid as one `NcclEnv`. Fails only when the managed log path
/// is not a valid env value.
pub(crate) fn spawn_env(
    remote_dir: &str,
    nccl: &NcclLevelEnvs,
    env: AgentEnv,
) -> Result<NcclEnv, NcclEnvError> {
    let level = match env {
        AgentEnv::Base => return Ok(nccl.global().clone()),
        AgentEnv::Nccl(level) => level,
    };
    let level_env = nccl.level(level);
    let capture = capture_env(
        DebugSettings::from_env(level_env),
        &managed_log_path(remote_dir, level),
    )?;
    Ok(level_env.overlay(&capture))
}

/// Environment words for a spawn of kind `env`, each a quoted POSIX-sh
/// word: `spawn_env` through `agent_env_words`, the single quoting path.
pub(crate) fn spawn_env_words(
    remote_dir: &str,
    nccl: &NcclLevelEnvs,
    env: AgentEnv,
) -> Result<Vec<String>, NcclEnvError> {
    Ok(agent_env_words(
        remote_dir,
        &spawn_env(remote_dir, nccl, env)?,
    ))
}

/// Every remote word after the `env` program of an agent spawn, each already
/// a quoted POSIX-sh word (passed through `raw_args`): the env assignments,
/// the agent binary, then `agent <args>`. The deployed binary is the full
/// multi-command CLI; node-side modes all live under its `agent`
/// subcommand.
///
/// `env` execs the binary, so the agent process's own argv is exactly
/// `<agent_path> agent <args>` — the assignments never appear in it, and
/// `agent_kill_pattern` (`[g]auntlet-agent agent <args>`) keeps matching
/// with any NCCL env.
pub(crate) fn agent_spawn_args(
    env_words: &[String],
    agent_path: &str,
    args: &[&str],
) -> Vec<String> {
    env_words
        .iter()
        .cloned()
        .chain([single_quote(agent_path), "agent".to_string()])
        .chain(args.iter().map(|arg| single_quote(arg)))
        .collect()
}

/// `KEY='value'` words for the NCCL env, in key order; empty for an empty
/// env (no prefix at all).
pub(crate) fn nccl_env_words(nccl_env: &NcclEnv) -> Vec<String> {
    nccl_env
        .iter()
        .map(|(key, value)| format!("{key}={}", single_quote(value)))
        .collect()
}

/// Quote `path` as a single POSIX-sh word. A leading `~` is turned into
/// `$HOME` inside double quotes so the *remote* shell expands it; every other
/// path is single-quoted verbatim.
pub(crate) fn shell_path(path: &str) -> String {
    if path == "~" {
        return "\"$HOME\"".to_string();
    }
    match path.strip_prefix("~/") {
        Some(rest) => format!("\"$HOME/{}\"", escape_double_quoted(rest)),
        None => single_quote(path),
    }
}

/// Quote an arbitrary string as a literal POSIX-sh word.
pub(crate) fn single_quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('\'');
    for ch in value.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

fn escape_double_quoted(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if matches!(ch, '"' | '\\' | '$' | '`') {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tilde_expands_through_home_on_the_remote() {
        assert_eq!(shell_path("~"), "\"$HOME\"");
        assert_eq!(shell_path("~/.gauntlet"), "\"$HOME/.gauntlet\"");
        assert_eq!(shell_path("~/a b"), "\"$HOME/a b\"");
    }

    #[test]
    fn absolute_paths_are_literal() {
        assert_eq!(shell_path("/opt/gauntlet"), "'/opt/gauntlet'");
        assert_eq!(shell_path("/opt/g x"), "'/opt/g x'");
        // A `~` that is not the first character is not a home reference.
        assert_eq!(shell_path("/opt/~x"), "'/opt/~x'");
    }

    fn nccl_env(entries: &[(&str, &str)]) -> NcclEnv {
        let raw = entries
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        NcclEnv::from_map(&raw).expect("valid env")
    }

    #[test]
    fn empty_nccl_env_adds_no_words() {
        assert!(nccl_env_words(&NcclEnv::default()).is_empty());
        assert_eq!(
            agent_env_words("/home/u/.gauntlet", &NcclEnv::default()),
            vec!["LD_LIBRARY_PATH='/home/u/.gauntlet/lib'".to_string()]
        );
    }

    #[test]
    fn env_words_follow_ld_library_path_in_key_order() {
        let raw = [("NCCL_IB_HCA", "mlx5_0,mlx5_1"), ("NCCL_DEBUG", "WARN")]
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        let env = NcclEnv::resolve(Some("bond0"), &raw).expect("valid env");
        assert_eq!(
            agent_env_words("/opt/g", &env),
            vec![
                "LD_LIBRARY_PATH='/opt/g/lib'".to_string(),
                "NCCL_DEBUG='WARN'".to_string(),
                "NCCL_IB_HCA='mlx5_0,mlx5_1'".to_string(),
                "NCCL_SOCKET_IFNAME='bond0'".to_string(),
            ]
        );
    }

    fn level_envs() -> NcclLevelEnvs {
        let global = nccl_env(&[("NCCL_IB_HCA", "mlx5_0")]);
        let overrides = std::collections::BTreeMap::from([
            (NcclLevel::Intranode, nccl_env(&[("NCCL_ALGO", "Ring")])),
            (NcclLevel::OverlapFleet, nccl_env(&[("NCCL_DEBUG", "WARN")])),
        ]);
        NcclLevelEnvs::resolve(global, &overrides)
    }

    fn words_for(remote_dir: &str, envs: &NcclLevelEnvs, env: AgentEnv) -> Vec<String> {
        spawn_env_words(remote_dir, envs, env).expect("valid words")
    }

    #[test]
    fn spawn_words_are_precomputed_per_agent_env() {
        let envs = level_envs();
        let words = SpawnWords::build("/opt/g", &envs).expect("valid words");
        assert_eq!(
            words.get(AgentEnv::Base),
            words_for("/opt/g", &envs, AgentEnv::Base)
        );
        for level in NcclLevel::ALL {
            assert_eq!(
                words.get(AgentEnv::Nccl(level)),
                words_for("/opt/g", &envs, AgentEnv::Nccl(level)),
                "{level}"
            );
        }
        // A remote_dir that cannot be an env value fails at connect, not
        // at some later spawn.
        assert!(SpawnWords::build("/opt/g\nx", &envs).is_err());
    }

    #[test]
    fn base_spawns_carry_the_global_env_without_capture_variables() {
        let envs = level_envs();
        assert_eq!(
            words_for("/opt/g", &envs, AgentEnv::Base),
            agent_env_words("/opt/g", envs.global())
        );
        assert_eq!(
            words_for("/opt/g", &envs, AgentEnv::Base),
            vec![
                "LD_LIBRARY_PATH='/opt/g/lib'".to_string(),
                "NCCL_IB_HCA='mlx5_0'".to_string(),
            ]
        );
    }

    #[test]
    fn nccl_spawns_carry_their_levels_env_plus_capture_variables() {
        let envs = level_envs();
        assert_eq!(
            words_for("/opt/g", &envs, AgentEnv::Nccl(NcclLevel::Intranode)),
            vec![
                "LD_LIBRARY_PATH='/opt/g/lib'".to_string(),
                "NCCL_ALGO='Ring'".to_string(),
                "NCCL_DEBUG='INFO'".to_string(),
                "NCCL_DEBUG_FILE='/opt/g/nccl-logs/intranode.%h.log'".to_string(),
                "NCCL_DEBUG_SUBSYS='INIT,NET'".to_string(),
                "NCCL_IB_HCA='mlx5_0'".to_string(),
            ]
        );
        // The intra-node override stays out of the fleet process.
        let fleet = words_for("/opt/g", &envs, AgentEnv::Nccl(NcclLevel::Fleet));
        assert!(
            !fleet.iter().any(|word| word.starts_with("NCCL_ALGO=")),
            "{fleet:?}"
        );
        assert!(
            fleet.contains(&"NCCL_DEBUG_FILE='/opt/g/nccl-logs/fleet.%h.log'".to_string()),
            "{fleet:?}"
        );
    }

    #[test]
    fn a_configured_nccl_debug_is_never_overridden() {
        let words = words_for(
            "/opt/g",
            &level_envs(),
            AgentEnv::Nccl(NcclLevel::OverlapFleet),
        );
        assert_eq!(
            words,
            vec![
                "LD_LIBRARY_PATH='/opt/g/lib'".to_string(),
                "NCCL_DEBUG='WARN'".to_string(),
                "NCCL_IB_HCA='mlx5_0'".to_string(),
            ]
        );
        // Every key appears once, even with capture additions.
        for level in NcclLevel::ALL {
            let words = words_for("/opt/g", &level_envs(), AgentEnv::Nccl(level));
            let keys: Vec<&str> = words
                .iter()
                .filter_map(|word| word.split_once('=').map(|(key, _)| key))
                .collect();
            let mut unique = keys.clone();
            unique.dedup();
            assert_eq!(keys, unique, "{level}: {words:?}");
        }
    }

    #[test]
    fn the_debug_file_pattern_reaches_nccl_unexpanded() {
        // `%h` is NCCL's placeholder, not the shell's: it must arrive as is.
        let words = words_for(
            "/home/u/.gauntlet dir",
            &NcclLevelEnvs::default(),
            AgentEnv::Nccl(NcclLevel::Barrier),
        );
        let script = format!("env {} printenv NCCL_DEBUG_FILE", words.join(" "));
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .output()
            .expect("run sh");
        assert!(output.status.success(), "{script}");
        assert_eq!(
            String::from_utf8(output.stdout).expect("utf8"),
            "/home/u/.gauntlet dir/nccl-logs/barrier.%h.log\n"
        );
    }

    /// The real contract: after a POSIX shell parses the words (as the
    /// remote login shell does), the process sees each value byte for byte
    /// and nothing expands or executes.
    #[test]
    fn adversarial_values_survive_a_real_shell_verbatim() {
        let values = [
            "mlx5_0,mlx5_1",
            "a b  c",
            "it's",
            "'''",
            "\"double\"",
            "$HOME",
            "${PATH}",
            "$(touch /nonexistent/pwned)",
            "`id`",
            "a;b|c&d>e<f",
            "*",
            "~",
            "back\\slash",
            "=mlx5_0:1",
            "-x",
            "^docker0,lo",
            "\u{e9}t\u{e9}",
        ];
        for value in values {
            let words = nccl_env_words(&nccl_env(&[("NCCL_TEST_VALUE", value)]));
            let script = format!("env {} printenv NCCL_TEST_VALUE", words.join(" "));
            let output = std::process::Command::new("sh")
                .arg("-c")
                .arg(&script)
                .output()
                .expect("run sh");
            assert!(output.status.success(), "{value:?}: {script}");
            let stdout = String::from_utf8(output.stdout).expect("utf8");
            assert_eq!(
                stdout.strip_suffix('\n'),
                Some(value),
                "value mangled by the shell: {script}"
            );
        }
    }

    /// Control characters never reach a command line: single quoting only
    /// holds across login shells (csh/tcsh break on a quoted newline) for
    /// one-line values, so `NcclEnv` refuses them up front.
    #[test]
    fn control_character_values_are_rejected_before_quoting() {
        for value in ["line\nbreak", "tab\there", "cr\r", "nul\0", "del\u{7f}"] {
            let raw = [("NCCL_TEST_VALUE".to_string(), value.to_string())]
                .into_iter()
                .collect();
            assert!(
                matches!(
                    NcclEnv::from_map(&raw),
                    Err(crate::nccl_env::NcclEnvError::InvalidValue {
                        reason: crate::nccl_env::NcclEnvValueError::ControlCharacter,
                        ..
                    })
                ),
                "{value:?} must be rejected"
            );
        }
    }

    #[test]
    fn quoting_neutralizes_metacharacters() {
        assert_eq!(single_quote("a'b"), "'a'\\''b'");
        assert_eq!(single_quote("$(rm -rf /)"), "'$(rm -rf /)'");
        assert_eq!(shell_path("~/$x`y`"), "\"$HOME/\\$x\\`y\\`\"");
    }

    #[test]
    fn detail_surfaces_the_panic_message_line() {
        use std::os::unix::process::ExitStatusExt;
        let output = RemoteOutput {
            status: std::process::ExitStatus::from_raw(1 << 8),
            stdout: String::new(),
            stderr: "thread 'main' (123) panicked at cudarc-0.19.8/src/lib.rs:200:5:\n\
                     Unable to dynamically load the \"nccl\" shared library\n\
                     note: run with RUST_BACKTRACE=1"
                .to_string(),
        };
        let detail = output.detail();
        assert!(detail.contains("panicked:"), "{detail}");
        assert!(detail.contains("nccl"), "{detail}");
        assert!(
            !detail.contains("lib.rs:200"),
            "header line must be replaced: {detail}"
        );
    }

    #[test]
    fn detail_keeps_ordinary_first_lines() {
        use std::os::unix::process::ExitStatusExt;
        let output = RemoteOutput {
            status: std::process::ExitStatus::from_raw(2 << 8),
            stdout: String::new(),
            stderr: "error: unrecognized subcommand 'probe'\nUsage: gauntlet ...".to_string(),
        };
        assert_eq!(
            output.detail(),
            "exit 2: error: unrecognized subcommand 'probe'"
        );
    }
}
