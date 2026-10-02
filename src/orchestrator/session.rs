//! One session per host, over whichever transport the run launches with
//! (`transport::Launcher`: persistent ssh, or srun job steps). Every agent
//! interaction — run to completion streaming events, capture a JSON
//! document, start in the background, kill — goes through here, so the
//! rest of the orchestrator never knows which transport is in use.

use std::path::Path;
use std::process::ExitStatus;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, warn};

use super::deploy::AGENT_RELPATH;
use super::transport::{AgentChild, AgentCommand, HostTransport, Pipe, SpawnStdio};
use crate::config::HostConfig;
use crate::launch::LaunchMode;
use crate::nccl_env::NcclEnv;
use crate::proto::{AgentEvent, PROTO_VERSION, decode_event};

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
    /// The agent directory after node-side resolution; always absolute.
    remote_dir: String,
    /// `<remote_dir>/bin/gauntlet-agent`.
    agent_path: String,
    /// The environment every agent spawn starts under (`AgentEnv`).
    agent_env: AgentEnv,
}

impl HostSession {
    /// Finish opening a session over an established `transport`: create
    /// and resolve the agent directory on the node (`dir` as configured;
    /// a leading `~` is the *node's* home), bounded by `timeout`.
    pub(crate) async fn establish(
        host: HostConfig,
        transport: HostTransport,
        dir: &str,
        nccl_env: &NcclEnv,
        timeout: Duration,
    ) -> Result<HostSession> {
        let remote_dir =
            tokio::time::timeout(timeout, resolve_remote_dir(&transport, &host.addr, dir))
                .await
                .map_err(|_| {
                    anyhow!(
                        "resolving the agent directory {dir} on {} timed out after {}s",
                        host.addr,
                        timeout.as_secs()
                    )
                })??;
        let agent_path = format!("{remote_dir}/{AGENT_RELPATH}");
        let agent_env = AgentEnv::new(&remote_dir, nccl_env);
        debug!(host = %host.addr, remote_dir = %remote_dir, "session established");
        Ok(HostSession {
            host,
            transport,
            remote_dir,
            agent_path,
            agent_env,
        })
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

    /// Absolute agent directory on the node (no `~`).
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
    /// fleet-wide through `deploy::ensure_fleet`). Staged then renamed, so
    /// a running copy of the old binary cannot make the write fail with
    /// `ETXTBSY` and readers never observe a half-written file.
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

    async fn spawn(&self, args: &[&str], stdio: SpawnStdio) -> Result<AgentChild> {
        self.transport
            .spawn(
                AgentCommand {
                    host: &self.host.addr,
                    env: &self.agent_env,
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
        args: &[&str],
        stdin_doc: Option<String>,
    ) -> Result<AgentChild> {
        let mut child = self
            .spawn(
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

    /// Spawn the deployed agent with `args`, write `stdin_doc` (a single
    /// JSON line) to its stdin, and stream decoded events to `on_event`
    /// until EOF. Returns the remote exit status.
    pub async fn run_agent(
        &self,
        args: &[&str],
        stdin_doc: Option<String>,
        mut on_event: impl FnMut(AgentEvent) + Send,
    ) -> Result<ExitStatus> {
        let mut child = self.spawn_with_document(args, stdin_doc).await?;

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

    /// Run the agent to completion and capture its stdout verbatim. Used for
    /// the sub-commands that print a single JSON document rather than the
    /// event stream (`probe`, `peer latency|bandwidth`, `nccl` id relay).
    pub async fn run_agent_capture(
        &self,
        args: &[&str],
        stdin_doc: Option<String>,
    ) -> Result<RemoteOutput> {
        let mut child = self.spawn_with_document(args, stdin_doc).await?;
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

    /// Start the agent in the background and hand back the child. The handle
    /// owns what it needs (a session clone, or the local srun process), so
    /// it outlives this borrow (phase 3 keeps a `peer serve` running while
    /// it drives the other end of the pair).
    pub async fn spawn_agent(&self, args: &[&str]) -> Result<AgentChild> {
        let mut child = self
            .spawn(
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
    /// node; srun: `scancel --signal=KILL` of the matching step. Best
    /// effort: a failure is logged, not raised.
    pub async fn kill_agent(&self, args: &str) {
        match self.transport.kill_agent(&self.host.addr, args).await {
            Ok(action) => debug!(host = %self.addr(), args, action, "remote agent cleanup sent"),
            Err(error) => warn!(host = %self.addr(), args, %error, "remote agent cleanup failed"),
        }
    }

    /// `kill_agent` for many hosts at once: over srun one squeue listing
    /// and one scancel cover all of them (the NCCL early-abort path kills
    /// every surviving host of a world). Best effort.
    pub async fn kill_agents(sessions: &[std::sync::Arc<HostSession>], args: &str) {
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

/// `mkdir -p` the agent directory and report its absolute path. The
/// expansion happens on the node: `~` means the *node's* home directory.
async fn resolve_remote_dir(transport: &HostTransport, addr: &str, dir: &str) -> Result<String> {
    let quoted = shell_path(dir);
    let script = format!("mkdir -p {quoted} && cd {quoted} && pwd");
    let output = transport.exec_capture(addr, &script).await?;
    if !output.success() {
        bail!(
            "cannot prepare agent directory {dir} on {addr}: {}",
            output.detail()
        );
    }
    let path = output.stdout.trim().to_string();
    if path.is_empty() {
        bail!("agent directory {dir} on {addr} resolved to an empty path");
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

/// The environment every agent invocation starts under:
///
/// - `lib_dir` (`<remote_dir>/lib`) on LD_LIBRARY_PATH: it holds shim
///   symlinks bootstrap may have created for runtime-only libraries (e.g.
///   libnccl.so -> libnccl.so.2); dlopen consults LD_LIBRARY_PATH as
///   captured at process start, so it must be set at spawn time. ssh sets
///   `LD_LIBRARY_PATH=<lib_dir>` on the remote `env` line (unchanged);
///   srun runs `env LD_LIBRARY_PATH=<lib_dir>:<inherited>` *inside* the
///   step, so the local srun client keeps the orchestrator's own path.
/// - `nccl`: one `NCCL_*=<value>` per resolved NCCL env entry, in key
///   order. NCCL reads its knobs with `getenv` at communicator init;
///   setting them at spawn puts them in the environment before the agent
///   starts any thread. The agent itself never calls `set_var` — it runs
///   on a multi-threaded tokio runtime, where mutating the environment is
///   unsound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AgentEnv {
    pub lib_dir: String,
    pub nccl: Vec<(String, String)>,
}

impl AgentEnv {
    pub(crate) fn new(remote_dir: &str, nccl_env: &NcclEnv) -> Self {
        Self {
            lib_dir: format!("{remote_dir}/lib"),
            nccl: nccl_env
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect(),
        }
    }

    /// `KEY='value'` words for the ssh `env` command line, each a single
    /// quoted POSIX-sh word (they go through `raw_arg`, unescaped by
    /// openssh): `LD_LIBRARY_PATH` first, then the NCCL env. Keys are
    /// literal shell words (validated `^NCCL_[A-Z0-9_]+$`); values are
    /// single-quoted, so spaces, quotes, `$`, backticks and globs stay
    /// literal.
    pub(crate) fn ssh_words(&self) -> Vec<String> {
        std::iter::once(format!("LD_LIBRARY_PATH={}", single_quote(&self.lib_dir)))
            .chain(
                self.nccl
                    .iter()
                    .map(|(key, value)| format!("{key}={}", single_quote(value))),
            )
            .collect()
    }
}

/// `AgentEnv` rendered as ssh `env` words (what `transport::ssh` puts on
/// the remote command line).
#[cfg(test)]
pub(crate) fn agent_env_words(remote_dir: &str, nccl_env: &NcclEnv) -> Vec<String> {
    AgentEnv::new(remote_dir, nccl_env).ssh_words()
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
#[cfg(test)]
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
