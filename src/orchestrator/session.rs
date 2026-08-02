//! One persistent ssh session per host, via the `openssh` crate
//! (native-mux). Sessions honor ~/.ssh/config (jump hosts, agent auth,
//! aliases); `ssh.user` / `ssh.key` from the fleet config override when set.

use std::path::Path;
use std::process::ExitStatus;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use openssh::{KnownHosts, Session, SessionBuilder, Stdio};
use openssh_sftp_client::{Sftp, SftpOptions};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, warn};

use super::deploy::AGENT_RELPATH;
use crate::config::{HostConfig, SshConfig};
use crate::proto::{AgentEvent, PROTO_VERSION, decode_event};

/// Captured result of a remote command that was allowed to fail.
#[derive(Debug, Clone)]
pub struct RemoteOutput {
    pub status: ExitStatus,
    pub stdout: String,
    pub stderr: String,
}

impl RemoteOutput {
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
    session: Arc<Session>,
    /// `ssh.remote_dir` after remote `~` expansion; always absolute.
    remote_dir: String,
    /// `<remote_dir>/bin/gauntlet-agent`.
    agent_path: String,
}

impl HostSession {
    /// Establish a session. `connect_timeout_secs` applies; errors carry the
    /// host address for attribution.
    pub async fn connect(host: HostConfig, ssh: &SshConfig) -> Result<HostSession> {
        let timeout = Duration::from_secs(ssh.connect_timeout_secs.max(1));

        let mut builder = SessionBuilder::default();
        builder
            .known_hosts_check(KnownHosts::Add)
            .connect_timeout(timeout)
            .server_alive_interval(Duration::from_secs(30));
        if let Some(user) = &ssh.user {
            builder.user(user.clone());
        }
        if let Some(key) = &ssh.key {
            builder.keyfile(key);
        }

        let session = tokio::time::timeout(timeout, builder.connect_mux(&host.addr))
            .await
            .map_err(|_| {
                anyhow!(
                    "ssh connect to {} timed out after {}s",
                    host.addr,
                    timeout.as_secs()
                )
            })?
            .with_context(|| format!("ssh connect to {}", host.addr))?;
        let session = Arc::new(session);

        let remote_dir = tokio::time::timeout(
            timeout,
            resolve_remote_dir(&session, &host.addr, &ssh.remote_dir),
        )
        .await
        .map_err(|_| {
            anyhow!(
                "resolving remote_dir on {} timed out after {}s",
                host.addr,
                timeout.as_secs()
            )
        })??;

        let agent_path = format!("{remote_dir}/{AGENT_RELPATH}");
        debug!(host = %host.addr, remote_dir = %remote_dir, "ssh session established");
        Ok(HostSession {
            host,
            session,
            remote_dir,
            agent_path,
        })
    }

    pub fn addr(&self) -> &str {
        &self.host.addr
    }

    /// Absolute scratch directory on the node (no `~`).
    pub fn remote_dir(&self) -> &str {
        &self.remote_dir
    }

    /// Absolute path of the deployed agent binary.
    pub fn agent_path(&self) -> &str {
        &self.agent_path
    }

    /// Environment for every agent invocation. `<remote_dir>/lib` holds
    /// shim symlinks bootstrap may have created for runtime-only libraries
    /// (e.g. libnccl.so -> libnccl.so.2); dlopen consults LD_LIBRARY_PATH
    /// as captured at process start, so it must be set at spawn time.
    fn agent_env(&self) -> String {
        format!("LD_LIBRARY_PATH={}/lib", self.remote_dir)
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
        run_shell(&self.session, &self.host.addr, command).await
    }

    /// Upload a local file to `remote_path` (sftp), creating parent dirs,
    /// setting the executable bit when `executable`.
    ///
    /// The bytes land in a sibling temp file that is then renamed into place,
    /// so a concurrently running copy of the old binary cannot make the write
    /// fail with `ETXTBSY` and readers never observe a half-written file.
    pub async fn upload(&self, local: &Path, remote_path: &str, executable: bool) -> Result<()> {
        let bytes = tokio::fs::read(local)
            .await
            .with_context(|| format!("reading local file {}", local.display()))?;
        let parent = remote_path
            .rsplit_once('/')
            .map(|(parent, _)| parent)
            .filter(|parent| !parent.is_empty())
            .unwrap_or(".");
        self.exec(&format!("mkdir -p {}", single_quote(parent)))
            .await
            .with_context(|| format!("creating remote directory {parent}"))?;

        let staging = format!("{remote_path}.staging");
        self.sftp_write(&staging, &bytes)
            .await
            .with_context(|| format!("uploading {} to {}", local.display(), staging))?;

        let mode = if executable { "755" } else { "644" };
        self.exec(&format!(
            "chmod {mode} {staging} && mv -f {staging} {dest}",
            staging = single_quote(&staging),
            dest = single_quote(remote_path),
        ))
        .await
        .with_context(|| format!("installing {remote_path}"))?;
        debug!(
            host = %self.host.addr,
            remote_path,
            bytes = bytes.len(),
            "uploaded file"
        );
        Ok(())
    }

    async fn sftp_write(&self, remote_path: &str, bytes: &[u8]) -> Result<()> {
        let mut child = self
            .session
            .subsystem("sftp")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .await
            .context("spawning sftp subsystem")?;
        let stdin = child.stdin().take().context("sftp stdin unavailable")?;
        let stdout = child.stdout().take().context("sftp stdout unavailable")?;

        let sftp = Sftp::new(stdin, stdout, SftpOptions::default())
            .await
            .context("sftp handshake")?;
        let result = async {
            let mut options = sftp.options();
            options.write(true).create(true).truncate(true);
            let mut file = options
                .open(remote_path)
                .await
                .with_context(|| format!("opening {remote_path} for write"))?;
            file.write_all(bytes)
                .await
                .with_context(|| format!("writing {remote_path}"))?;
            file.close().await.context("closing remote file")?;
            Ok::<(), anyhow::Error>(())
        }
        .await;
        sftp.close().await.context("closing sftp session")?;
        result?;
        child.wait().await.context("waiting for sftp subsystem")?;
        Ok(())
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
        let mut command = self.session.command("env");
        command
            .arg(self.agent_env())
            .arg(self.agent_path.clone())
            // The deployed binary is the full multi-command CLI; node-side
            // modes all live under its `agent` subcommand.
            .arg("agent")
            .args(args.iter().copied())
            .stdin(if stdin_doc.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .await
            .with_context(|| format!("spawning agent {args:?} on {}", self.host.addr))?;

        if let Some(doc) = stdin_doc {
            let mut stdin = child.stdin().take().context("agent stdin unavailable")?;
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

        let stderr_task = child
            .stderr()
            .take()
            .map(|stderr| tokio::spawn(drain_stderr(self.host.addr.clone(), stderr)));

        let stdout = child.stdout().take().context("agent stdout unavailable")?;
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
        let mut command = self.session.command("env");
        command
            .arg(self.agent_env())
            .arg(self.agent_path.clone())
            // The deployed binary is the full multi-command CLI; node-side
            // modes all live under its `agent` subcommand.
            .arg("agent")
            .args(args.iter().copied())
            .stdin(if stdin_doc.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .await
            .with_context(|| format!("spawning agent {args:?} on {}", self.host.addr))?;

        if let Some(doc) = stdin_doc {
            let mut stdin = child.stdin().take().context("agent stdin unavailable")?;
            stdin
                .write_all(doc.as_bytes())
                .await
                .context("writing agent stdin document")?;
            stdin
                .write_all(b"\n")
                .await
                .context("writing agent stdin")?;
            stdin.flush().await.context("flushing agent stdin")?;
            drop(stdin);
        }

        let output = child
            .wait_with_output()
            .await
            .with_context(|| format!("running agent {args:?} on {}", self.host.addr))?;
        Ok(RemoteOutput {
            status: output.status,
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }

    /// Start the agent in the background and hand back the child. The handle
    /// owns a clone of the session, so it outlives this borrow (phase 3 keeps
    /// a `peer serve` running while it drives the other end of the pair).
    pub async fn spawn_agent(&self, args: &[&str]) -> Result<openssh::Child<Arc<Session>>> {
        let mut command = Arc::clone(&self.session).arc_command("env");
        command
            .arg(self.agent_env())
            .arg(self.agent_path.clone())
            .arg("agent")
            .args(args.iter().copied())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .await
            .with_context(|| format!("spawning agent {args:?} on {}", self.host.addr))?;
        if let Some(stderr) = child.stderr().take() {
            tokio::spawn(drain_stderr(self.host.addr.clone(), stderr));
        }
        Ok(child)
    }
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

async fn drain_stderr(host: String, stderr: openssh::ChildStderr) {
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

/// `mkdir -p` the scratch directory and report its absolute path. The
/// expansion happens on the node: `~` means the *remote* home directory.
async fn resolve_remote_dir(session: &Session, addr: &str, remote_dir: &str) -> Result<String> {
    let quoted = shell_path(remote_dir);
    let script = format!("mkdir -p {quoted} && cd {quoted} && pwd");
    let output = run_shell(session, addr, &script).await?;
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

async fn run_shell(session: &Session, addr: &str, script: &str) -> Result<RemoteOutput> {
    let output = session
        .command("sh")
        .arg("-c")
        .arg(script)
        .output()
        .await
        .with_context(|| format!("running `{script}` on {addr}"))?;
    Ok(RemoteOutput {
        status: output.status,
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
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
