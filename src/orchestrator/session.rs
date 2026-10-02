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
use crate::nccl_env::NcclEnv;
use crate::proto::{AgentEvent, PROTO_VERSION, decode_event};
use crate::remote_dir::{RemoteDir, RemoteDirPart};

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
    /// Pre-quoted `KEY=value` words placed between `env` and the agent
    /// binary on every spawn (`agent_env_words`).
    env_words: Vec<String>,
}

impl HostSession {
    /// Establish a session. `connect_timeout_secs` applies; errors carry the
    /// host address for attribution.
    /// `nccl_env` is set on every agent spawn of this session.
    pub async fn connect(
        host: HostConfig,
        ssh: &SshConfig,
        nccl_env: &NcclEnv,
    ) -> Result<HostSession> {
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
        let env_words = agent_env_words(&remote_dir, nccl_env);
        debug!(host = %host.addr, remote_dir = %remote_dir, "ssh session established");
        Ok(HostSession {
            host,
            session,
            remote_dir,
            agent_path,
            env_words,
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

    /// The remote words after the `env` program for an `agent <args>` spawn
    /// (`agent_spawn_args`), shared by every spawn path.
    fn spawn_args(&self, args: &[&str]) -> Vec<String> {
        agent_spawn_args(&self.env_words, &self.agent_path, args)
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
    /// The bytes land in a sibling temp file with a per-upload random suffix
    /// (`staging_path`) that is then renamed into place (`install_command`),
    /// so a concurrently running copy of the old binary cannot make the
    /// write fail with `ETXTBSY`, readers never observe a half-written file,
    /// and concurrent uploads to one shared path (several nodes on one NFS
    /// `remote_dir`) never write into the same temp file: each renames a
    /// complete file, the last rename wins. A failed upload removes its temp
    /// file (best effort); a killed one leaves only an unreferenced
    /// `*.tmp.*` sibling, never a torn destination.
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

        let staging = staging_path(remote_path, upload_nonce());
        let mode = if executable {
            FileMode::Executable
        } else {
            FileMode::Regular
        };
        let installed = async {
            self.sftp_write(&staging, &bytes)
                .await
                .with_context(|| format!("uploading {} to {}", local.display(), staging))?;
            self.exec(&install_command(&staging, remote_path, mode))
                .await
                .with_context(|| format!("installing {remote_path}"))?;
            Ok::<(), anyhow::Error>(())
        }
        .await;
        if let Err(error) = installed {
            if let Err(cleanup) = self
                .exec_capture(&format!("rm -f {}", single_quote(&staging)))
                .await
            {
                debug!(host = %self.host.addr, staging, %cleanup, "staging cleanup failed");
            }
            return Err(error);
        }
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
            .raw_args(self.spawn_args(args))
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
            .raw_args(self.spawn_args(args))
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
            .raw_args(self.spawn_args(args))
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
/// expansion happens on the node: `~` means the *remote* home directory,
/// `$USER` the remote login name (`remote_dir_script`).
async fn resolve_remote_dir(
    session: &Session,
    addr: &str,
    remote_dir: &RemoteDir,
) -> Result<String> {
    let script = remote_dir_script(remote_dir);
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

/// Environment words for every agent invocation, in order, each already a
/// single quoted POSIX-sh word (they go through `raw_arg`, unescaped by
/// openssh):
///
/// - `LD_LIBRARY_PATH=<remote_dir>/lib`: `<remote_dir>/lib` holds shim
///   symlinks bootstrap may have created for runtime-only libraries (e.g.
///   libnccl.so -> libnccl.so.2); dlopen consults LD_LIBRARY_PATH as
///   captured at process start, so it must be set at spawn time.
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

/// Upload file mode, applied before the rename so the destination is
/// never visible with the wrong bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FileMode {
    Executable,
    Regular,
}

impl FileMode {
    fn octal(self) -> &'static str {
        match self {
            FileMode::Executable => "755",
            FileMode::Regular => "644",
        }
    }
}

/// Sibling temp name for one upload: `<dest>.tmp.<16 hex>`. Same directory
/// as the destination, so the final `mv` is a same-filesystem rename(2).
pub(crate) fn staging_path(dest: &str, nonce: u64) -> String {
    format!("{dest}.tmp.{nonce:016x}")
}

/// A fresh nonce per upload: randomly keyed std hasher over the clock, the
/// pid and a process-wide counter. Collisions only matter between
/// concurrent uploaders of one path (different processes or hosts), which
/// differ in key, pid or clock.
fn upload_nonce() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or_default(),
    );
    hasher.write_u32(std::process::id());
    hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    hasher.finish()
}

/// Set the mode on the staged file, then atomically rename it over `dest`.
pub(crate) fn install_command(staging: &str, dest: &str, mode: FileMode) -> String {
    format!(
        "chmod {mode} {staging} && mv -f {staging} {dest}",
        mode = mode.octal(),
        staging = single_quote(staging),
        dest = single_quote(dest),
    )
}

/// `remote_dir` as one POSIX-sh word: literal runs quoted by `shell_path`
/// (the first may carry a leading `~`) or `single_quote`, each `$USER`
/// replaced by `"$(id -un)"` — the remote login name, whether or not the
/// login environment exports USER.
pub(crate) fn remote_dir_word(remote_dir: &RemoteDir) -> String {
    remote_dir
        .parts()
        .into_iter()
        .enumerate()
        .map(|(index, part)| match part {
            RemoteDirPart::Literal(text) if index == 0 => shell_path(text),
            RemoteDirPart::Literal(text) => single_quote(text),
            RemoteDirPart::User => "\"$(id -un)\"".to_string(),
        })
        .collect()
}

/// Create the scratch directory (mode 700 when this creates it), refuse one
/// owned by another user — a world-writable parent like /tmp lets anyone
/// pre-create `/tmp/gauntlet-<you>` and plant a binary there — and print
/// its absolute path.
pub(crate) fn remote_dir_script(remote_dir: &RemoteDir) -> String {
    let word = remote_dir_word(remote_dir);
    format!(
        "mkdir -p -m 700 {word} && cd {word} && \
         {{ [ -O . ] || {{ echo \"remote_dir $(pwd) is not owned by $(id -un)\" >&2; exit 1; }}; }} && \
         pwd"
    )
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

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gauntlet-session-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn sh(script: &str, home: &std::path::Path) -> std::process::Output {
        std::process::Command::new("sh")
            .arg("-c")
            .arg(script)
            .env("HOME", home)
            .output()
            .expect("run sh")
    }

    fn login_name() -> String {
        let output = std::process::Command::new("id")
            .arg("-un")
            .output()
            .expect("id -un");
        String::from_utf8(output.stdout)
            .expect("utf8")
            .trim()
            .to_string()
    }

    #[test]
    fn user_expands_on_the_remote_shell_and_nothing_else_does() {
        let dir = RemoteDir::parse("/tmp/gauntlet-$USER/a b/'q'").expect("valid");
        let word = remote_dir_word(&dir);
        assert_eq!(word, r#"'/tmp/gauntlet-'"$(id -un)"'/a b/'\''q'\'''"#);
        let home = scratch("word");
        let output = sh(&format!("printf '%s' {word}"), &home);
        assert!(output.status.success());
        assert_eq!(
            String::from_utf8(output.stdout).expect("utf8"),
            format!("/tmp/gauntlet-{}/a b/'q'", login_name())
        );
        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn remote_dir_script_creates_a_private_per_user_dir() {
        use std::os::unix::fs::PermissionsExt;
        let base = scratch("script");
        let template = format!("{}/g-$USER", base.display());
        let dir = RemoteDir::parse(&template).expect("valid");
        let output = sh(&remote_dir_script(&dir), &base);
        assert!(output.status.success(), "{output:?}");
        let resolved = String::from_utf8(output.stdout).expect("utf8");
        let expected = base.join(format!("g-{}", login_name()));
        assert_eq!(
            std::path::Path::new(resolved.trim())
                .canonicalize()
                .expect("exists"),
            expected.canonicalize().expect("exists")
        );
        let mode = std::fs::metadata(&expected)
            .expect("dir")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o700, "{mode:o}");
        // Idempotent: a second resolve of the existing, owned dir succeeds.
        assert!(sh(&remote_dir_script(&dir), &base).status.success());
        std::fs::remove_dir_all(&base).expect("cleanup");
    }

    #[test]
    fn an_explicit_home_remote_dir_still_resolves_under_home() {
        let home = scratch("home");
        let dir = RemoteDir::parse("~/.gauntlet").expect("valid");
        let output = sh(&remote_dir_script(&dir), &home);
        assert!(output.status.success(), "{output:?}");
        let resolved = String::from_utf8(output.stdout).expect("utf8");
        assert_eq!(
            std::path::Path::new(resolved.trim())
                .canonicalize()
                .expect("exists"),
            home.join(".gauntlet").canonicalize().expect("exists")
        );
        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn a_scratch_dir_owned_by_someone_else_is_refused() {
        // `/` is owned by root; unless the tests run as root, `[ -O ]`
        // fails and the script must refuse instead of printing a path.
        if login_name() == "root" {
            return;
        }
        let dir = RemoteDir::parse("/").expect("valid");
        let output = sh(&remote_dir_script(&dir), &std::env::temp_dir());
        assert!(!output.status.success(), "{output:?}");
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8(output.stderr).expect("utf8");
        assert!(stderr.contains("not owned by"), "{stderr}");
    }

    #[test]
    fn staging_names_are_unique_siblings_of_the_destination() {
        let dest = "/tmp/gauntlet-u/bin/gauntlet-agent";
        let a = staging_path(dest, upload_nonce());
        let b = staging_path(dest, upload_nonce());
        assert_ne!(a, b);
        for staging in [&a, &b] {
            let (dir, name) = staging.rsplit_once('/').expect("path");
            assert_eq!(dir, "/tmp/gauntlet-u/bin");
            assert!(name.starts_with("gauntlet-agent.tmp."), "{name}");
            assert_eq!(name.len(), "gauntlet-agent.tmp.".len() + 16, "{name}");
        }
        assert_eq!(staging_path("/x/y", 0xab), "/x/y.tmp.00000000000000ab");
    }

    #[test]
    fn install_renames_a_complete_file_into_place() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("install");
        let dest = dir.join("bin dir/gauntlet-agent");
        std::fs::create_dir_all(dest.parent().expect("parent")).expect("mkdir");
        let dest = dest.to_str().expect("utf8").to_string();
        std::fs::write(&dest, b"old binary").expect("old");

        // Two uploaders racing on one shared path each stage their own file;
        // whichever renames last wins, and the destination is always one
        // complete file, never a mix.
        let first = staging_path(&dest, upload_nonce());
        let second = staging_path(&dest, upload_nonce());
        std::fs::write(&first, b"new binary").expect("stage 1");
        std::fs::write(&second, b"new binary").expect("stage 2");
        // A reader holding the old file keeps its contents across the rename.
        let held = std::fs::File::open(&dest).expect("hold old");

        for staging in [&first, &second] {
            let output = sh(&install_command(staging, &dest, FileMode::Executable), &dir);
            assert!(output.status.success(), "{output:?}");
            assert!(!std::path::Path::new(staging).exists(), "{staging}");
            assert_eq!(std::fs::read(&dest).expect("dest"), b"new binary");
            let mode = std::fs::metadata(&dest).expect("dest").permissions().mode();
            assert_eq!(mode & 0o777, 0o755, "{mode:o}");
        }
        let mut old = String::new();
        std::io::Read::read_to_string(&mut &held, &mut old).expect("read held");
        assert_eq!(old, "old binary");

        let regular = staging_path(&dest, upload_nonce());
        std::fs::write(&regular, b"data").expect("stage");
        assert!(
            sh(&install_command(&regular, &dest, FileMode::Regular), &dir)
                .status
                .success()
        );
        let mode = std::fs::metadata(&dest).expect("dest").permissions().mode();
        assert_eq!(mode & 0o777, 0o644, "{mode:o}");
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn a_failed_install_leaves_the_destination_untouched() {
        let dir = scratch("failed");
        let dest = dir
            .join("gauntlet-agent")
            .to_str()
            .expect("utf8")
            .to_string();
        std::fs::write(&dest, b"old binary").expect("old");
        // The staged file never arrived (upload killed before close).
        let staging = staging_path(&dest, upload_nonce());
        let output = sh(
            &install_command(&staging, &dest, FileMode::Executable),
            &dir,
        );
        assert!(!output.status.success());
        assert_eq!(std::fs::read(&dest).expect("dest"), b"old binary");
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}
