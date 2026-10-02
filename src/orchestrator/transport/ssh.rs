//! ssh transport: one persistent session per host via the `openssh` crate
//! (native-mux). Sessions honor ~/.ssh/config (jump hosts, agent auth,
//! aliases); `ssh.user` / `ssh.key` from the fleet config override when set.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use openssh::{KnownHosts, Session, SessionBuilder, Stdio};
use openssh_sftp_client::{Sftp, SftpOptions};
use tracing::debug;

use super::{AgentCommand, Pipe, SpawnStdio};
use crate::config::SshConfig;
use crate::orchestrator::session::{
    RemoteOutput, agent_kill_pattern, agent_spawn_args, env_words, single_quote,
};

pub(super) fn connect_timeout(ssh: &SshConfig) -> Duration {
    Duration::from_secs(ssh.connect_timeout_secs.max(1))
}

/// Establish a session. `connect_timeout_secs` applies; errors carry the
/// host address for attribution.
pub(super) async fn connect(addr: &str, ssh: &SshConfig) -> Result<Arc<Session>> {
    let timeout = connect_timeout(ssh);

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

    let session = tokio::time::timeout(timeout, builder.connect_mux(addr))
        .await
        .map_err(|_| {
            anyhow!(
                "ssh connect to {addr} timed out after {}s",
                timeout.as_secs()
            )
        })?
        .with_context(|| format!("ssh connect to {addr}"))?;
    debug!(host = %addr, "ssh session established");
    Ok(Arc::new(session))
}

pub(super) async fn exec_capture(
    session: &Session,
    addr: &str,
    script: &str,
) -> Result<RemoteOutput> {
    let output = session
        .command("sh")
        .arg("-c")
        .arg(script)
        .output()
        .await
        .with_context(|| format!("running `{script}` on {addr}"))?;
    Ok(RemoteOutput::from_output(output))
}

fn stdio(pipe: Pipe) -> Stdio {
    match pipe {
        Pipe::Piped => Stdio::piped(),
        Pipe::Null => Stdio::null(),
    }
}

/// `env <KEY='value'...> '<agent>' agent '<args>'...` as one remote command.
/// The child owns a clone of the session, so it outlives the caller's
/// borrow (phase 3 keeps a `peer serve` running while it drives the other
/// end of the pair).
pub(super) async fn spawn(
    session: &Arc<Session>,
    command: &AgentCommand<'_>,
    pipes: SpawnStdio,
) -> Result<openssh::Child<Arc<Session>>> {
    let words = env_words(command.env);
    let mut remote = Arc::clone(session).arc_command("env");
    remote
        .raw_args(agent_spawn_args(&words, command.agent_path, command.args))
        .stdin(stdio(pipes.stdin))
        .stdout(stdio(pipes.stdout))
        .stderr(stdio(pipes.stderr));
    remote
        .spawn()
        .await
        .with_context(|| format!("spawning agent {:?} on {}", command.args, command.host))
}

/// Kill a remote agent by its command line (`pkill -f`). Best effort at
/// the caller; here a failed command is an error.
pub(super) async fn kill_agent(session: &Session, addr: &str, args: &str) -> Result<String> {
    let pattern = agent_kill_pattern(args);
    exec_capture(
        session,
        addr,
        &format!("pkill -f {}", single_quote(&pattern)),
    )
    .await?;
    Ok(format!("pkill -f {pattern}"))
}

/// Upload a local file to `remote_path` (sftp), creating parent dirs,
/// setting the executable bit when `executable`.
///
/// The bytes land in a sibling temp file that is then renamed into place,
/// so a concurrently running copy of the old binary cannot make the write
/// fail with `ETXTBSY` and readers never observe a half-written file.
pub(crate) async fn upload(
    session: &Arc<Session>,
    addr: &str,
    local: &Path,
    remote_path: &str,
    executable: bool,
) -> Result<()> {
    let bytes = tokio::fs::read(local)
        .await
        .with_context(|| format!("reading local file {}", local.display()))?;
    let parent = remote_path
        .rsplit_once('/')
        .map(|(parent, _)| parent)
        .filter(|parent| !parent.is_empty())
        .unwrap_or(".");
    checked(
        exec_capture(session, addr, &format!("mkdir -p {}", single_quote(parent))).await,
        addr,
    )
    .with_context(|| format!("creating remote directory {parent}"))?;

    let staging = format!("{remote_path}.staging");
    sftp_write(session, &staging, &bytes)
        .await
        .with_context(|| format!("uploading {} to {}", local.display(), staging))?;

    let mode = if executable { "755" } else { "644" };
    checked(
        exec_capture(
            session,
            addr,
            &format!(
                "chmod {mode} {staging} && mv -f {staging} {dest}",
                staging = single_quote(&staging),
                dest = single_quote(remote_path),
            ),
        )
        .await,
        addr,
    )
    .with_context(|| format!("installing {remote_path}"))?;
    debug!(host = %addr, remote_path, bytes = bytes.len(), "uploaded file");
    Ok(())
}

fn checked(output: Result<RemoteOutput>, addr: &str) -> Result<RemoteOutput> {
    let output = output?;
    if !output.success() {
        anyhow::bail!("remote command failed on {addr}: {}", output.detail());
    }
    Ok(output)
}

async fn sftp_write(session: &Session, remote_path: &str, bytes: &[u8]) -> Result<()> {
    let mut child = session
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
