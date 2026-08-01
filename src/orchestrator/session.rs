//! One persistent ssh session per host, via the `openssh` crate
//! (native-mux). Sessions honor ~/.ssh/config (jump hosts, agent auth,
//! aliases); `ssh.user` / `ssh.key` from the fleet config override when set.

use std::path::Path;

use anyhow::Result;

use crate::config::{HostConfig, SshConfig};
use crate::proto::AgentEvent;

pub struct HostSession {
    pub host: HostConfig,
    // agent A: openssh::Session + remote_dir inside.
}

impl HostSession {
    /// Establish a session. `connect_timeout_secs` applies; errors carry the
    /// host address for attribution.
    pub async fn connect(host: HostConfig, ssh: &SshConfig) -> Result<HostSession> {
        let _ = (host, ssh);
        todo!("agent A: implement")
    }

    /// Run a short remote command, capturing stdout (used by deploy for
    /// hash checks and by bootstrap tuning).
    pub async fn exec(&self, command: &str) -> Result<String> {
        let _ = command;
        todo!("agent A: implement")
    }

    /// Upload a local file to `remote_path` (sftp), creating parent dirs,
    /// setting the executable bit when `executable`.
    pub async fn upload(&self, local: &Path, remote_path: &str, executable: bool) -> Result<()> {
        let _ = (local, remote_path, executable);
        todo!("agent A: implement")
    }

    /// Spawn the deployed agent with `args`, write `stdin_doc` (a single
    /// JSON line) to its stdin, and stream decoded events to `on_event`
    /// until EOF. Returns the remote exit status.
    pub async fn run_agent(
        &self,
        args: &[&str],
        stdin_doc: Option<String>,
        on_event: impl FnMut(AgentEvent) + Send,
    ) -> Result<std::process::ExitStatus> {
        let _ = (args, stdin_doc, on_event);
        todo!("agent A: implement")
    }
}
