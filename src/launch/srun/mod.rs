//! srun launch: typed configuration and the pure command-line builders for
//! `srun`, `squeue`, `scancel` and `sbcast`.
//!
//! Every agent process is its own job step:
//!
//! ```text
//! srun <flags...> --nodes=1 --ntasks=1 --nodelist=<host> \
//!      --job-name=gauntlet:<host>:agent <args> --export=ALL \
//!      env LD_LIBRARY_PATH=<dir>/lib:<inherited> <agent> agent <args...>
//! ```
//!
//! srun forwards stdin to the (single) task and the task's stdout/stderr
//! back, so the JSON-lines protocol and stdin directives work exactly as
//! they do over ssh. No shell is involved on either side: arguments are
//! argv words, the NCCL env is set on the local srun process and exported
//! with `--export=ALL`, and LD_LIBRARY_PATH is applied by `env` inside the
//! step so srun itself keeps the orchestrator's library path.
//!
//! - `flags`: `SrunFlag` validation (no managed, stdio or signal options).
//! - `steps`: step names, kill targets, `StepId`, squeue/scancel.
//! - `command`: step / carrier / agent / sbcast argv builders.

mod command;
mod flags;
mod steps;

use serde::{Deserialize, Serialize};

use crate::remote_dir::RemoteDir;

pub use self::command::{
    agent_step_command, carrier_step_args, env_to_strip, sbcast_args, sbcast_staging_path,
    step_args,
};
pub use self::flags::{FlagRejection, SrunFlag, SrunFlagError};
pub use self::steps::{
    STEP_NAME_PREFIX, StepEntry, StepId, StepIdError, StepName, StepTarget, find_step, parse_steps,
    scancel_args, squeue_args, step_matches, steps_to_kill,
};

/// The overridable flags every step gets unless `[launch.srun] flags`
/// replaces them. See docs/features/slurm_launch.md for the reasoning.
///
/// - `--overlap`: steps share the allocation's CPUs, memory and GRES with
///   every other step (Slurm >= 22.05). Without it a second concurrent
///   step on a node blocks until the first ends.
/// - `--cpu-bind=none`: the agent pins its own per-core workers; a
///   task-affinity mask from srun would confine it to a subset.
/// - `--kill-on-bad-exit=1`: a step whose task fails is torn down at once
///   instead of lingering under a site `KillOnBadExit=0`.
pub const DEFAULT_FLAGS: [&str; 3] = ["--overlap", "--cpu-bind=none", "--kill-on-bad-exit=1"];

pub fn default_flags() -> Vec<SrunFlag> {
    DEFAULT_FLAGS
        .iter()
        .map(|flag| SrunFlag::trusted(flag))
        .collect()
}

/// How the agent binary reaches the nodes when there is no sftp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SrunDeploy {
    /// `sbcast` (Slurm's tree broadcast), scoped by a carrier step to the
    /// stale nodes only, to a staging file inside the resolved `dir`; then
    /// a per-node atomic install into `dir/bin`. The default: works with
    /// node-local `dir`.
    #[default]
    Sbcast,
    /// `dir` is a shared filesystem visible from every node and from the
    /// orchestrator: the orchestrator installs the binary once, locally,
    /// and nodes only verify its hash.
    Shared,
}

/// `[launch.srun]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SrunConfig {
    /// srun options on every step; replaces `DEFAULT_FLAGS` when set.
    pub flags: Vec<SrunFlag>,
    /// Appended after `flags` (e.g. `--gres=gpu:8` for an exclusive
    /// allocation that did not request GPUs).
    pub extra_flags: Vec<SrunFlag>,
    /// Agent directory on every node, a `remote_dir` template (`~` and
    /// `$USER` expanded on the node; created mode 700, owner-checked,
    /// resolved to its physical path). Default `/tmp/gauntlet-$USER`.
    pub dir: Option<RemoteDir>,
    pub deploy: SrunDeploy,
}

impl Default for SrunConfig {
    fn default() -> Self {
        Self {
            flags: default_flags(),
            extra_flags: Vec::new(),
            dir: None,
            deploy: SrunDeploy::default(),
        }
    }
}

impl SrunConfig {
    /// `flags` then `extra_flags`, in order.
    pub fn all_flags(&self) -> impl Iterator<Item = &SrunFlag> {
        self.flags.iter().chain(&self.extra_flags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flag(text: &str) -> SrunFlag {
        SrunFlag::parse(text).expect("valid flag")
    }

    #[test]
    fn defaults_are_valid_user_flags_too() {
        for default in DEFAULT_FLAGS {
            assert_eq!(flag(default).as_str(), default);
        }
    }

    #[test]
    fn flags_round_trip_through_toml_as_strings() {
        #[derive(Debug, Deserialize, Serialize)]
        struct Doc {
            flags: Vec<SrunFlag>,
        }
        let doc: Doc = toml::from_str("flags = [\"--overlap\", \"--gres=gpu:4\"]").expect("toml");
        assert_eq!(doc.flags, vec![flag("--overlap"), flag("--gres=gpu:4")]);
        assert!(toml::from_str::<Doc>("flags = [\"--nodelist=x\"]").is_err());
        assert!(toml::from_str::<Doc>("flags = [\"--label\"]").is_err());
        assert!(toml::to_string(&doc).expect("ser").contains("--gres=gpu:4"));
    }

    #[test]
    fn replacing_the_default_flags_drops_them() {
        let config = SrunConfig {
            flags: vec![flag("--overlap")],
            ..SrunConfig::default()
        };
        let flags: Vec<&str> = config.all_flags().map(SrunFlag::as_str).collect();
        assert_eq!(flags, ["--overlap"]);
    }

    #[test]
    fn the_dir_is_a_remote_dir_template() {
        #[derive(Debug, Deserialize)]
        struct Doc {
            srun: SrunConfig,
        }
        let doc: Doc =
            toml::from_str("[srun]\ndir = \"/scratch/gauntlet-$USER\"\n").expect("template dir");
        assert_eq!(
            doc.srun.dir.as_ref().map(RemoteDir::as_str),
            Some("/scratch/gauntlet-$USER")
        );
        // Same validation as `ssh.remote_dir`: no other `$` expansion.
        assert!(toml::from_str::<Doc>("[srun]\ndir = \"/scratch/$HOME\"\n").is_err());
        assert_eq!(SrunConfig::default().dir, None);
    }
}
