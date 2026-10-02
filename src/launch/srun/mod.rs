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
use thiserror::Error;

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

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SrunDirError {
    #[error(
        "[launch.srun] dir {dir:?} must be an absolute path (it is used verbatim on every \
         node and as the sbcast destination; `~` is not expanded)"
    )]
    NotAbsolute { dir: String },
    #[error("[launch.srun] dir {dir:?} must not contain control characters")]
    ControlCharacter { dir: String },
    #[error("cannot derive the default [launch.srun] dir: neither USER nor LOGNAME is set")]
    NoUser,
}

/// Agent directory on every node in srun mode: an absolute path used
/// verbatim on each node (no `~`: there is no login shell to expand it,
/// and an sbcast destination must be literal). No trailing slash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SrunDir(String);

impl SrunDir {
    pub fn parse(dir: &str) -> Result<Self, SrunDirError> {
        if dir.chars().any(char::is_control) {
            return Err(SrunDirError::ControlCharacter {
                dir: dir.to_string(),
            });
        }
        let trimmed = dir.trim_end_matches('/');
        if !dir.starts_with('/') || trimmed.is_empty() {
            return Err(SrunDirError::NotAbsolute {
                dir: dir.to_string(),
            });
        }
        Ok(Self(trimmed.to_string()))
    }

    /// `/tmp/gauntlet-<user>`: node-local, so every node writes its own
    /// copy (no shared-home write race between nodes) and two users never
    /// share one.
    pub fn default_for(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, SrunDirError> {
        let user = ["USER", "LOGNAME"]
            .into_iter()
            .filter_map(&lookup)
            .find(|user| !user.trim().is_empty())
            .ok_or(SrunDirError::NoUser)?;
        // A user name never contains '/', but the path must not be steered
        // elsewhere if the environment says otherwise.
        let user: String = user
            .chars()
            .map(|ch| {
                if ch == '/' || ch.is_control() {
                    '_'
                } else {
                    ch
                }
            })
            .collect();
        Self::parse(&format!("/tmp/gauntlet-{user}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for SrunDir {
    type Error = SrunDirError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

impl From<SrunDir> for String {
    fn from(dir: SrunDir) -> Self {
        dir.0
    }
}

/// How the agent binary reaches the nodes when there is no sftp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SrunDeploy {
    /// `sbcast` (Slurm's tree broadcast), scoped by a carrier step to the
    /// stale nodes only, to a staging file next to `dir`; then a per-node
    /// atomic install into `dir/bin`. The default: works with node-local
    /// `dir`.
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
    /// Agent directory on every node; default `/tmp/gauntlet-$USER`.
    pub dir: Option<SrunDir>,
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
    fn dirs_are_absolute_literals() {
        assert_eq!(
            SrunDir::parse("/tmp/gauntlet-u/").expect("dir").as_str(),
            "/tmp/gauntlet-u"
        );
        assert_eq!(
            SrunDir::parse("/scratch/a b").expect("dir").as_str(),
            "/scratch/a b"
        );
        for bad in ["~/.gauntlet", "relative/dir", "", "/", "//"] {
            assert!(
                matches!(SrunDir::parse(bad), Err(SrunDirError::NotAbsolute { .. })),
                "{bad:?}"
            );
        }
        assert!(matches!(
            SrunDir::parse("/tmp/x\ny"),
            Err(SrunDirError::ControlCharacter { .. })
        ));
    }

    #[test]
    fn default_dir_is_node_local_and_per_user() {
        let lookup = |key: &str| (key == "USER").then(|| "alice".to_string());
        assert_eq!(
            SrunDir::default_for(lookup).expect("dir").as_str(),
            "/tmp/gauntlet-alice"
        );
        let logname = |key: &str| (key == "LOGNAME").then(|| "bob".to_string());
        assert_eq!(
            SrunDir::default_for(logname).expect("dir").as_str(),
            "/tmp/gauntlet-bob"
        );
        let hostile = |key: &str| (key == "USER").then(|| "../../etc".to_string());
        assert_eq!(
            SrunDir::default_for(hostile).expect("dir").as_str(),
            "/tmp/gauntlet-.._.._etc"
        );
        assert_eq!(SrunDir::default_for(|_| None), Err(SrunDirError::NoUser));
    }
}
