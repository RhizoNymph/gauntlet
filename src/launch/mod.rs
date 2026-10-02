//! How agents are started on the nodes: over ssh (the default), or as
//! `srun` job steps inside the Slurm allocation the orchestrator runs in.
//!
//! This module holds the transport-independent, pure part: the `[launch]`
//! config section, the Slurm allocation / nodelist handling (`slurm`), and
//! the srun/squeue/scancel/sbcast command-line builders (`srun`). The
//! orchestrator's `transport` module executes them.

pub mod slurm;
pub mod srun;

use std::fmt;

use serde::{Deserialize, Serialize};

pub use self::slurm::{SlurmAllocation, SlurmError, SlurmJobId};
pub use self::srun::{SrunConfig, SrunDeploy, SrunDir, SrunFlag};

/// `[launch] mode`, overridable with `--launch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum LaunchMode {
    /// Persistent ssh sessions to every host (ControlMaster multiplexing),
    /// agent deployed by sftp.
    #[default]
    Ssh,
    /// `srun` job steps inside the current Slurm allocation; hosts default
    /// to the allocation's nodes, agent deployed by sbcast.
    Srun,
}

impl fmt::Display for LaunchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LaunchMode::Ssh => "ssh",
            LaunchMode::Srun => "srun",
        })
    }
}

/// `[launch]`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LaunchConfig {
    pub mode: LaunchMode,
    pub srun: SrunConfig,
}

impl LaunchConfig {
    /// The mode in effect: the CLI override when given, else the config.
    pub fn effective_mode(&self, cli_override: Option<LaunchMode>) -> LaunchMode {
        cli_override.unwrap_or(self.mode)
    }
}

/// How a run's agents were launched, as recorded in `RunResults.launch`
/// (schema v13). An srun run always names its job: a record cannot claim
/// srun without one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum LaunchRecord {
    Ssh,
    Srun { job_id: SlurmJobId },
}

impl fmt::Display for LaunchRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LaunchRecord::Ssh => f.write_str("ssh"),
            LaunchRecord::Srun { job_id } => write!(f, "srun (slurm job {job_id})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_override_wins_over_config() {
        let config = LaunchConfig {
            mode: LaunchMode::Srun,
            ..LaunchConfig::default()
        };
        assert_eq!(config.effective_mode(None), LaunchMode::Srun);
        assert_eq!(
            config.effective_mode(Some(LaunchMode::Ssh)),
            LaunchMode::Ssh
        );
        assert_eq!(
            LaunchConfig::default().effective_mode(None),
            LaunchMode::Ssh
        );
    }

    #[test]
    fn records_serialize_with_a_mode_tag() {
        assert_eq!(
            serde_json::to_string(&LaunchRecord::Ssh).expect("json"),
            r#"{"mode":"ssh"}"#
        );
        let srun = LaunchRecord::Srun {
            job_id: SlurmJobId::new(31337),
        };
        let json = serde_json::to_string(&srun).expect("json");
        assert_eq!(json, r#"{"mode":"srun","job_id":31337}"#);
        assert_eq!(
            serde_json::from_str::<LaunchRecord>(&json).expect("back"),
            srun
        );
        // srun without a job id is not a representable record.
        assert!(serde_json::from_str::<LaunchRecord>(r#"{"mode":"srun"}"#).is_err());
        assert_eq!(srun.to_string(), "srun (slurm job 31337)");
    }
}
