//! `[launch.srun] flags` validation: an `SrunFlag` is one srun option word
//! that cannot change which node a step targets, how its stdio is wired,
//! or how it reacts to the signals gauntlet uses to cancel it.
//!
//! srun parses options with `getopt_long`, so the checks cover every
//! spelling it accepts: `--name`, `--name=value`, an unambiguous *prefix*
//! of a long name (`--lab` is `--label`), and bundled short options
//! (`-Kl` is not `-K -l`, but `-Ql` is `-Q -l`). The short-option table
//! mirrors `src/common/slurm_opt.c` (which letters take an argument, and
//! so end the bundle).

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Why an option is not allowed as a user flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagRejection {
    /// gauntlet sets it on every step (or it retargets nodes/jobs).
    Managed,
    /// It changes the step's stdin/stdout/stderr, which carry the agent's
    /// JSON-lines protocol and stdin directives.
    Stdio,
    /// It changes how srun reacts to the SIGTERM gauntlet uses to cancel
    /// an abandoned step.
    Signals,
    /// The step would not run attached to srun (no stdio at all).
    Detached,
}

impl fmt::Display for FlagRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            FlagRejection::Managed => "is managed by gauntlet and cannot be overridden",
            FlagRejection::Stdio => "changes the step's stdio, which carries the agent protocol",
            FlagRejection::Signals => {
                "changes srun's signal handling, which gauntlet relies on to cancel steps"
            }
            FlagRejection::Detached => "detaches the step from srun's stdio",
        })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SrunFlagError {
    #[error("srun flag {flag:?} must start with '-' (write options as --name=value)")]
    NotAnOption { flag: String },
    #[error("srun flag {flag:?} must be a single word without whitespace or control characters")]
    NotOneWord { flag: String },
    #[error("srun flag {flag:?} (option {option}) {reason}")]
    Rejected {
        flag: String,
        option: String,
        reason: FlagRejection,
    },
}

/// Long options a user flag may not name (exactly or by prefix).
const REJECTED_LONG: [(&str, FlagRejection); 25] = [
    ("nodes", FlagRejection::Managed),
    ("ntasks", FlagRejection::Managed),
    ("ntasks-per-node", FlagRejection::Managed),
    ("nodelist", FlagRejection::Managed),
    ("nodefile", FlagRejection::Managed),
    ("exclude", FlagRejection::Managed),
    ("relative", FlagRejection::Managed),
    ("no-allocate", FlagRejection::Managed),
    ("job-name", FlagRejection::Managed),
    ("jobid", FlagRejection::Managed),
    ("het-group", FlagRejection::Managed),
    ("clusters", FlagRejection::Managed),
    ("export", FlagRejection::Managed),
    ("label", FlagRejection::Stdio),
    ("output", FlagRejection::Stdio),
    ("input", FlagRejection::Stdio),
    ("error", FlagRejection::Stdio),
    ("open-mode", FlagRejection::Stdio),
    ("pty", FlagRejection::Stdio),
    ("multi-prog", FlagRejection::Stdio),
    ("unbuffered", FlagRejection::Stdio),
    ("task-prolog", FlagRejection::Stdio),
    ("task-epilog", FlagRejection::Stdio),
    ("ignore-signals", FlagRejection::Signals),
    ("test-only", FlagRejection::Detached),
];

/// Long options rejected beyond the table above (kept separate so the
/// table stays a fixed-size array of the common cases).
const REJECTED_LONG_EXTRA: [(&str, FlagRejection); 3] = [
    ("async", FlagRejection::Detached),
    ("prolog", FlagRejection::Stdio),
    ("epilog", FlagRejection::Stdio),
];

/// Short options a user flag may not contain, with their long names.
const REJECTED_SHORT: [(char, &str, FlagRejection); 14] = [
    ('N', "nodes", FlagRejection::Managed),
    ('n', "ntasks", FlagRejection::Managed),
    ('w', "nodelist", FlagRejection::Managed),
    ('F', "nodefile", FlagRejection::Managed),
    ('x', "exclude", FlagRejection::Managed),
    ('r', "relative", FlagRejection::Managed),
    ('Z', "no-allocate", FlagRejection::Managed),
    ('J', "job-name", FlagRejection::Managed),
    ('M', "clusters", FlagRejection::Managed),
    ('l', "label", FlagRejection::Stdio),
    ('o', "output", FlagRejection::Stdio),
    ('i', "input", FlagRejection::Stdio),
    ('e', "error", FlagRejection::Stdio),
    ('u', "unbuffered", FlagRejection::Stdio),
];

/// srun short options that take an argument (required or optional): in a
/// bundle the rest of the word after one of these is its argument, not
/// more options. From `slurm_opt.c`.
const SHORT_WITH_ARGUMENT: &str = "AabDMCScdexBGiJLTFwNnoprqtmWIKk";

/// One extra srun option, validated: a single `-`-prefixed word (so the
/// option list can never swallow the agent command) that does not touch an
/// option gauntlet manages, the step's stdio, or its signal handling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SrunFlag(String);

impl SrunFlag {
    pub fn parse(flag: &str) -> Result<Self, SrunFlagError> {
        let owned = || flag.to_string();
        if flag.is_empty() || flag.chars().any(|ch| ch.is_whitespace() || ch.is_control()) {
            return Err(SrunFlagError::NotOneWord { flag: owned() });
        }
        if !flag.starts_with('-') || flag == "-" || flag == "--" {
            return Err(SrunFlagError::NotAnOption { flag: owned() });
        }
        if let Some((option, reason)) = rejection(flag) {
            return Err(SrunFlagError::Rejected {
                flag: owned(),
                option,
                reason,
            });
        }
        Ok(Self(flag.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// A flag gauntlet itself writes (defaults); never user input.
    pub(super) fn trusted(flag: &str) -> Self {
        Self(flag.to_string())
    }
}

/// The rejected option a flag word names, if any, in getopt's reading.
fn rejection(flag: &str) -> Option<(String, FlagRejection)> {
    if let Some(long) = flag.strip_prefix("--") {
        let name = long.split_once('=').map_or(long, |(name, _)| name);
        // getopt_long accepts any unambiguous prefix of a long name, so a
        // prefix of a rejected name may *be* that option.
        return REJECTED_LONG
            .iter()
            .chain(REJECTED_LONG_EXTRA.iter())
            .find(|(rejected, _)| rejected.starts_with(name))
            .map(|(rejected, reason)| (format!("--{rejected}"), *reason));
    }
    // A bundle of short options: each letter is an option until one that
    // takes an argument, which consumes the rest of the word.
    for letter in flag[1..].chars() {
        if let Some((_, long, reason)) = REJECTED_SHORT.iter().find(|(short, ..)| *short == letter)
        {
            return Some((format!("-{letter} (--{long})"), *reason));
        }
        if SHORT_WITH_ARGUMENT.contains(letter) {
            break;
        }
    }
    None
}

impl TryFrom<String> for SrunFlag {
    type Error = SrunFlagError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

impl From<SrunFlag> for String {
    fn from(flag: SrunFlag) -> Self {
        flag.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rejected(flag: &str) -> FlagRejection {
        match SrunFlag::parse(flag) {
            Err(SrunFlagError::Rejected { reason, .. }) => reason,
            other => panic!("{flag}: expected a rejection, got {other:?}"),
        }
    }

    #[test]
    fn ordinary_options_pass() {
        for good in [
            "--overlap",
            "--gres=gpu:8",
            "--gpus-per-node=8",
            "--gpu-bind=none",
            "--cpu-bind=none",
            "-K1",
            "--mpi=none",
            "--mem=0",
            "--exclusive",
            "--quiet",
            "-Q",
            "--nodes-extra",
            "--exports",
            "-t10",
            // `-K` takes an optional argument: `l` is its argument here,
            // not `--label`.
            "-Kl",
            // `-p` takes an argument: the partition `gpul`.
            "-pgpul",
        ] {
            assert_eq!(
                SrunFlag::parse(good).map(|f| f.0),
                Ok(good.to_string()),
                "{good}"
            );
        }
    }

    #[test]
    fn shape_errors() {
        assert!(matches!(
            SrunFlag::parse("overlap"),
            Err(SrunFlagError::NotAnOption { .. })
        ));
        assert!(matches!(
            SrunFlag::parse("--"),
            Err(SrunFlagError::NotAnOption { .. })
        ));
        for spaced in ["--gres gpu:8", "", "--x=\n", "--a\tb"] {
            assert!(
                matches!(
                    SrunFlag::parse(spaced),
                    Err(SrunFlagError::NotOneWord { .. })
                ),
                "{spaced:?}"
            );
        }
    }

    #[test]
    fn managed_options_are_rejected_in_every_spelling() {
        for flag in [
            "--nodes=2",
            "--nodelist=other",
            "--ntasks=4",
            "--job-name=x",
            "--export=NONE",
            "--jobid=5",
            "--ntasks-per-node=2",
            "--exclude=n1",
            "--nodefile=f",
            "--relative=1",
            "--no-allocate",
            "--het-group=0",
            "--clusters=c",
            "-N2",
            "-n4",
            "-wnode3",
            "-Jname",
            "-xn1",
            "-Z",
            // Abbreviations getopt_long expands.
            "--node=2",
            "--nodel=x",
            "--expo=NONE",
            "--job-n=x",
        ] {
            assert_eq!(rejected(flag), FlagRejection::Managed, "{flag}");
        }
        let message = SrunFlag::parse("--nodelist=x")
            .expect_err("managed")
            .to_string();
        assert!(message.contains("managed by gauntlet"), "{message}");
    }

    #[test]
    fn stdio_changing_options_are_rejected_in_every_spelling() {
        for flag in [
            "--label",
            "-l",
            "--output=out.txt",
            "-oout.txt",
            "-o",
            "--input=none",
            "-inone",
            "--error=err.txt",
            "-eerr",
            "--pty",
            "--pty=/bin/bash",
            "--multi-prog",
            "--unbuffered",
            "-u",
            "--open-mode=append",
            "--task-prolog=/x",
            "--task-epilog=/x",
            "--prolog=/x",
            "--epilog=/x",
            // Abbreviations.
            "--lab",
            "--outp=f",
            "--unbuf",
            "--multi",
            // Bundles: a rejected letter before any argument-taking one.
            "-Ql",
            "-Xu",
            "-vl",
            "-lK",
        ] {
            assert_eq!(rejected(flag), FlagRejection::Stdio, "{flag}");
        }
        let message = SrunFlag::parse("-l").expect_err("stdio").to_string();
        assert!(message.contains("--label"), "{message}");
        assert!(message.contains("stdio"), "{message}");
    }

    #[test]
    fn signal_and_detaching_options_are_rejected() {
        assert_eq!(rejected("--ignore-signals=TERM"), FlagRejection::Signals);
        assert_eq!(rejected("--ignore-sig=15"), FlagRejection::Signals);
        assert_eq!(rejected("--test-only"), FlagRejection::Detached);
        assert_eq!(rejected("--async"), FlagRejection::Detached);
    }
}
