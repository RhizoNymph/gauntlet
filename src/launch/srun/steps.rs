//! Job steps gauntlet launches: their names, how a kill selects them from
//! `squeue --steps`, and the step ids handed back to `scancel`/`sbcast`.
//!
//! Step ids are kept exactly as squeue printed them (a validated
//! `StepId`), so array-task and het-job steps (`1237_3.4`, `1234+1.0`) are
//! cancelled by the same string Slurm uses for them; nothing is rebuilt
//! from `SLURM_JOB_ID`, which for an array task is the *raw* job id
//! (`1240`) rather than what squeue prints (`1237_3`).

use std::fmt;

use thiserror::Error;

use crate::launch::slurm::SlurmJobId;

/// Prefix of every step name gauntlet launches.
pub const STEP_NAME_PREFIX: &str = "gauntlet";

/// The name a step carries (`--job-name`). Every step gauntlet starts is
/// named, so every step can be found again by name:
///
/// - agent processes: `gauntlet:<host>:agent <args>`
/// - short shell commands (`exec_capture`): `gauntlet:<host>:exec`
/// - sbcast carrier steps: `gauntlet-bcast:<token>`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepName(String);

impl StepName {
    pub fn agent(host: &str, args: &[&str]) -> Self {
        Self(format!(
            "{STEP_NAME_PREFIX}:{host}:agent {}",
            args.join(" ")
        ))
    }

    pub fn exec(host: &str) -> Self {
        Self(format!("{STEP_NAME_PREFIX}:{host}:exec"))
    }

    /// A step that only exists to scope an sbcast to its nodes.
    pub fn carrier(token: &str) -> Self {
        Self(format!("{STEP_NAME_PREFIX}-bcast:{token}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for StepName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Which of a host's steps a kill targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepTarget<'a> {
    /// The agent running `agent <args...>`, matched on a word-boundary
    /// prefix: `barrier serve --port 29500` matches the full
    /// `barrier serve --port 29500 --world 4 --iters 2000`, but `peer serve
    /// --port 2950` does not match `... --port 29500`.
    Agent(&'a str),
    /// Every `exec_capture` step on the host.
    Exec,
}

/// Does step `name` belong to `host` and match `target`?
pub fn step_matches(name: &str, host: &str, target: StepTarget<'_>) -> bool {
    match target {
        StepTarget::Exec => name == StepName::exec(host).as_str(),
        StepTarget::Agent(args) => {
            let key = StepName::agent(host, &[args]);
            match name.strip_prefix(key.as_str()) {
                Some(rest) => rest.is_empty() || rest.starts_with(' '),
                None => false,
            }
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("{text:?} is not a numbered Slurm job step id")]
pub struct StepIdError {
    pub text: String,
}

/// A numbered job step exactly as squeue prints it:
/// `<job>.<step>`, `<array_job>_<task>.<step>` or `<leader>+<offset>.<step>`.
/// The job part is ASCII alphanumeric (numeric ids, and SLUIDs); special
/// steps (`.batch`, `.extern`, `.interactive`) are not steps gauntlet
/// launched and are rejected.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct StepId {
    raw: String,
    het_offset: Option<u32>,
    step: u32,
}

impl StepId {
    pub fn parse(text: &str) -> Result<Self, StepIdError> {
        let error = || StepIdError {
            text: text.to_string(),
        };
        let raw = text.trim();
        let (job, step) = raw.rsplit_once('.').ok_or_else(error)?;
        let step: u32 = digits(step).ok_or_else(error)?;
        let (base, het_offset) = match (job.split_once('_'), job.split_once('+')) {
            (Some(_), Some(_)) => return Err(error()),
            (Some((base, task)), None) => {
                digits(task).ok_or_else(error)?;
                (base, None)
            }
            (None, Some((base, offset))) => (base, Some(digits(offset).ok_or_else(error)?)),
            (None, None) => (job, None),
        };
        if base.is_empty() || !base.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
            return Err(error());
        }
        Ok(Self {
            raw: raw.to_string(),
            het_offset,
            step,
        })
    }

    /// The id as squeue printed it; what `scancel` is given.
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    pub fn step(&self) -> u32 {
        self.step
    }

    /// The `--jobid` for `sbcast`, which takes `<job>[+<offset>].<step>`
    /// (no array syntax): the allocation's own `SLURM_JOB_ID` — the raw id
    /// of an array task — plus the het offset squeue printed, if any.
    pub fn sbcast_jobid(&self, job: SlurmJobId) -> String {
        match self.het_offset {
            Some(offset) => format!("{job}+{offset}.{}", self.step),
            None => format!("{job}.{}", self.step),
        }
    }
}

impl fmt::Display for StepId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

fn digits(text: &str) -> Option<u32> {
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// One line of `squeue --steps` output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepEntry {
    pub id: StepId,
    pub name: String,
}

/// Separator between the id and name fields of `squeue_args`' format. Step
/// names never contain it (hosts and agent args are `|`-free).
const SQUEUE_SEPARATOR: char = '|';

/// List the steps of `job`, one `<id>|<name>` line each. Without a field
/// width squeue prints whole names, never truncating. `--jobs` takes the
/// allocation's own `SLURM_JOB_ID` (for an array task, its raw id).
pub fn squeue_args(job: SlurmJobId) -> Vec<String> {
    vec![
        "--noheader".into(),
        "--steps".into(),
        format!("--jobs={job}"),
        format!("--format=%i{SQUEUE_SEPARATOR}%j"),
    ]
}

/// Parse `squeue_args` output; lines that are not a numbered step of the
/// form `<id>|<name>` are skipped.
pub fn parse_steps(stdout: &str) -> Vec<StepEntry> {
    stdout
        .lines()
        .filter_map(|line| {
            let (id, name) = line.trim().split_once(SQUEUE_SEPARATOR)?;
            Some(StepEntry {
                id: StepId::parse(id).ok()?,
                name: name.trim().to_string(),
            })
        })
        .collect()
}

/// The steps of one listing that any of `requests` (host, target) selects,
/// each once, in listing order. The listing is already restricted to the
/// allocation's job by `squeue --jobs`.
pub fn steps_to_kill(steps: &[StepEntry], requests: &[(&str, StepTarget<'_>)]) -> Vec<StepId> {
    steps
        .iter()
        .filter(|entry| {
            requests
                .iter()
                .any(|(host, target)| step_matches(&entry.name, host, *target))
        })
        .map(|entry| entry.id.clone())
        .collect()
}

/// The step carrying `name`, if listed.
pub fn find_step<'a>(steps: &'a [StepEntry], name: &StepName) -> Option<&'a StepId> {
    steps
        .iter()
        .find(|entry| entry.name == name.as_str())
        .map(|entry| &entry.id)
}

/// `scancel --signal=KILL <id>...`: SIGKILL to every listed step's tasks in
/// one RPC round. A task blocked in a collective cannot ignore it, and the
/// step ends with its last task. Ids are passed exactly as squeue printed
/// them (`job[_task|+offset].step`, all accepted by scancel).
pub fn scancel_args(steps: &[StepId]) -> Vec<String> {
    std::iter::once("--signal=KILL".to_string())
        .chain(steps.iter().map(|step| step.as_str().to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(text: &str) -> StepId {
        StepId::parse(text).expect("step id")
    }

    #[test]
    fn step_ids_parse_plain_array_and_het_forms() {
        let plain = id("4242.7");
        assert_eq!((plain.as_str(), plain.step()), ("4242.7", 7));
        assert_eq!(plain.sbcast_jobid(SlurmJobId::new(4242)), "4242.7");

        // Array task 3 of array job 1237, whose own (raw) job id is 1240.
        let array = id("1237_3.5");
        assert_eq!((array.as_str(), array.step()), ("1237_3.5", 5));
        assert_eq!(array.sbcast_jobid(SlurmJobId::new(1240)), "1240.5");

        let het = id("1234+1.0");
        assert_eq!((het.as_str(), het.step()), ("1234+1.0", 0));
        assert_eq!(het.sbcast_jobid(SlurmJobId::new(1234)), "1234+1.0");

        // SLUID-style alphanumeric job ids are carried verbatim.
        assert_eq!(id("sabc123.2").as_str(), "sabc123.2");
        assert_eq!(id(" 12.3 ").as_str(), "12.3");
    }

    #[test]
    fn special_and_malformed_step_ids_are_rejected() {
        for bad in [
            "12.batch",
            "12.extern",
            "12.interactive",
            "12",
            "12.",
            ".3",
            "12.-1",
            "12_.3",
            "12_x.3",
            "12+.3",
            "12_1+1.3",
            "1;2.3",
            "12.3.4x",
            "a b.1",
        ] {
            assert!(StepId::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn names_identify_host_and_kind() {
        assert_eq!(
            StepName::agent("n0", &["peer", "serve", "--port", "29500"]).as_str(),
            "gauntlet:n0:agent peer serve --port 29500"
        );
        assert_eq!(StepName::exec("n0").as_str(), "gauntlet:n0:exec");
        assert_eq!(StepName::carrier("77-1").as_str(), "gauntlet-bcast:77-1");
    }

    #[test]
    fn agent_targets_match_on_word_boundaries_per_host() {
        let serve = StepName::agent("n0", &["peer", "serve", "--port", "29500"]);
        let agent = |args| StepTarget::Agent(args);
        assert!(step_matches(
            serve.as_str(),
            "n0",
            agent("peer serve --port 29500")
        ));
        assert!(!step_matches(
            serve.as_str(),
            "n0",
            agent("peer serve --port 2950")
        ));
        assert!(!step_matches(
            serve.as_str(),
            "n1",
            agent("peer serve --port 29500")
        ));
        let barrier = StepName::agent(
            "n0",
            &["barrier", "serve", "--port", "29500", "--world", "4"],
        );
        assert!(step_matches(
            barrier.as_str(),
            "n0",
            agent("barrier serve --port 29500")
        ));
        // A host whose name extends another's is a different host.
        assert!(!step_matches(
            StepName::agent("n01", &["nccl"]).as_str(),
            "n0",
            agent("nccl")
        ));
        // Kinds never cross.
        assert!(!step_matches(
            StepName::exec("n0").as_str(),
            "n0",
            agent("nccl")
        ));
        assert!(!step_matches(
            StepName::agent("n0", &["nccl"]).as_str(),
            "n0",
            StepTarget::Exec
        ));
        assert!(step_matches(
            StepName::exec("n0").as_str(),
            "n0",
            StepTarget::Exec
        ));
        assert!(!step_matches(
            StepName::exec("n01").as_str(),
            "n0",
            StepTarget::Exec
        ));
    }

    #[test]
    fn listings_parse_every_id_form_and_select_kill_targets_in_one_pass() {
        let stdout = "1237_3.0|gauntlet:n0:agent nccl\n\
                      1237_3.1|gauntlet:n1:agent nccl\n\
                      1237_3.batch|batch\n\
                      1237_3.extern|extern\n\
                      1237_3.2|gauntlet:n0:agent peer serve --port 29500\n\
                      1237_3.3|gauntlet:n2:exec\n\
                      1237_3.4|gauntlet-bcast:9-1\n\
                      garbage line\n";
        let steps = parse_steps(stdout);
        assert_eq!(steps.len(), 5);
        assert_eq!(steps[2].name, "gauntlet:n0:agent peer serve --port 29500");

        let ids = |requests: &[(&str, StepTarget<'_>)]| -> Vec<String> {
            steps_to_kill(&steps, requests)
                .iter()
                .map(|step| step.to_string())
                .collect()
        };
        // The fleet abort: one listing, every host's nccl step.
        assert_eq!(
            ids(&[
                ("n0", StepTarget::Agent("nccl")),
                ("n1", StepTarget::Agent("nccl")),
                ("n5", StepTarget::Agent("nccl")),
            ]),
            ["1237_3.0", "1237_3.1"]
        );
        assert_eq!(ids(&[("n2", StepTarget::Exec)]), ["1237_3.3"]);
        assert!(ids(&[("n2", StepTarget::Agent("nccl"))]).is_empty());
        assert_eq!(
            find_step(&steps, &StepName::carrier("9-1")).map(StepId::as_str),
            Some("1237_3.4")
        );
    }

    #[test]
    fn slurm_tool_arguments() {
        assert_eq!(
            squeue_args(SlurmJobId::new(77)),
            ["--noheader", "--steps", "--jobs=77", "--format=%i|%j"]
        );
        assert_eq!(
            scancel_args(&[id("77.5"), id("1237_3.0"), id("9+1.2")]),
            ["--signal=KILL", "77.5", "1237_3.0", "9+1.2"]
        );
    }
}
