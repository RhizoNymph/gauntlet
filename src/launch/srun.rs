//! srun launch: typed configuration and the pure command-line builders for
//! `srun`, `squeue`, `scancel` and `sbcast`.
//!
//! Every agent process is its own job step:
//!
//! ```text
//! srun <flags...> --nodes=1 --ntasks=1 --nodelist=<host> \
//!      --job-name=gauntlet:<host>:<args> --export=ALL \
//!      <agent_path> agent <args...>
//! ```
//!
//! srun forwards stdin to the (single) task and the task's stdout/stderr
//! back, so the JSON-lines protocol and stdin directives work exactly as
//! they do over ssh. No shell is involved on either side: arguments are
//! passed as argv words, and the agent environment (LD_LIBRARY_PATH plus
//! the resolved NCCL env) is set on the local srun process and exported
//! with `--export=ALL`.

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::slurm::SlurmJobId;

/// Prefix of every step name gauntlet launches.
pub const STEP_NAME_PREFIX: &str = "gauntlet";

/// The overridable flags every step gets unless `[launch.srun] flags`
/// replaces them. See docs/features/slurm_launch.md for the reasoning.
///
/// - `--overlap`: steps share the allocation's CPUs, memory and GRES with
///   every other step (Slurm >= 22.05). Without it a second concurrent
///   step on a node (peer serve + an exec, the NCCL lead + an abort probe,
///   or simply the orchestrator's own step) blocks until the first ends.
/// - `--cpu-bind=none`: the agent pins its own per-core workers; a
///   task-affinity mask from srun would confine it to a subset.
/// - `--kill-on-bad-exit=1`: a step whose task fails is torn down at once
///   instead of lingering under a site `KillOnBadExit=0`.
pub const DEFAULT_FLAGS: [&str; 3] = ["--overlap", "--cpu-bind=none", "--kill-on-bad-exit=1"];

/// Options gauntlet sets itself on every step; a user flag naming one of
/// them would silently change which node or process a step targets.
const RESERVED_OPTIONS: [(&str, Option<char>); 7] = [
    ("nodes", Some('N')),
    ("ntasks", Some('n')),
    ("nodelist", Some('w')),
    ("job-name", Some('J')),
    ("export", None),
    ("jobid", None),
    ("ntasks-per-node", None),
];

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SrunFlagError {
    #[error("srun flag {flag:?} must start with '-' (write options as --name=value)")]
    NotAnOption { flag: String },
    #[error("srun flag {flag:?} must be a single word without whitespace or control characters")]
    NotOneWord { flag: String },
    #[error("srun flag {flag:?} is managed by gauntlet and cannot be overridden")]
    Reserved { flag: String },
}

/// One extra srun option, validated: a single `-`-prefixed word (so the
/// option list can never swallow the agent command) that does not touch an
/// option gauntlet manages (`RESERVED_OPTIONS`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SrunFlag(String);

impl SrunFlag {
    pub fn parse(flag: &str) -> Result<Self, SrunFlagError> {
        let owned = || flag.to_string();
        if flag.chars().any(|ch| ch.is_whitespace() || ch.is_control()) || flag.is_empty() {
            return Err(SrunFlagError::NotOneWord { flag: owned() });
        }
        if !flag.starts_with('-') || flag == "-" || flag == "--" {
            return Err(SrunFlagError::NotAnOption { flag: owned() });
        }
        if is_reserved(flag) {
            return Err(SrunFlagError::Reserved { flag: owned() });
        }
        Ok(Self(flag.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn is_reserved(flag: &str) -> bool {
    if let Some(long) = flag.strip_prefix("--") {
        let name = long.split_once('=').map_or(long, |(name, _)| name);
        return RESERVED_OPTIONS
            .iter()
            .any(|(reserved, _)| *reserved == name);
    }
    // Short option: `-N2`, `-w node`, `-Jname`.
    let short = flag[1..].chars().next();
    RESERVED_OPTIONS
        .iter()
        .any(|(_, letter)| letter.is_some() && *letter == short)
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

pub fn default_flags() -> Vec<SrunFlag> {
    DEFAULT_FLAGS
        .iter()
        .map(|flag| SrunFlag(flag.to_string()))
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
    /// `sbcast` (Slurm's tree broadcast) to a staging file next to `dir`
    /// on every allocated node, then a per-node atomic install into
    /// `dir/bin`. The default: works with node-local `dir`.
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

/// The name a step carries (`--job-name`): `gauntlet:<host>:<args>`. It
/// identifies the step for `kill`: `squeue --steps` lists names, and the
/// step whose name starts with the kill key is the one to cancel — the
/// same "match the agent command line" contract `pkill -f` gives the ssh
/// transport, without any process-local bookkeeping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepName(String);

impl StepName {
    /// Name for an agent step running `agent <args>` on `host`.
    pub fn agent(host: &str, args: &[&str]) -> Self {
        Self(format!("{STEP_NAME_PREFIX}:{host}:{}", args.join(" ")))
    }

    /// Name for a short shell command (`exec_capture`); never matched by
    /// an agent kill key (`agent` steps carry their subcommand instead).
    pub fn exec(host: &str) -> Self {
        Self(format!("{STEP_NAME_PREFIX}-exec:{host}"))
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

/// Does step `name` run `agent <args...>` on `host` where the args start
/// with the words of `kill_args`? Word-boundary prefix: `peer serve --port
/// 29500` matches `... --port 29500` but not `... --port 295001`, and
/// `barrier serve --port 29500` matches the full `barrier serve --port
/// 29500 --world 4 --iters 2000`.
pub fn step_matches(name: &str, host: &str, kill_args: &str) -> bool {
    let key = StepName::agent(host, &[kill_args]);
    match name.strip_prefix(key.as_str()) {
        Some(rest) => rest.is_empty() || rest.starts_with(' '),
        None => false,
    }
}

/// Full srun argv (without the `srun` program) for one step on `host`
/// running `command` (program first). User flags come first, managed
/// flags last, then the command.
pub fn step_args<'a>(
    flags: impl IntoIterator<Item = &'a SrunFlag>,
    host: &str,
    name: &StepName,
    command: &[String],
) -> Vec<String> {
    flags
        .into_iter()
        .map(|flag| flag.as_str().to_string())
        .chain([
            "--nodes=1".to_string(),
            "--ntasks=1".to_string(),
            format!("--nodelist={host}"),
            format!("--job-name={name}"),
            // Explicit: an sbatch `--export=NONE` leaks into srun through
            // SLURM_EXPORT_ENV and would drop the agent env.
            "--export=ALL".to_string(),
        ])
        .chain(command.iter().cloned())
        .collect()
}

/// Environment variables of the orchestrator's own environment that must
/// not reach an agent through `--export=ALL`:
///
/// - GPU visibility (`CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`,
///   `GPU_DEVICE_ORDINAL`): Slurm sets these per step from the GPUs bound
///   to it; a value inherited from the orchestrator's step (often only its
///   own GPUs, or none) would hide node GPUs from the agent.
/// - Any `NCCL_*`: the run records exactly the resolved `[nccl]` env as the
///   env every agent ran under; a stray `NCCL_*` in the operator's shell
///   must not make that record false.
pub fn env_to_strip<'a>(names: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    const GPU_VISIBILITY: [&str; 3] = [
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ];
    names
        .into_iter()
        .filter(|name| GPU_VISIBILITY.contains(name) || name.starts_with("NCCL_"))
        .map(str::to_string)
        .collect()
}

/// A numbered job step, `<job>.<step>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StepId {
    pub job: SlurmJobId,
    pub step: u32,
}

impl fmt::Display for StepId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.job, self.step)
    }
}

impl StepId {
    /// `123.4`; special steps (`123.batch`, `123.extern`, het-job ids) are
    /// not steps gauntlet launched and parse to `None`.
    pub fn parse(text: &str) -> Option<Self> {
        let (job, step) = text.trim().split_once('.')?;
        Some(Self {
            job: job.parse().ok()?,
            step: if step.bytes().all(|byte| byte.is_ascii_digit()) {
                step.parse().ok()?
            } else {
                return None;
            },
        })
    }
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

/// List the steps of `job`, one `<job>.<step>|<name>` line each. Without a
/// field width squeue prints whole names, never truncating.
pub fn squeue_args(job: SlurmJobId) -> Vec<String> {
    vec![
        "--noheader".into(),
        "--steps".into(),
        format!("--jobs={job}"),
        format!("--format=%i{SQUEUE_SEPARATOR}%j"),
    ]
}

/// Parse `squeue_args` output; lines that are not a numbered step of the
/// form `<job>.<step>|<name>` are skipped.
pub fn parse_steps(stdout: &str) -> Vec<StepEntry> {
    stdout
        .lines()
        .filter_map(|line| {
            let (id, name) = line.trim().split_once(SQUEUE_SEPARATOR)?;
            Some(StepEntry {
                id: StepId::parse(id)?,
                name: name.trim().to_string(),
            })
        })
        .collect()
}

/// The steps of a listing that a kill of `agent <kill_args>` on `host`
/// targets, restricted to `job` (the orchestrator's own allocation).
pub fn steps_to_kill(
    steps: &[StepEntry],
    job: SlurmJobId,
    host: &str,
    kill_args: &str,
) -> Vec<StepId> {
    steps
        .iter()
        .filter(|entry| entry.id.job == job && step_matches(&entry.name, host, kill_args))
        .map(|entry| entry.id)
        .collect()
}

/// `scancel --signal=KILL <job>.<step>`: SIGKILL to the step's task. A
/// task blocked in a collective cannot ignore it, and the step ends with
/// its last task.
pub fn scancel_args(step: StepId) -> Vec<String> {
    vec!["--signal=KILL".into(), step.to_string()]
}

/// `sbcast --force --jobid=<job> <source> <dest>`: broadcast to every node
/// of the allocation, replacing a stale copy.
pub fn sbcast_args(job: SlurmJobId, source: &str, dest: &str) -> Vec<String> {
    vec![
        "--force".into(),
        format!("--jobid={job}"),
        source.to_string(),
        dest.to_string(),
    ]
}

/// The sbcast destination: a sibling of `dir` (so its parent — `/tmp` by
/// default — exists on every allocated node, even nodes outside a
/// configured host subset), unique per job and binary so concurrent jobs of
/// one user never write the same file.
pub fn sbcast_staging_path(dir: &SrunDir, job: SlurmJobId, sha256: &str) -> String {
    let short: String = sha256.chars().take(12).collect();
    format!("{}.sbcast-{job}-{short}", dir.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flag(text: &str) -> SrunFlag {
        SrunFlag::parse(text).expect("valid flag")
    }

    #[test]
    fn flags_must_be_single_option_words() {
        for good in [
            "--overlap",
            "--gres=gpu:8",
            "--gpus-per-node=8",
            "--gpu-bind=none",
            "-K1",
            "--mpi=none",
            "--mem=0",
        ] {
            assert_eq!(flag(good).as_str(), good);
        }
        assert!(matches!(
            SrunFlag::parse("overlap"),
            Err(SrunFlagError::NotAnOption { .. })
        ));
        assert!(matches!(
            SrunFlag::parse("/bin/sh"),
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
    fn managed_options_cannot_be_overridden() {
        for reserved in [
            "--nodes=2",
            "--nodelist=other",
            "--ntasks=4",
            "--job-name=x",
            "--export=NONE",
            "--jobid=5",
            "--ntasks-per-node=2",
            "-N2",
            "-n4",
            "-wnode3",
            "-Jname",
        ] {
            assert!(
                matches!(
                    SrunFlag::parse(reserved),
                    Err(SrunFlagError::Reserved { .. })
                ),
                "{reserved}"
            );
        }
        // Prefix look-alikes are different options.
        assert!(SrunFlag::parse("--nodes-extra").is_ok());
        assert!(SrunFlag::parse("--exports").is_ok());
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
        assert!(toml::to_string(&doc).expect("ser").contains("--gres=gpu:4"));
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

    #[test]
    fn step_args_put_managed_flags_after_user_flags_then_the_command() {
        let config = SrunConfig {
            extra_flags: vec![flag("--gres=gpu:8")],
            ..SrunConfig::default()
        };
        let name = StepName::agent("gpu-a01", &["peer", "serve", "--port", "29500"]);
        let command: Vec<String> = ["/tmp/g/bin/gauntlet-agent", "agent", "peer", "serve"]
            .map(String::from)
            .to_vec();
        let args = step_args(config.all_flags(), "gpu-a01", &name, &command);
        assert_eq!(
            args,
            [
                "--overlap",
                "--cpu-bind=none",
                "--kill-on-bad-exit=1",
                "--gres=gpu:8",
                "--nodes=1",
                "--ntasks=1",
                "--nodelist=gpu-a01",
                "--job-name=gauntlet:gpu-a01:peer serve --port 29500",
                "--export=ALL",
                "/tmp/g/bin/gauntlet-agent",
                "agent",
                "peer",
                "serve",
            ]
        );
        // Every option is one `-` word, so the first non-option word is
        // always the command (what srun — and the test shim — rely on).
        let first_command = args.iter().position(|arg| !arg.starts_with('-'));
        assert_eq!(first_command, Some(9));
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
    fn kill_keys_match_on_word_boundaries_per_host() {
        let serve = StepName::agent("n0", &["peer", "serve", "--port", "29500"]);
        assert!(step_matches(
            serve.as_str(),
            "n0",
            "peer serve --port 29500"
        ));
        assert!(!step_matches(
            serve.as_str(),
            "n0",
            "peer serve --port 2950"
        ));
        assert!(!step_matches(
            serve.as_str(),
            "n1",
            "peer serve --port 29500"
        ));
        assert!(!step_matches(
            StepName::agent("n0", &["peer", "serve", "--port", "295001"]).as_str(),
            "n0",
            "peer serve --port 29500"
        ));

        let barrier = StepName::agent(
            "n0",
            &[
                "barrier", "serve", "--port", "29500", "--world", "4", "--iters", "10",
            ],
        );
        assert!(step_matches(
            barrier.as_str(),
            "n0",
            "barrier serve --port 29500"
        ));

        let nccl = StepName::agent("n0", &["nccl"]);
        assert!(step_matches(nccl.as_str(), "n0", "nccl"));
        // A host whose name extends another's is a different host.
        assert!(!step_matches(
            StepName::agent("n01", &["nccl"]).as_str(),
            "n0",
            "nccl"
        ));
        // exec steps are never agent kill targets.
        assert!(!step_matches(StepName::exec("n0").as_str(), "n0", "nccl"));
    }

    #[test]
    fn squeue_output_parses_to_numbered_steps() {
        let job = SlurmJobId::new(4242);
        let stdout = "4242.0|gauntlet:n0:nccl\n\
                      4242.1|gauntlet:n1:nccl\n\
                      4242.batch|batch\n\
                      4242.extern|extern\n\
                      4242.2|gauntlet:n0:peer serve --port 29500\n\
                      garbage line\n\
                      4243.7|gauntlet:n0:nccl\n";
        let steps = parse_steps(stdout);
        assert_eq!(steps.len(), 4);
        assert_eq!(
            steps[2],
            StepEntry {
                id: StepId { job, step: 2 },
                name: "gauntlet:n0:peer serve --port 29500".into(),
            }
        );
        // Only this job's steps on this host with this command are targets.
        assert_eq!(
            steps_to_kill(&steps, job, "n0", "nccl"),
            vec![StepId { job, step: 0 }]
        );
        assert_eq!(
            steps_to_kill(&steps, job, "n0", "peer serve --port 29500"),
            vec![StepId { job, step: 2 }]
        );
        assert!(steps_to_kill(&steps, job, "n2", "nccl").is_empty());
    }

    #[test]
    fn step_ids_reject_special_steps() {
        assert_eq!(
            StepId::parse("12.3"),
            Some(StepId {
                job: SlurmJobId::new(12),
                step: 3
            })
        );
        for special in ["12.batch", "12.extern", "12+1.0", "12", "x.1", "12.-1"] {
            assert_eq!(StepId::parse(special), None, "{special}");
        }
    }

    #[test]
    fn slurm_tool_arguments() {
        let job = SlurmJobId::new(77);
        assert_eq!(
            squeue_args(job),
            ["--noheader", "--steps", "--jobs=77", "--format=%i|%j"]
        );
        assert_eq!(
            scancel_args(StepId { job, step: 5 }),
            ["--signal=KILL", "77.5"]
        );
        assert_eq!(
            sbcast_args(job, "/home/u/gauntlet", "/tmp/gauntlet-u.sbcast-77-abc"),
            [
                "--force",
                "--jobid=77",
                "/home/u/gauntlet",
                "/tmp/gauntlet-u.sbcast-77-abc"
            ]
        );
        let dir = SrunDir::parse("/tmp/gauntlet-u").expect("dir");
        assert_eq!(
            sbcast_staging_path(&dir, job, "0123456789abcdef0123"),
            "/tmp/gauntlet-u.sbcast-77-0123456789ab"
        );
    }

    #[test]
    fn orchestrator_gpu_visibility_and_nccl_vars_are_stripped() {
        let names = [
            "PATH",
            "HOME",
            "CUDA_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "NCCL_DEBUG",
            "NCCL_SOCKET_IFNAME",
            "SLURM_JOB_ID",
            "MY_NCCL_THING",
        ];
        assert_eq!(
            env_to_strip(names),
            [
                "CUDA_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL",
                "NCCL_DEBUG",
                "NCCL_SOCKET_IFNAME",
            ]
        );
    }
}
