//! Pure argv builders for the srun transport: steps, sbcast carrier steps,
//! the agent command run inside a step, and sbcast itself.

use super::SrunDir;
use super::flags::SrunFlag;
use super::steps::StepName;
use crate::launch::slurm::SlurmJobId;

/// The options gauntlet sets on every step, after the user's flags. Every
/// option is one `-`-prefixed word, so the first non-option word is always
/// the command.
fn managed_flags(nodes: &[&str], name: &StepName) -> Vec<String> {
    let count = nodes.len().max(1);
    let mut flags = vec![format!("--nodes={count}"), format!("--ntasks={count}")];
    if count > 1 {
        flags.push("--ntasks-per-node=1".to_string());
    }
    flags.extend([
        format!("--nodelist={}", nodes.join(",")),
        format!("--job-name={name}"),
        // Explicit: an sbatch `--export=NONE` leaks into srun through
        // SLURM_EXPORT_ENV and would drop the agent env.
        "--export=ALL".to_string(),
    ]);
    flags
}

/// Full srun argv (without the `srun` program) for one single-task step on
/// `host` running `command` (program first). User flags come first,
/// managed flags last, then the command.
pub fn step_args<'a>(
    flags: impl IntoIterator<Item = &'a SrunFlag>,
    host: &str,
    name: &StepName,
    command: &[String],
) -> Vec<String> {
    flags
        .into_iter()
        .map(|flag| flag.as_str().to_string())
        .chain(managed_flags(&[host], name))
        .chain(command.iter().cloned())
        .collect()
}

/// srun argv for an sbcast *carrier* step: one task per node on exactly
/// `hosts`, running `sleep <secs>`. `sbcast --jobid=<job>.<step>` then
/// transmits to that step's nodes only (supported since at least Slurm
/// 20.11), so a broadcast never touches allocation nodes outside the
/// stale, connected set.
pub fn carrier_step_args<'a>(
    flags: impl IntoIterator<Item = &'a SrunFlag>,
    hosts: &[&str],
    name: &StepName,
    lifetime_secs: u64,
) -> Vec<String> {
    flags
        .into_iter()
        .map(|flag| flag.as_str().to_string())
        .chain(managed_flags(hosts, name))
        .chain(["sleep".to_string(), lifetime_secs.max(1).to_string()])
        .collect()
}

/// The command a step runs for `agent <args>`:
/// `env LD_LIBRARY_PATH=<lib_dir>[:<inherited>] <agent> agent <args...>`.
///
/// `LD_LIBRARY_PATH` is applied inside the step rather than on the local
/// srun process, so srun itself keeps the orchestrator's library path
/// (module-provided Slurm/CUDA/NCCL paths), and the agent gets `lib_dir`
/// *prepended* to the same inherited value `--export=ALL` would hand it.
/// No shell is involved: each element is one argv word, verbatim. `env`
/// execs the agent, so the agent's own argv is still `<agent> agent
/// <args>`.
pub fn agent_step_command(
    agent_path: &str,
    args: &[&str],
    lib_dir: &str,
    inherited_ld_library_path: Option<&str>,
) -> Vec<String> {
    let library_path = match inherited_ld_library_path.filter(|value| !value.is_empty()) {
        Some(inherited) => format!("LD_LIBRARY_PATH={lib_dir}:{inherited}"),
        None => format!("LD_LIBRARY_PATH={lib_dir}"),
    };
    [
        "env".to_string(),
        library_path,
        agent_path.to_string(),
        "agent".to_string(),
    ]
    .into_iter()
    .chain(args.iter().map(|arg| arg.to_string()))
    .collect()
}

/// Environment variables of the orchestrator's own environment that must
/// not reach a step through `--export=ALL`:
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

/// `sbcast --force --jobid=<jobid> <source> <dest>`; `jobid` is a step's
/// `StepId::sbcast_jobid`, scoping the broadcast to that step's nodes.
pub fn sbcast_args(jobid: &str, source: &str, dest: &str) -> Vec<String> {
    vec![
        "--force".into(),
        format!("--jobid={jobid}"),
        source.to_string(),
        dest.to_string(),
    ]
}

/// The sbcast destination: a sibling of `dir` (so its parent — `/tmp` by
/// default — exists on every node), unique per job and binary so
/// concurrent jobs of one user never write the same file.
pub fn sbcast_staging_path(dir: &SrunDir, job: SlurmJobId, sha256: &str) -> String {
    let short: String = sha256.chars().take(12).collect();
    format!("{}.sbcast-{job}-{short}", dir.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launch::srun::SrunConfig;

    fn flag(text: &str) -> SrunFlag {
        SrunFlag::parse(text).expect("valid flag")
    }

    #[test]
    fn step_args_put_managed_flags_after_user_flags_then_the_command() {
        let config = SrunConfig {
            extra_flags: vec![flag("--gres=gpu:8")],
            ..SrunConfig::default()
        };
        let name = StepName::agent("gpu-a01", &["peer", "serve"]);
        let command: Vec<String> = ["env", "LD_LIBRARY_PATH=/g/lib", "/g/bin/gauntlet-agent"]
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
                "--job-name=gauntlet:gpu-a01:agent peer serve",
                "--export=ALL",
                "env",
                "LD_LIBRARY_PATH=/g/lib",
                "/g/bin/gauntlet-agent",
            ]
        );
        assert_eq!(args.iter().position(|arg| !arg.starts_with('-')), Some(9));
    }

    #[test]
    fn carrier_steps_span_exactly_the_target_nodes_one_task_each() {
        let config = SrunConfig::default();
        let name = StepName::carrier("7-1");
        let args = carrier_step_args(config.all_flags(), &["n0", "n2"], &name, 90);
        assert_eq!(
            args,
            [
                "--overlap",
                "--cpu-bind=none",
                "--kill-on-bad-exit=1",
                "--nodes=2",
                "--ntasks=2",
                "--ntasks-per-node=1",
                "--nodelist=n0,n2",
                "--job-name=gauntlet-bcast:7-1",
                "--export=ALL",
                "sleep",
                "90",
            ]
        );
        let single = carrier_step_args(config.all_flags(), &["n0"], &name, 0);
        assert!(single.contains(&"--nodes=1".to_string()));
        assert!(
            !single
                .iter()
                .any(|arg| arg.starts_with("--ntasks-per-node"))
        );
        assert_eq!(single.last().map(String::as_str), Some("1"));
    }

    #[test]
    fn agent_commands_prepend_the_lib_dir_to_the_inherited_library_path() {
        assert_eq!(
            agent_step_command(
                "/tmp/g/bin/gauntlet-agent",
                &["peer", "serve", "--port", "29500"],
                "/tmp/g/lib",
                Some("/opt/cuda/lib64:/opt/slurm/lib"),
            ),
            [
                "env",
                "LD_LIBRARY_PATH=/tmp/g/lib:/opt/cuda/lib64:/opt/slurm/lib",
                "/tmp/g/bin/gauntlet-agent",
                "agent",
                "peer",
                "serve",
                "--port",
                "29500",
            ]
        );
        for inherited in [None, Some("")] {
            assert_eq!(
                agent_step_command("/a", &["probe"], "/d/lib", inherited)[1],
                "LD_LIBRARY_PATH=/d/lib"
            );
        }
    }

    /// The argv words reach the agent verbatim through a real `env` (no
    /// shell parses them), including a library path with shell
    /// metacharacters.
    #[test]
    fn agent_command_words_survive_env_verbatim() {
        let lib = "/tmp/a b/$HOME/`id`/lib";
        let inherited = "/opt/x;y:/opt/'q'";
        let mut command = agent_step_command("/bin/sh", &[], lib, Some(inherited));
        // Swap the agent for a probe that prints the variable.
        command.truncate(2);
        command.extend(["printenv".to_string(), "LD_LIBRARY_PATH".to_string()]);
        let output = std::process::Command::new(&command[0])
            .args(&command[1..])
            .output()
            .expect("run env");
        assert!(output.status.success());
        assert_eq!(
            String::from_utf8(output.stdout).expect("utf8").trim_end(),
            format!("{lib}:{inherited}")
        );
    }

    #[test]
    fn orchestrator_gpu_visibility_and_nccl_vars_are_stripped() {
        let names = [
            "PATH",
            "LD_LIBRARY_PATH",
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

    #[test]
    fn sbcast_targets_a_step_and_stages_next_to_the_dir() {
        assert_eq!(
            sbcast_args("77.5", "/home/u/gauntlet", "/tmp/gauntlet-u.sbcast-77-abc"),
            [
                "--force",
                "--jobid=77.5",
                "/home/u/gauntlet",
                "/tmp/gauntlet-u.sbcast-77-abc"
            ]
        );
        let dir = SrunDir::parse("/tmp/gauntlet-u").expect("dir");
        assert_eq!(
            sbcast_staging_path(&dir, SlurmJobId::new(77), "0123456789abcdef0123"),
            "/tmp/gauntlet-u.sbcast-77-0123456789ab"
        );
    }
}
