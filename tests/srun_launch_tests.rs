//! srun launch mode, end to end on localhost against a fake Slurm.
//!
//! `tests/fake_slurm/{srun,squeue,scancel,sbcast,scontrol}` are small shell
//! scripts installed into a temp dir per test. Fake nodes have Slurm
//! NodeNames that do not resolve (`node-a` ...) and NodeAddrs on distinct
//! loopback addresses (127.0.0.x): srun runs the real agent locally (stdio
//! passed through, SIGTERM cancels the step like real srun), registers the
//! step for squeue/scancel/sbcast, and records its argv and client
//! environment. The real `gauntlet` binary drives `run --launch srun` and
//! `bootstrap` through them, so the whole path — allocation detection,
//! nodelist and NodeAddr resolution, scoped sbcast deploy, the JSON-lines
//! protocol over srun stdio, stdin directives, step kills and cancellation
//! of abandoned steps — is exercised without Slurm.

use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use gauntlet::config::HostConfig;
use gauntlet::launch::{LaunchRecord, SlurmJobId, SrunConfig, SrunDir};
use gauntlet::nccl_env::NcclEnv;
use gauntlet::orchestrator::bootstrap::{BootstrapReport, CheckStatus, HostReadiness};
use gauntlet::orchestrator::session::HostSession;
use gauntlet::orchestrator::transport::{Launcher, SlurmTools, SrunLauncher};
use gauntlet::report::{RunResults, SCHEMA_VERSION};

const JOB_ID: u64 = 4242;
const SHIMS: [(&str, &str); 5] = [
    ("srun", include_str!("fake_slurm/srun")),
    ("squeue", include_str!("fake_slurm/squeue")),
    ("scancel", include_str!("fake_slurm/scancel")),
    ("sbcast", include_str!("fake_slurm/sbcast")),
    ("scontrol", include_str!("fake_slurm/scontrol")),
];
const SEP: char = '\u{1f}';

/// A value no shell, srun option parser or env transport may alter.
const ADVERSARIAL: &str = "mlx5_0,mlx5_1 'q' \"d\" $HOME `id` $(touch /nonexistent/x) ;|&*~ \\b =x";

/// The orchestrator's own library path, which srun must keep and the
/// agent must get with the agent lib dir prepended.
const ORCHESTRATOR_LD_PATH: &str = "/opt/module/cuda/lib64:/opt/slurm/lib";

/// (NodeName, NodeAddr) of the fake allocation.
const NODES: [(&str, &str); 3] = [
    ("node-a", "127.0.0.1"),
    ("node-b", "127.0.0.2"),
    ("node-c", "127.0.0.3"),
];

fn node_names() -> Vec<&'static str> {
    NODES.iter().map(|(name, _)| *name).collect()
}

struct FakeSlurm {
    root: PathBuf,
}

impl FakeSlurm {
    /// A fake allocation of `nodes` (NodeName, NodeAddr); an empty address
    /// means scontrol reports no NodeAddr for that node.
    fn new(tag: &str, nodes: &[(&str, &str)]) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "gauntlet-srun-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(root.join("bin")).expect("bin dir");
        std::fs::create_dir_all(root.join("state/steps")).expect("state dir");
        for (name, body) in SHIMS {
            let path = root.join("bin").join(name);
            std::fs::write(&path, body).expect("write shim");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .expect("chmod shim");
        }
        std::fs::write(root.join("job_id"), JOB_ID.to_string()).expect("job id");
        let names: String = nodes.iter().map(|(name, _)| format!("{name}\n")).collect();
        std::fs::write(root.join("nodes"), names).expect("nodes");
        let addrs: String = nodes
            .iter()
            .filter(|(_, addr)| !addr.is_empty())
            .map(|(name, addr)| format!("{name} {addr}\n"))
            .collect();
        std::fs::write(root.join("node_addrs"), addrs).expect("node addrs");
        Self { root }
    }

    /// Nodes on which sbcast transfers fail.
    fn mark_bad(&self, nodes: &[&str]) {
        let list: String = nodes.iter().map(|node| format!("{node}\n")).collect();
        std::fs::write(self.root.join("bad_nodes"), list).expect("bad nodes");
    }

    /// Make squeue print step ids in array-task form, `<array_id>.<step>`.
    fn as_array_task(&self, array_id: &str) {
        std::fs::write(self.root.join("array_id"), array_id).expect("array id");
    }

    fn bin(&self) -> PathBuf {
        self.root.join("bin")
    }

    /// The agent directory every fake node uses.
    fn node_dir(&self) -> PathBuf {
        self.root.join("node dir")
    }

    fn log(&self, name: &str) -> String {
        std::fs::read_to_string(self.root.join("state").join(name)).unwrap_or_default()
    }

    /// step id -> srun argv (without the program).
    fn srun_calls(&self) -> BTreeMap<String, Vec<String>> {
        parse_records(&self.log("srun.log"))
    }

    /// step id -> the selected env entries of the srun client process.
    fn srun_envs(&self) -> BTreeMap<String, Vec<String>> {
        parse_records(&self.log("srun-env.log"))
    }

    /// Nodes a step was launched on.
    fn step_nodes(&self, step: &str) -> Vec<String> {
        std::fs::read_to_string(self.root.join("state/steps").join(format!("{step}.nodes")))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    /// Whether a step's task process is still alive.
    fn step_alive(&self, step: &str) -> bool {
        let Ok(task) =
            std::fs::read_to_string(self.root.join("state/steps").join(format!("{step}.task")))
        else {
            return false;
        };
        Command::new("kill")
            .args(["-0", task.trim()])
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    fn write_config(&self, body: &str) -> PathBuf {
        let path = self.root.join("gauntlet.toml");
        std::fs::write(&path, body).expect("config");
        path
    }

    /// `gauntlet <args>` inside the fake allocation: shims first on PATH,
    /// SLURM_JOB_ID / SLURM_JOB_NODELIST set, plus orchestrator-side
    /// variables that must never reach an agent and a library path srun
    /// must keep.
    fn gauntlet(&self, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_gauntlet"));
        let path = format!(
            "{}:{}",
            self.bin().display(),
            std::env::var("PATH").unwrap_or_default()
        );
        command
            .args(args)
            .current_dir(&self.root)
            .env("PATH", path)
            .env("SLURM_JOB_ID", JOB_ID.to_string())
            .env("SLURM_JOB_NODELIST", "node-[a-c]")
            .env("CUDA_VISIBLE_DEVICES", "orchestrator-only")
            .env("NCCL_STRAY_FROM_SHELL", "1")
            .env("LD_LIBRARY_PATH", ORCHESTRATOR_LD_PATH)
            .env("RUST_LOG", "info");
        command
    }

    /// An in-process launcher over the shims (transport-level tests).
    fn launcher(&self) -> Launcher {
        let srun = SrunLauncher::new(
            SlurmJobId::new(JOB_ID),
            SrunConfig::default(),
            SrunDir::parse(self.node_dir().to_str().expect("utf8")).expect("dir"),
            Vec::new(),
            None,
            SlurmTools::in_dir(&self.bin()),
        );
        Launcher::Srun(Arc::new(srun))
    }

    fn install_agent(&self) {
        let bin = self.node_dir().join("bin");
        std::fs::create_dir_all(&bin).expect("bin");
        std::fs::copy(env!("CARGO_BIN_EXE_gauntlet"), bin.join("gauntlet-agent")).expect("copy");
    }
}

impl Drop for FakeSlurm {
    fn drop(&mut self) {
        // No fake step may outlive its test.
        for (step, _) in self.srun_calls() {
            if self.step_alive(&step)
                && let Ok(task) = std::fs::read_to_string(
                    self.root.join("state/steps").join(format!("{step}.task")),
                )
            {
                let _ = Command::new("kill").args(["-KILL", task.trim()]).status();
            }
        }
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn parse_records(text: &str) -> BTreeMap<String, Vec<String>> {
    text.lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let mut fields = line.split(SEP);
            let step = fields
                .next()
                .and_then(|head| head.strip_prefix("step="))
                .expect("record starts with step=")
                .to_string();
            (step, fields.map(str::to_string).collect())
        })
        .collect()
}

fn toml_string(value: &str) -> String {
    toml::Value::String(value.to_string()).to_string()
}

/// A port base unlikely to collide with other tests on this machine.
fn port_base(offset: u16) -> u16 {
    20_000 + (std::process::id() % 20_000) as u16 + offset * 50
}

/// Run `command` to completion, killing it after `deadline` so a hang
/// fails the test with everything it said instead of wedging the suite.
fn output_within(command: &mut Command, deadline: Duration, slurm: &FakeSlurm) -> Output {
    use std::io::Read;
    use std::process::Stdio;
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn gauntlet");
    let mut stdout = child.stdout.take().expect("stdout");
    let mut stderr = child.stderr.take().expect("stderr");
    let out_reader = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = stdout.read_to_end(&mut buffer);
        buffer
    });
    let err_reader = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = stderr.read_to_end(&mut buffer);
        buffer
    });
    let started = std::time::Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            break Some(status);
        }
        if started.elapsed() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let Some(status) = status else {
        let stderr = String::from_utf8_lossy(&err_reader.join().unwrap_or_default()).into_owned();
        panic!(
            "gauntlet did not finish within {deadline:?}\n--- stderr\n{stderr}\n--- srun.log\n{}\n--- squeue.log\n{}",
            slurm.log("srun.log").replace(SEP, " "),
            slurm.log("squeue.log"),
        );
    };
    Output {
        status,
        stdout: out_reader.join().unwrap_or_default(),
        stderr: err_reader.join().unwrap_or_default(),
    }
}

fn show(output: &Output) -> String {
    format!(
        "status {:?}\n--- stdout\n{}\n--- stderr\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn value_of<'a>(args: &'a [String], option: &str) -> Option<&'a str> {
    args.iter().find_map(|arg| {
        arg.strip_prefix(option)
            .and_then(|rest| rest.strip_prefix('='))
    })
}

fn bootstrap_report(slurm: &FakeSlurm, config: &str) -> (Output, BootstrapReport) {
    let output = output_within(
        &mut slurm.gauntlet(&["bootstrap", "--json", "--config", config]),
        Duration::from_secs(300),
        slurm,
    );
    let report: BootstrapReport = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("{error}: {}", show(&output)));
    (output, report)
}

fn check<'a>(
    row: &'a HostReadiness,
    name: &str,
) -> &'a gauntlet::orchestrator::bootstrap::ReadinessCheck {
    row.checks
        .iter()
        .find(|check| check.name == name)
        .unwrap_or_else(|| panic!("{}: no {name} check: {:?}", row.host, row.checks))
}

fn host(addr: &str) -> HostConfig {
    HostConfig {
        addr: addr.into(),
        data_addr: None,
        labels: Default::default(),
    }
}

/// `gauntlet run --launch srun` with no hosts configured: the fleet is the
/// allocation (NodeNames as step targets, NodeAddrs for peer traffic), the
/// agent arrives by one step-scoped sbcast, the inventory and network
/// phases (event streams, stdin task specs, pairwise peers, the TCP
/// barrier) all run through srun steps, every step carries exactly the
/// configured NCCL env, and srun itself keeps the orchestrator's library
/// path while the agent gets the agent lib dir prepended to it.
#[test]
fn run_over_srun_infers_hosts_and_runs_phases_end_to_end() {
    let slurm = FakeSlurm::new("run", &NODES);
    let dir = slurm.node_dir();
    let config = slurm.write_config(&format!(
        r#"
[launch.srun]
dir = {dir}
extra_flags = ["--mpi=none"]

[tests]
phases = ["inventory", "network"]
phase_timeout_secs = 120
net_latency_secs = 1
net_bandwidth_secs = 1
net_port_base = {port}
nccl_intranode = false
barrier_iters = 20

[nccl]
socket_ifname = "lo"
env = {{ NCCL_DEBUG = "WARN", NCCL_TEST_VALUE = {value} }}
"#,
        dir = toml_string(dir.to_str().expect("utf8")),
        port = port_base(0),
        value = toml_string(ADVERSARIAL),
    ));
    let out = slurm.root.join("out.json");
    let output = output_within(
        &mut slurm.gauntlet(&[
            "run",
            "--launch",
            "srun",
            "--config",
            config.to_str().expect("utf8"),
            "--out",
            out.to_str().expect("utf8"),
        ]),
        Duration::from_secs(300),
        &slurm,
    );
    // 0 = clean, 1 = stragglers (three "nodes" sharing one machine may
    // well disagree); 2 would be host failures.
    assert!(
        matches!(output.status.code(), Some(0) | Some(1)),
        "{}",
        show(&output)
    );

    // (An orchestrator error also exits 1, so the document must exist.)
    let text = std::fs::read_to_string(&out)
        .unwrap_or_else(|error| panic!("no results ({error}):\n{}", show(&output)));
    let results: RunResults = serde_json::from_str(&text).expect("json");
    assert_eq!(results.schema_version, SCHEMA_VERSION);
    assert_eq!(
        results.launch,
        Some(LaunchRecord::Srun {
            job_id: SlurmJobId::new(JOB_ID)
        })
    );
    // Hosts are keyed by NodeName; their pairwise peers were reached at
    // NodeAddr (the names do not resolve, so any metric proves it).
    let hosts: BTreeSet<&str> = results.hosts.keys().map(String::as_str).collect();
    assert_eq!(
        hosts,
        node_names().into_iter().collect(),
        "{}",
        show(&output)
    );
    for (host, observations) in &results.hosts {
        assert!(
            observations.errors.is_empty(),
            "{host}: {:?}\n{}",
            observations.errors,
            show(&output)
        );
        assert!(observations.inventory.is_some(), "{host}: no inventory");
        assert!(
            observations
                .metrics
                .iter()
                .any(|metric| metric.name == "rtt_p50"),
            "{host}: no pairwise latency"
        );
    }
    assert_eq!(
        results
            .nccl_env
            .as_ref()
            .and_then(|env| env.get("NCCL_TEST_VALUE"))
            .map(String::as_str),
        Some(ADVERSARIAL)
    );
    let table = String::from_utf8_lossy(&output.stdout);
    assert!(table.contains("launch: srun (slurm job 4242)"), "{table}");

    // Hosts and addresses: one scontrol call each.
    assert_eq!(
        slurm.log("scontrol.log").lines().collect::<Vec<_>>(),
        [
            "show hostnames node-[a-c]",
            "--oneliner show node node-[a-c]"
        ]
    );

    let calls = slurm.srun_calls();
    let envs = slurm.srun_envs();
    let agent = format!("{}/bin/gauntlet-agent", dir.display());
    let mut agent_steps = 0;
    for (step, args) in &calls {
        for flag in [
            "--overlap",
            "--cpu-bind=none",
            "--kill-on-bad-exit=1",
            "--mpi=none",
            "--export=ALL",
        ] {
            assert!(
                args.iter().any(|arg| arg == flag),
                "{step}: {flag} missing in {args:?}"
            );
        }
        let name = value_of(args, "--job-name").expect("job name");
        let nodes = value_of(args, "--nodelist").expect("nodelist");
        let command = args
            .iter()
            .position(|arg| !arg.starts_with('-'))
            .expect("command word");
        if name.starts_with("gauntlet-bcast:") {
            // The broadcast carrier spans exactly the stale nodes.
            assert_eq!(nodes, "node-a,node-b,node-c", "{args:?}");
            assert_eq!(args[command], "sleep");
            continue;
        }
        assert!(node_names().contains(&nodes), "{step}: {nodes}");
        assert!(args.iter().any(|arg| arg == "--nodes=1"), "{args:?}");
        assert!(args.iter().any(|arg| arg == "--ntasks=1"), "{args:?}");
        let env = envs.get(step).expect("env record");
        if name.starts_with(&format!("gauntlet:{nodes}:agent ")) {
            agent_steps += 1;
            // env LD_LIBRARY_PATH=<lib>:<orchestrator's> <agent> agent ...
            assert_eq!(
                &args[command..command + 4],
                [
                    "env".to_string(),
                    format!(
                        "LD_LIBRARY_PATH={}/lib:{ORCHESTRATOR_LD_PATH}",
                        dir.display()
                    ),
                    agent.clone(),
                    "agent".to_string(),
                ],
                "{args:?}"
            );
            // srun's own environment: the orchestrator's library path, the
            // resolved NCCL env byte for byte, and nothing stray.
            assert_eq!(
                env,
                &vec![
                    format!("LD_LIBRARY_PATH={ORCHESTRATOR_LD_PATH}"),
                    "NCCL_DEBUG=WARN".to_string(),
                    "NCCL_SOCKET_IFNAME=lo".to_string(),
                    format!("NCCL_TEST_VALUE={ADVERSARIAL}"),
                ],
                "step {step}"
            );
        } else {
            assert_eq!(name, format!("gauntlet:{nodes}:exec"));
            assert_eq!(&args[command..command + 2], ["sh", "-c"], "{args:?}");
            assert_eq!(
                env,
                &vec![format!("LD_LIBRARY_PATH={ORCHESTRATOR_LD_PATH}")]
            );
        }
    }
    // inventory + counters + pairs + barrier on every node, at least.
    assert!(agent_steps >= 9, "{agent_steps}");

    // One sbcast, scoped to the carrier step, to a job-unique staging file
    // that is cleaned up after the per-node installs.
    let sbcast = slurm.log("sbcast.log");
    let lines: Vec<&str> = sbcast.lines().collect();
    assert_eq!(lines.len(), 1, "{sbcast}");
    assert!(
        lines[0].starts_with(&format!("--force --jobid={JOB_ID}.")),
        "{sbcast}"
    );
    let staging_prefix = format!("{}.sbcast-{JOB_ID}-", dir.display());
    assert!(lines[0].contains(&staging_prefix), "{sbcast}");
    assert!(staging_files(&slurm).is_empty(), "staging file left behind");
    assert!(Path::new(&agent).exists());

    // Every squeue call lists this job's steps.
    let squeue = slurm.log("squeue.log");
    assert!(!squeue.is_empty());
    assert!(
        squeue
            .lines()
            .all(|line| line == format!("--noheader --steps --jobs={JOB_ID} --format=%i|%j")),
        "{squeue}"
    );
    // No step outlived the run.
    for step in calls.keys() {
        assert!(!slurm.step_alive(step), "step {step} still running");
    }
}

/// `gauntlet bootstrap` in srun mode: configured hosts are a checked subset
/// of the allocation in config order; a broken node *outside* the subset
/// never fails the deploy (the broadcast is scoped to the stale fleet
/// nodes); connectivity names the job; a second bootstrap uploads nothing.
#[test]
fn bootstrap_over_srun_scopes_the_broadcast_and_is_idempotent() {
    let mut nodes = NODES.to_vec();
    nodes.push(("node-bad", "127.0.0.9"));
    let slurm = FakeSlurm::new("bootstrap", &nodes);
    slurm.mark_bad(&["node-bad"]);
    let dir = slurm.node_dir();
    let config = slurm.write_config(&format!(
        r#"
hosts = ["node-c", "node-a"]

[launch]
mode = "srun"

[launch.srun]
dir = {dir}
"#,
        dir = toml_string(dir.to_str().expect("utf8")),
    ));
    let config = config.to_str().expect("utf8").to_string();

    let (output, first) = bootstrap_report(&slurm, &config);
    let rows: Vec<&str> = first.hosts.iter().map(|row| row.host.as_str()).collect();
    assert_eq!(rows, ["node-c", "node-a"], "{}", show(&output));
    for row in &first.hosts {
        assert_eq!(check(row, "connectivity").status, CheckStatus::Ok);
        assert!(
            check(row, "connectivity")
                .detail
                .starts_with(&format!("srun step in job {JOB_ID}: ")),
            "{:?}",
            check(row, "connectivity")
        );
        assert_eq!(check(row, "arch").status, CheckStatus::Ok);
        assert_eq!(
            check(row, "deploy").status,
            CheckStatus::Ok,
            "{:?}",
            row.checks
        );
        assert_eq!(check(row, "deploy").detail, "installed via sbcast");
        assert_eq!(
            check(row, "probe").status,
            CheckStatus::Ok,
            "{:?}",
            row.checks
        );
        assert!(row.inventory.is_some());
    }
    // One sbcast, targeting the carrier step on exactly the fleet nodes.
    let sbcast = slurm.log("sbcast.log");
    assert_eq!(sbcast.lines().count(), 1, "{sbcast}");
    let step = sbcast
        .split_whitespace()
        .find_map(|word| word.strip_prefix(&format!("--jobid={JOB_ID}.")))
        .expect("step-scoped jobid");
    assert_eq!(slurm.step_nodes(step), ["node-c", "node-a"]);
    // Only the configured nodes ever got steps.
    let targeted: BTreeSet<String> = slurm
        .srun_calls()
        .values()
        .filter_map(|args| value_of(args, "--nodelist").map(str::to_string))
        .flat_map(|list| list.split(',').map(str::to_string).collect::<Vec<_>>())
        .collect();
    assert_eq!(
        targeted,
        ["node-a", "node-c"].map(String::from).into_iter().collect()
    );

    let (output, second) = bootstrap_report(&slurm, &config);
    for row in &second.hosts {
        assert_eq!(
            check(row, "deploy").detail,
            "up to date",
            "{}",
            show(&output)
        );
    }
    assert_eq!(slurm.log("sbcast.log").lines().count(), 1);
}

/// A broken node *inside* the fleet fails only its own deploy: the fleet
/// broadcast fails, deploy falls back to one sbcast per node, and the
/// staging file is cleaned up from every targeted node anyway.
#[test]
fn one_bad_fleet_node_fails_only_its_own_deploy() {
    let mut nodes = NODES.to_vec();
    nodes.push(("node-bad", "127.0.0.9"));
    let slurm = FakeSlurm::new("badnode", &nodes);
    slurm.mark_bad(&["node-bad"]);
    let dir = slurm.node_dir();
    let config = slurm.write_config(&format!(
        "hosts = [\"node-a\", \"node-bad\", \"node-b\"]\n[launch]\nmode = \"srun\"\n[launch.srun]\ndir = {}\n",
        toml_string(dir.to_str().expect("utf8")),
    ));
    let (output, report) = bootstrap_report(&slurm, config.to_str().expect("utf8"));
    let deploy: Vec<(&str, CheckStatus)> = report
        .hosts
        .iter()
        .map(|row| (row.host.as_str(), check(row, "deploy").status))
        .collect();
    assert_eq!(
        deploy,
        [
            ("node-a", CheckStatus::Ok),
            ("node-bad", CheckStatus::Fail),
            ("node-b", CheckStatus::Ok),
        ],
        "{}",
        show(&output)
    );
    let bad = &report.hosts[1];
    let detail = &check(bad, "deploy").detail;
    assert!(
        detail.contains("sbcast") && detail.contains("node-bad"),
        "{detail}"
    );
    // The fleet attempt, then one per node.
    assert_eq!(
        slurm.log("sbcast.log").lines().count(),
        4,
        "{}",
        slurm.log("sbcast.log")
    );
    assert!(staging_files(&slurm).is_empty(), "staging left behind");
}

/// sbcast staging files (siblings of the agent dir) currently on disk.
fn staging_files(slurm: &FakeSlurm) -> Vec<PathBuf> {
    let prefix = format!("{}.sbcast-", slurm.node_dir().display());
    std::fs::read_dir(&slurm.root)
        .expect("root")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.to_string_lossy().starts_with(prefix.as_str()))
        .collect()
}

#[test]
fn srun_mode_outside_an_allocation_is_a_clear_error() {
    let slurm = FakeSlurm::new("noalloc", &NODES);
    let config = slurm.write_config("[launch]\nmode = \"srun\"\n");
    let output = output_within(
        slurm
            .gauntlet(&["run", "--config", config.to_str().expect("utf8")])
            .env_remove("SLURM_JOB_ID"),
        Duration::from_secs(60),
        &slurm,
    );
    assert!(!output.status.success(), "{}", show(&output));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("SLURM_JOB_ID is not set"), "{stderr}");
    assert!(stderr.contains("salloc"), "{stderr}");
    assert!(slurm.log("srun.log").is_empty(), "no step may launch");
}

#[test]
fn configured_hosts_outside_the_allocation_are_rejected() {
    let slurm = FakeSlurm::new("outside", &NODES);
    let config = slurm.write_config("hosts = [\"node-a\", \"elsewhere\"]\n");
    let output = output_within(
        &mut slurm.gauntlet(&[
            "bootstrap",
            "--launch",
            "srun",
            "--config",
            config.to_str().expect("utf8"),
        ]),
        Duration::from_secs(60),
        &slurm,
    );
    assert!(!output.status.success(), "{}", show(&output));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("elsewhere"), "{stderr}");
    assert!(stderr.contains("not part of Slurm job 4242"), "{stderr}");
    assert!(slurm.log("srun.log").is_empty(), "no step may launch");
}

#[test]
fn ssh_mode_still_requires_hosts() {
    let slurm = FakeSlurm::new("nohosts", &NODES);
    let config = slurm.write_config("[launch]\nmode = \"srun\"\n");
    // The config is valid for srun; overriding to ssh leaves no hosts.
    let output = output_within(
        &mut slurm.gauntlet(&[
            "run",
            "--launch",
            "ssh",
            "--config",
            config.to_str().expect("utf8"),
        ]),
        Duration::from_secs(60),
        &slurm,
    );
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no hosts configured"), "{stderr}");
}

// ---------------------------------------------------------------------------
// Transport-level: kills, batching, and abandoned steps
// ---------------------------------------------------------------------------

async fn connect(launcher: &Launcher, addr: &str) -> Arc<HostSession> {
    Arc::new(
        launcher
            .connect(host(addr), &NcclEnv::default())
            .await
            .unwrap_or_else(|error| panic!("connect {addr}: {error:#}")),
    )
}

/// The steps named `name`, by step id.
fn steps_named(slurm: &FakeSlurm, name: &str) -> Vec<String> {
    slurm
        .srun_calls()
        .into_iter()
        .filter(|(_, args)| value_of(args, "--job-name") == Some(name))
        .map(|(step, _)| step)
        .collect()
}

/// The peer-teardown backstop: a background agent step is found by name
/// and SIGKILLed with scancel; a matching command on another node is left
/// alone. srun reports the killed task as exit 128 + 9.
#[tokio::test(flavor = "multi_thread")]
async fn kill_cancels_only_the_named_step_on_that_node() {
    let slurm = FakeSlurm::new("kill", &NODES);
    let launcher = slurm.launcher();
    let first = connect(&launcher, "node-a").await;
    let second = connect(&launcher, "node-b").await;
    slurm.install_agent();

    let port = port_base(1).to_string();
    let other_port = (port_base(1) + 1).to_string();
    let child = first
        .spawn_agent(&["peer", "serve", "--port", &port])
        .await
        .expect("spawn first");
    let other = second
        .spawn_agent(&["peer", "serve", "--port", &other_port])
        .await
        .expect("spawn second");
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Same command, other node: untouched.
    second
        .kill_agent(&format!("peer serve --port {port}"))
        .await;
    first.kill_agent(&format!("peer serve --port {port}")).await;
    let status = tokio::time::timeout(Duration::from_secs(10), child.wait())
        .await
        .expect("step must end after scancel")
        .expect("wait");
    assert_eq!(status.code(), Some(137), "{status:?}");
    let scancel = slurm.log("scancel.log");
    assert_eq!(scancel.lines().count(), 1, "{scancel}");
    assert!(
        scancel.starts_with(&format!("--signal=KILL {JOB_ID}.")),
        "{scancel}"
    );

    second
        .kill_agent(&format!("peer serve --port {other_port}"))
        .await;
    let status = tokio::time::timeout(Duration::from_secs(10), other.wait())
        .await
        .expect("second step ends")
        .expect("wait");
    assert_eq!(status.code(), Some(137));
}

/// In an `sbatch --array` job squeue prints `<array_job>_<task>.<step>`
/// while SLURM_JOB_ID is the task's raw id: the kill must scancel the id
/// exactly as printed (the fake scancel rejects the raw form).
#[tokio::test(flavor = "multi_thread")]
async fn kill_cancels_array_task_steps_by_their_printed_id() {
    let slurm = FakeSlurm::new("array", &NODES);
    slurm.as_array_task("1237_3");
    let launcher = slurm.launcher();
    let session = connect(&launcher, "node-c").await;
    slurm.install_agent();
    let port = port_base(4).to_string();
    let child = session
        .spawn_agent(&["peer", "serve", "--port", &port])
        .await
        .expect("spawn");
    tokio::time::sleep(Duration::from_millis(500)).await;
    session.kill_agent("peer serve").await;
    let status = tokio::time::timeout(Duration::from_secs(10), child.wait())
        .await
        .expect("array-task step cancelled")
        .expect("wait");
    assert_eq!(status.code(), Some(137));
    let scancel = slurm.log("scancel.log");
    assert!(scancel.starts_with("--signal=KILL 1237_3."), "{scancel}");
}

/// The fleet kill (the NCCL early-abort path): one squeue listing and one
/// scancel cover every host.
#[tokio::test(flavor = "multi_thread")]
async fn fleet_kill_lists_once_and_cancels_every_host_in_one_scancel() {
    let slurm = FakeSlurm::new("fleetkill", &NODES);
    let launcher = slurm.launcher();
    let mut sessions = Vec::new();
    for name in node_names() {
        sessions.push(connect(&launcher, name).await);
    }
    slurm.install_agent();
    let mut children = Vec::new();
    for (index, session) in sessions.iter().enumerate() {
        let port = (port_base(5) + index as u16).to_string();
        children.push(
            session
                .spawn_agent(&["peer", "serve", "--port", &port])
                .await
                .expect("spawn"),
        );
    }
    tokio::time::sleep(Duration::from_millis(500)).await;
    let squeue_before = slurm.log("squeue.log").lines().count();

    HostSession::kill_agents(&sessions, "peer serve").await;

    for child in children {
        let status = tokio::time::timeout(Duration::from_secs(10), child.wait())
            .await
            .expect("every step cancelled")
            .expect("wait");
        assert_eq!(status.code(), Some(137));
    }
    assert_eq!(slurm.log("squeue.log").lines().count(), squeue_before + 1);
    let scancel = slurm.log("scancel.log");
    let lines: Vec<&str> = scancel.lines().collect();
    assert_eq!(lines.len(), 1, "{scancel}");
    assert_eq!(
        lines[0].split_whitespace().count(),
        1 + NODES.len(),
        "{scancel}"
    );
}

/// Concurrent single-host kills (every host of a world hitting the same
/// phase timeout) are batched into one listing and one scancel too.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_kills_are_batched() {
    let slurm = FakeSlurm::new("batch", &NODES);
    let launcher = slurm.launcher();
    let a = connect(&launcher, "node-a").await;
    let b = connect(&launcher, "node-b").await;
    slurm.install_agent();
    let port_a = port_base(6).to_string();
    let port_b = (port_base(6) + 1).to_string();
    let child_a = a
        .spawn_agent(&["peer", "serve", "--port", &port_a])
        .await
        .expect("a");
    let child_b = b
        .spawn_agent(&["peer", "serve", "--port", &port_b])
        .await
        .expect("b");
    tokio::time::sleep(Duration::from_millis(500)).await;
    let squeue_before = slurm.log("squeue.log").lines().count();

    tokio::join!(a.kill_agent("peer serve"), b.kill_agent("peer serve"));

    for child in [child_a, child_b] {
        let status = tokio::time::timeout(Duration::from_secs(10), child.wait())
            .await
            .expect("cancelled")
            .expect("wait");
        assert_eq!(status.code(), Some(137));
    }
    assert_eq!(slurm.log("squeue.log").lines().count(), squeue_before + 1);
    assert_eq!(slurm.log("scancel.log").lines().count(), 1);
}

/// The NCCL early-abort shape: an agent blocked mid-run (a barrier
/// coordinator waiting for ranks that never join) while its stdout is
/// being consumed is killed from another task; the run call returns
/// promptly with a failure instead of holding until a phase timeout.
#[tokio::test(flavor = "multi_thread")]
async fn kill_aborts_an_agent_blocked_mid_run() {
    let slurm = FakeSlurm::new("abort", &NODES);
    let launcher = slurm.launcher();
    let session = connect(&launcher, "node-c").await;
    slurm.install_agent();

    let port = port_base(2).to_string();
    let running = tokio::spawn({
        let session = Arc::clone(&session);
        let port = port.clone();
        async move {
            session
                .run_agent_capture(
                    &[
                        "barrier", "serve", "--port", &port, "--world", "2", "--iters", "5",
                    ],
                    None,
                )
                .await
        }
    });
    tokio::time::sleep(Duration::from_millis(1000)).await;
    assert!(!running.is_finished(), "coordinator should be blocked");

    session
        .kill_agent(&format!("barrier serve --port {port}"))
        .await;
    let output = tokio::time::timeout(Duration::from_secs(10), running)
        .await
        .expect("aborted run returns promptly")
        .expect("join")
        .expect("run");
    assert!(!output.success());
    assert_eq!(output.status.code(), Some(137));
}

/// A shell step that hangs past its caller's timeout is cancelled, not
/// orphaned: dropping the future SIGTERMs srun, which kills the task.
#[tokio::test(flavor = "multi_thread")]
async fn an_exec_step_abandoned_by_a_timeout_is_cancelled() {
    let slurm = FakeSlurm::new("exectimeout", &NODES);
    let launcher = slurm.launcher();
    let session = connect(&launcher, "node-b").await;
    let before: BTreeSet<String> = steps_named(&slurm, "gauntlet:node-b:exec")
        .into_iter()
        .collect();

    let abandoned =
        tokio::time::timeout(Duration::from_secs(1), session.exec_capture("sleep 60")).await;
    assert!(
        abandoned.is_err(),
        "the exec must still be running at the timeout"
    );

    let hung: Vec<String> = steps_named(&slurm, "gauntlet:node-b:exec")
        .into_iter()
        .filter(|step| !before.contains(step))
        .collect();
    assert_eq!(hung.len(), 1, "{hung:?}");
    let step = &hung[0];
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while slurm.step_alive(step) && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(!slurm.step_alive(step), "abandoned exec step still running");
    assert!(
        slurm.log("cancelled.log").lines().any(|line| line == step),
        "srun was not told to cancel step {step}: {}",
        slurm.log("cancelled.log")
    );
}
