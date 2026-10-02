//! srun launch mode, end to end on localhost against a fake Slurm.
//!
//! `tests/fake_slurm/{srun,squeue,scancel,sbcast,scontrol}` are small shell
//! scripts installed into a temp dir per test. Each fake "node" is a
//! distinct loopback address (127.0.0.x) on this machine: srun execs the
//! real agent locally (stdio passed straight through, like srun's I/O
//! forwarding for one task), registers the step for squeue/scancel, and
//! records its argv and the task environment. The real `gauntlet` binary
//! drives `run --launch srun` and `bootstrap` through them, so the whole
//! path — allocation detection, nodelist expansion, sbcast deploy, the
//! JSON-lines protocol over srun stdio, stdin directives, step kills — is
//! exercised without Slurm.

use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use gauntlet::launch::{LaunchRecord, SlurmJobId, SrunConfig, SrunDir};
use gauntlet::nccl_env::NcclEnv;
use gauntlet::orchestrator::bootstrap::{BootstrapReport, CheckStatus};
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

struct FakeSlurm {
    root: PathBuf,
}

impl FakeSlurm {
    fn new(tag: &str, nodes: &[&str]) -> Self {
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
        let mut list = nodes.join("\n");
        list.push('\n');
        std::fs::write(root.join("nodes"), list).expect("nodes");
        Self { root }
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

    /// step id -> the selected env entries the task inherited.
    fn srun_envs(&self) -> BTreeMap<String, Vec<String>> {
        parse_records(&self.log("srun-env.log"))
    }

    fn write_config(&self, body: &str) -> PathBuf {
        let path = self.root.join("gauntlet.toml");
        std::fs::write(&path, body).expect("config");
        path
    }

    /// `gauntlet <args>` inside the fake allocation: shims first on PATH,
    /// SLURM_JOB_ID / SLURM_JOB_NODELIST set, plus orchestrator-side
    /// variables that must never reach an agent.
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
            .env("SLURM_JOB_NODELIST", "fake[1-3]")
            .env("CUDA_VISIBLE_DEVICES", "orchestrator-only")
            .env("NCCL_STRAY_FROM_SHELL", "1")
            .env("RUST_LOG", "info");
        command
    }
}

impl Drop for FakeSlurm {
    fn drop(&mut self) {
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
        // Kill whatever fake steps are still alive so the readers finish.
        for step in std::fs::read_dir(slurm.root.join("state/steps"))
            .into_iter()
            .flatten()
            .flatten()
        {
            let _ = Command::new("kill")
                .arg("-KILL")
                .arg(step.file_name())
                .status();
        }
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

const NODES: [&str; 3] = ["127.0.0.1", "127.0.0.2", "127.0.0.3"];

/// `gauntlet run --launch srun` with no hosts configured: the fleet is the
/// allocation, the agent arrives by sbcast, the inventory and network
/// phases (event streams, stdin task specs, pairwise peers, the TCP
/// barrier) all run through srun steps, and every step carries exactly
/// the configured NCCL env.
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

    // Results: the allocation's nodes, all healthy, launch recorded.
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
    let hosts: BTreeSet<&str> = results.hosts.keys().map(String::as_str).collect();
    assert_eq!(hosts, NODES.into_iter().collect(), "{}", show(&output));
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

    // Hosts came from `scontrol show hostnames $SLURM_JOB_NODELIST`.
    assert_eq!(slurm.log("scontrol.log").trim(), "show hostnames fake[1-3]");

    // Every step: managed flags, defaults plus extra flags, one node of
    // the allocation, and a gauntlet step name.
    let calls = slurm.srun_calls();
    assert!(calls.len() >= 9, "too few steps: {calls:?}");
    let agent = format!("{}/bin/gauntlet-agent", dir.display());
    let mut agent_steps = Vec::new();
    for (step, args) in &calls {
        for flag in [
            "--overlap",
            "--cpu-bind=none",
            "--kill-on-bad-exit=1",
            "--mpi=none",
            "--nodes=1",
            "--ntasks=1",
            "--export=ALL",
        ] {
            assert!(
                args.iter().any(|arg| arg == flag),
                "{step}: {flag} missing in {args:?}"
            );
        }
        let node = value_of(args, "--nodelist").expect("nodelist");
        assert!(NODES.contains(&node), "{step}: {node}");
        let name = value_of(args, "--job-name").expect("job name");
        assert!(name.starts_with("gauntlet"), "{step}: {name}");
        let command = args
            .iter()
            .position(|arg| !arg.starts_with('-'))
            .expect("command word");
        if args[command] == agent {
            assert_eq!(args[command + 1], "agent", "{args:?}");
            assert!(
                name.starts_with(&format!("gauntlet:{node}:")),
                "{name} vs {node}"
            );
            agent_steps.push(step.clone());
        } else {
            assert_eq!(&args[command..command + 2], ["sh", "-c"], "{args:?}");
        }
    }
    // inventory + counters + pairs + barrier on every node, at least.
    assert!(agent_steps.len() >= 9, "{agent_steps:?}");

    // The agent environment: exactly LD_LIBRARY_PATH and the resolved
    // NCCL env, values byte for byte; the orchestrator's GPU visibility
    // and stray NCCL_* never leak in.
    let envs = slurm.srun_envs();
    for step in &agent_steps {
        let env = envs.get(step).expect("env record");
        assert_eq!(
            env,
            &vec![
                format!("LD_LIBRARY_PATH={}/lib", dir.display()),
                "NCCL_DEBUG=WARN".to_string(),
                "NCCL_SOCKET_IFNAME=lo".to_string(),
                format!("NCCL_TEST_VALUE={ADVERSARIAL}"),
            ],
            "step {step}"
        );
    }

    // One sbcast for the whole allocation, to a job-unique staging file
    // that is cleaned up after the per-node installs.
    let sbcast = slurm.log("sbcast.log");
    let lines: Vec<&str> = sbcast.lines().collect();
    assert_eq!(lines.len(), 1, "{sbcast}");
    let staging_prefix = format!("{}.sbcast-{JOB_ID}-", dir.display());
    assert!(
        lines[0].starts_with(&format!("--force --jobid={JOB_ID} ")),
        "{sbcast}"
    );
    assert!(lines[0].contains(&staging_prefix), "{sbcast}");
    let staging = lines[0]
        .rsplit_once(' ')
        .map(|(_, dest)| dest)
        .expect("dest");
    assert!(!Path::new(staging).exists(), "staging file left behind");
    assert!(Path::new(&agent).exists());

    // Every peer teardown looks its step up by name.
    assert!(
        slurm
            .log("squeue.log")
            .lines()
            .all(|line| line == format!("--noheader --steps --jobs={JOB_ID} --format=%i|%j")),
        "{}",
        slurm.log("squeue.log")
    );
    assert!(!slurm.log("squeue.log").is_empty());
}

/// `gauntlet bootstrap` in srun mode (config `mode = "srun"`): configured
/// hosts are a checked subset of the allocation in config order, the
/// connectivity column names the job, deploy goes through sbcast, and a
/// second bootstrap uploads nothing.
#[test]
fn bootstrap_over_srun_reports_srun_checks_and_is_idempotent() {
    let slurm = FakeSlurm::new("bootstrap", &NODES);
    let dir = slurm.node_dir();
    let config = slurm.write_config(&format!(
        r#"
hosts = ["127.0.0.3", "127.0.0.1"]

[launch]
mode = "srun"

[launch.srun]
dir = {dir}
"#,
        dir = toml_string(dir.to_str().expect("utf8")),
    ));
    let config = config.to_str().expect("utf8").to_string();
    let bootstrap = || {
        let output = slurm
            .gauntlet(&["bootstrap", "--json", "--config", &config])
            .output()
            .expect("run bootstrap");
        let report: BootstrapReport = serde_json::from_slice(&output.stdout)
            .unwrap_or_else(|error| panic!("{error}: {}", show(&output)));
        (output, report)
    };

    let (output, first) = bootstrap();
    let rows: Vec<&str> = first.hosts.iter().map(|row| row.host.as_str()).collect();
    assert_eq!(rows, ["127.0.0.3", "127.0.0.1"], "{}", show(&output));
    for row in &first.hosts {
        let check = |name: &str| {
            row.checks
                .iter()
                .find(|check| check.name == name)
                .unwrap_or_else(|| panic!("{}: no {name} check: {:?}", row.host, row.checks))
        };
        assert_eq!(check("connectivity").status, CheckStatus::Ok);
        assert!(
            check("connectivity")
                .detail
                .starts_with(&format!("srun step in job {JOB_ID}: ")),
            "{:?}",
            check("connectivity")
        );
        assert_eq!(check("arch").status, CheckStatus::Ok);
        assert_eq!(check("deploy").status, CheckStatus::Ok, "{:?}", row.checks);
        assert_eq!(check("deploy").detail, "installed via sbcast");
        assert_eq!(check("probe").status, CheckStatus::Ok, "{:?}", row.checks);
        assert!(row.inventory.is_some());
    }
    // Only the configured nodes got steps.
    let nodes: BTreeSet<String> = slurm
        .srun_calls()
        .values()
        .filter_map(|args| value_of(args, "--nodelist").map(str::to_string))
        .collect();
    assert_eq!(
        nodes,
        ["127.0.0.1", "127.0.0.3"]
            .map(String::from)
            .into_iter()
            .collect()
    );

    let (output, second) = bootstrap();
    for row in &second.hosts {
        let deploy = row
            .checks
            .iter()
            .find(|check| check.name == "deploy")
            .unwrap_or_else(|| panic!("{}", show(&output)));
        assert_eq!(deploy.detail, "up to date");
    }
    assert_eq!(slurm.log("sbcast.log").lines().count(), 1);
}

#[test]
fn srun_mode_outside_an_allocation_is_a_clear_error() {
    let slurm = FakeSlurm::new("noalloc", &NODES);
    let config = slurm.write_config("[launch]\nmode = \"srun\"\n");
    let output = slurm
        .gauntlet(&["run", "--config", config.to_str().expect("utf8")])
        .env_remove("SLURM_JOB_ID")
        .output()
        .expect("run gauntlet");
    assert!(!output.status.success(), "{}", show(&output));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("SLURM_JOB_ID is not set"), "{stderr}");
    assert!(stderr.contains("salloc"), "{stderr}");
    assert!(slurm.log("srun.log").is_empty(), "no step may launch");
}

#[test]
fn configured_hosts_outside_the_allocation_are_rejected() {
    let slurm = FakeSlurm::new("outside", &NODES);
    let config = slurm.write_config("hosts = [\"127.0.0.1\", \"elsewhere\"]\n");
    let output = slurm
        .gauntlet(&[
            "bootstrap",
            "--launch",
            "srun",
            "--config",
            config.to_str().expect("utf8"),
        ])
        .output()
        .expect("run gauntlet");
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
    let output = slurm
        .gauntlet(&[
            "run",
            "--launch",
            "ssh",
            "--config",
            config.to_str().expect("utf8"),
        ])
        .output()
        .expect("run gauntlet");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no hosts configured"), "{stderr}");
}

// ---------------------------------------------------------------------------
// Transport-level: the kill paths the NCCL early-abort and peer teardown use
// ---------------------------------------------------------------------------

fn launcher(slurm: &FakeSlurm) -> Launcher {
    let srun = SrunLauncher::new(
        SlurmJobId::new(JOB_ID),
        SrunConfig::default(),
        SrunDir::parse(slurm.node_dir().to_str().expect("utf8")).expect("dir"),
        Vec::new(),
        SlurmTools::in_dir(&slurm.bin()),
    );
    Launcher::Srun(Arc::new(srun))
}

fn install_agent(slurm: &FakeSlurm) {
    let bin = slurm.node_dir().join("bin");
    std::fs::create_dir_all(&bin).expect("bin");
    std::fs::copy(env!("CARGO_BIN_EXE_gauntlet"), bin.join("gauntlet-agent")).expect("copy");
}

/// The peer-teardown backstop: a background agent step is found by name
/// and SIGKILLed with scancel; a matching command on another node is left
/// alone.
#[tokio::test(flavor = "multi_thread")]
async fn kill_cancels_only_the_named_step_on_that_node() {
    let slurm = FakeSlurm::new("kill", &NODES);
    let launcher = launcher(&slurm);
    let env = NcclEnv::default();
    let first = launcher
        .connect(
            gauntlet::config::HostConfig {
                addr: "127.0.0.1".into(),
                data_addr: None,
                labels: Default::default(),
            },
            &env,
        )
        .await
        .expect("connect first");
    let second = launcher
        .connect(
            gauntlet::config::HostConfig {
                addr: "127.0.0.2".into(),
                data_addr: None,
                labels: Default::default(),
            },
            &env,
        )
        .await
        .expect("connect second");
    install_agent(&slurm);

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
    assert_eq!(status.signal(), Some(9), "{status:?}");
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
    assert_eq!(status.signal(), Some(9));
}

/// The NCCL early-abort shape: an agent blocked mid-run (a barrier
/// coordinator waiting for ranks that never join) while its stdout is
/// being consumed is killed from another task; the run call returns
/// promptly with a failure instead of holding until a phase timeout.
#[tokio::test(flavor = "multi_thread")]
async fn kill_aborts_an_agent_blocked_mid_run() {
    let slurm = FakeSlurm::new("abort", &NODES);
    let launcher = launcher(&slurm);
    let session = Arc::new(
        launcher
            .connect(
                gauntlet::config::HostConfig {
                    addr: "127.0.0.3".into(),
                    data_addr: None,
                    labels: Default::default(),
                },
                &NcclEnv::default(),
            )
            .await
            .expect("connect"),
    );
    install_agent(&slurm);

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
    assert_eq!(output.status.signal(), Some(9));
}
