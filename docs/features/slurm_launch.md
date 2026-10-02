# Slurm launch (`--launch srun`)

## Scope
Start every agent as an `srun` job step of the Slurm allocation the
orchestrator itself runs in, instead of over ssh. This removes three
requirements on Slurm clusters: key-based ssh between compute nodes,
`pam_slurm_adopt` admitting the job owner into the node, and a
hand-rendered host list (hosts come from the allocation).

Behind one transport abstraction, so phases, NCCL world driving, failure
attribution, reporting and bootstrap are identical in both modes.

Non-scope:
- Submitting the allocation. The operator runs `salloc` (then `gauntlet
  run --launch srun` in the shell it opens) or an `sbatch` script that
  calls gauntlet; gauntlet never calls `sbatch`/`salloc`.
- Heterogeneous jobs and job arrays (a numeric `SLURM_JOB_ID` is required).
- Parsing Slurm's bracket nodelist syntax: `scontrol show hostnames`
  expands it.
- MPI/PMI: steps are plain single-task steps; NCCL rendezvous is still the
  orchestrator-relayed unique id.
- A cross-architecture allocation (same rule as ssh: bootstrap's arch
  check fails it).

## Usage
```sh
salloc -N 4 --gpus-per-node=8 --exclusive -t 1:00:00
gauntlet bootstrap --launch srun        # readiness matrix over srun steps
gauntlet run --launch srun              # hosts = the 4 allocated nodes
```
Or in the config: `[launch] mode = "srun"`. `hosts` may then be omitted
(the default: every allocated node, in `scontrol show hostnames` order,
which becomes fleet and NCCL rank order). When `hosts` is given in srun
mode it must be a subset of the allocation (labels and `data_addr` are
kept); a host outside it is a typed `SlurmError::HostsOutsideAllocation`.

Running in srun mode outside an allocation (no `SLURM_JOB_ID`) fails
before anything launches with `SlurmError::NotInAllocation`, whose message
says to use `salloc`/`sbatch` or `--launch ssh`.

## Data / control flow
1. `FleetConfig::load_for(path, --launch)` → `validate_for(effective
   mode)`: an empty `hosts` is allowed only when the effective mode is
   srun (`require_hosts_for`), so `gauntlet run --launch srun` works on a
   config with no hosts and no `[launch]` section. Plain `load`/`validate`
   use the configured mode.
2. `transport::resolve_launch(config, --launch)`: effective mode = CLI
   override else config.
   - ssh: `require_hosts_for(Ssh)` (a `--launch ssh` override of an
     srun config without hosts errors here), `Launcher::Ssh`.
   - srun: `SlurmAllocation::from_env` (SLURM_JOB_ID, SLURM_JOB_NODELIST)
     → `scontrol show hostnames <nodelist>` → `parse_hostnames` (pure:
     one name per line, order kept, blanks ignored; duplicates, inner
     whitespace/control chars and empty output are typed errors) →
     `select_hosts` → `config.with_hosts(..)` (revalidated). Agent dir =
     `[launch.srun] dir` or `/tmp/gauntlet-$USER` (`SrunDir::default_for`).
     The orchestrator env names to strip are computed once
     (`env_to_strip`). `Launcher::Srun(SrunLauncher)`.
3. `Launcher::connect(host)` → `HostSession::establish`: `mkdir -p <dir>
   && cd && pwd` through the transport's `exec_capture` (srun: `srun ...
   sh -c <script>`, step name `gauntlet-exec:<host>`), bounded by
   `STEP_SETUP_TIMEOUT` (60 s; slurmctld step creation is slower than an
   ssh mux handshake).
4. Deploy: `deploy::ensure_fleet` (see below).
5. Phases: unchanged. `HostSession::{run_agent, run_agent_capture,
   spawn_agent}` spawn through `HostTransport::spawn`; for srun that is a
   local `srun` child whose stdin/stdout/stderr are the task's (srun
   forwards stdin to the single task and closes it at EOF), so the
   JSON-lines event stream, the hello/proto check and stdin directives
   (task specs, NCCL directives) work byte for byte as over ssh.
6. Kills (`HostSession::kill_agent(args)`, used by peer teardown, the TCP
   barrier timeout, and every NCCL abort/timeout path through
   `kill_remote_agent`): ssh → `pkill -f '[g]auntlet-agent agent <args>'`;
   srun → `squeue --noheader --steps --jobs=<job> --format=%i|%j` →
   `parse_steps` → `steps_to_kill(job, host, args)` → `scancel
   --signal=KILL <job>.<step>` per match. The local srun process then
   exits with the task's signal, so an in-flight `run_agent` returns
   promptly (the early-abort contract in `orchestrator/nccl/`).
7. `RunResults.launch = Some(launcher.record())` (also on partial
   snapshots); the table prints `launch: srun (slurm job N)`.

### Step command line
```
srun <flags> <extra_flags> --nodes=1 --ntasks=1 --nodelist=<host> \
     --job-name=gauntlet:<host>:<agent args> --export=ALL \
     <dir>/bin/gauntlet-agent agent <args...>
```
Built by `launch::srun::step_args` (pure). User flags first, managed
flags last; every option is one `-`-prefixed word, so the first
non-option word is always the command.

Default `flags` (`DEFAULT_FLAGS`, replaced wholesale by `[launch.srun]
flags`; `extra_flags` appends):
- `--overlap` — since Slurm 22.05 lets a step share CPUs, memory *and
  GRES* with every other step. Gauntlet always has concurrent steps on a
  node (a `peer serve` next to the TCP-barrier join or another pair's
  probe, the barrier coordinator next to its own host's join, any agent
  next to the orchestrator's own batch/interactive step), and without it
  a step blocks ("step
  creation temporarily disabled") until resources free up. On 20.11/21.08
  `--overlap` shares CPUs only, so concurrent GPU steps on one node can
  still block there; 22.05+ is the supported baseline.
- `--cpu-bind=none` — the agent pins its own per-core workers; a task
  affinity mask would confine it. (Since `--exact` is not the default, a
  step already gets all of the job's CPUs on its node.)
- `--kill-on-bad-exit=1` — tear the step down when its task fails,
  regardless of the site `KillOnBadExit` default.

Managed (never user-overridable; `SrunFlag` rejects them, long and short
forms): `--nodes/-N`, `--ntasks/-n`, `--nodelist/-w`, `--job-name/-J`,
`--export`, `--jobid`, `--ntasks-per-node`.
- `--export=ALL` is explicit because an sbatch `--export=NONE` propagates
  into srun through `SLURM_EXPORT_ENV` and would drop the agent env.
- `--job-name` names the *step* (`StepName`), which is what kill matches.

### GPU visibility
A step is allocated all of the GRES the *job* requested on its node by
default (srun `--gres` docs), and with `--overlap` concurrent steps share
them; Slurm sets `CUDA_VISIBLE_DEVICES` per step from the GPUs bound to
it. So the allocation must request the GPUs (`--gpus-per-node=N` or
`--gres=gpu:N`). An `--exclusive` allocation that requested *no* GPUs
gets none implicitly for its steps: add `extra_flags =
["--gres=gpu:N"]`. The orchestrator's own `CUDA_VISIBLE_DEVICES`,
`ROCR_VISIBLE_DEVICES` and `GPU_DEVICE_ORDINAL` (often only the batch
step's GPUs) are removed from the srun process env (`env_to_strip`), so a
step sees exactly what Slurm binds to it, never the orchestrator's view.
The phase-0 `cuda_visible_gpus` vs nvidia-smi comparison surfaces a step
that sees fewer GPUs than the node has.

### Agent environment (NCCL env)
`HostSession` holds the agent env as typed pairs (`agent_env_vars`:
`LD_LIBRARY_PATH=<dir>/lib`, then the resolved `[nccl]` env in key
order). ssh renders them as single-quoted `env` words (unchanged). srun
sets them on the local srun process (`Command::env`) and `--export=ALL`
hands that environment to the task: no shell parses them anywhere, so any
value `NcclEnv` accepts arrives byte for byte. `--export=ALL,NCCL_X=v` is
deliberately not used: srun splits `--export` on commas, which breaks
values like `NCCL_IB_HCA=mlx5_0,mlx5_1`. Every `NCCL_*` variable of the
orchestrator's own environment is stripped first, so
`RunResults.nccl_env` stays the exact env every agent ran under. The
agent still never calls `set_var`.

### Deploy (`deploy::ensure_fleet`)
There is no sftp. `[launch.srun] deploy`:
- `sbcast` (default): per host `sha256sum` (one step each); if any host
  is stale, one `sbcast --force --jobid=<job> <orchestrator binary>
  <dir>.sbcast-<job>-<sha12>` to every allocated node (Slurm's tree
  broadcast; the staging file is a *sibling* of `dir` so its parent —
  `/tmp` by default — exists on every node, including nodes outside a
  configured host subset, and is unique per job and binary), then per
  stale host one install step (`sbcast_install_script`: `mkdir -p bin;
  cp` to `<agent>.staging.$$`; `chmod 755`; `mv -f` into place; print
  the new digest, verified against the local one), then a best-effort
  `rm -f` of the staging file per host. sbcast is used rather than a
  shared path by default because the default `dir` is node-local: every
  node writes only its own copy, which avoids the concurrent shared-home
  write race (gap 6), and a running old agent keeps its inode
  (`ETXTBSY`-free).
- `shared`: `dir` is a shared filesystem visible to the orchestrator and
  all nodes; the orchestrator installs the binary once locally (temp name
  + rename) and each stale node only re-hashes it (a mismatch says the
  path is not actually shared).
sbcast has no per-node target before Slurm 23.x, which is why deploy is a
fleet-level operation (`ensure_fleet`) rather than a per-session one;
bootstrap was restructured into connect/arch → fleet deploy → probe.

## Files
- `src/launch/mod.rs` — `LaunchMode` (serde + clap ValueEnum),
  `LaunchConfig` (`[launch]`, `effective_mode`), `LaunchRecord`
  (`{"mode":"ssh"}` / `{"mode":"srun","job_id":N}`).
- `src/launch/slurm.rs` — `SlurmJobId`, `SlurmAllocation::from_env`,
  `scontrol_hostnames_args`, `parse_hostnames`, `select_hosts`,
  `SlurmError`.
- `src/launch/srun.rs` — `SrunFlag`, `SrunDir`, `SrunDeploy`,
  `SrunConfig`, `DEFAULT_FLAGS`, `StepName`, `step_matches`, `step_args`,
  `env_to_strip`, `StepId`, `StepEntry`, `squeue_args`, `parse_steps`,
  `steps_to_kill`, `scancel_args`, `sbcast_args`, `sbcast_staging_path`.
- `src/orchestrator/transport/mod.rs` — `Launcher` (`connect`, `record`,
  `mode`), `resolve_launch`, `HostTransport` (enum dispatch:
  `exec_capture`, `spawn`, `kill_agent`), `AgentChild`
  (`take_stdin/stdout/stderr`, `wait`).
- `src/orchestrator/transport/ssh.rs` — openssh connect, `sh -c` exec,
  `env ...` spawn, `pkill` kill, sftp `upload` (moved, unchanged).
- `src/orchestrator/transport/srun.rs` — `SrunLauncher` (step command
  builder, exec, spawn, squeue/scancel kill, sbcast `broadcast`),
  `SlurmTools` (program paths; bare names on PATH by default).
- `src/orchestrator/session.rs` — transport-agnostic `HostSession`
  (`establish`, `exec`, `exec_capture`, `run_agent`, `run_agent_capture`,
  `spawn_agent`, `kill_agent`), `agent_env_vars`, quoting helpers.
- `src/orchestrator/deploy.rs` — `ensure_fleet`, `DeployOutcome`,
  `DeployMethod`.
- `src/orchestrator/bootstrap.rs` — `Stage`, `connect_host`,
  `finish_host`, `connectivity_detail`.
- `src/config.rs` — `FleetConfig.launch`, optional `hosts`, `load_for`,
  `validate_for`, `require_hosts_for`, `with_hosts`,
  `ConfigError::NoHosts { mode }`.
- `src/cli.rs` — `--launch ssh|srun` on `run` and `bootstrap`.
- `tests/srun_launch_tests.rs`, `tests/fake_slurm/*` — end-to-end runs
  against shell-script fakes of srun/squeue/scancel/sbcast/scontrol.

## Invariants
- Nothing above `HostSession`/`Launcher` branches on the transport.
- An srun step targets exactly one node of the orchestrator's own job
  (`--nodes=1 --ntasks=1 --nodelist`, managed flags not overridable), and
  only allocated nodes are ever targeted.
- Kill matches by step name with a word-boundary prefix on this job,
  this host and `agent <args>` only; exec steps are never kill targets.
- The agent env is exactly `LD_LIBRARY_PATH` + the resolved NCCL env:
  orchestrator `NCCL_*` and GPU-visibility variables never reach a step.
- A `LaunchRecord::Srun` always carries its job id.
- `PROTO_VERSION` is unchanged (the wire is identical); `SCHEMA_VERSION`
  is 13 (`RunResults.launch`).

## Testing
Unit: flag validation, dir validation/defaults, step args, kill matching,
squeue parsing, Slurm tool args, nodelist parsing, host selection,
allocation detection, launch records, config (`tests/config_tests.rs`),
the sbcast install script run through a real `sh`.
Integration (`tests/srun_launch_tests.rs`): `run --launch srun` with no
hosts over a 3-"node" fake allocation (127.0.0.1-3) through inventory
and network phases, asserting results, step flags, the exact agent env
(adversarial NCCL value), one sbcast, squeue lookups; `bootstrap` in
srun mode (subset order, connectivity/deploy cells, idempotence); typed
errors outside an allocation and for hosts outside it; scancel killing
only the named step on the named node; an abort of a blocked agent
mid-run.
