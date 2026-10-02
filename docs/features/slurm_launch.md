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
- Parsing Slurm's bracket nodelist syntax: `scontrol show hostnames`
  expands it.
- MPI/PMI: steps are plain single-task steps; NCCL rendezvous is still the
  orchestrator-relayed unique id.
- A cross-architecture allocation (same rule as ssh: bootstrap's arch
  check fails it).

## Slurm versions
- **22.05+ required** for concurrent GPU steps: `--overlap` shares CPUs,
  memory *and GRES* between steps only since 22.05 (20.11/21.08 share
  CPUs only, so a second GPU step on a node blocks).
- Step-scoped `sbcast --jobid=<job>.<step>`: documented since at least
  20.11, so available on every supported version. (`sbcast --nodelist`
  only exists from 24.11 and is not used.)
- The SIGTERM behaviour of srun relied on below is in
  `src/srun/signals.c` (`_forward_signal`) of current Slurm.

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
Array-task and heterogeneous jobs work (steps are matched by name and
cancelled by the id squeue prints, see Kills).

Running in srun mode outside an allocation (no `SLURM_JOB_ID`) fails
before anything launches with `SlurmError::NotInAllocation`, whose message
says to use `salloc`/`sbatch` or `--launch ssh`.

## Data / control flow
1. `FleetConfig::load_for(path, --launch)` → `validate_for(effective
   mode)`: an empty `hosts` is allowed only when the effective mode is
   srun (`require_hosts_for`), so `gauntlet run --launch srun` works on a
   config with no hosts and no `[launch]` section.
2. `transport::resolve_launch(config, --launch)`:
   - ssh: `require_hosts_for(Ssh)`, `Launcher::Ssh`.
   - srun: `SlurmAllocation::from_env` (SLURM_JOB_ID, SLURM_JOB_NODELIST)
     → `scontrol show hostnames <nodelist>` → `parse_hostnames` (pure;
     duplicates, inner whitespace/control chars and empty output are
     typed errors) → `select_hosts` → `scontrol --oneliner show node
     <nodelist>` (one call for the whole allocation) → `parse_node_addrs`
     → `apply_node_addrs` → `config.with_hosts(..)`. Agent dir =
     `[launch.srun] dir` or `/tmp/gauntlet-$USER`. The orchestrator env
     names to strip (`env_to_strip`) and its own `LD_LIBRARY_PATH` are
     captured once. `SrunLauncher::new` starts the launcher's kill
     batcher task.
3. `Launcher::connect(host)` → `HostSession::establish`: `mkdir -p <dir>
   && cd && pwd` through `exec_capture` (srun: `srun ... sh -c <script>`,
   step `gauntlet:<host>:exec`), bounded by `STEP_SETUP_TIMEOUT` (60 s).
   A failed connect also cancels the host's exec steps by name.
4. Deploy: `deploy::ensure_fleet` (see below).
5. Phases: unchanged. `HostSession::{run_agent, run_agent_capture,
   spawn_agent}` spawn through `HostTransport::spawn`; for srun that is a
   local `srun` child (`SrunChild`) whose stdin/stdout/stderr are the
   task's (srun forwards stdin to the single task and closes it at EOF),
   so the JSON-lines event stream, the hello/proto check and stdin
   directives work byte for byte as over ssh. srun exits with the task's
   code, or 128 + signal for a signalled task.
6. Kills: see below.
7. `RunResults.launch = Some(launcher.record())` (also on partial
   snapshots); the table prints `launch: srun (slurm job N)`.

### Host addresses (NodeName vs NodeAddr)
`HostConfig.addr` stays the Slurm **NodeName** — what `srun --nodelist`
targets and what results are keyed by. NodeNames need not resolve, so
each node's **NodeAddr** (`scontrol --oneliner show node`, the first
` NodeAddr=` field of each `NodeName=` record, which precedes the
free-text OS/Reason fields) becomes `data_addr` — the address pairwise
TCP tests, the TCP barrier and peers connect to — when it differs from
the NodeName. A configured `data_addr` always wins; a node without a
NodeAddr (or a failed lookup, logged as a warning) keeps using its name.

### Step command line
```
srun <flags> <extra_flags> --nodes=1 --ntasks=1 --nodelist=<host> \
     --job-name=gauntlet:<host>:agent <args> --export=ALL \
     env LD_LIBRARY_PATH=<dir>/lib:<orchestrator's LD_LIBRARY_PATH> \
     <dir>/bin/gauntlet-agent agent <args...>
```
Built by `launch::srun::{step_args, agent_step_command}` (pure). User
flags first, managed flags last; every option is one `-`-prefixed word, so
the first non-option word is always the command. Every step is named:
`gauntlet:<host>:agent <args>`, `gauntlet:<host>:exec`, or
`gauntlet-bcast:<token>` (sbcast carrier).

Default `flags` (`DEFAULT_FLAGS`, replaced wholesale by `[launch.srun]
flags`; `extra_flags` appends):
- `--overlap` — steps share CPUs, memory and GRES with every other step
  (22.05+). Gauntlet always has concurrent steps on a node (a `peer
  serve` next to a probe, the barrier coordinator next to its own host's
  join, any agent next to the orchestrator's own step); without it a step
  blocks ("step creation temporarily disabled").
- `--cpu-bind=none` — the agent pins its own per-core workers; a task
  affinity mask would confine it. (Since `--exact` is not the default, a
  step already gets all of the job's CPUs on its node.)
- `--kill-on-bad-exit=1` — tear the step down when its task fails,
  regardless of the site `KillOnBadExit` default.

Rejected user flags (`SrunFlag::parse` → `SrunFlagError::Rejected` with a
typed `FlagRejection` reason). srun parses with `getopt_long`, so every
spelling is checked: `--name`, `--name=value`, any *prefix* of a rejected
long name (`--lab` is `--label`), and bundled short options (letters are
scanned until one that takes an argument, per `slurm_opt.c`: `-Ql` is
rejected, `-Kl` is `-K` with argument `l`):
- Managed: `--nodes/-N`, `--ntasks/-n`, `--ntasks-per-node`,
  `--nodelist/-w`, `--nodefile/-F`, `--exclude/-x`, `--relative/-r`,
  `--no-allocate/-Z`, `--job-name/-J`, `--jobid`, `--het-group`,
  `--clusters/-M`, `--export`. (`--export=ALL` is explicit because an
  sbatch `--export=NONE` propagates through `SLURM_EXPORT_ENV`.)
- Stdio (the protocol channel): `--label/-l`, `--output/-o`,
  `--input/-i`, `--error/-e`, `--open-mode`, `--pty`, `--multi-prog`,
  `--unbuffered/-u`, `--task-prolog`, `--task-epilog`, `--prolog`,
  `--epilog`.
- Signals: `--ignore-signals` (would defeat the SIGTERM cancellation).
- Detached: `--test-only`, `--async` (no attached stdio at all).

### GPU visibility
A step is allocated all of the GRES the *job* requested on its node by
default, and with `--overlap` concurrent steps share them; Slurm sets
`CUDA_VISIBLE_DEVICES` per step from the GPUs bound to it. So the
allocation must request the GPUs (`--gpus-per-node=N` or `--gres=gpu:N`).
An `--exclusive` allocation that requested *no* GPUs gets none implicitly
for its steps: add `extra_flags = ["--gres=gpu:N"]`. The orchestrator's
own `CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` and
`GPU_DEVICE_ORDINAL` are removed from the srun process env.

### Agent environment
`HostSession` holds an `AgentEnv { lib_dir, nccl }`.
- NCCL env: set on the local srun process (`Command::env`) and exported
  with `--export=ALL`; no shell parses it, so any value `NcclEnv` accepts
  arrives byte for byte. (`--export=ALL,NCCL_X=v` is not used: srun splits
  `--export` on commas.) Every `NCCL_*` of the orchestrator's own
  environment is stripped first, so `RunResults.nccl_env` stays exact.
- `LD_LIBRARY_PATH`: applied *inside* the step by `env` (argv words, no
  shell), as `<dir>/lib` prepended to the orchestrator's own value — the
  same value `--export=ALL` hands the task. The local srun client keeps
  the orchestrator's `LD_LIBRARY_PATH` untouched (module-provided
  Slurm/CUDA/NCCL paths; libslurm for srun itself). Over ssh the remote
  `env` line still sets `LD_LIBRARY_PATH='<dir>/lib'` (unchanged
  behaviour; it does not prepend).
- The agent never calls `set_var`.

### Kills and abandoned steps
No step is ever orphaned:
- **Abandoned srun** (a caller's timeout drops the future): every local
  srun is held by an `SrunChild` whose `Drop` sends srun **SIGTERM** if it
  is still running. srun handles SIGTERM (and SIGHUP) as "forcing job
  termination": it forwards SIGKILL to every task of the step and exits
  (`signals.c`). A single SIGINT only prints task status (a second within
  one second is needed to abort), and SIGKILL to srun — tokio's
  `kill_on_drop` — would leave the step running with nobody to cancel it.
- **Kill by name** (`HostSession::kill_agent`, peer teardown, the TCP
  barrier timeout, every NCCL timeout/abort path): `squeue --noheader
  --steps --jobs=<SLURM_JOB_ID> --format=%i|%j` → `parse_steps` →
  `steps_to_kill([(host, StepTarget)])` → `scancel --signal=KILL <id>...`.
  Ids are kept exactly as squeue printed them (`StepId`, validated:
  `<job>.<step>`, `<array_job>_<task>.<step>`, `<leader>+<offset>.<step>`;
  `.batch`/`.extern` rejected) and handed to scancel verbatim, so array
  and het jobs work — `SLURM_JOB_ID` of an array task is its *raw* id,
  which is what squeue `--jobs` takes but not what it prints.
  `StepTarget::Exec` matches exec steps (used after a failed connect).
- **Batching**: kill requests go over an mpsc channel to the launcher's
  kill batcher task; everything arriving within 100 ms becomes one squeue
  listing and one scancel (union of matches), each requester getting its
  own ids back. The NCCL early abort calls the fleet-level
  `HostSession::kill_agents` (one request for every surviving host), and
  near-simultaneous phase timeouts of a world's hosts coalesce the same
  way.

### Deploy (`deploy::ensure_fleet`)
There is no sftp. `[launch.srun] deploy`:
- `sbcast` (default): per host `sha256sum`; the stale hosts (connected
  and mismatched — never other allocation nodes) get the binary via
  `SrunLauncher::broadcast`: a **carrier step** (`srun --nodes=K
  --ntasks=K --ntasks-per-node=1 --nodelist=<stale> sleep <t>`) is
  started, found by name in squeue, and used as `sbcast --force
  --jobid=<SLURM_JOB_ID>[+<het offset>].<step> <binary>
  <dir>.sbcast-<job>-<sha12>`, which transmits to that step's nodes only;
  the carrier is then cancelled (scancel, plus SIGTERM on drop). If the
  fleet broadcast fails (sbcast, carrier, squeue or timeout error), it is
  retried **per node**, so one bad node fails only its own deploy. Then
  per host an install step (`sbcast_install_script`: `mkdir -p bin; cp`
  to `<agent>.staging.$$`; `chmod 755`; `mv -f`; print the digest,
  verified against the local one). Finally the staging file is removed
  (`rm -f`, best effort) from **every** targeted node, whatever the
  broadcast or install outcome. The staging file is a sibling of `dir` so
  its parent (`/tmp` by default) exists; unique per job and binary.
  Node-local `dir` means every node writes only its own copy (no
  shared-home write race, gap 6), and a running old agent keeps its
  inode.
- `shared`: `dir` is a shared filesystem visible to the orchestrator and
  all nodes; the orchestrator installs the binary once locally (temp name
  + rename) and each stale node only re-hashes it.
srun transport failures are typed (`SrunTransportError`: Spawn, Io,
Timeout, Squeue, Scancel, Sbcast, Carrier, Scontrol, ...), which is how
deploy decides whether a failed fleet broadcast is worth isolating per
node.

## Files
- `src/launch/mod.rs` — `LaunchMode`, `LaunchConfig`, `LaunchRecord`.
- `src/launch/slurm.rs` — `SlurmJobId`, `SlurmAllocation::from_env`,
  `parse_hostnames`, `select_hosts`, `scontrol_show_nodes_args`,
  `parse_node_addrs`, `apply_node_addrs`, `SlurmError`.
- `src/launch/srun/mod.rs` — `SrunConfig`, `SrunDir`, `SrunDeploy`,
  `DEFAULT_FLAGS`.
- `src/launch/srun/flags.rs` — `SrunFlag`, `SrunFlagError`,
  `FlagRejection`.
- `src/launch/srun/steps.rs` — `StepName`, `StepTarget`, `step_matches`,
  `StepId`, `StepEntry`, `squeue_args`, `parse_steps`, `steps_to_kill`,
  `find_step`, `scancel_args`.
- `src/launch/srun/command.rs` — `step_args`, `carrier_step_args`,
  `agent_step_command`, `env_to_strip`, `sbcast_args`,
  `sbcast_staging_path`.
- `src/orchestrator/transport/mod.rs` — `Launcher`, `resolve_launch`,
  `HostTransport`, `AgentChild`, `kill_agents`.
- `src/orchestrator/transport/ssh.rs` — openssh transport (unchanged
  behaviour).
- `src/orchestrator/transport/srun.rs` — `SrunLauncher` (exec, spawn,
  `kill_steps` via the kill batcher, `broadcast` via a carrier step),
  `SrunChild` (SIGTERM on drop), `SrunTransportError`, `SlurmTools`,
  `local`.
- `src/orchestrator/session.rs` — `HostSession` (`kill_agent`,
  `kill_agents`), `AgentEnv`.
- `src/orchestrator/fanout.rs` — `fan_out`, the bounded-concurrency
  helper shared by deploy and bootstrap.
- `src/orchestrator/deploy.rs` — `ensure_fleet`, `scoped_broadcast`.
- `src/orchestrator/bootstrap.rs` — `Stage`, `connect_host`,
  `finish_host`.
- `src/config.rs` — `FleetConfig.launch`, `load_for`, `validate_for`,
  `with_hosts`.
- `tests/srun_launch_tests.rs`, `tests/fake_slurm/*` — end-to-end runs
  against shell-script fakes of srun/squeue/scancel/sbcast/scontrol.

## Invariants
- Nothing above `HostSession`/`Launcher` branches on the transport.
- An agent or exec step targets exactly one allocated node
  (`--nodes=1 --ntasks=1 --nodelist`); a carrier step exactly the stale
  fleet nodes. Managed, stdio and signal options are not overridable.
- Every step is named, so every step can be found and cancelled; a
  dropped srun cancels its own step (SIGTERM).
- Step ids reach scancel exactly as squeue printed them.
- The agent env is exactly the resolved NCCL env plus `<dir>/lib`
  prepended to the inherited `LD_LIBRARY_PATH`; orchestrator `NCCL_*` and
  GPU-visibility variables never reach a step; srun keeps the
  orchestrator's own `LD_LIBRARY_PATH`.
- A broadcast never targets nodes outside the stale, connected set, and
  its staging file is removed from every node it targeted.
- `PROTO_VERSION` is unchanged; `SCHEMA_VERSION` is 13
  (`RunResults.launch`).

## Testing
Unit: flag validation in every spelling (managed/stdio/signal/detached,
prefixes, bundles), step ids in plain/array/het form, kill selection,
carrier and agent command lines (the agent command run through a real
`env`), NodeAddr parsing (including NodeAddr == NodeName and records
without one), host selection, allocation detection, config, the fan-out
helper, the sbcast install script through a real `sh`.

Integration (`tests/srun_launch_tests.rs`; fake nodes `node-a..c` whose
names do not resolve, NodeAddrs 127.0.0.1-3; the fake srun forwards
stdin, cancels its task on SIGTERM, and reports 128 + signal):
- `run --launch srun` with no hosts through inventory and network phases:
  results keyed by NodeName with peers reached via NodeAddr, every step's
  flags, srun's own env (orchestrator `LD_LIBRARY_PATH`, exact NCCL env,
  nothing stray) and the agent command (`env
  LD_LIBRARY_PATH=<dir>/lib:<orchestrator's>`), one step-scoped sbcast,
  staging cleaned, no step left running;
- bootstrap with a subset and a broken node outside it (deploy unaffected,
  carrier spans the subset only), idempotence;
- a broken node inside the fleet fails only its own deploy (fleet sbcast,
  then per node), staging cleaned;
- typed errors outside an allocation / for hosts outside it;
- kills: named step on the named node only; array-task ids
  (`1237_3.<step>`) cancelled verbatim; a fleet kill and concurrent
  single kills each cost one squeue and one scancel; a blocked agent
  aborted mid-run; an exec step abandoned by a timeout is cancelled.
