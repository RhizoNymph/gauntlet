# Bootstrap

## Scope
`gauntlet bootstrap`: take a fleet from "ssh works" (or, with `--launch
srun`, "I am inside a Slurm allocation") to "ready for `gauntlet run`"
with no manual node setup. Non-scope: OS/driver installation, user creation,
ssh key distribution (in ssh mode the operator must already be able to ssh
in; srun mode needs no ssh at all).

## Control flow
1. Load `FleetConfig`, resolve the launcher (`transport::resolve_launch`:
   `--launch` over `[launch] mode`; srun mode takes hosts from the Slurm
   allocation, see docs/features/slurm_launch.md), fan out per host with
   `ssh.max_concurrent` bound (`tokio::sync::Semaphore`).
2. Three stages, each preserving host order (`Stage`: `Ready` with the
   session and checks so far, or `Done` with a final row):
   connect (`Launcher::connect`) → arch check (`uname -m` must equal
   orchestrator arch, else Fail), per host (`connect_host`); then deploy
   fleet-wide (`deploy::ensure_fleet` over the Ready sessions: sha256
   compare, then per-host sftp upload over ssh, or one step-scoped sbcast
   to the stale nodes (per-node fallback on failure) plus per-node
   installs over srun — a broadcast covers many nodes at once, which is
   why deploy is not a per-host step); then per host
   (`finish_host`): run `agent probe`, parse
   `InventorySnapshot` into readiness checks (gpu_driver, gpu_libs,
   gpu_idle, clock_sync, ib_ports, governor, persistence_mode) → with `--tune`, apply
   `nvidia-smi -pm 1` and set the performance governor (sudo-gated; refusal
   is a Warn, never fatal).
   The first four steps are also matrix columns (`connectivity`, `arch`,
   `deploy`, `probe`); a Fail in any of them short-circuits the rest of that
   host's sequence, so its later columns render as `-`.
   Between probe and the derived checks, a runtime-only NCCL install
   (libnccl.so.2 present, no name cudarc searches — `nccl_shim_needed`)
   triggers the shim step: `<remote_dir>/lib/libnccl.so -> libnccl.so.2`
   (`ln -sf`, no sudo), then a re-probe. Every agent invocation runs under
   `env LD_LIBRARY_PATH=<remote_dir>/lib`, so cudarc's dlopen("libnccl.so")
   resolves through the shim. Matrix column `nccl_shim` reports it.
3. Render a host × check readiness matrix (comfy-table); exit non-zero iff
   any host has a Fail. Columns appear in execution order; cells are `ok`,
   or `warn:`/`fail:` plus a one-line detail.

## Check policy
Only these produce a Fail (i.e. a non-zero exit):
- `connectivity`: session or remote_dir setup failed. Over ssh the ok
  detail is the resolved remote dir; over srun it is `srun step in job
  <id>: <dir>` (a step reached the node inside that allocation). Running
  in srun mode outside an allocation, or naming hosts outside it, fails
  the whole command before any row (typed `SlurmError`).
- `arch`: node arch known and different from the orchestrator's. An
  unrecognized `uname -m` is a Warn — the deploy step fails loudly anyway.
- `deploy`, `probe`: the agent could not be installed or did not report.
  The deploy ok detail names the path: `up to date`, `uploaded` (sftp),
  `installed via sbcast`, `installed on the shared path`.
- `clock_sync`: |offset| ≥ 1000 ms (≥ 100 ms warns, absent offset warns).
- `ib_ports`: ports exist and none is Active (a partial outage warns; a node
  with no IB ports warns).

Everything else is advisory: `gpu_driver` (missing driver, no GPUs, or Xid
errors since boot), `gpu_libs` (dlopen probe: any of libcuda/libcublas/
libnccl not loadable on a GPU-bearing node warns — nccl-only absence calls
out that the NCCL sweep is unavailable; n/a without GPUs), `governor` (anything but `performance`),
`persistence_mode` (off on any GPU; reported `ok`/n-a when the node has no
GPUs), `gpu_idle` (see below), and both `tune_*` steps.

`gpu_idle` (`src/orchestrator/bootstrap/gpu_idle.rs`) applies the run's
`gpu_idle` policy (`proto::assess_gpu_idle`, same
`thresholds.gpu_idle_max_used_mib`) to the probe's per-GPU occupancy:
`ok` "<n> idle gpu(s)" (or n/a without GPUs); `warn` naming each busy GPU
and its processes, e.g. "gpu0: in use by VLLM::EngineCore (pid 2102873,
23232 MiB)"; `warn` "occupancy unknown on gpu <list>" when nvidia-smi did
not say. Advisory rather than Fail: a busy GPU does not stop a run from
starting (bootstrap's Fail set is "the run cannot proceed"), it makes its
GPU numbers meaningless — which the run itself reports as a Failed
`gpu_idle` outcome. See docs/features/phase0_inventory.md.

## Tuning (`--tune`)
Both steps use `sudo -n` so a node without passwordless sudo reports a Warn
instead of hanging on a password prompt. `tune_persistence` runs
`sudo -n nvidia-smi -pm 1` (skipped as n/a with no GPUs); `tune_governor`
writes `performance` into every `cpufreq/scaling_governor` via
`sudo -n tee` and then reads cpu0's governor back to confirm.

## Files
- `src/orchestrator/bootstrap.rs` — `run`, `CheckStatus`, `HostReadiness`,
  `ReadinessCheck`, `Stage`, `connect_host`, `finish_host`,
  `connectivity_detail`.
- `src/orchestrator/bootstrap/gpu_idle.rs` — `gpu_idle_check`.
- `src/orchestrator/deploy.rs` — `ensure_fleet`, `DeployOutcome`,
  `DeployMethod`, `local_sha256`, `AGENT_RELPATH`.
- `src/orchestrator/session.rs` — `HostSession` (establish/exec/upload).
- `src/orchestrator/transport/` — `Launcher`, `resolve_launch` (ssh or
  srun; docs/features/slurm_launch.md).
- `src/agent/mod.rs` — `probe()` prints `InventorySnapshot` JSON.

## Invariants
- Idempotent: a second bootstrap on a healthy fleet uploads nothing and
  changes nothing.
- Per-host failures never abort other hosts; every host appears in the
  matrix exactly once, in config order.
- Tuning steps run only under `--tune` and each reports ok/warn/fail
  independently.
- `ssh.remote_dir` is expanded on the *node*: a leading `~` becomes `$HOME`
  inside the remote shell word, never the operator's home. `HostSession`
  resolves and `mkdir -p`s it at connect time and caches the absolute path.
- Uploads are staged (`<path>.staging` → `chmod` → `mv -f`) so replacing a
  running agent cannot fail with `ETXTBSY` or leave a truncated binary.
  srun installs stage per process (`<path>.staging.$$`) from the sbcast
  copy, so nodes sharing a filesystem never write the same temp file.
- srun mode: `[launch.srun] dir` is an absolute literal path (no `~`),
  default `/tmp/gauntlet-$USER`, identical on every node.

## Machine interface (`--json`)

`gauntlet bootstrap --json` prints a `BootstrapReport` document to stdout
instead of the table: `{ schema_version, finished_epoch_secs, hosts:
[HostReadiness] }`, where each `HostReadiness` carries the ordered
`ReadinessCheck` list (`name`, `status`: ok|warn|fail, one-line `detail`)
plus the probed `InventorySnapshot` when available (including each GPU's
`occupancy` since proto v10; additive and serde-defaulted, so
`BOOTSTRAP_SCHEMA_VERSION` stays 1). The exit code contract
is unchanged (non-zero when any host's worst status is fail). The GUI
viewer runs bootstrap through this interface and renders the same matrix;
`BOOTSTRAP_SCHEMA_VERSION` bumps on field renames.
