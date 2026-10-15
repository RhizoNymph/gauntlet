# Bootstrap

## Scope
`gauntlet bootstrap`: take a fleet from "ssh works" to "ready for `gauntlet run`"
with no manual node setup. Non-scope: OS/driver installation, user creation,
ssh key distribution (the operator must already be able to ssh in).

## Control flow
1. Load `FleetConfig`, fan out per host with `ssh.max_concurrent` bound
   (`tokio::sync::Semaphore`; the permit covers session establishment only).
2. Per host: connect (`HostSession::connect`) → arch check (`uname -m` must
   equal orchestrator arch, else Fail) → `deploy::ensure_agent` (sha256
   compare, upload on mismatch) → run `agent probe`, parse
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
   resolves through the shim. Matrix column `nccl_shim` reports it. The
   logic lives in `orchestrator/nccl_shim.rs` and is shared with `run`:
   with the node-local default remote_dir the shim can vanish on reboot or
   tmpfiles aging, so a run whose phases use NCCL (network, overlap)
   checks for it after deploy (`ensure_fleet_shims`: one `[ -e ]` when
   present; otherwise probe, rebuild via the same `ensure_shim`, re-probe).
   Advisory there too: failures are logged, never a failed host.
3. Render a host × check readiness matrix (comfy-table); exit non-zero iff
   any host has a Fail. Columns appear in execution order; cells are `ok`,
   or `warn:`/`fail:` plus a one-line detail.

## Check policy
Only these produce a Fail (i.e. a non-zero exit):
- `connectivity`: session or remote_dir setup failed.
- `arch`: node arch known and different from the orchestrator's. An
  unrecognized `uname -m` is a Warn — the deploy step fails loudly anyway.
- `deploy`, `probe`: the agent could not be installed or did not report.
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
  `ReadinessCheck`.
- `src/orchestrator/bootstrap/gpu_idle.rs` — `gpu_idle_check`.
- `src/orchestrator/deploy.rs` — `ensure_agent`, `local_sha256`,
  `AGENT_RELPATH`.
- `src/orchestrator/session.rs` — `HostSession` (connect/exec/upload).
- `src/orchestrator/session/remote_fs.rs` — `remote_dir_word`,
  `remote_dir_script`, `staging_path`, `upload_nonce`, `install_command`,
  `FileMode`, `sweep_stale_staging_command`, `STALE_STAGING_MINUTES`.
- `src/orchestrator/nccl_shim.rs` — `NCCL_SHIM_SCRIPT`,
  `nccl_shim_needed`, `ensure_shim`, `ShimOutcome`, `probe`,
  `ensure_shim_for_run`, `ensure_fleet_shims`, `phases_use_nccl`.
- `src/remote_dir.rs` — `RemoteDir`, `RemoteDirPart`, `RemoteDirError`,
  `DEFAULT_REMOTE_DIR`.
- `src/build_info.rs` — `BuildInfo`, `GitRevision` (bootstrap `--json`).
- `src/agent/mod.rs` — `probe()` prints `InventorySnapshot` JSON.

## Invariants
- Idempotent: a second bootstrap on a healthy fleet uploads nothing and
  changes nothing.
- Per-host failures never abort other hosts; every host appears in the
  matrix exactly once, in config order.
- Tuning steps run only under `--tune` and each reports ok/warn/fail
  independently.
- `ssh.remote_dir` is a validated `RemoteDir` template
  (`src/remote_dir.rs`), default `/tmp/gauntlet-$USER`: node-local, since a
  home directory is often shared NFS on clusters and a shared remote_dir
  makes every node upload onto the same file at once. It is expanded on the
  *node*, never on the operator's machine: a leading `~` becomes `$HOME`,
  `$USER` / `${USER}` becomes `"$(id -un)"` (the remote login name, set or
  not in the login environment); every other `$` is a config error, and
  all other text is single-quoted literally (`remote_fs::remote_dir_word`).
  An explicit value such as `~/.gauntlet` keeps working. `HostSession`
  resolves it at connect time (`remote_dir_script`: `mkdir -p -m 700`;
  refuse it when the last component is a symlink the login user does not
  own (`find -user` does not follow it); `cd`, then refuse a directory not
  owned by the login user — in a world-writable parent like /tmp anyone
  could pre-create `/tmp/gauntlet-<you>` — and print `pwd -P`). The
  *physical* path is cached and used for every upload and exec, so the
  directory checked is the directory used, not a symlink that could be
  repointed later. A remote_dir on a `noexec` mount
  cannot run the agent; point it elsewhere.
- Uploads are staged under a per-upload random sibling name
  (`<dest>.tmp.<16 hex>`, `remote_fs::staging_path`), `chmod`ed, then
  renamed over the destination (`remote_fs::install_command`: `mv -f`, a
  same-directory rename(2)). Replacing a running agent therefore cannot
  fail with `ETXTBSY`, a killed or failed upload never leaves a torn
  binary (a failed one removes its temp file; a killed one leaves only an
  unreferenced `*.tmp.*`), and concurrent deploys to one shared path never
  share a temp file: each renames a complete, identical binary and the last
  rename wins. Each upload first sweeps `<dest>.tmp.*` regular files not
  modified for `STALE_STAGING_MINUTES` (15) — leftovers of killed uploads,
  each a full-size binary often on tmpfs — while younger ones, which may
  be another node's upload in flight on a shared directory, are never
  touched. Before uploading, `deploy::ensure_agent` compares the local
  sha256 with the remote `sha256sum` and skips the upload on a match.

## Machine interface (`--json`)

`gauntlet bootstrap --json` prints a `BootstrapReport` document to stdout
instead of the table: `{ schema_version, gauntlet_version,
finished_epoch_secs, hosts: [HostReadiness] }`, where each
`HostReadiness` carries the ordered `ReadinessCheck` list (`name`,
`status`: ok|warn|fail, one-line `detail`)
plus the probed `InventorySnapshot` when available (including each GPU's
`occupancy` since proto v10). `BOOTSTRAP_SCHEMA_VERSION` 2 adds
`gauntlet_version` (`build_info::BuildInfo`: crate version plus the git
revision captured at build time, `unknown` outside a checkout — the binary
that bootstrapped, which is also the agent it deployed;
serde-defaulted, `None` in v1 documents). The exit code contract
is unchanged (non-zero when any host's worst status is fail). The GUI
viewer runs bootstrap through this interface and renders the same matrix;
`BOOTSTRAP_SCHEMA_VERSION` bumps on field renames.
