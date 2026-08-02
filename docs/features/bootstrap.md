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
   `InventorySnapshot` into readiness checks (gpu_driver, clock_sync,
   ib_ports, governor, persistence_mode) → with `--tune`, apply
   `nvidia-smi -pm 1` and set the performance governor (sudo-gated; refusal
   is a Warn, never fatal).
   The first four steps are also matrix columns (`connectivity`, `arch`,
   `deploy`, `probe`); a Fail in any of them short-circuits the rest of that
   host's sequence, so its later columns render as `-`.
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
GPUs), and both `tune_*` steps.

## Tuning (`--tune`)
Both steps use `sudo -n` so a node without passwordless sudo reports a Warn
instead of hanging on a password prompt. `tune_persistence` runs
`sudo -n nvidia-smi -pm 1` (skipped as n/a with no GPUs); `tune_governor`
writes `performance` into every `cpufreq/scaling_governor` via
`sudo -n tee` and then reads cpu0's governor back to confirm.

## Files
- `src/orchestrator/bootstrap.rs` — `run`, `CheckStatus`, `HostReadiness`,
  `ReadinessCheck`.
- `src/orchestrator/deploy.rs` — `ensure_agent`, `local_sha256`,
  `AGENT_RELPATH`.
- `src/orchestrator/session.rs` — `HostSession` (connect/exec/upload).
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
