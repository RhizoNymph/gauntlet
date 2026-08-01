# Bootstrap

## Scope
`gauntlet bootstrap`: take a fleet from "ssh works" to "ready for `gauntlet run`"
with no manual node setup. Non-scope: OS/driver installation, user creation,
ssh key distribution (the operator must already be able to ssh in).

## Control flow
1. Load `FleetConfig`, fan out per host with `ssh.max_concurrent` bound.
2. Per host: connect (`HostSession::connect`) → arch check (`uname -m` must
   equal orchestrator arch, else Fail) → `deploy::ensure_agent` (sha256
   compare, upload on mismatch) → run `agent probe`, parse
   `InventorySnapshot` into readiness checks (gpu_driver, clock_sync,
   ib_ports, governor, persistence_mode) → with `--tune`, apply
   `nvidia-smi -pm 1` and set the performance governor (sudo-gated; refusal
   is a Warn, never fatal).
3. Render a host × check readiness matrix (comfy-table); exit non-zero iff
   any host has a Fail.

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
  matrix exactly once.
- Tuning steps run only under `--tune` and each reports ok/warn/fail
  independently.
