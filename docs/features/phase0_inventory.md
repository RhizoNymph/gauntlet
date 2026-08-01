# Phase 0 — Inventory & Sanity

## Scope
Per-node hardware/software snapshot plus health counters; fleet-level
consistency analysis happens in reporting. Non-scope: fixing anything.

## Data flow
`agent run` (phase `inventory`) → `inventory::collect()` →
`AgentEvent::Inventory { snapshot }` → collector stores per host →
`report::build` extracts `proto::consistency_fields` and flags dissenters
from the majority value per field.

## Probe sources (all best effort — missing tool/permission ⇒ None/empty,
never an error; only unreadable /proc fails)
- /proc: hostname, cpuinfo (model, logical cores), meminfo.
- /sys/devices/system/node → NUMA node count.
- /sys/class/net/*/{mtu,speed} → NicInventory (skip lo).
- /sys/class/infiniband/*/ports/*/{state,rate,counters/link_downed} →
  IbPortInventory.
- /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor.
- `nvidia-smi --query-gpu=index,name,uuid,vbios_version,memory.total,ecc.errors.uncorrected.volatile.total,remapped_rows.pending,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,persistence_mode,driver_version --format=csv,noheader,nounits`
  — absent binary ⇒ no GPUs; per-field "N/A" / "[N/A]" / "[Not Supported]" ⇒
  None. `driver_version` is folded into this one query rather than costing a
  second process spawn; it is taken from the first row. `memory.total` is
  MiB under `nounits` and is scaled to bytes.
- CUDA version: `/usr/local/cuda/version.json` (`cuda.version`), falling back
  to the `CUDA Version:` field of the `nvidia-smi` banner. Only probed when
  at least one GPU was found.
- Clock offset: `chronyc tracking` ("Last offset : ±N seconds"), then
  `timedatectl timesync-status` ("Offset: +1.234ms"). Note this is not
  `timedatectl show`, which exposes no offset at all.
- Xid errors: `journalctl -k --no-pager -o cat` grep for "NVRM: Xid"
  (fallback /var/log/kern.log), codes deduplicated and sorted ascending.

## Probe execution model
Every external command runs through one bounded helper: stdin `/dev/null`,
stdout piped and drained on a companion thread (a blocking `wait()` would
deadlock behind a full pipe), the child killed once its deadline passes.
Budgets are 2s for `nvidia-smi` / `journalctl`, 1s for `chronyc`,
`timedatectl` and `uname`; captured output is capped at 8 MiB. A missing
binary, a non-zero exit and a timeout are all indistinguishable at the call
site: they yield `None`.

## Field fallbacks
- `kernel`: third token of /proc/version → `uname -r` → `"unknown"`.
- `cpu_model`: /proc/cpuinfo `model name` / `Model Name` / `cpu model` /
  `Hardware` → `uname -m` → `"unknown"`. Consistency analysis keys on both
  fields, so neither is ever empty.
- `logical_cores`: `processor` lines in /proc/cpuinfo → `available_parallelism`.
- NIC `speed`: unreadable (EINVAL on wireless/virtual links) or ≤ 0 (carrier
  down) ⇒ None, never 0.
- `nvlinks_active`: always None. It needs a separate, much slower
  `nvidia-smi nvlink -s` call, and phase 2 measures the links directly.

## Files
- `src/agent/inventory.rs` — `collect`, `run`.
- `src/proto.rs` — `InventorySnapshot`, `GpuInventory`, `NicInventory`,
  `IbPortInventory`, `consistency_fields`.

## Invariants
- `collect()` must complete in < 5s on a healthy node.
- Emits exactly one `Inventory` event per run.
- No stdout writes outside `EventSink`.
