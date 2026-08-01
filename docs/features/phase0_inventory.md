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
- `nvidia-smi --query-gpu=index,name,uuid,vbios_version,memory.total,ecc.errors.uncorrected.volatile.total,remapped_rows.pending,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,persistence_mode --format=csv,noheader,nounits`
  — absent binary ⇒ no GPUs; per-field "N/A" ⇒ None. Driver/CUDA versions
  from `nvidia-smi --query-gpu=driver_version ...` + `/usr/local/cuda/version.json`
  fallback to `nvidia-smi` banner.
- Clock offset: `chronyc tracking` then `timedatectl show` fallback.
- Xid errors: `journalctl -k` grep for "NVRM: Xid" (fallback /var/log/kern.log).

## Files
- `src/agent/inventory.rs` — `collect`, `run`.
- `src/proto.rs` — `InventorySnapshot`, `GpuInventory`, `NicInventory`,
  `IbPortInventory`, `consistency_fields`.

## Invariants
- `collect()` must complete in < 5s on a healthy node.
- Emits exactly one `Inventory` event per run.
- No stdout writes outside `EventSink`.
