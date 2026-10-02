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
- `nvidia-smi --query-gpu=index,name,uuid,vbios_version,memory.total,ecc.errors.uncorrected.volatile.total,remapped_rows.pending,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,persistence_mode,driver_version,pci.bus_id,memory.used --format=csv,noheader,nounits`
  — absent binary ⇒ no GPUs; per-field "N/A" / "[N/A]" / "[Not Supported]" ⇒
  None. `driver_version` is folded into this one query rather than costing a
  second process spawn; it is taken from the first row. `memory.total` is
  MiB under `nounits` and is scaled to bytes. `pci.bus_id` and `memory.used`
  (proto v10) are appended last so every older position is unchanged; they
  feed GPU occupancy (below).
- `nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits`
  — per-GPU compute processes, see "GPU occupancy and gpu_idle".
- CUDA version: `/usr/local/cuda/version.json` (`cuda.version`), falling back
  to the `CUDA Version:` field of the `nvidia-smi` banner. Only probed when
  at least one GPU was found.
- Clock offset: `chronyc tracking` ("Last offset : ±N seconds"), then
  `timedatectl timesync-status` ("Offset: +1.234ms"). Note this is not
  `timedatectl show`, which exposes no offset at all.
- Xid errors: `journalctl -k --no-pager -o cat` grep for "NVRM: Xid"
  (fallback /var/log/kern.log), codes deduplicated and sorted ascending.
- GPU library stack: dlopen probe (libloading) for libcuda / libcublas /
  libnccl -> `gpu_libs` map. The "nccl" entry deliberately mirrors cudarc's
  search list, which does NOT include libnccl.so.2 (NCCL's actual runtime
  soname); a companion "nccl_runtime" entry probes libnccl.so.2 itself so
  bootstrap can tell "not installed" apart from "runtime-only install" and
  build a shim for the latter. This measures what the GPU/NCCL phases
  will actually experience; handles are leaked deliberately (some driver
  stacks misbehave under dlclose; the agent is short-lived). Feeds the
  bootstrap `gpu_libs` readiness check, the orchestrator's NCCL-sweep
  gating, and (on GPU-bearing hosts) consistency fields `lib:<name>`.
- CUDA-visible GPUs: `cuda_visible_gpus` = `cuDeviceGetCount` via cudarc
  (gpu feature; guarded, so a missing libcuda yields `None`, never a
  crash; `None` without the gpu feature). This — not the nvidia-smi
  `gpus` list — sizes each host's fleet NCCL rank block. When the two
  disagree (a GPU fell off the bus, MIG enabled, CUDA_VISIBLE_DEVICES set
  in the agent's environment, or no CUDA count on a host that lists
  GPUs), `proto::gpu_visibility_mismatch` produces a per-host finding:
  the inventory phase emits a Failed `inventory` outcome (Node scope)
  with the reason, making the verdict at least Failures.

## GPU occupancy and gpu_idle (proto v10 / schema v11)

### Scope
Detect, before any GPU phase runs, that a GPU is already in use by
something else — an inference server, a training job, or a leftover
gauntlet agent — and say so by name ("node1 gpu0 is in use by
VLLM::EngineCore (pid 2102873, 23232 MiB)"). Without it a shared GPU shows
up only indirectly: OOM in gpu_mem_bandwidth, ncclUnhandledCudaError at
fleet NCCL init, and silently depressed GEMM numbers.

Non-scope: acting on it. A busy host still runs the GPU/NCCL phases (no
auto-skip); gauntlet never starts or stops processes on a node.

### Types (`src/proto/occupancy.rs`, re-exported from `proto`)
- `GpuInventory.occupancy: GpuOccupancy` (serde default).
- `GpuOccupancy { memory_used_mib: Option<u64>, memory_total_mib:
  Option<u64>, compute_processes: Option<Vec<GpuProcess>> }`. `None` is
  always "unknown", never zero: `compute_processes = None` means the
  compute-apps query failed or this GPU's bus id is unknown;
  `Some(empty)` means verified none. The default (everything `None`) is
  what a pre-v10 inventory decodes to.
- `GpuProcess { pid, name, used_mib: Option<u64>, owner: ProcessOwner }`,
  `ProcessOwner::{Foreign, StaleGauntletAgent}`. `describe()` renders
  "VLLM::EngineCore (pid 2102873, 23232 MiB)" / "stale gauntlet agent
  <name> (pid N, M MiB)".
- `GpuIdleAssessment::{Idle, Busy { processes, memory_over:
  Option<MemoryOverage> }, Unknown { missing }}` from
  `assess_gpu_idle(&GpuOccupancy, max_used_mib)`; `.outcome()` maps to
  Passed / Failed { reason = detail } / Skipped { reason }.
- `gpu_idle_outcomes(&InventorySnapshot, max_used_mib) -> Vec<(Scope,
  TestOutcome)>`: one per GPU, `Scope::Gpu { index }`, index order.

### Collection (agent, `src/agent/gpu_occupancy.rs` + `inventory.rs`)
1. `probe_gpus` runs the GPU query, then the compute-apps query (2s budget
   each). `gpus_from_nvidia_smi(gpu_csv, Option<apps_csv>, identity,
   exe_of)` is the pure core, unit-tested with fixture text.
2. `parse_gpu_query` fills `memory_used_mib` / `memory_total_mib` and pairs
   each GPU with its `PciBusId` (typed; `FromStr` normalizes
   `00000000:01:00.0` and `0000:01:00.0` to the same value, typed
   `PciBusIdError` otherwise).
3. `parse_compute_apps` reads rows `bus_id, pid, name..., used_memory`
   (the name is everything between pid and the last field, so commas in a
   title survive; unparseable rows and banners are dropped; empty output =
   no compute processes).
4. `processes_by_bus` classifies each row with `classify`: a pid in the
   agent's own lineage (`AgentIdentity::current()`: this pid plus its
   `/proc/<pid>/stat` ppid chain) is dropped; a process whose reported name
   or `/proc/<pid>/exe` basename (" (deleted)" stripped — a replaced
   binary) is `gauntlet-agent` (or the running agent's own exe name) is
   `StaleGauntletAgent`; anything else is `Foreign`. An unreadable exe link
   (another user's process) falls back to the name.
5. `processes_for` attaches the list by bus id: a GPU absent from the
   compute-apps output gets `Some(empty)`; a failed query or an unknown bus
   id gets `None`.

Graphics-only clients (Xorg, sddm-greeter, type G in nvidia-smi's table)
are never listed by the compute-apps query, so they cannot trip the check;
their memory counts toward `memory.used`, which the threshold's headroom
absorbs.

### Policy (`assess_gpu_idle`)
- Busy (Failed) on any evidence: a listed process (foreign or stale
  agent), or `memory_used_mib > thresholds.gpu_idle_max_used_mib`
  (default 1024). Evidence wins over ignorance: a listed process fails the
  GPU even when memory.used is N/A.
- Idle (Passed) only when the process list is known-empty and memory used
  is known and ≤ the threshold (inclusive boundary).
- Otherwise Unknown (Skipped) with the missing signal named.

### Where outcomes are made
Orchestrator-side, because the threshold is orchestrator config and the
agent reports facts only: `orchestrator::node_phase` derives
`gpu_idle_outcomes` from each `Inventory` event as it arrives and forwards
them as `TestId::GpuIdle` outcomes right behind the snapshot (so partial
snapshots show them too). Failed feeds `report::verdict` like any failed
test (Failures, never HostFailures: the host is reachable and the finding
is actionable). Bootstrap applies the same `assess_gpu_idle` with the same
threshold to its probe (`gpu_idle` matrix column); the report renders a
"gpus in use (gpu_idle)" section (docs/features/reporting.md).

### Invariants
- Unknown is never reported as idle; the agent's own process tree is
  never reported; a stale gauntlet agent is never hidden.
- `TestId::GpuIdle` is never emitted by an agent.
- Old inventories/results decode with occupancy unknown (serde default).

## Probe execution model
Every external command runs through one bounded helper: stdin `/dev/null`,
stdout piped and drained on a companion thread (a blocking `wait()` would
deadlock behind a full pipe), the child killed once its deadline passes.
Budgets are 2s for each `nvidia-smi` query / `journalctl`, 1s for `chronyc`,
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
- `src/agent/inventory.rs` — `collect`, `run`, `probe_cuda_visible_gpus`,
  `GPU_QUERY`, `gpus_from_nvidia_smi`, `parse_gpu_query`.
- `src/agent/gpu_occupancy.rs` — `COMPUTE_APPS_QUERY`, `PciBusId`,
  `PciBusIdError`, `ComputeApp`, `parse_compute_apps`, `AgentIdentity`,
  `parse_stat_ppid`, `exe_basename`, `classify`, `processes_by_bus`,
  `processes_for`.
- `src/proto/mod.rs` — `InventorySnapshot`, `GpuInventory`, `NicInventory`,
  `IbPortInventory`, `consistency_fields`, `gpu_visibility_mismatch`.
- `src/proto/occupancy.rs` — `GpuOccupancy`, `GpuProcess`,
  `ProcessOwner`, `GpuIdleAssessment`, `MemoryOverage`, `assess_gpu_idle`,
  `gpu_idle_outcomes`.
- `src/orchestrator/mod.rs` (`node_phase`) — derives the gpu_idle outcomes.

## Invariants
- `collect()` must complete in < 5s on a healthy node.
- Emits exactly one `Inventory` event per run, plus one Failed
  `inventory` outcome when the GPU-visibility counts disagree. The
  orchestrator adds one `gpu_idle` outcome per listed GPU.
- No stdout writes outside `EventSink`.
