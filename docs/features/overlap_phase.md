# Overlap Phase — Compute + Comms Under Combined Load

## Scope
Two combined-load steps, both scheduled last so their retention ratios
divide isolated baselines from the same run:

1. **Node-local overlap**: sustained cuBLAS GEMM on every GPU of a node
   running concurrently with an intra-node NCCL all-reduce across those
   same GPUs, for a fixed window (default 30s).
2. **Fleet overlap**: the same GEMM load on every GPU of every node while
   one rank per node drives a *cross-node* all-reduce over the real network
   fabric (proto v6 / schema v7; `tests.overlap_fleet`, default on, gated
   on ≥ 2 NCCL-capable hosts).

Each step reports overlapped GEMM GFLOPS per GPU, all-reduce bus bandwidth
isolated and overlapped, and — the primary straggler signal — *retention
ratios* (overlapped/isolated) fed into the standard MAD outlier analysis.
Rationale: real training overlaps compute and communication; stragglers
born of PCIe contention, power steering, and NIC/GPU NUMA misplacement only
show under combined load. The fleet step is what catches GPU↔NIC PCIe
contention, GPUDirect RDMA degradation under compute load, and NIC/network
-stack behavior on hot, busy hosts — paths the intra-node form structurally
cannot see.

Non-scope: pair-level (tournament) overlap — retention is per-node and the
MAD comparison across nodes localizes the bad node from the single
fleet-wide group, at flat wall time in n; pair topologies would add
rounds × window cost for little extra attribution.

## Baselines and why they are what they are
- **GEMM retention** (both steps) divides by the phase-2 sustained number
  (`gpu_gemm_perf.gflops_<dtype>`) of the same GPU, same dtype, same
  `gemm_dim`, same repeat iteration, from the same run. That is why the
  phase is scheduled last.
- **All-reduce retention** (both steps) divides by an isolated baseline the
  step itself measures (`baseline_secs`, default 5s) seconds before the
  combined window, on the *same communicator*. The phase-3 NCCL sweep
  cannot serve as the denominator: it measures a different message-size
  regime and (for the intra-node step) a different topology, and it emits
  only on rank 0, so no per-node isolated number exists there. The two
  steps keep separate baselines — intra-node and fleet communicators are
  not comparable.

## Data / control flow (node-local step)
1. Orchestrator: `Phase::Overlap` is node-local; the phase driver in
   `orchestrator/mod.rs` routes it through `node_phase` (same as phases
   0-2), sending `AgentTaskSpec` (with `OverlapTaskSpec`) to `agent run` on
   every host simultaneously.
2. Agent (`agent/gpu/overlap.rs::run`):
   - `< 2` GPUs → Skipped outcomes (a world of one moves nothing);
     driver-init failure → Failed outcomes; all cudarc entry goes through
     `gpu::guard` (dlopen panics become findings, per the phase-2 pattern).
   - `execute`: one `CudaContext` per GPU; NCCL communicator via
     `Comm::from_devices` (`ncclCommInitAll`, single process, one rank per
     GPU, each rank on its device's default stream). Warmup rounds, then
     the **isolated baseline**: `timed_rounds` drives grouped all-reduces
     (`ncclGroupStart`/`End`, `ROUND_ITERS` per sync) for `baseline_secs`.
   - **Combined window**: one plain thread per GPU
     (`gemm_worker`) binds the context, creates its own stream
     (`ctx.new_stream()` — the default stream carries the collective) and
     cuBLAS handle, uploads fixed-seed operands (reusing `gemm.rs`
     machinery: `fill_matrix`, `upload_operands`, `launch_gemm`), warms
     once, then meets a `Barrier` with the driver thread. Workers hammer
     GEMM batches until an `AtomicBool` stop flag; the driver runs
     `timed_rounds` for `duration_secs`, sets the flag, joins. Workers
     reach the barrier on *every* path, including failed setup, so the
     driver can never deadlock; the driver stops and joins workers before
     propagating any collective error, so threads never leak.
   - Emit under `Scope::Node`: `overlap_all_reduce.{msg_bytes,
     isolated_bus_gib_per_sec, overlap_bus_gib_per_sec}` (bus bandwidth via
     `nccl::all_reduce_bus_gib_per_sec`, factor `2(n-1)/n`, n = GPU count);
     per `Scope::Gpu`: `overlap_gemm.gflops_<dtype>` (via
     `sustained_gflops_value`). One GPU's worker failing yields a Failed
     outcome for that GPU only.
3. Report (`report/mod.rs::derive_overlap_retention`, called at the top of
   `build`): per host, appends derived `overlap_retention` records —
   `gemm_<dtype>` per GPU and `all_reduce` per node, `Unit::Ratio`, joined
   within each repeat index. Ratios form only over finite, positive
   baselines; the function is idempotent (a document already carrying
   retention records derives nothing). The derived records then ride the
   existing aggregate → MAD-outlier → threshold machinery and appear in the
   JSON under `hosts.<h>.metrics` and in `aggregates` as
   `overlap_retention.gemm_<dtype>` / `overlap_retention.all_reduce`.
4. Terminal table: the hosts section gains an `overlap ret (min)` column —
   the worst per-subject median retention on that host; retention outliers
   surface in the standard outliers section.

## Fleet overlap step

### Topology
One fleet-wide NCCL group with the phase-3 sweep's shape: one rank per
NCCL-capable node, rank ordinal = fleet order, the collective on GPU 0's
default stream. NOT tournament pairs — retention is per-node, so the MAD
comparison across nodes localizes the bad node from a single group, and
one ~(baseline_secs + duration_secs) window keeps wall time flat in n. The
compute leg spans *every* local GPU (`gpu::worker`, one thread + one extra
stream per GPU), so GPU 0's collective path contends with the whole node's
compute, power, and PCIe pressure — the "every GPU loaded, fabric busy"
regime training actually runs in. Rendezvous reuses the `agent nccl` relay
(rank 0 mints the `NcclId`, the orchestrator relays it).

### Window consensus without clock sync
Both measurement windows (isolated baseline, then overlapped) must end
between the same two iterations on every rank, or ranks would blend
baseline and loaded iterations differently. Host clocks cannot arbitrate
this (NTP-grade offsets). Instead the boundary travels *through the
collective*: after every `OVERLAP_CONTROL_INTERVAL` (4) payload
all-reduces, all ranks all-reduce a one-element control word with MIN —
rank 0 contributes 0.0 once its local clock says the window is over, 1.0
before that; every other rank always contributes 1.0. The MIN is 0 exactly
when the lead closed the window, and a collective returns the same value
everywhere, so all ranks leave in the same control step. Only rank 0's
clock ever matters.

Refinements, all in the pure module (`src/agent/window.rs`, unit-tested
without a GPU):
- **Alignment round**: each window opens with one untallied payload batch,
  so whatever skew the previous boundary (or the worker start release)
  left behind is absorbed before the first tallied batch — the figure
  starts from a synchronized point on every rank.
- **Iteration floor** (`MIN_WINDOW_ITERS`, 16): a close signal is honored
  only once that many payload iterations are tallied, so a degraded link
  cannot reduce the baseline denominator to a single cold batch. Every
  rank runs identical batches, so the floor decides identically everywhere
  and can never split the group.
- **Follower failsafe** (`failsafe_secs`: 2× the window budget, floor
  10s): a follower whose lead never closes the window ends the protocol
  with a structured error instead of hammering the fabric until an
  external kill. It only helps while collectives still complete; a rank
  blocked *inside* a collective is reaped by the orchestrator's phase
  timeout, as before.
- Control steps and the alignment round stay outside the bandwidth tally
  (`WindowTally`), so per-iteration timings measure the payload collective
  only.

### Flow
1. Orchestrator (`orchestrator/nccl.rs::overlap_fleet_sweep`, inside the
   Overlap phase arm after the node-local fan-out, gated on
   `tests.overlap_fleet` and ≥ 2 hosts from `nccl_world`): builds an
   `NcclJob` with `NcclWorkload::Overlap(spec)` and runs it through
   `drive_fleet_nccl` — the rendezvous-relay + participant-supervision
   driver shared with the phase-3 sweep — in `WarnOnly` failure mode.
2. Agent (`agent/nccl.rs::overlap_fleet`, gpu feature): the workload enum
   replaces the sweep wholesale. All fallible GEMM setup happens *before*
   the communicator enters the first window: `local_contexts` (a bad GPU
   degrades to a Failed report entry) then `GemmLoad::spawn` +
   `wait_ready` (workers upload, warm up, and park quiet at the start
   barrier). Then warmup collectives, the isolated baseline window (quiet
   GPUs), `start()` — the only between-window action — the overlapped
   window on the same communicator, and `finish` (stop + join) on every
   path before any error propagates. A rank can no longer abort *between*
   the windows and strand the fleet mid-collective.
3. Every rank emits `AgentEvent::OverlapFleetReport` (barrier-timings
   pattern: participants emit Hello, then the typed event; the driver
   intercepts and merges; a stray at the collector is debug-ignored).
4. Orchestrator (`fleet_overlap_records`) turns each report into metrics
   attributed to that rank's host — `overlap_fleet_all_reduce.{msg_bytes,
   isolated_bus_gib_per_sec, overlap_bus_gib_per_sec}` under `Scope::Node`,
   `overlap_fleet_gemm.gflops_<dtype>` per `Scope::Gpu` (names via
   `proto::overlap_metric`) — plus Passed outcomes, and per-GPU Failed
   outcomes for failed workers.
5. Everything short of running lands in the results document: the gated
   skip (< 2 NCCL-capable hosts) records Skipped outcomes for both fleet
   tests on each eligible host; a rank that fails, times out, or never
   reports records Failed outcomes with the reason — but never a
   failed-*host* verdict. Only `overlap_fleet = false` leaves no trace.
6. Report: `derive_overlap_retention` also derives
   `overlap_retention.fleet_gemm_<dtype>` (per GPU, vs the phase-2
   baseline, same GPU/repeat) and `overlap_retention.fleet_all_reduce`
   (per node, overlapped/isolated from this step's own baseline window).

## Configuration
`[tests]` keys (see `gauntlet.example.toml`): `overlap_secs` (30),
`overlap_baseline_secs` (5), `overlap_msg_mib` (64) — shared by both
steps — and `overlap_fleet` (true) to disable the fleet step. The compute
leg reuses `gemm_dim` and the *first* entry of `gemm_dtypes` (contention
measurement, not dtype coverage). CLI: `--phases overlap` (needs the gpu
phase from the same run for GEMM retention; the all-reduce ratios are
self-contained).

## Files
- `src/agent/gpu/overlap.rs` — node-local agent-side driver, timed rounds.
- `src/agent/gpu/worker.rs` — shared GEMM-load machinery (`GemmLoad`:
  spawn / wait_ready / start / finish; operands filled once and `Arc`d
  across workers), used by both steps.
- `src/agent/window.rs` — pure window-consensus logic (control word,
  closure predicate + iteration floor, follower failsafe, payload tally);
  no GPU dependency.
- `src/agent/nccl.rs` — `all_reduce_bus_gib_per_sec`, `message_elements`
  (shared); `overlap_fleet` + `consensus_window` (gpu feature).
- `src/agent/gpu/gemm.rs` — shared kernels/helpers (`launch_gemm`,
  `upload_operands`, `fill_matrix`, `sustained_gflops_value` — pub(crate)).
- `src/proto.rs` — `Phase::Overlap`, the shared `OverlapSpec` (embedded in
  both `AgentTaskSpec.overlap` and `NcclWorkload::Overlap`),
  `NcclWorkload` (sweep and overlap mutually exclusive by construction),
  `OverlapFleetReport`/`OverlapGpuGemm`, `overlap_metric` name consts,
  `TestId::{OverlapGemm, OverlapAllReduce, OverlapFleetGemm,
  OverlapFleetAllReduce, OverlapRetention}`; PROTO_VERSION 6.
- `src/config.rs` — `overlap_*` keys, the single `overlap_spec` mapper
  (feeds both steps).
- `src/agent/mod.rs` — phase dispatch (`overlap_phase`, gpu/non-gpu).
- `src/orchestrator/mod.rs` — Overlap phase arm (node-local fan-out then
  the fleet step).
- `src/orchestrator/nccl.rs` — `nccl_world` / `NcclJob` /
  `drive_fleet_nccl` (returns the per-host failure map) shared with the
  phase-3 sweep; `overlap_fleet_sweep`, `fleet_overlap_records`,
  `step_outcomes`.
- `src/orchestrator/collect.rs` — stray `OverlapFleetReport` ignored.
- `src/report/mod.rs` — `derive_overlap_retention` (both steps), display
  names, `min_overlap_retention` table column; SCHEMA_VERSION 7.

## Invariants
- The overlap phase runs after gpu/network (`Phase::ALL` order; the
  default config keeps that order) so retention divides isolated baselines
  from the same run and repeat; the fleet step runs after the node-local
  fan-out.
- Agents never emit `TestId::OverlapRetention`; it is derived
  orchestrator-side and only from finite values over positive baselines.
  Fleet and intra-node all-reduce retentions never share a baseline.
- Each isolated all-reduce baseline is measured with quiet SMs on the same
  communicator as its overlapped window: intra-node, the workers are not
  yet spawned; in the fleet step they are already set up but parked at the
  start barrier (`wait_ready`), so a rank never performs fallible GEMM
  setup between the windows.
- The GEMM leg and the collective run on different streams of the same
  device; neither is ever serialized behind the other by construction.
- Worker threads always reach both barriers and are always joined — no
  deadlock, no leaked threads, regardless of error path; `finish` releases
  workers still parked at the start barrier (`GemmLoad` contract, both
  steps).
- Fleet windows close for every rank in the same control step, driven
  solely by rank 0's clock through the MIN-reduced control word (subject
  to the deterministic iteration floor); followers never consult their own
  clocks except for the dead-lead failsafe, which ends the protocol with
  an error rather than closing a window.
- Every tallied window figure rests on at least `MIN_WINDOW_ITERS` payload
  iterations, measured from an aligned start (untallied alignment round).
- A failed fleet-overlap NCCL group, rank, or gate is visible in the
  results document as Skipped/Failed outcomes on the fleet overlap tests —
  never as a failed-host verdict on its own. A failed GEMM worker (or
  unusable GPU context) inside a successful group is a per-GPU Failed
  outcome (mirrors the node-local step). Only `overlap_fleet = false`
  leaves no trace.
- Everything compiles and unit-tests without a GPU: cudarc is
  dynamic-loading, all entry points are guarded, and the consensus/ratio/
  serde/wiring logic is pure and tested with synthetic data.
