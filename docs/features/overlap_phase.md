# Overlap Phase — Compute + Comms Under Combined Load

## Scope
Sustained cuBLAS GEMM on every GPU of a node running concurrently with an
intra-node NCCL all-reduce across those same GPUs, for a fixed window
(default 30s). Reports overlapped GEMM GFLOPS per GPU, overlapped intra-node
all-reduce bus bandwidth per node, and — the primary straggler signal —
*retention ratios* (overlapped/isolated) fed into the standard MAD outlier
analysis. Rationale: real training overlaps compute and communication;
stragglers born of PCIe contention, power steering, and NIC/GPU NUMA
misplacement only show under combined load.

Non-scope (documented follow-up): multi-node overlap (GEMM under the
fleet-wide or pair-level NCCL worlds). The fleet NCCL machinery is one
process per node driven by orchestrator-relayed rendezvous; overlapping it
with GEMM would need a combined directive (`agent nccl` + compute leg) and
per-rank duration consensus. The single-process intra-node form needs
neither, so it ships first.

## Baselines and why they are what they are
- **GEMM retention** divides by the phase-2 sustained number
  (`gpu_gemm_perf.gflops_<dtype>`) of the same GPU, same dtype, same
  `gemm_dim`, same repeat iteration, from the same run. That is why the
  phase is scheduled last.
- **All-reduce retention** divides by an isolated intra-node baseline the
  overlap phase itself measures (`baseline_secs`, default 5s) seconds before
  the combined window, on the *same communicator*. The phase-3 NCCL sweep
  cannot serve as the denominator: it measures the fleet-wide world (a
  different topology) and emits only on rank 0, so no per-node isolated
  number exists there.

## Data / control flow
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

## Configuration
`[tests]` keys (see `gauntlet.example.toml`): `overlap_secs` (30),
`overlap_baseline_secs` (5), `overlap_msg_mib` (64). The compute leg reuses
`gemm_dim` and the *first* entry of `gemm_dtypes` (contention measurement,
not dtype coverage). CLI: `--phases overlap` (needs the gpu phase from the
same run for GEMM retention; the all-reduce ratio is self-contained).

## Files
- `src/agent/gpu/overlap.rs` — agent-side driver, workers, timed rounds.
- `src/agent/gpu/gemm.rs` — shared kernels/helpers (`launch_gemm`,
  `upload_operands`, `fill_matrix`, `sustained_gflops_value` — pub(crate)).
- `src/agent/nccl.rs` — `all_reduce_bus_gib_per_sec` (shared formula).
- `src/proto.rs` — `Phase::Overlap`, `OverlapTaskSpec`,
  `TestId::{OverlapGemm, OverlapAllReduce, OverlapRetention}`;
  PROTO_VERSION 2 (AgentTaskSpec is deny_unknown_fields, so old agents
  cannot decode the new spec; the bump triggers re-deploy).
- `src/config.rs` — `overlap_*` keys, `task_spec` mapping.
- `src/agent/mod.rs` — phase dispatch (`overlap_phase`, gpu/non-gpu).
- `src/orchestrator/mod.rs` — Overlap in the node-local phase arm.
- `src/report/mod.rs` — `derive_overlap_retention`, display names,
  `min_overlap_retention` table column; SCHEMA_VERSION 3.

## Invariants
- The overlap phase runs after gpu/network (`Phase::ALL` order; the
  default config keeps that order) so retention divides isolated baselines
  from the same run and repeat.
- Agents never emit `TestId::OverlapRetention`; it is derived
  orchestrator-side and only from finite values over positive baselines.
- The isolated all-reduce baseline is measured with quiet SMs (workers not
  yet spawned) on the same communicator as the overlapped window.
- The GEMM leg and the collective run on different streams of the same
  device; neither is ever serialized behind the other by construction.
- Worker threads always reach the start barrier and are always joined —
  no deadlock, no leaked threads, regardless of error path.
- Everything compiles and unit-tests without a GPU: cudarc is
  dynamic-loading, all entry points are guarded, and the ratio/serde/
  wiring logic is pure and tested with synthetic data.
