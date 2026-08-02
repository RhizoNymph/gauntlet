# Reporting & Analysis

## Scope
Fleet statistics (median/MAD outliers), absolute thresholds, consistency
findings, calibration extraction, JSON persistence, terminal rendering,
exit codes. Non-scope: collecting data (orchestrator), emitting it (agent).

## Data flow
`Collector::into_observations()` → `report::build(config, observations,
started, finished)`:
1. Group every metric into "<test>.<metric>" buckets (`metric_key`).
   Sample key = `sample_key(host, scope)` (see Key formats). Unit is a
   function of (test, name), so a bucket never mixes units.
2. `stats::flag_outliers(samples, thresholds.mad_k)` per *fleet-comparable*
   group → `fleet.outliers`. Only groups with at least one hit are
   inserted, so `outliers.is_empty()` means a clean fleet.
3. Absolute `thresholds.absolute` bounds → `fleet.threshold_violations`
   (offending sample keys; only non-empty groups inserted).
4. `proto::consistency_fields` majority vote → `fleet.consistency`
   (fields where ≥1 host dissents; majority = most common value, ties
   broken by the lexicographically smallest value). Optional fields
   (nvidia_driver, cuda_version) report "(absent)" rather than being
   skipped, so present-vs-absent skew dissents; `lib:<name>` fields carry
   the dlopen probe results on GPU-bearing hosts.
5. Rooflines per host (min across the host's GPUs for GPU metrics; the
   straggler defines the node) and `links` alpha-beta fits → `calibration`.
6. `verdict()`: HostFailures if any host has errors; else Stragglers if
   any test outcome is Failed or any outliers/violations exist; else Clean.
   Exit codes 2/1/0.

`run_id` = "<started_epoch_secs>-<6 lowercase hex>", the hex being FNV-1a
over the finish timestamp and the host set (deterministic, no rand dep).

## Key formats (contract with the orchestrator)
- `metric_key(test, name)` = "<test_display_name>.<name>", e.g.
  `mem_bandwidth.triad`, `nccl_all_reduce.elapsed_us`.
- `scope_label(scope)`: `Node` → `None`; `Core{3}` → `core3`;
  `Numa{0}` → `numa0`; `Gpu{1}` → `gpu1`; `GpuPair{0,2}` → `gpupair0-2`;
  `Disk{"/tmp"}` → `disk:/tmp`; `HostPair{"n7"}` → `pair:n7`.
- `sample_key(host, scope)` = `host` for `Node`, else `host:<label>`
  (e.g. `n1:gpu2`, `n1:pair:n2`).

## Fleet-comparability rule
A group is a fleet comparison only when every sample key in it is
distinct. A repeated key means the metric is a per-host *series*, not one
reading per subject: the NCCL sweeps emit `msg_bytes`/`elapsed_us` once per
message size, so their spread is the design, not a straggler signal. Such
groups are skipped by MAD analysis (they feed `calibration.links` instead).
Absolute thresholds still apply to them, since a configured bound is an
explicit per-value opt-in.

## Statistics contracts (stats.rs)
- `median`: ignores non-finite; None on empty (after filtering).
- `mad`: scaled by 1.4826; None for <2 finite values.
- `flag_outliers`: nothing when MAD == 0 or fewer than 4 finite samples;
  deviation is signed MADs from median; output order follows input order;
  non-finite samples never become outliers.

## Fit contract (fit.rs)
OLS over all (bytes, us) points, accumulated about the mean for numerical
stability across a sweep spanning several decades; `TooFewPoints` unless
≥2 distinct sizes; `NonFinite` on bad input; r_squared ∈ [0,1] (clamped;
1.0 when every timing is identical).

## Calibration extraction
Per-host `NodeRoofline`:
- `gpu_gflops["gflops_*"]`: min across the host's `gpu_gemm_perf.gflops_*`
  metrics — the slowest GPU defines the node.
- `cpu_gflops_allcore`: min of `cpu_gflops.gflops_allcore`.
- `dram_gib_per_sec`: `mem_bandwidth.triad_allnode`, falling back to the
  max per-NUMA `mem_bandwidth.triad`.
- `gpu_hbm_gib_per_sec`: min of `gpu_mem_bandwidth.d2d`.
- `pcie_h2d_gib_per_sec`: min of `gpu_mem_bandwidth.h2d_pinned`.
- `disk_read/write_gib_per_sec`: max of `disk_io.seq_read` / `.seq_write`
  (distinct mount points are not comparable, so the best is the headline).

`calibration.links`:
- `nccl_allreduce_fleet` / `nccl_allgather_fleet`: within each host's
  metric list the sweep emits `msg_bytes` and `elapsed_us` as parallel
  metrics, one of each per size, so they are joined by emission order and
  fitted with `fit_alpha_beta` over the whole fleet's points.
- `tcp_pairwise`: a two-point *synthesis*, not a regression — the peer
  tests measure latency and streaming bandwidth separately, with no size
  sweep. `alpha_us` = median `net_latency.rtt_p50`, `beta_us_per_byte` =
  1e6 / (median `net_bandwidth.gib_per_sec` × 2^30), `r_squared` = 1.0 by
  construction (not evidence of fit quality). Emitted only when both
  medians exist and the bandwidth is positive.

## Rendering
`render_table` (comfy-table, writer-injected for tests) emits a header line
then sections: per-host summary (pass/fail/skip counts + key rooflines),
outliers (subject, metric, value vs fleet median, MADs), absolute-threshold
violations, inventory consistency dissenters, failed hosts, and the link
alpha-beta digest. Empty sections either state "none" (outliers,
consistency) or are omitted (violations, failures, links). `render_saved`
loads via `history::load` and prints the table, or pretty JSON with
`--json`.

## Persistence (history.rs)
`runs/<run_id>.json`, pretty-printed. `list` returns `*.json` paths sorted
by file name (chronological, since run_id leads with the epoch) and treats
a missing directory as an empty list.

## Files
`src/analysis/{stats,fit,schedule}.rs`, `src/report/{mod,history}.rs`.

## Invariants
- SCHEMA_VERSION bumps on any field rename/removal in `RunResults`.
- The table is a projection of the JSON; no analysis happens at render
  time.
- Outlier grouping never compares across different units, and never
  compares a per-host series against itself.
