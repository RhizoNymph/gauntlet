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
over a timestamp and the host set (deterministic, no rand dep). Two seeds
produce the same shape:
- `build` seeds from the *finish* timestamp and the hosts actually heard
  from (`observations.keys()`, sorted) — the id of a document built in
  isolation, e.g. by tests or an offline re-analysis.
- `make_run_id(started_epoch_secs, hosts)` seeds from the *start* timestamp
  and the configured host addresses in config order. `gauntlet run` computes
  this once at startup and stamps it over `build`'s id, so a run has one
  identity from its first partial snapshot through its final document.

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
`runs/<run_id>.json`, pretty-printed. `list` returns completed `*.json`
paths sorted by file name (chronological, since run_id leads with the
epoch) and treats a missing directory as an empty list. It skips in-flight
snapshots (`*.partial.json`) and anything whose file name starts with `.`
(hidden files and the snapshot staging file).

### Partial snapshots (contract with the viewer)
While a run is in flight the collector republishes the whole results
document every `PARTIAL_SNAPSHOT_INTERVAL` (2s) so a separate process can
tail progress:
- **Filename**: `runs/<run_id>.partial.json` (`history::PARTIAL_SUFFIX` =
  `".partial.json"`). Same `RunResults` schema as a finished run — a
  snapshot is just an early build over the observations so far, with
  `finished_epoch_secs` set to the snapshot time.
- **Atomicity**: `save_partial` writes `runs/.<run_id>.partial.json.tmp`
  then renames it over the destination. A reader therefore sees either the
  previous snapshot or the new one, never a torn document. The staging name
  is hidden and `.tmp`-suffixed, so neither `list` nor `list_live` shows it.
- **Stable id**: the snapshots and the final document share the `run_id`
  from `make_run_id(started, configured hosts)`, computed before the first
  event arrives. A viewer keys on it and follows a run across completion.
- **Cadence**: tick-driven (`tokio::time::interval`,
  `MissedTickBehavior::Delay`) and gated on a dirty flag, so an idle run
  rewrites nothing. A snapshot failure is logged (warn) and dropped; it can
  never abort a run.
- **Cleanup**: after the final `runs/<run_id>.json` is saved,
  `remove_partial(run_id, dir)` deletes the snapshot (absent is Ok), so a
  live listing only ever contains runs that are genuinely in flight. A
  crashed run leaves its last snapshot behind by design.
- **Only the default directory**: `--out <path>` is a one-shot destination,
  not a directory anyone tails, so it disables snapshots entirely.
- `list_live(dir)` returns the `*.partial.json` paths, sorted, with a
  missing directory yielding an empty list — the viewer's discovery call.

## Files
`src/analysis/{stats,fit,schedule}.rs`, `src/report/{mod,history}.rs`,
`src/orchestrator/mod.rs` (`PartialWriter`, snapshot cadence),
`src/orchestrator/collect.rs` (`Collector::snapshot`).

## Invariants
- SCHEMA_VERSION bumps on any field rename/removal in `RunResults`.
- A run's `run_id` never changes once the run has started: the partial
  snapshots and the final document are the same file stem.
- `list` and `list_live` partition the visible documents in a run
  directory; a `<run_id>` appearing in both is a transient overlap only if
  cleanup failed.
- Partial snapshots are advisory. Nothing in the run path reads them, and
  no snapshot error is ever fatal.
- The table is a projection of the JSON; no analysis happens at render
  time.
- Outlier grouping never compares across different units, and never
  compares a per-host series against itself.

## Repeats and distribution moments (schema v2)

`gauntlet run --repeat N` executes the measurement phases N times over the
held sessions (inventory once); the orchestrator stamps each metric record
with its repeat index (`MetricRecord.repeat`, serde-defaulted so old wire
output still decodes). `report::build` reduces raw records into
`RunResults.aggregates`: group -> subject -> `{unit, moments}` with
`Moments {n, median, mad, min, max, mean, stddev}` (robust median/MAD as
the headline; mean/stddev for Gaussian consumers). A group where any
(subject, repeat) pair occurs twice is a per-host sweep series and is
excluded — series feed `calibration.links` exactly as before.

Downstream effects: fleet outliers are flagged on per-subject *medians*
(centered run-to-run noise can no longer masquerade as slowness);
`fleet.jitter_outliers` flags subjects whose spread is a high-side fleet
outlier (informational, not part of the verdict); absolute thresholds
check the median (sweep series keep per-value semantics); rooflines reduce
over per-subject medians. Everything degrades gracefully at n = 1.

## Overlap retention (schema v3)

`build` starts by appending derived `overlap_retention` records to each
host's metric list (`derive_overlap_retention`): `gemm_<dtype>` per GPU
(overlapped `overlap_gemm.gflops_<dtype>` over the phase-2
`gpu_gemm_perf.gflops_<dtype>` of the same GPU and repeat) and
`all_reduce` per node (overlapped over isolated
`overlap_all_reduce.*_bus_gib_per_sec` of the same repeat). Ratios form
only over finite, positive baselines; derivation is idempotent (skipped if
retention records already exist, e.g. a rebuilt document). Because the
records land before grouping, they flow through aggregates, MAD outliers,
jitter, and absolute thresholds like measured metrics. The hosts table
adds an `overlap ret (min)` column: the worst per-subject median retention
on that host. See docs/features/overlap_phase.md.
