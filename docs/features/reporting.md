# Reporting & Analysis

## Scope
Fleet statistics (median/MAD outliers), absolute thresholds, consistency
findings, calibration extraction, JSON persistence, terminal rendering,
exit codes. Non-scope: collecting data (orchestrator), emitting it (agent).

## Data flow
`Collector::into_observations()` → `report::build(config, observations,
started, finished)`:
1. Group metrics by "<test>.<metric>" within comparable scope kinds (Core
   metrics compare across all cores fleet-wide; Gpu metrics across all
   GPUs; Node metrics across nodes; HostPair across pairs). Sample key =
   "host" or "host:scope" rendered stably.
2. `stats::flag_outliers(samples, thresholds.mad_k)` per group →
   `fleet.outliers`.
3. Absolute `thresholds.absolute` bounds → `fleet.threshold_violations`.
4. `proto::consistency_fields` majority vote → `fleet.consistency`
   (fields where ≥1 host dissents; majority = most common value).
5. Rooflines per host (min across the host's GPUs for GPU metrics; the
   straggler defines the node) and `links` alpha-beta fits → `calibration`.
6. `verdict()`: HostFailures if any host has errors; else Stragglers if
   any outliers/violations; else Clean. Exit codes 2/1/0.

## Statistics contracts (stats.rs)
- `median`: ignores non-finite; None on empty (after filtering).
- `mad`: scaled by 1.4826; None for <2 finite values.
- `flag_outliers`: nothing when MAD == 0 or sample count < 4; deviation is
  signed MADs from median.

## Fit contract (fit.rs)
OLS over all (bytes, us) points; `TooFewPoints` unless ≥2 distinct sizes;
`NonFinite` on bad input; r_squared ∈ [0,1] (clamp tiny negatives).

## Rendering
`render_table`: sections — per-host summary (pass/fail counts, key
rooflines), outliers (host, metric, value vs fleet median, MADs),
consistency dissenters, failed hosts, calibration digest. comfy-table,
writer-injected for tests. `render_saved` loads via `history::load`.

## Persistence (history.rs)
`runs/<run_id>.json`, pretty-printed; `run_id` =
"<started_epoch_secs>-<6 hex>". `list` sorts by filename (chronological).

## Files
`src/analysis/{stats,fit}.rs`, `src/report/{mod,history}.rs`.

## Invariants
- SCHEMA_VERSION bumps on any field rename/removal in `RunResults`.
- The table is a projection of the JSON; no analysis happens at render
  time.
- Outlier grouping never compares across different units.
