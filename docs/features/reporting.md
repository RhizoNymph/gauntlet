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
   the dlopen probe results on GPU-bearing hosts. Each host's
   `NcclNicSummary::consistency_fields` (nccl_ib_ports /
   nccl_ib_link_layer / nccl_ib_ceiling_gbps) join the vote.
5. Rooflines per host (min across the host's GPUs for GPU metrics; the
   straggler defines the node), `links` alpha-beta fits and per-host
   `nccl_nics` summaries → `calibration`.
6. `verdict()`: HostFailures if any host has errors; else Stragglers if
   any test outcome is Failed (including `gpu_idle` and `nccl_nics`) or any
   outliers/violations exist; else Clean.
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
  `mem_bandwidth.triad`, `nccl_all_reduce.elapsed_us`,
  `nccl_intra_all_reduce.bus_gib_per_sec_peak_8gpu`. NCCL sweep metric names
  are the `proto::nccl_metric` consts, shared by emitters and extraction.
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
series groups are skipped by MAD analysis (they feed `calibration.links`
instead). Absolute thresholds still apply to them, since a configured
bound is an explicit per-value opt-in.

The intra-node sweep's `bus_gib_per_sec_peak_<n>gpu` and `ranks` are one
value per node, so they *are* fleet comparisons — the peak is the
intra-node straggler headline. Its name carries the GPU count, so nodes
are only compared within their own topology (the same keying as the
`_<n>gpu` link classes); a topology with fewer than 4 nodes never flags.

## Statistics contracts (stats.rs)
- `median`: ignores non-finite; None on empty (after filtering).
- `mad`: scaled by 1.4826; None for <2 finite values.
- `flag_outliers`: nothing when MAD == 0 or fewer than 4 finite samples;
  deviation is signed MADs from median; output order follows input order;
  non-finite samples never become outliers.

## Fit contract (fit.rs)
Least squares over all (bytes, us) points, constrained to alpha ≥ 0 and
beta ≥ 0 (schema v12), because a negative latency or inverse bandwidth
would have a simulator predict negative time. Plain OLS on a real
rank-per-GPU all-reduce sweep gave alpha = -46.59 us: small messages are
latency-bound, large ones bandwidth-bound, and the large sizes dominate the
line. Closed form, exact for two parameters:
1. OLS, accumulated about the mean for numerical stability across a sweep
   spanning several decades. If alpha ≥ 0 and beta ≥ 0, that is the fit
   (`bound: None`).
2. Otherwise the optimum lies on the boundary of the feasible quadrant (RSS
   is strictly convex, so an interior optimum would be the infeasible OLS
   point). Both boundary rays are evaluated: through the origin
   (alpha = 0, beta = max(0, Σxy/Σx²), `AlphaZero`) and horizontal
   (beta = 0, alpha = max(0, mean y), `BetaZero`). A ray whose parameter
   also clamps collapses to the origin (`BothZero`, only possible with
   non-positive timings). The ray with the lower RSS wins. The proof is
   the doc comment on `fit_alpha_beta`.
3. r_squared is recomputed for the chosen parameters (it can only fall
   relative to OLS) and clamped to [0,1]; 1.0 when every timing is
   identical.

`TooFewPoints` unless ≥2 distinct sizes; `NonFinite` on bad input; both
are checked before fitting, exactly as before the constraint.

`AlphaBetaFit.bound: Option<FitBound>` (`alpha_zero` / `beta_zero` /
`both_zero` in JSON, `null` for a clean fit) records which constraint was
active. It is serde-defaulted, so pre-v12 documents load with it `None`,
which on those documents means "not recorded", not "clean". Invariant:
`AlphaZero` ⇒ alpha_us == 0, `BetaZero` ⇒ beta_us_per_byte == 0 (so
`bandwidth_gib_per_sec` is infinite), `BothZero` ⇒ both.

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
- `nccl_allreduce_rank_per_gpu` / `nccl_allgather_rank_per_gpu` (named
  `nccl_*_fleet` before schema v8; renamed because the world changed
  meaning, see below): within each host's
  metric list the sweep emits `msg_bytes` and `elapsed_us` as parallel
  metrics, one of each per size, so they are joined by emission order and
  fitted with `fit_alpha_beta` over the whole fleet's points. These are
  the fits most likely to bind: an all-reduce sweep over sizes from 1 KiB
  to 256 MiB is not linear, and OLS through its large sizes undershoots
  zero at the origin. Such a fit comes out with alpha = 0 and
  `bound: alpha_zero`; a simulator that needs the small-message latency
  should treat that alpha as unknown rather than zero.
- `nccl_allreduce_intranode_<n>gpu` / `nccl_allgather_intranode_<n>gpu`
  (schema v9, `report/intranode.rs`): the intra-node sweep's
  `nccl_intra_all_*` series, joined by emission order within each (host,
  repeat) and bucketed by that run's `ranks` value (local GPU count), one
  fit per bucket. Keyed by GPU count because different counts are
  different links; every node's data is kept. Runs without a valid
  `ranks` record (sweep failed part-way) contribute nothing.
- `tcp_pairwise`: a two-point *synthesis*, not a regression — the peer
  tests measure latency and streaming bandwidth separately, with no size
  sweep. `alpha_us` = median `net_latency.rtt_p50`, `beta_us_per_byte` =
  1e6 / (median `net_bandwidth.gib_per_sec` × 2^30), `r_squared` = 1.0 by
  construction (not evidence of fit quality). Emitted only when both
  medians exist and the bandwidth is positive. `bound` is always `None`:
  both parameters are measured, never solved for.

## Rendering
`render_table` (comfy-table, writer-injected for tests) emits a header line
then sections: per-host summary (pass/fail/skip counts + key rooflines),
silent data corruption, gpus in use (gpu_idle), nccl nics (only when
some host has an IB/RoCE port), outliers (subject, metric, value vs fleet median, MADs), absolute-threshold
violations, inventory consistency dissenters, failed hosts, and the link
alpha-beta digest (whose `bound` column shows the active constraint's
label, e.g. `alpha=0`, or `-` for a clean fit). Empty sections either state "none" (outliers,
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
`src/analysis/{stats,fit,schedule}.rs`,
`src/report/{mod,history,intranode,nccl_env,nccl_nics,gpu_idle}.rs`,
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
- Every `calibration.links` fit has alpha_us ≥ 0 and beta_us_per_byte ≥ 0
  (schema v12+); `bound` is `Some` exactly when a constraint changed the
  result away from OLS.
- Outlier grouping never compares across different units, and never
  compares a per-host series against itself.

## Error-counter findings (schema v4)

`hosts.*.counter_deltas` carries the full per-node error-counter delta
list across the load phases (zeros and resets included);
`fleet.counter_findings` keeps only counters with `after > before`, keyed
by host. Any finding makes the verdict at least Stragglers. `render_table`
adds an "error-counter deltas (across load phases)" section (host, domain,
device, counter, before, after, +increment), omitted entirely when there
are no findings. See docs/features/counter_deltas.md for collection and
scheduling.

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

## GPUs in use (schema v11)

Inventory GPUs carry `occupancy` (memory used/total and the compute
processes other than the reporting agent; stale gauntlet agents marked),
and each host gains one `gpu_idle` outcome per GPU, derived
orchestrator-side from the snapshot against
`thresholds.gpu_idle_max_used_mib` (docs/features/phase0_inventory.md).
A Failed `gpu_idle` feeds the verdict like any failed test (Stragglers).
Both additions are serde-defaulted: a v10 document decodes with occupancy
unknown and simply has no `gpu_idle` outcomes.

The host table only counts failed outcomes, so `report/gpu_idle.rs` adds a
"gpus in use (gpu_idle)" section right after the SDC section: one row per
Failed `gpu_idle` outcome (`gpus_in_use(&RunResults) -> Vec<GpuInUse>`,
host then GPU order) with `host:gpuN`, "used / total MiB", and one line
per compute process ("VLLM::EngineCore (pid 2102873, 23232 MiB)", "stale
gauntlet agent ..."). It is a projection: failed GPUs come from
`hosts.*.outcomes`, the detail from the same host's inventory; with no
process to name (memory over the threshold only, or an old document) the
outcome reason stands in. Omitted when no GPU is busy.

## NCCL NICs and the NIC ceiling (schema v13)
`calibration.nccl_nics: {host: NcclNicSummary}` (serde-defaulted; absent
in pre-v13 documents) — for every host with an inventory,
`nccl_ib::summarize(inventory, NcclIbConfig::from_env(run env))`:
`selected` ports (device, port, link_layer, rate_gbps, payload_gbps,
netdevs, numa_node, gpu_locality), `excluded` ports with a tagged
`reason` (`ib_disabled` / `not_active` {state} / `unsupported_link_layer`
/ `filtered_by_hca`), and `ceiling_gib_per_sec` (summed payload line
rate; null when a selected rate is unknown, 0 when nothing is selected).
Details and NCCL_IB_HCA semantics: docs/features/nccl_nics.md.

Fed into the analysis three ways:
- Outcome: hosts carry one Node-scope `nccl_nics` outcome (derived by the
  orchestrator as the inventory arrives). Failed — IB/RoCE ports exist but
  NCCL_IB_HCA selects no active one, so NCCL would fall back to sockets —
  makes the verdict Stragglers. No ports at all, or NCCL_IB_DISABLE, is
  Skipped.
- Metric: `nccl_nics.ceiling_gib_per_sec` (node scope, GibPerSec) on
  hosts with IB/RoCE ports, transport enabled, ceiling known; normal
  aggregate/MAD/threshold handling.
- Consistency: `consistency_findings` chains each host's
  `NcclNicSummary::consistency_fields` (`nccl_ib_ports`,
  `nccl_ib_link_layer`, `nccl_ib_ceiling_gbps`) after the inventory
  fields, so one host with fewer/slower selected NICs dissents even when
  the ceiling's MAD is zero.

`report/nccl_nics.rs` renders "nccl nics (NCCL_IB_HCA=… | unset)": per
host the selected ports ("mlx5_0:1 infiniband 200G (ibp28s0, gpu0
pcie_switch)" — netdevs and the nearest GPUs), selection link layer,
ceiling GiB/s, excluded ports with reasons, and the outcome note (FAIL /
skip reason). Omitted when no host has an IB/RoCE port.

## Non-negative link fits (schema v12)

`fit_alpha_beta` became a bounded least-squares fit (alpha ≥ 0, beta ≥ 0;
see Fit contract) and `calibration.links.*.bound` records which constraint
was active. Numbers change only for fits OLS would have made non-physical;
every other fit is bit-for-bit the OLS result it was before. The viewer's
link card shows the bound label next to the link class.

## NCCL env (schema v10)

`RunResults.nccl_env` records, once per run, the resolved NCCL
environment every communicator was created under (`[nccl] env` plus
`socket_ifname` as NCCL_SOCKET_IFNAME; see docs/features/nccl_env.md).
It is run-level because it is fleet-uniform by construction: the
orchestrator sends one map to every host. It is an
`Option<BTreeMap<String, String>>` with a serde default. `None` means not
recorded, which is how pre-v10 documents decode, so old history keeps
loading. `Some(empty)` means an untuned run. The terminal table prints
`nccl env: K=V ...`, `(none)` or `(not recorded)` under its header line.
`report::nccl_env::nccl_env_drift` lists the keys added, removed or
changed between two runs, and returns `None` if either run did not record
its env. The viewer's diff mode uses it.

## Barrier stragglers (schema v6)

The barrier-skew microbenchmark (docs/features/barrier_skew.md) emits
per-host `nccl_barrier.*` / `tcp_barrier.*` metrics that flow through the
ordinary MAD machinery, plus a dedicated rule:
`report::barrier_straggler_flags` populates `fleet.barrier_stragglers`
(group -> flagged hosts) when a host's median `slowest_frac` exceeds
`thresholds.barrier_slowest_frac` and its median `slowest_considered` is
at least `analysis::skew::MIN_TALLY_ITERS`. Barrier straggler flags count
toward the `Stragglers` verdict and render as the "barrier stragglers"
table section. The field is serde-defaulted, so pre-v6 documents load
with it empty.

## Overlap retention (schema v5)

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

Schema v7 adds the fleet overlap step's groups (`overlap_fleet_gemm`,
`overlap_fleet_all_reduce`) and the corresponding derived names:
`overlap_retention.fleet_gemm_<dtype>` (per GPU, same phase-2 baseline)
and `overlap_retention.fleet_all_reduce` (per node, over the fleet step's
own isolated window — the two steps' communicators are not comparable, so
their baselines never cross). No field changed shape; pre-v7 documents
decode unchanged.

## Rank-per-GPU granularity (schema v8)

The fleet NCCL world became one rank per GPU (proto v7; see
docs/features/phase3_network.md), which changes the granularity of three
metric families and the *meaning* of others, without changing any
field's shape:
- `nccl_barrier.{p50_us,p90_us,p99_us,max_us}`: `host` → `host:gpuN`
  subjects. The tally (`slowest_frac`, `slowest_considered`) stays one
  node-scope value per host — a host's ranks share one arrival group, so
  per-GPU copies would weight hosts by GPU count — and so do
  `fleet.barrier_stragglers.nccl_barrier` keys (one row per late host;
  docs/features/barrier_skew.md). `fleet_span_*` stays one node-scope
  series on the lead host. `tcp_barrier.*` is unchanged.
- `overlap_fleet_all_reduce.*`: per GPU (each GPU is a rank with its own
  isolated and overlapped windows).
- `overlap_retention.fleet_all_reduce`: per GPU, each dividing that GPU's
  own isolated window. `derive_overlap_retention` joins bus baselines on
  (step, repeat, scope label), so node-scope inputs (the node-local
  step, older documents) still derive node-scope ratios.
- Meaning change: the phase-3 sweep series (`nccl_all_reduce.*`,
  `nccl_all_gather.*`) keep their names and node scope on the lead host
  (timed on global rank 0), but now measure a world of n = total GPUs
  whose ring mixes NVLink with the fabric; they are not comparable with
  v7 numbers. The fits were therefore renamed
  `nccl_{allreduce,allgather}_fleet` →
  `nccl_{allreduce,allgather}_rank_per_gpu`, so a v7 baseline cannot line
  up with them under one key, and the viewer's diff mode marks any
  baseline with a different `schema_version` as not directly comparable
  (docs/features/viewer.md).
- `hosts.*.inventory` gains `cuda_visible_gpus`, and a mismatch with the
  nvidia-smi GPU count is a Failed `inventory` outcome on that host
  (verdict at least Stragglers).
- Fleet NCCL failures are attributed (docs/features/phase3_network.md):
  `fleet.failed_hosts` lists only the host that caused a sweep failure;
  hosts it aborted are warnings, not host failures.
