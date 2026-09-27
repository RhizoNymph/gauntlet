# Barrier Skew — Straggler Microbenchmark

## Scope
Many iterations (default 2000, `tests.barrier_iters`) of a tiny collective,
with per-rank timing, to identify which hosts consistently arrive late.
This is the most direct measurement of straggler-ness as training
experiences it: OS jitter, clock throttling, and network variance
integrate into one per-rank number. Two probes share one analysis:

- **NCCL barrier** (`nccl_barrier`): a 4–8 byte all-reduce over the same
  fleet-wide communicator the phase-3 sweep already sets up — one rank
  per GPU (proto v7), so per-rank results are per GPU. GPU/NCCL fleets
  only.
- **TCP barrier** (`tcp_barrier`): a star barrier over plain TCP, run
  across the *whole* fleet regardless of GPUs — the CPU-only fallback, and
  a fabric-independent complement on GPU fleets.

Non-scope: sweep-style bandwidth measurement (phase-3 sweeps), IB-verbs
barriers, and distinguishing *arrival* order among the GPUs of one host —
one process drives a host's whole rank block from one thread, so its GPUs
share a launch instant (see "Arrival groups" below).

## Measurement design and reasoning

The problem with timing a collective is attribution: with NCCL + host
timestamps there is no per-rank "arrival timestamp" visible anywhere, and
cross-rank wall-clock comparison was rejected because phase 0's
`clock_offset_ms` is chrony/NTP-grade (milliseconds) — three orders of
magnitude coarser than barrier iterations (tens of µs). The two probes
therefore extract the per-rank signal without any cross-host clock:

**NCCL: the wait-time inversion.** Each rank times its own
launch-to-completion elapsed for every iteration, with every local stream
drained per iteration. The host's process launches the grouped all-reduce
on all its local ranks at one instant and stamps each rank when its
completion event is first observed done (round-robin polling,
`agent/nccl/completion.rs` — sequential per-stream syncs would stamp GPU 0
first on every iteration and bias the per-GPU distributions by index). A tiny all-reduce completes at (nearly) the
same instant on every rank, but each rank's timer *starts* when that rank
arrives. An early arriver waits out the stragglers and records a long
elapsed; the straggler arrives last and completes almost immediately, so
it records the *shortest* elapsed. Per iteration, `argmin(elapsed)` is the
late rank (`SkewPolarity::LateIsMin`). The per-iteration sync aligns
iteration boundaries across ranks (everyone leaves iteration k together),
so a rank that is slow to launch iteration k+1 — scheduler jitter, clock
throttling, slow interconnect path into it — is exactly the one that shows
up at the minimum. This is stronger than any single-rank statistic: the
discriminating signal is the *cross-rank comparison within an iteration*,
which cancels the (shared) collective cost.

**TCP: single-observer arrival times.** The coordinator releases all
ranks at once (one task per rank; a watch channel fires them together)
and measures each rank's release-to-response time on its own clock —
`SkewPolarity::LateIsMax`. Each rank's round is timed by its own socket
task from its own write, so sequential-send bias cannot enter. The number
is server→client latency + client scheduling delay + client→server
latency: path variance and host jitter both included, deliberately, since
a training-step barrier experiences both. A systematically longer path
raises a rank's *baseline*; the margin rule (below) keeps a baseline
offset from being tallied unless the gap is nontrivial, and the per-rank
distribution shape (p50 vs p99) separates "far away" from "jittery".

**Arrival groups (NCCL, proto v7).** Ranks launched by one thread share
their arrival: which sibling GPU shows the minimum in an iteration is
completion noise, not lateness. Tallying per rank would split a late
host's blame across its GPUs (8 GPUs → ~1/8 each) and keep every one of
them under the straggler threshold. `analysis::skew::analyze_grouped`
therefore takes a rank → group map (the host index, from
`RankLayout::arrival_group`): per iteration each group's value is its
members' extreme under the polarity, the late arriver is the unique
extreme *group*, the margin is measured against the median of the group
values, and every rank of the late group is tallied. With singleton
groups this is exactly the per-rank analysis (`analyze`, which the TCP
barrier keeps using — one process per rank there). Consequently a late
host flags *all* its GPUs (`host:gpu0..gpuN`); per-GPU localization of a
bad path comes from the per-GPU p50/p99 distributions, the sweep and the
fleet overlap retention, not from the tally.

**Per-rank outputs** (both probes): p50/p90/p99/max of the per-iteration
value, plus the slowest-rank tally. An iteration feeds the tally only
when the late arrival group's deviation from the iteration median exceeds
`max(0.25 × |median|, 10µs)` (`analysis::skew::Margin`); tied extremes
blame nobody. `slowest_frac` = tallied iterations / margin-passing
("considered") iterations.

**Fleet output**: distribution (p50/p90/p99/max) of the per-iteration
barrier span — the max across ranks, i.e. what every rank but the
straggler experiences.

## Data / control flow

1. `orchestrator::network_phase` → `nccl_sweep` (existing) → new
   `tcp_barrier_sweep`, both after `pairwise_sweep`.
2. **NCCL path**: `nccl_sweep` attaches `BarrierSpec { iters, bytes }`
   (from `tests.barrier_iters` / `tests.barrier_bytes`, only when the NCCL
   world spans ≥ 2 *hosts* — one host's ranks share one arrival, so a
   single-host world has no skew to measure) to the `Lead` and
   `Participate` directives. After the sweep, every host's process runs
   warmup + `iters` timed tiny grouped all-reduces over its local ranks
   and emits one `AgentEvent::NcclBarrierTimings { rank, elapsed_us }` per
   local rank (participants included — the one time they speak). The
   per-host driver callbacks intercept these into a shared
   `Arc<Mutex<Vec<RankSeries>>>` (they never reach the collector; a stray
   one is debug-ignored, like `NcclId`). Once all host tasks join, the
   orchestrator runs `analysis::skew::analyze_grouped(LateIsMin)` with
   the layout's arrival groups and emits metric records, mapping each
   rank to (host, local GPU) through `RankLayout::locate`.
3. **TCP path**: `tcp_barrier_sweep` runs whenever `barrier_iters > 0`
   and ≥ 2 hosts are usable. Host 0 runs `agent barrier serve --port
   <net_port_base> --world n --iters k` (via `run_agent_capture` in a
   spawned task; the pairwise sweep is over, so the port base is free);
   every host — including host 0, as a separate process — runs `agent
   barrier join <endpoint:port> --rank <fleet index> --iters k` targeting
   the coordinator's `data_addr` when set. The serving agent computes the
   skew itself (it is the only process that sees every rank on one clock,
   and raw vectors would be megabytes at fleet scale) and prints one
   `TcpBarrierReport` JSON line, which the orchestrator parses and turns
   into the same metric records.
4. **Metrics** (per rank, under `nccl_barrier.*` / `tcp_barrier.*`):
   `p50_us`, `p90_us`, `p99_us`, `max_us` (Micros), `slowest_frac`
   (Ratio), `slowest_considered` (Count). NCCL ranks are attributed to
   their host under `Scope::Gpu { index: local }` (sample key
   `host:gpuN`, schema v8); TCP ranks to their host under `Scope::Node`.
   Attribution is `orchestrator/barrier.rs::barrier_records`, pure over
   a rank → `RankSubject { host, scope }` locator. Fleet-level,
   attributed to the lead / coordinator host under `Scope::Node` like
   the sweep metrics: `fleet_span_p50_us`, `fleet_span_p90_us`,
   `fleet_span_p99_us`, `fleet_span_max_us`.
5. **Analysis**: the per-rank metrics are one-value-per-host groups, so
   they flow through the ordinary MAD outlier machinery (under
   `nccl_barrier` a straggler is a *low*-side `p50_us` outlier, compared
   across every GPU of the fleet; under `tcp_barrier` a *high*-side one —
   the signed `deviation_mads` carries the direction). The tally
   additionally gets its own flagging rule:
   `report::barrier_straggler_flags` flags a subject (`host:gpuN` for
   NCCL, `host` for TCP) into
   `fleet.barrier_stragglers` when its median `slowest_frac` exceeds
   `thresholds.barrier_slowest_frac` (default 0.5) *and* at least
   `skew::MIN_TALLY_ITERS` (100) iterations cleared the margin. Flags
   count toward the `Stragglers` verdict and render as their own table
   section.

## Wire / protocol changes (proto v5, schema v6)

- `PROTO_VERSION` 4 → 5: `NcclDirective::{Lead,Participate}` gain
  `barrier: Option<BarrierSpec>` (serde-defaulted, so old documents still
  decode), and `AgentEvent::NcclBarrierTimings` is new. Participants now
  emit `Hello` before their timings.
- `TestId::{NcclBarrier,TcpBarrier}` ("nccl_barrier" / "tcp_barrier").
- `SCHEMA_VERSION` 5 → 6: `RunResults.fleet.barrier_stragglers`
  (serde-defaulted; pre-v3 documents load with it empty).
- TCP barrier wire format: client hello `[0xB7][rank: u32 be]`, then per
  iteration one release byte (0x52) answered by one ack byte (0x41);
  TCP_NODELAY, single-byte frames.

## Rank-per-GPU change (proto v7, schema v8)

- The NCCL world is one rank per GPU (docs/features/phase3_network.md,
  "Fleet NCCL world"); the barrier probe therefore reports one series per
  GPU and its per-rank metrics move from `host` to `host:gpuN` keys.
- Straggler flags stay per subject — `host:gpuN` — and a late host flags
  all of its GPUs (arrival groups, above).
- `fleet_span_*` stays one node-scope series on the lead host.
- The TCP barrier is unchanged (one rank per host, `analyze`).

## Files
- `src/analysis/skew.rs` — polarity-aware skew statistics: `analyze`
  (independent ranks), `analyze_grouped` (arrival groups),
  `RankSeries`, `RankSkew`, `FleetBarrier`, `BarrierSkew`, `Margin`,
  `MIN_TALLY_ITERS`. Pure; unit-tested with synthetic iteration data.
- `src/orchestrator/barrier.rs` — `RankSubject`, `barrier_records`
  (pure attribution), `emit_barrier_metrics`.
- `src/agent/barrier.rs` — TCP star barrier: `run_server` (accept, release
  loop via per-rank tasks + watch channel, analysis), `join`,
  `TcpBarrierReport`. Localhost-tested.
- `src/agent/nccl/sweep.rs` — barrier loop after the sweep over all
  local ranks (gpu feature only); every host emits Hello + one
  `NcclBarrierTimings` per local rank.
- `src/agent/nccl/completion.rs` — unbiased per-rank completion stamps.
- `src/proto/mod.rs` — `BarrierSpec`, `NcclBarrierTimings`, new `TestId`s,
  `PROTO_VERSION` 5.
- `src/orchestrator/mod.rs` — `tcp_barrier_sweep`.
- `src/orchestrator/nccl/mod.rs` — barrier spec on the sweep workload,
  timing interception in the fleet NCCL driver, grouped analysis +
  per-GPU attribution via the rank layout.
- `src/orchestrator/collect.rs` — stray `NcclBarrierTimings` ignored.
- `src/report/mod.rs` — `barrier_stragglers` field + flagging rule +
  verdict + table section, `SCHEMA_VERSION` 6.
- `src/config.rs` — `tests.barrier_iters` (2000), `tests.barrier_bytes`
  (8), `thresholds.barrier_slowest_frac` (0.5, validated in (0, 1]).
- `src/cli.rs` — `agent barrier serve|join`.

## Invariants
- The raw per-iteration vectors never enter `RunResults`; only derived
  per-rank summaries and tallies do.
- `slowest_frac` sums (over arrival groups — over ranks for the TCP
  barrier) to ≤ 1.0 exactly: each considered iteration blames exactly one
  group, ties and sub-margin iterations blame none. Every rank of a group
  carries the group's value.
- Percentiles are nearest-rank over sorted samples, so p50 ≤ p90 ≤ p99 ≤
  max holds by construction (same convention as the peer latency probe).
- Skew analysis needs ≥ 2 distinct ranks, ≥ 2 arrival groups, a group
  for every rank, and ≥ 1 fully-finite aligned iteration; anything less
  yields `None` / a structured error, never a fabricated flag.
- No straggler flag without both `slowest_frac > threshold` and
  `considered ≥ MIN_TALLY_ITERS`.
- The NCCL barrier only runs where the existing sweep gates already
  passed (GPU-bearing hosts with loadable libnccl, ≥ 2 hosts); on this
  path everything sits behind the `gpu` feature. The TCP barrier has no
  such gate and is the CPU-only fleet's source of barrier data.
- `barrier_iters = 0` disables both probes entirely.
