# Barrier Skew — Straggler Microbenchmark

## Scope
Many iterations (default 2000, `tests.barrier_iters`) of a tiny collective,
with per-rank timing, to identify which hosts consistently arrive late.
This is the most direct measurement of straggler-ness as training
experiences it: OS jitter, clock throttling, and network variance
integrate into one per-rank number. Two probes share one analysis:

- **NCCL barrier** (`nccl_barrier`): a 4–8 byte all-reduce over the same
  fleet-wide communicator the phase-3 sweep already sets up. GPU/NCCL
  fleets only.
- **TCP barrier** (`tcp_barrier`): a star barrier over plain TCP, run
  across the *whole* fleet regardless of GPUs — the CPU-only fallback, and
  a fabric-independent complement on GPU fleets.

Non-scope: sweep-style bandwidth measurement (phase-3 sweeps), IB-verbs
barriers, per-GPU (intra-node) skew — the NCCL barrier runs one rank per
node, like the sweep.

## Measurement design and reasoning

The problem with timing a collective is attribution: with NCCL + host
timestamps there is no per-rank "arrival timestamp" visible anywhere, and
cross-rank wall-clock comparison was rejected because phase 0's
`clock_offset_ms` is chrony/NTP-grade (milliseconds) — three orders of
magnitude coarser than barrier iterations (tens of µs). The two probes
therefore extract the per-rank signal without any cross-host clock:

**NCCL: the wait-time inversion.** Each rank times its own
enqueue-to-stream-sync elapsed for every iteration, with the stream
synchronized per iteration. A tiny all-reduce completes at (nearly) the
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

**Per-rank outputs** (both probes): p50/p90/p99/max of the per-iteration
value, plus the slowest-rank tally. An iteration feeds the tally only
when the late rank's deviation from the iteration median exceeds
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
   world has ≥ 2 ranks) to the `Lead` and `Participate` directives. After
   the sweep, every rank (participants included — the one time they speak)
   runs warmup + `iters` timed tiny all-reduces and emits
   `AgentEvent::NcclBarrierTimings { rank, elapsed_us }`. The per-host
   driver callbacks intercept these into a shared
   `Arc<Mutex<Vec<RankSeries>>>` (they never reach the collector; a stray
   one is debug-ignored, like `NcclId`). Once all rank tasks join, the
   orchestrator runs `analysis::skew::analyze(LateIsMin)` and emits
   metric records.
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
4. **Metrics** (per rank, attributed to that rank's host, `Scope::Node`,
   under `nccl_barrier.*` / `tcp_barrier.*`): `p50_us`, `p90_us`,
   `p99_us`, `max_us` (Micros), `slowest_frac` (Ratio),
   `slowest_considered` (Count). Fleet-level, attributed to the lead /
   coordinator host like the sweep metrics: `fleet_span_p50_us`,
   `fleet_span_p90_us`, `fleet_span_p99_us`, `fleet_span_max_us`.
5. **Analysis**: the per-rank metrics are one-value-per-host groups, so
   they flow through the ordinary MAD outlier machinery (under
   `nccl_barrier` a straggler is a *low*-side `p50_us` outlier; under
   `tcp_barrier` a *high*-side one — the signed `deviation_mads` carries
   the direction). The tally additionally gets its own flagging rule:
   `report::barrier_straggler_flags` flags a host into
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

## Files
- `src/analysis/skew.rs` — polarity-aware skew statistics: `analyze`,
  `RankSeries`, `RankSkew`, `FleetBarrier`, `BarrierSkew`, `Margin`,
  `MIN_TALLY_ITERS`. Pure; unit-tested with synthetic iteration data.
- `src/agent/barrier.rs` — TCP star barrier: `run_server` (accept, release
  loop via per-rank tasks + watch channel, analysis), `join`,
  `TcpBarrierReport`. Localhost-tested.
- `src/agent/nccl.rs` — barrier loop after the sweep (gpu feature only);
  participants emit Hello + `NcclBarrierTimings`.
- `src/proto.rs` — `BarrierSpec`, `NcclBarrierTimings`, new `TestId`s,
  `PROTO_VERSION` 5.
- `src/orchestrator/mod.rs` — `emit_barrier_metrics`, `tcp_barrier_sweep`.
- `src/orchestrator/nccl.rs` — barrier spec on the sweep workload, timing
  interception in the fleet NCCL driver.
- `src/orchestrator/collect.rs` — stray `NcclBarrierTimings` ignored.
- `src/report/mod.rs` — `barrier_stragglers` field + flagging rule +
  verdict + table section, `SCHEMA_VERSION` 6.
- `src/config.rs` — `tests.barrier_iters` (2000), `tests.barrier_bytes`
  (8), `thresholds.barrier_slowest_frac` (0.5, validated in (0, 1]).
- `src/cli.rs` — `agent barrier serve|join`.

## Invariants
- The raw per-iteration vectors never enter `RunResults`; only derived
  per-rank summaries and tallies do.
- `slowest_frac` sums (over ranks) to ≤ 1.0 of `considered_iters`
  exactly: each considered iteration blames exactly one rank, ties and
  sub-margin iterations blame none.
- Percentiles are nearest-rank over sorted samples, so p50 ≤ p90 ≤ p99 ≤
  max holds by construction (same convention as the peer latency probe).
- Skew analysis needs ≥ 2 distinct ranks and ≥ 1 fully-finite aligned
  iteration; anything less yields `None` / a structured error, never a
  fabricated flag.
- No straggler flag without both `slowest_frac > threshold` and
  `considered ≥ MIN_TALLY_ITERS`.
- The NCCL barrier only runs where the existing sweep gates already
  passed (GPU-bearing hosts with loadable libnccl, world ≥ 2); on this
  path everything sits behind the `gpu` feature. The TCP barrier has no
  such gate and is the CPU-only fleet's source of barrier data.
- `barrier_iters = 0` disables both probes entirely.
