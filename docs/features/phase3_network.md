# Phase 3 — Network

## Scope
The network hierarchy, innermost level first: the intra-node NCCL sweep
(every local GPU of one node, NVLink/PCIe), pairwise TCP latency/bandwidth
between nodes (tournament-scheduled), then the fleet-wide NCCL sweep.
Non-scope: intra-node pairwise p2p copies (phase 2, `gpu_p2p`), IB-verbs
microbenchmarks (post-v1; NCCL measures what training experiences),
pair-level NCCL sweeps (full fleet + TCP pairwise usually localizes).

## Control flow
Orchestrator phase-3 driver (`network_phase` in `orchestrator/mod.rs`):
0. Intra-node NCCL sweep (`tests.nccl_intranode`, default on) — see
   "Intra-node level" below. Runs first, before any cross-node traffic.
1. `analysis::schedule::tournament_rounds(n)` (or `sampled_rounds` with
   `--sample-pairs`) over the host index space.
2. Per round, per pair (a,b) concurrently: spawn `agent peer serve --port
   <base+pair_slot>` on a; on b run `agent peer latency <a:port>` then
   `agent peer bandwidth <a:port>`; parse the client's JSON reports; send
   `shutdown_peer`. Attribute metrics to both hosts with
   `Scope::HostPair{peer}` (`net_latency.rtt_p50/p99/max`,
   `net_bandwidth.gib_per_sec`) before forwarding to the collector.
3. NCCL sweeps, hierarchical: fleet-wide world (one rank per GPU-bearing
   node); rendezvous = orchestrator relays `NcclUniqueId` from rank 0's
   GenerateId to all Participate stdin docs. The per-size timing loop is
   `agent::sweep::run_plan`, shared with the intra-node level; only the
   launch adapter (`RankCollectives`: one communicator on device 0)
   differs.
   The world selection (`nccl_world`: GPU-bearing + loadable libnccl) and
   the rendezvous-relay/participant-supervision driver
   (`NcclJob`/`drive_fleet_nccl`, in `src/orchestrator/nccl.rs`) are
   shared with the overlap phase's fleet step
   (docs/features/overlap_phase.md), which runs the same world shape under
   combined GEMM load; the directive's `NcclWorkload` selects sweep vs
   overlap.
4. `analysis::fit::fit_alpha_beta` over (size, elapsed_us) → calibration
   `links` entries ("tcp_pairwise" from latency+bandwidth points per pair
   class, "nccl_allreduce_fleet" / "nccl_allgather_fleet" from the fleet
   sweep, "nccl_{allreduce,allgather}_intranode_<n>gpu" from the
   intra-node sweep).
5. Barrier-skew microbenchmark: a tiny-collective straggler probe riding
   the same NCCL communicator, plus a TCP star-barrier fallback after it.
   See docs/features/barrier_skew.md.

## Intra-node level (proto v7 / schema v8)

### Why
Pairwise p2p copies (phase 2) move one buffer between two GPUs at a time.
Collective traffic drives every NVLink/PCIe path at once, which is where a
degraded NVLink, a downtrained PCIe switch shared by several GPUs, or a
missing P2P path (NCCL silently falling back to SHM through host memory)
shows. The level also calibrates the simulator's intra-node link class.

### Flow
1. Orchestrator (`orchestrator/intranode.rs::intranode_sweep`, called from
   `network_phase` when `tests.nccl_intranode`): selects GPU-bearing hosts
   (`gpu_bearing_hosts`, probing hosts without an inventory) and
   partitions them on `nccl::nccl_loadable` (the inventory dlopen probe,
   shared with the fleet world selection). Hosts without a loadable
   libnccl get Skipped outcomes for both tests from the orchestrator; the
   rest run `node_phase(.., Phase::Network, ..)` — the ordinary per-node
   fan-out, all hosts at once, no rendezvous, no relay. CPU-only hosts are
   not dispatched to (no intra-node level; a missing driver there is not a
   finding for this test).
2. Task spec: `AgentTaskSpec.nccl_intranode: Option<NcclSweepSpec>`
   (`config::intranode_sweep_spec`: `nccl_sizes` + `nccl_iters_per_size`,
   `None` when the toggle is off). The agent's `Phase::Network` arm runs
   the sweep iff it is `Some` (gpu build) or emits Skipped outcomes (no-gpu
   build).
3. Agent gating (`agent/intranode.rs`, GPU-independent): `run_gated` asks
   the guarded driver for the device count; `gate` → `Run(MultiGpuWorld)`
   (>= 2 GPUs, the only way to reach the sweep), `Skip` (< 2 GPUs, reason
   names the count) or `Fail` (driver init failed). Skip/Fail — and any
   error or loader panic from the sweep itself — land as node-scope
   outcomes on both `nccl_intra_all_reduce` and `nccl_intra_all_gather`.
4. Agent sweep (`agent/gpu/intranode.rs`): `NodeComm::init` (one context
   per GPU, default streams, `Comm::from_devices` = `ncclCommInitAll`;
   shared with the overlap phase), then `run_plan` over
   `SweepPlan::new(sizes, world)` through `NodeCollectives`, which issues
   each collective on every rank inside one `ncclGroupStart/End` from a
   single thread and syncs all streams. Same warmup (5 full-size
   all-reduces), same per-size loop, same all-gather sharding as the fleet
   sweep.
5. Emission, `Scope::Node`, per size (streamed as measured):
   `elapsed_us`, `msg_bytes`, `bus_gib_per_sec` under
   `nccl_intra_all_reduce` / `nccl_intra_all_gather` (`sweep::point_records`,
   the fleet sweep's exact shape). After the sweep, per collective
   (`intranode::summary`): `bus_gib_per_sec_peak` (headline) and `ranks`
   (communicator size) plus a Passed outcome — or Skipped when no
   configured size could run for that collective (all-gather sizes too
   small to shard across the GPUs).

### Headline metric choice
`bus_gib_per_sec_peak` is the **maximum** bus bandwidth across the sweep's
sizes, not the value at the largest size: the largest configured size is
not guaranteed to be the saturating one (a sweep capped below the knee, an
all-gather leg skipped at the top), and the best achieved figure is what
operators compare to the link's peak. Non-finite values are ignored; no
finite value means no headline (and the Skipped outcome above). It is one
value per node per repeat, so the fleet-comparability rule MADs it across
nodes — the straggler signal. The per-size groups repeat their sample key
and are excluded from MAD (they feed calibration).

### Bus bandwidth
`sweep::Collective::bus_gib_per_sec` delegates to the existing
`nccl::{all_reduce,all_gather}_bus_gib_per_sec`: all-reduce `2(n-1)/n`,
all-gather `(n-1)/n` over the gathered size, n = local GPU count.

### socket_ifname does not apply
`[nccl] socket_ifname` is not exported for the intra-node sweep. The
communicator is created with `ncclCommInitAll` inside one process: its
bootstrap handshake connects the process to itself, and the data path is
P2P/NVLink/SHM — no socket transport carries payload, so the interface
choice cannot change what is measured. Exporting it would also mean a
`set_var` inside `agent run`, which (unlike `agent nccl`) executes on the
multi-threaded tokio runtime, where mutating the environment is unsound.
The overlap phase's node-local communicator has the same shape and
likewise does not set it.

### Calibration link classes
`report/intranode.rs`: per (host, repeat), `msg_bytes` and `elapsed_us`
are joined by emission order and bucketed by that run's `ranks` value;
each bucket is fitted with `fit_alpha_beta` into
`nccl_allreduce_intranode_<n>gpu` / `nccl_allgather_intranode_<n>gpu`.
Keyed by GPU count rather than restricted to the majority count: different
GPU counts are different links (8-GPU NVSwitch vs 4-GPU PCIe), keying
keeps every node's data, and a heterogeneous fleet gets one calibration per
topology the simulator may need; a homogeneous fleet simply gets one class
per collective. A (host, repeat) without a valid `ranks` value (a sweep
that failed part-way) contributes no points.

## Management vs data plane
`HostConfig.data_addr`, when set, is the target for peer latency/bandwidth
probes; ssh control traffic stays on `addr`. Without it, the target is
`addr` stripped of user/port. NCCL interface selection is orthogonal:
`[nccl] socket_ifname` pins NCCL_SOCKET_IFNAME (e.g. "bond0").

## Peer wire format (net.rs)
First byte from client selects mode: 0x01 latency, 0x02 bandwidth, 0xFF
shutdown. Latency: 8-byte payload echoed, TCP_NODELAY, client times each
round trip until duration elapses. Bandwidth: client streams 4 MiB writes
until duration elapses, then half-closes; server reports bytes received
back on the same socket in a trailing 8-byte frame (client computes GiB/s
from received-side count). Server handles clients sequentially; exits on
shutdown frame.

The bandwidth window runs from the first write until the trailing count
arrives, not until half-close: bytes still in flight when the client stops
writing are part of the transfer, and crediting them for free would inflate
high-BDP links. Counting on the receiver is what makes `bytes_sent` mean
bytes that actually landed. `serve` binds 0.0.0.0, sets TCP_NODELAY on both
ends, logs and skips a client that dies mid-session, and returns Ok only on
the shutdown frame; the client modes print their report as one JSON line on
stdout while `serve` prints nothing. Connects are bounded by a 10s timeout,
so an unreachable peer errors instead of stalling its round. Percentiles are
nearest-rank over the sorted sample vector, which makes p50 ≤ p99 ≤ max hold
by construction.

## Scheduling invariants (schedule.rs)
- `tournament_rounds(n)`: every unordered pair exactly once; no host twice
  in a round; round count n-1 (even n) / n (odd n); pairs `(a,b)` a<b.
- `sampled_rounds(n,k)`: ring offsets 1..=k, deduped, packed disjointly.

## Files
- `src/agent/net.rs` — peer mode (TCP latency/bandwidth).
- `src/agent/sweep.rs` — shared, GPU-independent sweep driver:
  `Collective` (bus factors), `SweepLevel` (level → TestId), `SweepPlan`
  (sizes → steps, all-gather sharding, buffer size), `SweepCollectives`
  (launch/sync trait), `run_plan` (warmup + per-size timing loop),
  `point_records` (per-size metric records).
- `src/agent/nccl.rs` — fleet sweep (`run_sweep` + `RankCollectives`),
  bus-bandwidth helpers, fleet overlap.
- `src/agent/intranode.rs` — intra-node gating (`MultiGpuWorld`, `gate`,
  `run_gated`), headline selection (`peak_bus_gib_per_sec`), `summary`,
  `skip_without_gpu`.
- `src/agent/gpu/intranode.rs` — intra-node sweep on the GPU
  (`run`, `NodeCollectives`).
- `src/agent/gpu/node_comm.rs` — `NodeComm` (contexts, streams,
  `ncclCommInitAll`, `grouped`, `sync_all`, `alloc_per_rank`) and
  `nccl_error`; shared with the overlap phase.
- `src/agent/mod.rs` — `Phase::Network` arm → `network_phase`.
- `src/proto.rs` — `NcclSweepSpec`, `AgentTaskSpec.nccl_intranode`,
  `TestId::{NcclIntraAllReduce, NcclIntraAllGather}`, `nccl_metric` name
  consts; PROTO_VERSION 7.
- `src/config.rs` — `tests.nccl_intranode`, `intranode_sweep_spec`.
- `src/analysis/schedule.rs`, `src/analysis/fit.rs`.
- `src/orchestrator/mod.rs` — `network_phase` (hierarchy order), pairwise
  + TCP barrier; `src/orchestrator/intranode.rs` — intra-node step;
  `src/orchestrator/nccl.rs` — fleet NCCL jobs, `nccl_loadable`.
- `src/report/intranode.rs` — intra-node link classes;
  `src/report/mod.rs` — display names, `link_fits`; SCHEMA_VERSION 8.

## Invariants
- RTT metrics report distribution (p50/p99/max), never mean-only.
- A failed pair marks both directions' metrics absent and records the error
  against the client host; it never stalls the round (per-pair timeout).
- Port allocation: `net_port_base + slot` where slot < pairs-per-round;
  serve side binds 0.0.0.0.
- Hierarchy order within the phase: intra-node, then pairwise, then fleet.
- Every sweep level runs the same per-size loop (`sweep::run_plan`) with
  the same sizes, iterations, warmup, and all-gather sharding; levels
  differ only in how a collective is launched and which TestIds they
  emit under.
- The intra-node sweep only ever runs on a `MultiGpuWorld` (>= 2 GPUs);
  every other path on a dispatched host ends in explicit Skipped/Failed
  outcomes for both intra-node tests, never a silent gap and never a
  failed-host verdict (the agent itself exits cleanly).
- Intra-node per-size series are never MAD-compared; the
  `bus_gib_per_sec_peak` headline (one per node per repeat) always is.
- Intra-node link classes never mix GPU counts, and intra-node points
  never enter the fleet classes (distinct TestIds).
