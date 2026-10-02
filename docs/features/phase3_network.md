# Phase 3 — Network

## Scope
The network hierarchy, innermost level first: the intra-node NCCL sweep
(every local GPU of one node, NVLink/PCIe), pairwise TCP latency/bandwidth
between nodes (tournament-scheduled), then the fleet-wide NCCL sweep.
Non-scope: intra-node pairwise p2p copies (phase 2, `gpu_p2p`), IB-verbs
microbenchmarks (post-v1; NCCL measures what training experiences),
pair-level NCCL sweeps (full fleet + TCP pairwise usually localizes).

## Step selection (`[tests] net_steps`, `--net-steps`)
The phase is four independent steps, a typed `NetStep` set
(`src/net_steps.rs`): `intranode`, `pairwise` (alias `tcp`), `nccl` (the
fleet sweep) and `barrier` (both barrier-skew probes). Default: all.
`--net-steps` (comma-separated, help generated from
`NetStep::PARSE_TABLE`) overrides `tests.net_steps`;
`tests.nccl_intranode = false` always removes `intranode`.
`FleetConfig::resolve_net_steps` is the only constructor of the resolved
`NetSteps`, so the set the orchestrator sees is validated (unknown names
are `ConfigError::UnknownNetStep`) and non-empty (`ConfigError::NoNetSteps`
— drop the network phase instead). `validate` resolves it once, so a bad
`net_steps` fails at load.

A quick NCCL-only check is `gauntlet run --phases network --net-steps
nccl` (add `intranode` for both NCCL levels). Before any step runs,
`network_phase` records `Skipped { reason: "disabled by config" }`
(`net_steps::DISABLED_REASON`) at node scope on every host for each test a
deselected step would have produced (`disabled_outcomes`, from
`NetStep::tests`): intranode → `nccl_intra_all_reduce` /
`nccl_intra_all_gather`; pairwise → `net_latency` / `net_bandwidth`;
nccl → `nccl_all_reduce` / `nccl_all_gather`; barrier → `nccl_barrier` /
`tcp_barrier`. The NCCL barrier rides the fleet sweep's communicator, so
`barrier` without `nccl` runs only the TCP barrier and records
`nccl_barrier` as Skipped (`NetSteps::nccl_barrier`). Skipped outcomes
never affect the verdict. `barrier_iters = 0` still disables the barrier
probes without a trace (it is a size knob, not the step switch).

## Control flow
Orchestrator phase-3 driver (`network_phase` in `orchestrator/mod.rs`),
each step gated on the resolved `NetSteps`:
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
3. NCCL sweeps, hierarchical: fleet-wide world with **one rank per GPU**
   (proto v7; see "Fleet NCCL world" below); rendezvous = orchestrator
   relays `NcclUniqueId` from the lead host's in-process mint to every
   other host's Participate stdin doc. The per-size timing loop is
   `agent::sweep::run_plan`, shared with the intra-node level; this level
   supplies only its launch (grouped, once per local rank) and its timer
   (global rank 0's stream) via `RankBlockCollectives`
   (`agent/nccl/sweep.rs`).
   The world selection (`nccl_world`: GPU-bearing + loadable libnccl,
   laid out by `RankLayout`) and the rendezvous-relay/host-supervision
   driver (`NcclJob`/`drive_fleet_nccl`, in `src/orchestrator/nccl/`) are
   shared with the overlap phase's fleet step
   (docs/features/overlap_phase.md), which runs the same world under
   combined GEMM load; the directive's `NcclWorkload` selects sweep vs
   overlap. The sweep's timing comes from global rank 0 only: the lead
   host times each size until local rank 0's stream (= global rank 0)
   completes and emits the node-scope `nccl_all_reduce.*` /
   `nccl_all_gather.*` series; bus bandwidth uses the full world size
   (total GPUs).
4. `analysis::fit::fit_alpha_beta` over (size, elapsed_us) → calibration
   `links` entries ("tcp_pairwise" from latency+bandwidth points per pair
   class, "nccl_allreduce_rank_per_gpu" / "nccl_allgather_rank_per_gpu"
   from the sweep — renamed from `*_fleet` in schema v8 because the
   world changed meaning: n is total GPUs and the ring mixes NVLink with
   the fabric, so v7 per-node fits must not line up under the same key),
   and "nccl_{allreduce,allgather}_intranode_<n>gpu" from the intra-node
   sweep.
5. Barrier-skew microbenchmark: a tiny-collective straggler probe riding
   the same NCCL communicator, plus a TCP star-barrier fallback after it.
   See docs/features/barrier_skew.md.

## Intra-node level (proto v8 / schema v9)

### Why
Pairwise p2p copies (phase 2) move one buffer between two GPUs at a time.
Collective traffic drives every NVLink/PCIe path at once, which is where a
degraded NVLink, a downtrained PCIe switch shared by several GPUs, or a
missing P2P path (NCCL silently falling back to SHM through host memory)
shows. The level also calibrates the simulator's intra-node link class.

### Flow
1. Orchestrator (`orchestrator/intranode.rs::intranode_sweep`, called from
   `network_phase` when `tests.nccl_intranode`): selects GPU-bearing hosts
   (`gpu_bearing_hosts`, probing hosts without an inventory) and decides
   per host with `intranode::eligibility`: a host whose libnccl cannot
   load (`nccl::nccl_loadable`, the inventory dlopen probe shared with the
   fleet world selection) or whose inventory `cuda_visible_gpus` is below
   2 (the same CUDA-visible count the fleet world sizes rank blocks from,
   never the nvidia-smi count) gets Skipped outcomes for both tests from
   the orchestrator, with the reason. A host without a CUDA-visible count
   is still dispatched: the agent's own gate is authoritative, and a
   driver that cannot count GPUs on a GPU-bearing host is a Failed
   finding. The rest run `node_phase(.., Phase::Network, ..)` — the ordinary per-node
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
   single thread and syncs all streams (the clock stops when every local
   rank is done — the default `wait_timed` — unlike the fleet level,
   which times global rank 0). Same warmup (5 full-size
   all-reduces), same per-size loop, same all-gather sharding as the fleet
   sweep.
5. Emission, `Scope::Node`, per size (streamed as measured):
   `elapsed_us`, `msg_bytes`, `bus_gib_per_sec` under
   `nccl_intra_all_reduce` / `nccl_intra_all_gather` (`sweep::point_records`,
   the fleet sweep's exact shape). After the sweep, per collective
   (`intranode::summary`): `bus_gib_per_sec_peak_<n>gpu` (headline, named
   by `proto::nccl_metric::bus_peak`) and `ranks`
   (communicator size) plus a Passed outcome — or Skipped when no
   configured size could run for that collective (all-gather sizes too
   small to shard across the GPUs).

### Headline metric choice
`bus_gib_per_sec_peak_<n>gpu` is the **maximum** bus bandwidth across the sweep's
sizes, not the value at the largest size: the largest configured size is
not guaranteed to be the saturating one (a sweep capped below the knee, an
all-gather leg skipped at the top), and the best achieved figure is what
operators compare to the link's peak. Non-finite values are ignored; no
finite value means no headline (and the Skipped outcome above). It is one
value per node per repeat, so the fleet-comparability rule MADs it across
nodes — the straggler signal.

The name carries the communicator size (`_<n>gpu`, the same suffix as the
calibration link classes, `nccl_metric::gpu_class_suffix`), so every
topology is its own MAD group: an 8-GPU NVLink node (~200 GiB/s) is never
compared against a 4-GPU PCIe node (~20 GiB/s). Under one shared group a
healthy minority topology would be flagged wholesale and a degraded node
inside it could not be told from its peers. The consequence is that a
topology needs at least 4 nodes (`flag_outliers`' minimum) before its
nodes can be flagged at all; a smaller group reports its values but never
flags. Keying happens at emission rather than in report grouping so the
report stays generic — no metric-specific grouping rule. The per-size
groups repeat their sample key and are excluded from MAD (they feed
calibration).

### Bus bandwidth
`sweep::Collective::bus_gib_per_sec` delegates to the existing
`nccl::{all_reduce,all_gather}_bus_gib_per_sec`: all-reduce `2(n-1)/n`,
all-gather `(n-1)/n` over the gathered size, n = local GPU count.

### NCCL env in the intra-node sweep
The resolved `[nccl]` env, including `socket_ifname` as
NCCL_SOCKET_IFNAME, reaches the intra-node sweep in the same way as every
other agent process. The orchestrator sets it on the remote `env ...
gauntlet agent run` command line, so it is already in the process
environment before `agent run` builds its multi-threaded tokio runtime.
The agent never calls `set_var`. The same applies to the overlap phase's
node-local communicator (docs/features/nccl_env.md).

Some knobs have no effect here:
- NCCL_SOCKET_IFNAME changes nothing measurable. The communicator is
  created with `ncclCommInitAll` inside one process, its bootstrap
  handshake connects the process to itself, and the data path is
  P2P/NVLink/SHM, so no socket transport carries payload.
- The same holds for the IB/NET knobs.

Other knobs do act on intra-node paths: NCCL_P2P_LEVEL, NCCL_ALGO,
NCCL_PROTO and NCCL_DEBUG.

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

## Fleet NCCL world (proto v7)

Every GPU on every NCCL-capable host is a rank, so the sweep, the barrier
probe and the fleet overlap step exercise every GPU's PCIe/NIC path —
not just GPU 0's, which is all the v6 one-rank-per-node world ever
touched (a bad NIC, riser or PCIe switch behind any other GPU was
invisible). Real training runs one rank per GPU; so does this.

- **Layout** (`orchestrator/nccl/layout.rs`, pure, generic over the
  member type): hosts in fleet order, zero-GPU hosts excluded, each host
  a contiguous `RankBlock { base, count }` ordered by local GPU index
  (local GPU `i` = global rank `base + i`); `world_size` = total GPUs.
  Global rank 0 is local GPU 0 of the first NCCL-capable host. A host's
  rank count is what *CUDA* can open there — the phase-0 inventory's
  `cuda_visible_gpus` (`cuDeviceGetCount` in the agent; probed when the
  inventory is missing), never the nvidia-smi `gpus.len()`. The two
  differ when a GPU fell off the bus, MIG is on, or CUDA_VISIBLE_DEVICES
  is set in the ssh environment; a block sized from nvidia-smi would make
  that host fail before init and strand the world. The mismatch is a
  per-host phase-0 finding (Failed `inventory` outcome,
  `proto::gpu_visibility_mismatch`); a host whose agent could not ask
  CUDA contributes no ranks. Local GPU index = CUDA device ordinal, the
  same mapping phase 2 uses for `Scope::Gpu`. `locate(rank)` maps a rank back to (host, local GPU);
  `arrival_group(rank)` gives the host index.
- **Directive** (`proto::ranks`): `Lead`/`Participate` carry a
  `RankAssignment { block, world_size }`. `RankBlock` rejects `count = 0`
  and u32 overflow; `RankAssignment` rejects blocks reaching past the
  world. Both validate on construction *and* on deserialization (serde
  `try_from`), so a decoded directive is a sensible one. The agent
  additionally refuses a `Lead` whose block does not start at 0 and a
  `Participate` whose block does. `agent nccl` emits `Hello` before
  anything can fail (reading/parsing the directive, these checks, the id
  decode, device checks) and reports every later failure as a typed
  `Fatal` naming the real reason — never as "first event was not
  hello".
- **Process model**: still one `agent nccl` process, one ssh session and
  one supervised task per host. The process drives its whole block from
  one thread (`agent/nccl/local.rs`), in two stages:
  1. `PreparedRanks::new` — no NCCL call: `check_local_devices` (block
     count ≤ CUDA device count, typed `LocalDeviceError`), a context per
     local GPU, a proven `bind_to_thread` for each, completion events;
     the workload's buffers are then allocated on those streams. The lead
     finishes this stage *before* minting the rendezvous id, so a lead
     that cannot run never recruits the followers.
  2. `PreparedRanks::connect` — every communicator via `ncclCommInitRank`
     (cudarc `Comm::from_rank`, device context bound so NCCL picks the
     right device) inside one `ncclGroupStart`/`ncclGroupEnd`, NCCL's
     documented multi-GPU-per-thread init. cudarc 0.19 exposes
     `group_start`/`group_end` and `Comm::from_rank`, so no
     thread-per-rank fallback was needed. Only `ncclCommInitRank` can
     fail inside the group. If rank k's init fails, the group is
     deliberately left open and the half-built communicators leaked
     (`abandon_partial_init`): `ncclGroupEnd` would run the queued inits
     of ranks 0..k-1, which block until the whole world joins, and
     cudarc's `Comm` drop calls `comm_abort(..).expect`, which can panic
     on a never-initialized communicator. The process fails and exits
     right after, which reclaims both.
  Every collective is then issued once per local rank inside one group
  per operation (collective groups are always closed), and the local
  streams are drained afterwards. `LocalRank`'s stream and context come
  from its communicator (single source).
- **Per-rank timing without ordering bias**: where timings are per rank
  (barrier, fleet overlap), the process records a completion event on
  every local stream and polls them round-robin
  (`agent/nccl/completion.rs`, pure), stamping each rank when first seen
  done. Sequential `stream.synchronize()` calls would stamp local GPU 0
  first on every iteration — an index-ordered bias the fleet-wide MAD
  would read as per-GPU skew.
- **Failure semantics** (`orchestrator/nccl/attribution.rs`, pure): one
  dead rank blocks every other rank inside a collective, so without care
  a single fault reads as a timeout on *every* host. Each host's failure
  is classified:
  - *primary* — a `Fatal` event, any other error or nonzero exit, or the
    lead failing before the id relay (whatever its own outcome);
  - *secondary* — a timeout, "never started" (no id), a cascade exit
    (`AGENT_EXIT_CASCADE` = 124: the agent stopped itself because the
    fleet stopped), or `Aborted` (killed by the driver, below).
  Primaries stay Failed on the culprit host; secondaries become Skipped
  with a reason naming the culprit(s) ("fleet overlap aborted: rank
  failure on 10.1.1.68 (…)"); with no primary at all (everyone timed
  out) every host stays Failed — there is nobody to blame. The sweep
  (`HostError` mode) records culprits as host errors and only *warns*
  about secondaries, so healthy hosts never read as HostFailures; the
  fleet overlap step (`WarnOnly`) turns both into outcomes. The driver
  captures `Fatal` events itself rather than forwarding them to the
  collector. Failure messages name the rank range (`nccl ranks 8..16
  exited with …`).
- **Early abort** (`AbortTracker`, pure): the first primary failure ends
  the job — the driver kills every still-running host's agent at once
  instead of letting them hang until the phase timeout, and whatever
  those hosts report afterwards is classified `Aborted` (secondary). A
  lead that mints no id within `NCCL_ID_WAIT` is killed immediately.
- **Remote kill**: every host abandoned by the phase timeout (and every
  host aborted early) is killed with `pkill -f '[g]auntlet-agent agent
  nccl'` over its ssh session (`orchestrator::kill_remote_agent`, the
  same mechanism the peer and TCP-barrier servers use). Dropping the ssh
  future alone left the remote process blocked in a collective — never
  writing to stdout again, so not even SIGPIPE ended it — with its GPUs
  still loaded. The shared helper also fixes the pattern those older
  kills used (`[g]auntlet-agent peer serve …` never matched the real
  command line `…/bin/gauntlet-agent agent peer serve …`). It kills any
  `agent nccl` on the host; only one runs per host at a time.
- **Rank ownership** (`orchestrator/nccl/ownership.rs`, pure): per-rank
  events (barrier timings, fleet-overlap reports) are accepted only from
  the host whose `RankBlock` holds the rank, first report wins.
  Out-of-block, unknown-host and duplicate reports are dropped and
  recorded as a structured host error against the sender (a broken
  agent, whichever step) — never misattributed, and a duplicate no
  longer voids the whole barrier analysis.

## Management vs data plane
`HostConfig.data_addr`, when set, is the target for peer latency/bandwidth
probes; ssh control traffic stays on `addr`. Without it, the target is
`addr` stripped of user/port. NCCL interface selection is orthogonal:
`[nccl] socket_ifname` pins NCCL_SOCKET_IFNAME (e.g. "bond0") so NCCL's
bootstrap and socket transport ride the data plane rather than whatever
interface it would pick first. It stays the typed first-class knob for
this; other NCCL tuning (NCCL_IB_HCA, NCCL_IB_GID_INDEX,
NCCL_NET_GDR_LEVEL, ...) goes in `[nccl] env`. The orchestrator folds
`socket_ifname` into that map as NCCL_SOCKET_IFNAME (setting it in both
places is a config error) and sets the resolved map on the remote
`env ... gauntlet agent ...` command line of every agent spawn, so it is
in each rank's environment before the process starts (the agent never
calls `set_var`; proto v9 dropped `socket_ifname` from the directives).
See docs/features/nccl_env.md.

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
Intra-node level and shared sweep loop:
- `src/agent/net.rs` — peer mode (TCP latency/bandwidth).
- `src/agent/sweep.rs` — shared, GPU-independent sweep driver:
  `Collective` (bus factors), `SweepLevel` (level → TestId), `SweepPlan`
  (sizes → steps, all-gather sharding, buffer size), `SweepCollectives`
  (launch/sync trait), `run_plan` (warmup + per-size timing loop),
  `point_records` (per-size metric records), `max_elements` (buffer
  sizing rule).
- `src/agent/intranode.rs` — intra-node gating (`MultiGpuWorld`, `gate`,
  `run_gated`), headline selection (`peak_bus_gib_per_sec`), `summary`,
  `skip_without_gpu`.
- `src/agent/gpu/intranode.rs` — intra-node sweep on the GPU
  (`run`, `NodeCollectives`).
- `src/agent/gpu/node_comm.rs` — `NodeComm` (contexts, streams,
  `ncclCommInitAll`, `grouped`, `sync_all`, `alloc_per_rank`) and
  `nccl_error`; shared with the overlap phase.
- `src/agent/mod.rs` — `Phase::Network` arm → `network_phase`.
- `src/proto/mod.rs` — `NcclSweepSpec`, `AgentTaskSpec.nccl_intranode`,
  `TestId::{NcclIntraAllReduce, NcclIntraAllGather}`, `nccl_metric` name
  consts and helpers (`bus_peak`, `gpu_class_suffix`); PROTO_VERSION 8.
- `src/config.rs` — `tests.nccl_intranode`, `intranode_sweep_spec`,
  `tests.net_steps`, `resolve_net_steps`.
- `src/net_steps.rs` — `NetStep` (`ALL`, `PARSE_TABLE`, `name`, `parse`,
  `help_list`, `tests`), `NetSteps` (`contains`, `iter`, `nccl_barrier`),
  `disabled_outcomes`, `DISABLED_REASON`.
- `src/cli.rs` — `RunArgs::net_steps` (`--net-steps`, generated help).
- `src/analysis/schedule.rs`, `src/analysis/fit.rs`.
- `src/orchestrator/mod.rs` — `network_phase` (hierarchy order), pairwise
  + TCP barrier; `src/orchestrator/intranode.rs` — intra-node step
  (`eligibility`, `intranode_sweep`); `src/orchestrator/nccl/mod.rs` —
  `nccl_loadable` (shared with the fleet world).
- `src/report/intranode.rs` — intra-node link classes;
  `src/report/mod.rs` — display names, `link_fits`; SCHEMA_VERSION 9.

`src/agent/net.rs`, `src/agent/nccl/`, `src/analysis/schedule.rs`,
`src/analysis/fit.rs`, orchestrator phase-3 driver in
`src/orchestrator/mod.rs` (pairwise + TCP barrier) and
`src/orchestrator/nccl/` (fleet NCCL jobs).

Fleet NCCL world:
- `src/proto/ranks.rs` — `RankBlock` (`new`, `base`, `count`, `end`,
  `ranks`, `contains`, `local_index`, `holds_lead`), `RankAssignment`
  (`new`, `block`, `world_size`), `RankError`.
- `src/proto/mod.rs` — `InventorySnapshot::cuda_visible_gpus`,
  `gpu_visibility_mismatch`, `AGENT_EXIT_CASCADE`.
- `src/orchestrator/nccl/layout.rs` — `RankLayout<M>` (`new`, `empty`,
  `members`, `world_size`, `member_count`, `locate`, `arrival_group`),
  `RankLocation`, `LayoutError`.
- `src/orchestrator/nccl/mod.rs` — `nccl_world` (`nccl_rank_count`),
  `NcclJob`, `drive_fleet_nccl` (`run_host`, `abort_rest`,
  `blame_the_lead`, `report_failures`), `block_failure`, `nccl_sweep`,
  `overlap_fleet_sweep`.
- `src/orchestrator/nccl/attribution.rs` — `FailureKind`, `HostFailure`,
  `Attribution`, `attribute`, `AbortTracker`, `AbortAction`.
- `src/orchestrator/nccl/ownership.rs` — `accept_owned`,
  `OwnershipViolation`.
- `src/orchestrator/nccl/records.rs` — per-GPU fleet-overlap records.
- `src/orchestrator/mod.rs` — `agent_kill_pattern`, `kill_remote_agent`.
- `src/agent/nccl/mod.rs` — directive entry (`run_from_stdin`: Hello,
  then `Fatal` on any failure; `imp::run`), `check_local_devices`,
  bus-bandwidth formulas.
- `src/agent/nccl/local.rs` — `PreparedRanks` (stage 1), `connect`
  (grouped init, `abandon_partial_init`), `LocalRanks` (grouped
  collectives, `sync_all`, `completion_secs`), id encode/decode.
- `src/agent/nccl/watchdog.rs` — `guarded` (hard-deadline watchdog),
  `exit_cascade`, `CascadeAbort`.
- `src/agent/nccl/sweep.rs` — fleet sweep over local ranks
  (`RankBlockCollectives`: grouped launch + rank-0 timer for the shared
  `run_plan`) and the barrier probe.
- `src/agent/nccl/fleet_overlap.rs` — fleet overlap protocol.
- `src/agent/nccl/completion.rs` — pure round-robin completion stamping.

## Invariants
- RTT metrics report distribution (p50/p99/max), never mean-only.
- A failed pair marks both directions' metrics absent and records the error
  against the client host; it never stalls the round (per-pair timeout).
- Port allocation: `net_port_base + slot` where slot < pairs-per-round;
  serve side binds 0.0.0.0.
- Fleet NCCL rank blocks tile `0..world_size` contiguously in fleet order,
  one rank per GPU, no zero-GPU host, global rank 0 on the first host's
  GPU 0; every directive's block lies inside its world by construction.
- The fleet sweep never runs on a world of fewer than 2 ranks
  (`orchestrator::nccl::sweep_gate`): a one-rank "all-reduce" is a local
  copy, whose timings would calibrate the `_rank_per_gpu` fits with a link
  that does not exist. Every member records Skipped for
  `nccl_all_reduce` / `nccl_all_gather` with the rank count as the reason.
  The gate is on ranks, not hosts — one host with 2+ GPUs is a real world.
- One `agent nccl` process per host, whatever its GPU count; grouped init
  is all-or-nothing per host, and nothing fallible but `ncclCommInitRank`
  runs inside the init group.
- Rank blocks are sized from CUDA-visible devices, never nvidia-smi.
- `agent nccl` always speaks `Hello` first; every failure after it is a
  typed `Fatal`.
- Sweep timing (and the `nccl_allreduce_rank_per_gpu` alpha/beta fit it
  feeds) comes from global rank 0 only.
- Only a primary failure is ever recorded against a host as Failed when
  a primary exists; hosts it aborted are Skipped/warned. No abandoned
  `agent nccl` process outlives its job: early abort or the timeout kill
  reaches every host.
- Hierarchy order within the phase: intra-node, then pairwise, then fleet.
- A deselected network step leaves one Skipped outcome ("disabled by
  config") per test per host per repeat, never a silent gap; the resolved
  step set is never empty.
- Every sweep level runs the same per-size loop (`sweep::run_plan`) with
  the same sizes, iterations, warmup, and all-gather sharding; levels
  differ only in how a collective is launched and which TestIds they
  emit under.
- The intra-node sweep only ever runs on a `MultiGpuWorld` (>= 2 GPUs);
  every other path on a dispatched host ends in explicit Skipped/Failed
  outcomes for both intra-node tests, never a silent gap and never a
  failed-host verdict (the agent itself exits cleanly).
- Intra-node per-size series are never MAD-compared; the
  `bus_gib_per_sec_peak_<n>gpu` headline (one per node per repeat) always
  is, and only against nodes with the same GPU count.
- Intra-node link classes never mix GPU counts, and intra-node points
  never enter the fleet classes (distinct TestIds).
