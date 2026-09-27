# Phase 3 — Network

## Scope
Pairwise TCP latency/bandwidth between nodes, NCCL collective sweeps, and
the tournament scheduler. Non-scope: intra-node p2p (phase 2), IB-verbs
microbenchmarks (post-v1; NCCL measures what training experiences).

## Control flow
Orchestrator phase-3 driver:
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
   other host's Participate stdin doc. Pair-level sweeps optional in v1
   (config flag) since full-fleet + TCP pairwise usually localizes.
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
   the fabric, so v7 per-node fits must not line up under the same key).
5. Barrier-skew microbenchmark: a tiny-collective straggler probe riding
   the same NCCL communicator, plus a TCP star-barrier fallback after it.
   See docs/features/barrier_skew.md.

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
- `src/agent/nccl/sweep.rs` — sweep + barrier probe over local ranks.
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
