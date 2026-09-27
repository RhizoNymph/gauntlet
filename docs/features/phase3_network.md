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
   class, "nccl_allreduce_fleet" from the sweep).
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
  Global rank 0 is local GPU 0 of the first NCCL-capable host. GPU counts
  come from the phase-0 inventory (`gpus.len()`, probed when missing);
  local GPU index = CUDA device ordinal, the same mapping phase 2 uses
  for `Scope::Gpu`. `locate(rank)` maps a rank back to (host, local GPU);
  `arrival_group(rank)` gives the host index.
- **Directive** (`proto::ranks`): `Lead`/`Participate` carry a
  `RankAssignment { block, world_size }`. `RankBlock` rejects `count = 0`
  and u32 overflow; `RankAssignment` rejects blocks reaching past the
  world. Both validate on construction *and* on deserialization (serde
  `try_from`), so a decoded directive is a sensible one. The agent
  additionally refuses a `Lead` whose block does not start at 0 and a
  `Participate` whose block does.
- **Process model**: still one `agent nccl` process, one ssh session and
  one supervised task per host. The process drives its whole block from
  one thread (`agent/nccl/local.rs`): a context per local GPU, then every
  communicator via `ncclCommInitRank` (cudarc `Comm::from_rank`, with the
  device's context bound so NCCL picks the right device) inside one
  `ncclGroupStart`/`ncclGroupEnd` — NCCL's documented multi-GPU-per-thread
  init. cudarc 0.19 exposes `group_start`/`group_end` and
  `Comm::from_rank`, so no thread-per-rank fallback was needed. The group
  is always closed, even when a rank's init call fails part-way. Every
  collective is then issued once per local rank inside one group per
  operation, and the local streams are drained afterwards.
- **Per-rank timing without ordering bias**: where timings are per rank
  (barrier, fleet overlap), the process records a completion event on
  every local stream and polls them round-robin
  (`agent/nccl/completion.rs`, pure), stamping each rank when first seen
  done. Sequential `stream.synchronize()` calls would stamp local GPU 0
  first on every iteration — an index-ordered bias the fleet-wide MAD
  would read as per-GPU skew.
- **Failure semantics** (unchanged in kind): an error anywhere in grouped
  init fails the whole host's participation — the world was sized for
  every local GPU, so a partial block cannot join. The sweep reports that
  as a host error (`HostError` mode), the fleet overlap step as outcomes
  only (`WarnOnly`). Failure messages name the rank range
  (`nccl ranks 8..16 timed out after 600s`).

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
  `ranks`, `local_index`, `global_rank`, `holds_lead`), `RankAssignment`
  (`new`, `block`, `world_size`), `RankError`.
- `src/orchestrator/nccl/layout.rs` — `RankLayout<M>` (`new`, `empty`,
  `members`, `world_size`, `member_count`, `locate`, `arrival_group`),
  `RankLocation`, `LayoutError`.
- `src/orchestrator/nccl/mod.rs` — `nccl_world`, `NcclJob`,
  `drive_fleet_nccl`, `block_failure`, `nccl_sweep`,
  `overlap_fleet_sweep`.
- `src/orchestrator/nccl/records.rs` — per-GPU fleet-overlap records.
- `src/agent/nccl/mod.rs` — directive entry (`imp::lead`/`participate`),
  bus-bandwidth formulas.
- `src/agent/nccl/local.rs` — `LocalRanks` (grouped init, grouped
  collectives, `sync_all`, `completion_secs`), id encode/decode.
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
  is all-or-nothing per host.
- Sweep timing (and the fleet `nccl_allreduce_fleet` alpha/beta fit it
  feeds) comes from global rank 0 only.
