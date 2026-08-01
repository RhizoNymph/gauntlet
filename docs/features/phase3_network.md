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
3. NCCL sweeps, hierarchical: fleet-wide world (one rank per GPU-bearing
   node); rendezvous = orchestrator relays `NcclUniqueId` from rank 0's
   GenerateId to all Participate stdin docs. Pair-level sweeps optional in
   v1 (config flag) since full-fleet + TCP pairwise usually localizes.
4. `analysis::fit::fit_alpha_beta` over (size, elapsed_us) → calibration
   `links` entries ("tcp_pairwise" from latency+bandwidth points per pair
   class, "nccl_allreduce_fleet" from the sweep).

## Peer wire format (net.rs)
First byte from client selects mode: 0x01 latency, 0x02 bandwidth, 0xFF
shutdown. Latency: 8-byte payload echoed, TCP_NODELAY, client times each
round trip until duration elapses. Bandwidth: client streams 4 MiB writes
until duration elapses, then half-closes; server reports bytes received
back on the same socket in a trailing 8-byte frame (client computes GiB/s
from received-side count). Server handles clients sequentially; exits on
shutdown frame.

## Scheduling invariants (schedule.rs)
- `tournament_rounds(n)`: every unordered pair exactly once; no host twice
  in a round; round count n-1 (even n) / n (odd n); pairs `(a,b)` a<b.
- `sampled_rounds(n,k)`: ring offsets 1..=k, deduped, packed disjointly.

## Files
`src/agent/net.rs`, `src/agent/nccl.rs`, `src/analysis/schedule.rs`,
`src/analysis/fit.rs`, orchestrator phase-3 driver in
`src/orchestrator/mod.rs`.

## Invariants
- RTT metrics report distribution (p50/p99/max), never mean-only.
- A failed pair marks both directions' metrics absent and records the error
  against the client host; it never stalls the round (per-pair timeout).
- Port allocation: `net_port_base + slot` where slot < pairs-per-round;
  serve side binds 0.0.0.0.
