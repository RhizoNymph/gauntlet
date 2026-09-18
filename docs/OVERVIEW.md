# Gauntlet — Cluster Pre-flight Benchmark & Health Check

```yaml
Overview:
  description: >
    Rust CLI that reads a fleet of hosts from a TOML config, deploys itself to
    each node over ssh, and runs a phased suite of correctness and performance
    tests (CPU, DRAM, disk, GPU, intra/inter-node network, NCCL collectives).
    Output is a schema-versioned JSON document plus a human table, with
    fleet-relative outlier detection (median + MAD) as the primary straggler
    signal and optional absolute thresholds as a secondary overlay. Results
    feed simulation calibration: per-node roofline numbers and alpha/beta
    network fits per link class.
  subsystems:
    orchestrator: >
      Runs on the operator's machine (`gauntlet run`). Tokio task per host,
      persistent ssh sessions via the `openssh` crate (native-mux
      ControlMaster multiplexing, respects ~/.ssh/config). Deploys the agent
      (sftp upload of the running binary, staged then renamed, skipped when
      the remote sha256 matches), drives the phase schedule, aggregates
      results over an mpsc channel into a single lock-free collector task.
    agent: >
      Same binary in `gauntlet agent` mode, executed on each node. Built for
      glibc (static musl cannot dlopen, which cudarc requires; a musl build
      only makes sense with --no-default-features). GPU tests use cudarc with
      dynamic-loading + the cuda-12040 API baseline: libcuda/libcublas/libnccl
      are dlopened at runtime (no build-time CUDA toolchain, no node install
      beyond the driver stack). Emits JSON-lines events over stdout. Has peer
      (two-sided TCP tests) and nccl subcommand modes.
    scheduler: >
      Phase DAG. Phases 0-2 and the overlap phase are embarrassingly
      parallel across nodes. Phase 3 pairwise tests use round-robin
      tournament scheduling: n-1 rounds of n/2 disjoint pairs, wall time
      linear in n. The overlap phase (compute + comms under combined load)
      runs last: its retention ratios divide the isolated phase-2 baselines
      from the same run. Target scale 32-256 nodes; a sampled mode exists
      for quick runs.
    reporting: >
      Collector computes fleet median/MAD per metric, flags outliers beyond k
      MADs, applies optional absolute thresholds, renders table + JSON, sets
      exit code. Runs persisted for diffing against last known-good. With
      --repeat N, metrics aggregate into per-subject distribution moments
      (median/MAD/min/max/mean/stddev); outliers flag on medians and
      high run-to-run spread flags as jitter. A run in
      flight also republishes itself every 2s as runs/<run_id>.partial.json so
      viewers can tail progress.
  data_flow: >
    config.toml -> orchestrator -> (scp agent, spawn `gauntlet agent` per
    host/pair) -> agent JSON-lines on stdout -> per-host tokio task decodes ->
    mpsc -> collector -> outlier analysis -> report.json + terminal table.
    Shared serde types in a proto module are the contract between orchestrator
    and agent; protocol is versioned and the agent announces its version first.
    The collector task also ticks on a 2s interval, rebuilding the results
    document from a snapshot of its state and writing it atomically to
    runs/<run_id>.partial.json; that file is removed when the final
    runs/<run_id>.json lands.

Features Index:
  bootstrap:
    description: >
      `gauntlet bootstrap`: connectivity check, arch check, agent deploy,
      capability probe (agent probe -> InventorySnapshot), optional --tune
      (GPU persistence mode, performance governor). Renders a host x check
      readiness matrix; idempotent. `--json` emits the same report as a
      schema-versioned document (the GUI viewer's interface).
    entry_points: [orchestrator/bootstrap.rs, orchestrator/deploy.rs]
    depends_on: []
    doc: docs/features/bootstrap.md
  phase0_inventory:
    description: >
      Inventory and sanity: kernel/driver/CUDA/NIC-firmware/MTU/governor/NUMA
      inventory; fleet consistency check (flag nodes differing from majority);
      GPU health counters (ECC, row remaps, dmesg Xid, PCIe link gen/width,
      NVLink status/errors); IB port state; clock sync offset.
    entry_points: [agent/inventory.rs]
    depends_on: []
    doc: docs/features/phase0_inventory.md
  phase1_cpu_mem_disk:
    description: >
      Per-core pinned correctness workloads (silent-data-corruption screen),
      per-core and all-core GFLOPS, NUMA-aware STREAM triad DRAM bandwidth,
      sequential disk read/write on dataset/checkpoint paths.
    entry_points: [agent/cpu.rs, agent/mem.rs, agent/disk.rs]
    depends_on: [phase0_inventory]
    doc: docs/features/phase1_cpu_mem_disk.md
  phase2_gpu:
    description: >
      Per-GPU fixed-seed GEMM correctness (tolerance vs FP64 reference, per
      training dtype), sustained 30-60s cuBLAS GEMM with clock/temp sampling
      (thermal-throttle detection), D2D + pinned H2D/D2H bandwidth (PCIe
      downtraining detection), NVLink p2p pair bandwidth/latency. All cudarc.
    entry_points: [agent/gpu/]
    depends_on: [phase0_inventory]
    doc: docs/features/phase2_gpu.md
  hot_sdc:
    description: >
      Silent-data-corruption screens under thermal load (SDC is
      temperature/voltage dependent): periodic bitwise verification of the
      sustained GEMM output during the loaded window (busy-time scheduled,
      throughput-neutral), and an all-core CPU screen interleaving checksum
      rounds with the power-heavy FMA workload. Mismatches are hard
      per-scope failures with clock/temp context; fleet.sdc_failures +
      dedicated table section.
    entry_points: [agent/gpu/sdc.rs, agent/gpu/gemm.rs, agent/cpu.rs]
    depends_on: [phase1_cpu_mem_disk, phase2_gpu]
    doc: docs/features/hot_sdc.md
  phase3_network:
    description: >
      Pairwise TCP RTT distribution (p50/p99) and bandwidth via agent peer
      mode, tournament-scheduled full mesh. NCCL all-reduce/all-gather message
      -size sweeps, hierarchical: intra-node, node pairs, full fleet. Fits
      t = alpha + beta*size per link class for simulator calibration.
    entry_points: [agent/net.rs, agent/nccl.rs, analysis/schedule.rs, orchestrator/mod.rs]
    depends_on: [phase0_inventory]
    doc: docs/features/phase3_network.md
  barrier_skew:
    description: >
      Straggler microbenchmark: ~2000 iterations of a tiny collective with
      per-rank timing. NCCL path (tiny all-reduce on the sweep's fleet
      communicator; straggler = min local elapsed, the wait-time inversion)
      plus a pure-TCP star-barrier fallback for CPU-only fleets (straggler
      = max release-to-response on the coordinator's clock). Per-rank
      p50/p90/p99/max feed MAD analysis; a slowest-rank tally with a noise
      margin gets its own flagging rule (fleet.barrier_stragglers, part of
      the verdict). Fleet-level per-iteration barrier-span distribution is
      recorded against the lead host.
    entry_points: [analysis/skew.rs, agent/barrier.rs, agent/nccl.rs, orchestrator/mod.rs]
    depends_on: [phase3_network]
    doc: docs/features/barrier_skew.md
  overlap_phase:
    description: >
      Combined-load straggler tests, run last. Node-local step: sustained
      GEMM concurrent with an intra-node NCCL all-reduce on the same GPUs
      (single process, one rank per GPU, ncclCommInitAll; GEMM on a second
      stream per device from one thread per GPU, gpu/worker.rs). Fleet
      step (proto v6/schema v7, tests.overlap_fleet, >= 2 GPU hosts): the
      same per-GPU GEMM load on every node while one rank per node drives
      a cross-node all-reduce over the real fabric, window boundaries
      agreed through a MIN-reduced control word (agent/window.rs) so no
      cross-host clock comparison is needed; every rank reports its own
      OverlapFleetReport (barrier-timings pattern). Both steps emit
      overlapped GFLOPS per GPU and isolated + overlapped all-reduce bus
      bandwidth; report::build derives retention ratios
      (overlapped/isolated) against the phase-2 GEMM baselines and each
      step's own collective baseline, which feed the MAD outlier analysis
      as the primary combined-load straggler signal.
    entry_points: [agent/gpu/overlap.rs, agent/nccl.rs, agent/window.rs, orchestrator/mod.rs, report/mod.rs]
    depends_on: [phase2_gpu, phase3_network]
    doc: docs/features/overlap_phase.md
  counter_deltas:
    description: >
      Error-counter delta detection across the load phases: the agent
      snapshots PCIe AER, GPU ECC/row-remap, dmesg Xid, NVLink, EDAC, IB
      port and NVMe error counters before the first load phase (baseline
      held by the orchestrator) and again after the last repeat, diffs on
      the agent, and emits per-node CounterDeltas. Any positive increment
      is a per-host finding (verdict Stragglers) rendered in its own table
      section; full deltas (zeros included) land in the JSON. Collection
      is best effort — nodes without a subsystem contribute nothing.
    entry_points: [agent/counters.rs, orchestrator/mod.rs]
    depends_on: [phase0_inventory]
    doc: docs/features/counter_deltas.md
  reporting:
    description: >
      JSON schema-versioned results, MAD outlier flags, absolute-threshold
      overlay, run history, terminal table, exit codes. Live runs publish
      periodic partial snapshots (runs/<run_id>.partial.json, written via
      temp+rename, removed on completion) under a run id fixed at startup;
      history::list excludes them, history::list_live enumerates them.
      Snapshots are disabled when --out redirects the run elsewhere.
    entry_points: [report/mod.rs, report/history.rs, analysis/stats.rs, analysis/fit.rs, orchestrator/collect.rs]
    depends_on: [phase0_inventory, phase1_cpu_mem_disk, phase2_gpu, phase3_network, overlap_phase, counter_deltas, barrier_skew]
    doc: docs/features/reporting.md
  viewer:
    description: >
      `gauntlet-view` (workspace member `viewer/`): native GPUI desktop app
      rendering runs as a fully connected fleet graph — node color = host
      health, edge color = pairwise-path health — plus per-direction edge
      details, roofline cards, link fits, and a filterable quantitative
      metric table. Sidebar lists all runs (selectable, baseline-pinnable);
      a 1s poll loop live-tails `<run_id>.partial.json` snapshots while a
      run executes; the ▶ button launches `gauntlet run` directly; diff
      mode recolors the graph by regression vs a baseline run (>5% warn,
      >15% bad, unit-aware direction of goodness). ⚙ runs bootstrap --json
      and renders the readiness matrix; ✕ cancels a launched child (and
      cleans up its orphaned partial); debug-build runs are chip-flagged.
    entry_points: [viewer/src/main.rs, viewer/src/model.rs, viewer/src/ui/]
    depends_on: [reporting]
    doc: docs/features/viewer.md
```

## Decisions (2026-08-01)

- GPU kernels implemented in-agent via cudarc (runtime dlopen), not external tools.
- Target scale 32–256 nodes; full-mesh pairwise viable via tournament rounds.
- v1 scope: phases 0–3. Soak mode (sustained mixed load watching clock decay,
  new ECC/Xid errors, link flaps) and richer run-history diffing deferred.
```
