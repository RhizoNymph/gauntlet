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
      Phase DAG. Phases 0-2 are embarrassingly parallel across nodes. Phase 3
      pairwise tests use round-robin tournament scheduling: n-1 rounds of n/2
      disjoint pairs, wall time linear in n. Target scale 32-256 nodes; a
      sampled mode exists for quick runs.
    reporting: >
      Collector computes fleet median/MAD per metric, flags outliers beyond k
      MADs, applies optional absolute thresholds, renders table + JSON, sets
      exit code. Runs persisted for diffing against last known-good.
  data_flow: >
    config.toml -> orchestrator -> (scp agent, spawn `gauntlet agent` per
    host/pair) -> agent JSON-lines on stdout -> per-host tokio task decodes ->
    mpsc -> collector -> outlier analysis -> report.json + terminal table.
    Shared serde types in a proto module are the contract between orchestrator
    and agent; protocol is versioned and the agent announces its version first.

Features Index:
  bootstrap:
    description: >
      `gauntlet bootstrap`: connectivity check, arch check, agent deploy,
      capability probe (agent probe -> InventorySnapshot), optional --tune
      (GPU persistence mode, performance governor). Renders a host x check
      readiness matrix; idempotent.
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
  phase3_network:
    description: >
      Pairwise TCP RTT distribution (p50/p99) and bandwidth via agent peer
      mode, tournament-scheduled full mesh. NCCL all-reduce/all-gather message
      -size sweeps, hierarchical: intra-node, node pairs, full fleet. Fits
      t = alpha + beta*size per link class for simulator calibration.
    entry_points: [agent/net.rs, agent/nccl.rs, analysis/schedule.rs, orchestrator/mod.rs]
    depends_on: [phase0_inventory]
    doc: docs/features/phase3_network.md
  reporting:
    description: >
      JSON schema-versioned results, MAD outlier flags, absolute-threshold
      overlay, run history, terminal table, exit codes.
    entry_points: [report/mod.rs, report/history.rs, analysis/stats.rs, analysis/fit.rs, orchestrator/collect.rs]
    depends_on: [phase0_inventory, phase1_cpu_mem_disk, phase2_gpu, phase3_network]
    doc: docs/features/reporting.md
```

## Decisions (2026-08-01)

- GPU kernels implemented in-agent via cudarc (runtime dlopen), not external tools.
- Target scale 32–256 nodes; full-mesh pairwise viable via tournament rounds.
- v1 scope: phases 0–3. Soak mode (sustained mixed load watching clock decay,
  new ECC/Xid errors, link flaps) and richer run-history diffing deferred.
```
