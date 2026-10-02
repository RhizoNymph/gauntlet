# Gauntlet — Cluster Pre-flight Benchmark & Health Check

```yaml
Overview:
  description: >
    Rust CLI that reads a fleet of hosts from a TOML config (or, in srun
    launch mode, from the Slurm allocation it runs in), deploys itself to
    each node over ssh or as srun job steps, and runs a phased suite of correctness and performance
    tests (CPU, DRAM, disk, GPU, intra/inter-node network, NCCL collectives).
    Output is a schema-versioned JSON document plus a human table, with
    fleet-relative outlier detection (median + MAD) as the primary straggler
    signal and optional absolute thresholds as a secondary overlay. Results
    feed simulation calibration: per-node roofline numbers and alpha/beta
    network fits per link class.
  subsystems:
    orchestrator: >
      Runs on the operator's machine (`gauntlet run`), or on a node inside
      a Slurm allocation in srun launch mode. Tokio task per host; every
      node operation goes through a `HostSession` over one transport
      (orchestrator/transport): persistent ssh sessions via the `openssh`
      crate (native-mux ControlMaster multiplexing, respects ~/.ssh/config),
      or `srun` job steps of the current allocation (`--overlap`, one step
      per agent process, killed with `scancel` of the named step). Deploys
      the agent (ssh: sftp upload of the running binary; srun: one sbcast to
      every allocated node then a per-node install; staged then renamed,
      skipped when the remote sha256 matches), drives the phase schedule,
      aggregates results over an mpsc channel into a single lock-free
      collector task.
    agent: >
      Same binary in `gauntlet agent` mode, executed on each node. At
      startup, before any thread exists, the agent moves its protocol
      channel to a private dup of stdout and points fd 1 at stderr, so
      library output (NCCL_DEBUG, CUDA) cannot corrupt events. Built for
      glibc (static musl cannot dlopen, which cudarc requires; a musl build
      only makes sense with --no-default-features). GPU tests use cudarc with
      dynamic-loading + the cuda-12040 API baseline: libcuda/libcublas/libnccl
      are dlopened at runtime (no build-time CUDA toolchain, no node install
      beyond the driver stack). Emits JSON-lines events over stdout. Has peer
      (two-sided TCP tests) and nccl subcommand modes.
    scheduler: >
      Phase DAG. Phases 0-2 and the overlap phase are embarrassingly
      parallel across nodes. Phase 3 is a hierarchy: the intra-node NCCL
      sweep rides the same per-node fan-out (node-local communicator, no
      rendezvous), then pairwise tests use round-robin tournament
      scheduling (n-1 rounds of n/2 disjoint pairs, wall time linear in
      n), then the fleet NCCL sweep. The overlap phase (compute + comms under combined load)
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
    config.toml (+ Slurm allocation in srun mode: SLURM_JOB_ID,
    `scontrol show hostnames $SLURM_JOB_NODELIST`) -> launcher resolution
    -> orchestrator -> (deploy agent, spawn `gauntlet agent` per
    host/pair/NCCL rank over ssh or as an srun step) -> agent JSON-lines on
    stdout (srun forwards the task's stdio unchanged) -> per-host tokio
    task decodes -> mpsc -> collector -> outlier analysis -> report.json +
    terminal table. The resolved `[nccl]` env (validated NCCL_* map) is the
    environment of every agent spawn — on the remote `env ... gauntlet
    agent ...` command line over ssh, on the local srun process (exported
    with --export=ALL) over srun; never on the wire, never via set_var in
    the agent — and recorded run-level in RunResults.nccl_env, next to
    RunResults.launch (ssh, or srun + job id).
    Shared serde types in a proto module are the contract between orchestrator
    and agent; protocol is versioned and the agent announces its version first.
    The collector task also ticks on a 2s interval, rebuilding the results
    document from a snapshot of its state and writing it atomically to
    runs/<run_id>.partial.json; that file is removed when the final
    runs/<run_id>.json lands.

Features Index:
  slurm_launch:
    description: >
      `[launch] mode = "srun"` / `--launch srun` (schema v13): agents start
      as srun job steps of the allocation the orchestrator runs in
      (`srun <flags> --nodes=1 --ntasks=1 --nodelist=<host>
      --job-name=gauntlet:<host>:<args> --export=ALL <agent> agent ...`;
      default flags --overlap, --cpu-bind=none, --kill-on-bad-exit=1,
      typed and config-overridable). Hosts default to the allocation
      (`scontrol show hostnames`, pure parser) or must be a subset of it;
      no SLURM_JOB_ID is a typed error. Agent env via the srun process env
      (orchestrator NCCL_* and GPU-visibility vars stripped). Deploy by one
      sbcast to /tmp/gauntlet-$USER (or a shared path). Kill = squeue
      lookup by step name + scancel --signal=KILL. Behind a transport enum
      (Ssh | Srun) so nothing above HostSession changes.
    entry_points: [launch/, orchestrator/transport/, orchestrator/session.rs, orchestrator/deploy.rs, config.rs, cli.rs]
    depends_on: [bootstrap, nccl_env, phase3_network]
    doc: docs/features/slurm_launch.md
  bootstrap:
    description: >
      `gauntlet bootstrap`: connectivity check, arch check, agent deploy
      (fleet-wide step: per-host sftp, or one sbcast in srun mode),
      capability probe (agent probe -> InventorySnapshot, including the
      gpu_idle column), optional --tune
      (GPU persistence mode, performance governor). Renders a host x check
      readiness matrix; idempotent. `--json` emits the same report as a
      schema-versioned document (the GUI viewer's interface). `--launch`
      selects ssh or srun (connectivity cell names the Slurm job).
    entry_points: [orchestrator/bootstrap.rs, orchestrator/deploy.rs]
    depends_on: [slurm_launch]
    doc: docs/features/bootstrap.md
  phase0_inventory:
    description: >
      Inventory and sanity: kernel/driver/CUDA/NIC-firmware/MTU/governor/NUMA
      inventory; fleet consistency check (flag nodes differing from majority);
      GPU health counters (ECC, row remaps, dmesg Xid, PCIe link gen/width,
      NVLink status/errors); IB port state; clock sync offset; per-GPU
      occupancy (memory used/total, compute processes by bus id).
    entry_points: [agent/inventory.rs, agent/gpu_occupancy.rs]
    depends_on: []
    doc: docs/features/phase0_inventory.md
  gpu_idle:
    description: >
      Pre-flight "is anyone else on this GPU" check (proto v10 / schema
      v11). The agent records per-GPU GpuOccupancy (nvidia-smi memory.used/
      total plus --query-compute-apps joined by PCI bus id; graphics-only
      clients excluded by construction; its own process lineage dropped;
      any other gauntlet-agent kept as StaleGauntletAgent). The
      orchestrator derives one TestId::GpuIdle outcome per GPU as the
      inventory arrives (proto::assess_gpu_idle against
      thresholds.gpu_idle_max_used_mib, default 1024): Failed names each
      process, pid and MiB and makes the verdict Stragglers; unknown is
      Skipped. Bootstrap shows the same policy as its gpu_idle column
      (warn); the report adds a "gpus in use" section. Detection only: no
      phase is auto-skipped.
    entry_points: [agent/gpu_occupancy.rs, proto/occupancy.rs, orchestrator/mod.rs, orchestrator/bootstrap/gpu_idle.rs, report/gpu_idle.rs]
    depends_on: [phase0_inventory]
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
      Hierarchical network tests, innermost first. Intra-node (proto
      v8/schema v9, tests.nccl_intranode): all-reduce + all-gather
      message-size sweep across every local GPU (single process,
      ncclCommInitAll, NVLink/PCIe) via the per-node fan-out, on hosts with
      a loadable libnccl and >= 2 CUDA-visible GPUs (others Skipped with the
      reason); per-size series feed calibration link classes
      nccl_{allreduce,allgather}_intranode_<n>gpu, and a per-node
      bus_gib_per_sec_peak_<n>gpu headline is MAD-compared across nodes of
      the same GPU count (degraded NVLink, downtrained PCIe switch, missing
      P2P path). Then pairwise TCP RTT distribution (p50/p99) and bandwidth
      via agent peer mode, tournament-scheduled full mesh. Then the fleet
      NCCL sweep. Both NCCL sweep levels share one per-size timing loop
      (agent/sweep.rs); each level supplies only its launch and timer. Fits
      t = alpha + beta*size per link class for simulator calibration
      (least squares constrained to alpha, beta >= 0, the active bound
      recorded per fit; schema v12). The
      fleet NCCL world is one rank per GPU (proto v7): each NCCL-capable host
      owns a contiguous, validated rank block (RankBlock/RankAssignment,
      laid out by RankLayout from the CUDA-visible GPU counts in the
      phase-0 inventory) ordered by local GPU index, and one agent nccl
      process per host drives its whole block (all fallible setup first,
      then grouped ncclCommInitRank + grouped collectives from one
      thread), so every GPU's PCIe/NIC path is exercised, not just GPU
      0's. Failures are attributed: the first primary failure aborts the
      rest of the world (remote kill), only the culprit is Failed, hosts
      it aborted are Skipped/warned; abandoned agents are always killed
      remotely; per-rank reports are accepted only from the owning host.
    entry_points: [agent/net.rs, agent/sweep.rs, agent/nccl/, agent/intranode.rs, agent/gpu/intranode.rs, analysis/schedule.rs, orchestrator/mod.rs, orchestrator/intranode.rs, orchestrator/nccl/, proto/ranks.rs, report/intranode.rs]
    depends_on: [phase0_inventory, phase2_gpu]
    doc: docs/features/phase3_network.md
  barrier_skew:
    description: >
      Straggler microbenchmark: ~2000 iterations of a tiny collective with
      per-rank timing. NCCL path (tiny all-reduce on the sweep's fleet
      communicator, one rank per GPU; straggler = min local elapsed, the
      wait-time inversion; a host's ranks share one arrival and are
      tallied as one arrival group, results keyed host:gpuN) plus a
      pure-TCP star-barrier fallback for CPU-only fleets (straggler = max
      release-to-response on the coordinator's clock, keyed per host).
      Per-rank p50/p90/p99/max feed MAD analysis; a slowest-rank tally
      with a noise margin gets its own flagging rule
      (fleet.barrier_stragglers, part of the verdict). Fleet-level
      per-iteration barrier-span distribution is recorded against the
      lead host.
    entry_points: [analysis/skew.rs, agent/barrier.rs, agent/nccl/sweep.rs, orchestrator/barrier.rs, orchestrator/mod.rs, orchestrator/nccl/]
    depends_on: [phase3_network]
    doc: docs/features/barrier_skew.md
  nccl_env:
    description: >
      `[nccl] env` passthrough (proto v9 / schema v10): a validated
      name -> value map restricted to `^NCCL_[A-Z0-9_]+$` keys (no
      LD_PRELOAD / CUDA_VISIBLE_DEVICES smuggling), non-empty NUL-free
      values; `socket_ifname` stays typed and folds in as
      NCCL_SOCKET_IFNAME (setting both is a ConfigError). The orchestrator
      puts the resolved map, single-quoted, on the `env` command line of
      every agent spawn (HostSession), so every NCCL communicator — fleet
      sweep/barrier, fleet overlap, intra-node overlap — starts under it
      without the multi-threaded agent mutating its environment. Recorded
      run-level in `RunResults.nccl_env`; the viewer's diff mode lists
      drift against the baseline.
    entry_points: [nccl_env.rs, config.rs, orchestrator/session.rs, report/nccl_env.rs]
    depends_on: [phase3_network, overlap_phase]
    doc: docs/features/nccl_env.md
  overlap_phase:
    description: >
      Combined-load straggler tests, run last. Node-local step: sustained
      GEMM concurrent with an intra-node NCCL all-reduce on the same GPUs
      (single process, one rank per GPU, ncclCommInitAll via the NodeComm
      helper shared with the intra-node sweep; GEMM on a second
      stream per device from one thread per GPU, gpu/worker.rs). Fleet
      step (tests.overlap_fleet, >= 2 GPU hosts): the same per-GPU GEMM
      load on every node while every GPU is also a rank of a cross-node
      all-reduce over the real fabric (one rank per GPU since proto
      v7/schema v8), window boundaries agreed through a MIN-reduced
      control word (agent/window.rs, exactly one lead: global rank 0) so
      no cross-host clock comparison is needed; every rank (= GPU)
      reports its own OverlapFleetReport (barrier-timings pattern). Both
      steps emit overlapped GFLOPS per GPU and isolated + overlapped
      all-reduce bus bandwidth (node-local: per node; fleet: per GPU);
      report::build derives retention ratios (overlapped/isolated)
      against the phase-2 GEMM baselines and each subject's own
      collective baseline, which feed the MAD outlier analysis as the
      primary combined-load straggler signal. A wedged step cannot leave
      load behind: GEMM workers stop at a spec-derived hard deadline and
      a watchdog terminates the fleet step's agent shortly after.
    entry_points: [agent/gpu/overlap.rs, agent/nccl/fleet_overlap.rs, agent/window.rs, orchestrator/nccl/, report/mod.rs]
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
    depends_on: [phase0_inventory, gpu_idle, phase1_cpu_mem_disk, phase2_gpu, phase3_network, overlap_phase, counter_deltas, barrier_skew, nccl_env, slurm_launch]
    doc: docs/features/reporting.md
  viewer:
    description: >
      `gauntlet-view` (workspace member `viewer/`): native GPUI desktop app
      rendering runs as a fully connected fleet graph (diff mode marks a
      baseline from another schema version as not directly comparable) —
      node color = host
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
