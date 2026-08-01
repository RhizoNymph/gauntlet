# Phase 1 — CPU, DRAM, Disk

## Scope
Per-core CPU correctness + throughput, NUMA-aware DRAM bandwidth,
sequential disk throughput. Non-scope: GPU anything, network anything.

## Control flow
`agent run` phase `cpu_mem` → `cpu::run` → `mem::run` → `disk::run`, all
emitting through `EventSink`.

### cpu.rs
- Correctness: golden values for `checksum_round` / `float_checksum_round`
  computed once on the main thread; then one pinned worker per logical core
  (core_affinity) repeats rounds for `correctness_secs_per_core`, comparing
  each round. Mismatch ⇒ `Outcome{CpuCorrectness, Core{id}, Failed}` with
  round count in the reason; else Passed per core. Float kernel must be
  bit-deterministic: scalar f64 FMA chain, no reassociation, no rayon.
- Throughput: per-core FMA loop over an L1-resident buffer for
  `gflops_secs / cores` each (sequential, so cores are measured unloaded),
  then one all-cores-parallel run. Metrics: `cpu_gflops.gflops` per Core
  scope + `cpu_gflops.gflops_allcore` per Node scope.

### mem.rs
- Triad `a[i] = b[i] + s*c[i]` with buffers ≥ `buffer_bytes_per_numa`
  (min 4× LLC), first-touch allocated by threads pinned to the NUMA node's
  cores. Best-of-`iters` timing. Metrics `mem_bandwidth.triad` per Numa
  scope + all-nodes-parallel `mem_bandwidth.triad_allnode` per Node scope.
  NUMA discovery from /sys; single-node machines report Numa{0} only.

### disk.rs
- Per configured path: write `file_bytes` sequentially (O_DIRECT if
  possible, else buffered + fsync in the timed region), then read it back
  (drop caches not assumed — O_DIRECT read, else a fresh-open sequential
  read; document which mode ran via a `mode` log event). Temp file removed
  on all paths (RAII guard). Metrics `disk_io.seq_write` / `.seq_read` per
  Disk scope.

## Files
`src/agent/{cpu,mem,disk}.rs`; shared types in `src/proto.rs`
(`CpuTaskSpec`, `MemTaskSpec`, `DiskTaskSpec`).

## Invariants
- Correctness workers are pinned; a failed pin is a Skipped outcome for
  that core, not a silent unpinned run.
- All emitted GiB/s and GFLOPS values are finite and positive.
- No panics on machines with 1 core, 1 NUMA node, or tmpfs disk paths.
