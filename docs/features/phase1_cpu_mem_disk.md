# Phase 1 — CPU, DRAM, Disk

## Scope
Per-core CPU correctness + throughput, NUMA-aware DRAM bandwidth,
sequential disk throughput. Non-scope: GPU anything, network anything.

## Control flow
`agent run` phase `cpu_mem` → `cpu::run` → `mem::run` → `disk::run`, all
emitting through `EventSink`.

### cpu.rs
- Correctness: golden values for `checksum_round` / `float_checksum_round`
  computed once on the main thread, before any worker starts — a worker that
  disagrees is the suspect, not the oracle. Then one pinned worker per
  logical core (core_affinity), **all running concurrently** for
  `correctness_secs_per_core` wall time, comparing every round. Mismatch ⇒
  `Outcome{CpuCorrectness, Core{id}, Failed}` with the clean-round count in
  the reason; else Passed per core.
- `Scope::Core{id}` is the 0..`available_parallelism()` enumeration index,
  not the OS CPU number, so there is exactly one outcome per logical core.
  An index with no corresponding entry in the affinity mask, or a
  `set_for_current` that returns false, yields Skipped — never a silent
  unpinned run.
- Float kernel is bit-deterministic: scalar f64 `mul_add` chain driven by an
  MMIX LCG, checksum via `to_bits()`. `mul_add` is kept here precisely
  because IEEE-754 fusedMultiplyAdd is correctly rounded and therefore
  identical whether it lowers to `vfmadd` or to a libm call.
- Throughput: multiply-add loop over a 16 KiB (L1-resident) f64 buffer with
  8 independent accumulators, `gflops_secs / cores` per core (floor 50 ms),
  run sequentially so each core is measured unloaded; then one
  all-cores-parallel run on the same per-core budget, reported as the sum.
  Metrics: `cpu_gflops.gflops` per Core scope + `cpu_gflops.gflops_allcore`
  per Node scope, value = `2 * multiply_adds / elapsed / 1e9`.
- The throughput kernel writes `acc*0.5 + x` rather than calling `mul_add`.
  Without a compile-time FMA target feature `f64::mul_add` lowers to a libm
  call: ~8x slower (measured 1.05 vs 11.9 GFLOP/s per core) and it measures
  the node's libm instead of its CPU, which is fatal for a fleet-relative
  comparison. Written out, both operations are plain hardware instructions
  and the accumulator lanes stay independent, so there is no reassociation.
- `std::hint::black_box` guards the buffer and the accumulators so the
  optimizer cannot delete the loop.

### mem.rs
- Triad `a[i] = b[i] + s*c[i]` with buffers ≥ `buffer_bytes_per_numa`
  (min 4× LLC), first-touch allocated by threads pinned to the NUMA node's
  cores. Best-of-`iters` timing. Metrics `mem_bandwidth.triad` per Numa
  scope + all-nodes-parallel `mem_bandwidth.triad_allnode` per Node scope.
  NUMA discovery from /sys; single-node machines report Numa{0} only.
- Each node is measured with one thread per core of that node, since a
  single core cannot saturate a memory controller. `buffer_bytes_per_numa`
  is split evenly across those threads (floor 1 MiB each), every thread
  first-touches its own arrays, a `Barrier` lines up each timed pass so the
  windows overlap, and the per-thread GiB/s are summed.
- Bandwidth convention: 3 × bytes per pass (2 reads + 1 write). The
  write-allocate read is deliberately not counted, matching published
  STREAM numbers.
- NUMA cpulists (`/sys/devices/system/node/node*/cpulist`) are intersected
  with the process affinity mask: inside a restricted cpuset sysfs still
  names CPUs the process may not run on, and pinning to those fails
  silently. No usable /sys view ⇒ a single node 0 over the allowed CPUs.

### disk.rs
- Per configured path: write `file_bytes` sequentially (O_DIRECT if
  possible, else buffered + fsync in the timed region), then read it back
  (drop caches not assumed — O_DIRECT read, else a fresh-open sequential
  read; document which mode ran via a `mode` log event). Temp file removed
  on all paths (RAII guard). Metrics `disk_io.seq_write` / `.seq_read` per
  Disk scope.
- O_DIRECT is attempted independently for each direction and only when
  `file_bytes` is 4096-aligned (the transfer length must be a whole number
  of blocks). Open or write failure — tmpfs and several network filesystems
  reject the flag — re-runs the whole pass buffered against a freshly
  truncated file. The 4 MiB transfer buffer is block-aligned by carving an
  aligned window out of an over-allocated `Vec<u8>`, so no `unsafe`
  allocator work is needed, and it is filled with a non-trivial pattern so
  compressing filesystems cannot fake the write.
- `sync_all` is inside the timed write region: a checkpoint that only
  reached the page cache has not been written, and excluding the flush
  would report DRAM bandwidth on the buffered path.
- The chosen modes are reported by `run` as an Info Log event
  `disk_io path=… bytes=… write_mode=… read_mode=…`, and `measure` returns
  them on `DiskThroughput` (`IoMode::{Direct,Buffered}`).
- A path that fails emits `Outcome{DiskIo, Disk{path}, Failed}` and the
  phase continues; it is never an agent-level error. An empty `paths` list
  emits Skipped under Node scope.

## Files
`src/agent/{cpu,mem,disk}.rs`; shared types in `src/proto.rs`
(`CpuTaskSpec`, `MemTaskSpec`, `DiskTaskSpec`).

## Invariants
- Correctness workers are pinned; a failed pin is a Skipped outcome for
  that core, not a silent unpinned run.
- All emitted GiB/s and GFLOPS values are finite and positive; a
  non-finite or non-positive measurement becomes a Failed outcome for that
  scope instead of a metric.
- No panics on machines with 1 core, 1 NUMA node, or tmpfs disk paths.
- No `unsafe` and no `unwrap`; a panicking worker thread degrades to a
  Failed/zero result for its scope, never to a poisoned phase.

## Local reference numbers (AMD Ryzen 7 7840U, 16 logical cores, 1 NUMA
node, DDR5, NVMe, release build) — a sanity anchor, not a threshold:
per-core 11.4–12.2 GFLOPS, all-core 86 GFLOPS, triad 52–57 GiB/s,
seq_write 2.1 GiB/s / seq_read 5.4 GiB/s at 512 MiB with O_DIRECT on
both directions.
