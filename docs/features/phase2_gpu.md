# Phase 2 — GPU (cudarc)

## Scope
Per-GPU GEMM correctness + sustained throughput, HBM/PCIe bandwidth,
intra-node p2p. Non-scope: inter-node anything (phase 3), driver install.

## Build/runtime model
cudarc `dynamic-loading` + `cuda-12040` API baseline: libcuda/libcublas/
libnccl are dlopened at runtime. The `gpu` cargo feature (default on) gates
all of `src/agent/gpu/` and `nccl::imp`; a `--no-default-features` build
emits Skipped outcomes for the phase. Driver init failure on a node whose
inventory saw GPUs is reported as Failed (broken driver is a finding).

## Control flow
`gpu::run` iterates visible GPUs; per GPU: `gemm::run_on_gpu` →
`bandwidth::run_on_gpu`; then `p2p::run_all_pairs` once.

- gemm.rs: fixed-seed host PRNG fills A,B; per dtype in `gemm_dtypes` run
  cuBLAS GEMM; compare a deterministic ~4096-element subsample against f64
  CPU recomputation; max relative error → `gpu_gemm_correctness.residual`
  metric + Passed/Failed vs `residual_tolerance(dtype)`. Sustained loop for
  `gemm_secs`; sustained GFLOPS → `gpu_gemm_perf.gflops_<dtype>`. Sample
  clocks/temp (nvidia-smi poll thread) → `clock_mhz_start`, `clock_mhz_end`,
  `temp_c_max` per GPU (throttle detection = clock_end ≪ clock_start).
- bandwidth.rs: best-of-N `bandwidth_bytes` copies: D2D, H2D pinned, D2H
  pinned → `gpu_mem_bandwidth.{d2d,h2d_pinned,d2h_pinned}` GiB/s.
- p2p.rs: for each ordered pair with p2p capability: enable access, large
  copy bandwidth + small copy latency → `gpu_p2p.{bandwidth,latency}` under
  GpuPair scope; no-p2p pairs ⇒ Skipped outcome.

## NCCL (phase 3 component, same owner)
`agent nccl` reads `NcclDirective` from stdin. GenerateId (rank 0) prints
`NcclUniqueId` JSON on stdout. Participate: set NCCL_SOCKET_IFNAME if
given, init communicator (one process/node, one GPU/rank v1), warmup, then
per size in `sizes`: `iters_per_size` timed all-reduce + all-gather; rank 0
emits `nccl_all_reduce.elapsed_us` + `msg_bytes` + `bus_gib_per_sec`
metrics per size. Bus bandwidth formula: allreduce factor 2(n-1)/n.

## Files
`src/agent/gpu/{mod,gemm,bandwidth,p2p}.rs`, `src/agent/nccl.rs`;
specs/types in `src/proto.rs` (`GpuTaskSpec`, `GemmDtype`, `NcclDirective`).

## Invariants
- Reduced-precision correctness is tolerance-based, never bit-exact; the
  residual value is always emitted even when passing.
- One GPU failing must not stop tests on the remaining GPUs.
- All CUDA errors surface as structured Failed outcomes with the CUDA
  error string, not panics.
