# Phase 2 — GPU (cudarc)

## Scope
Per-GPU GEMM correctness + sustained throughput, HBM/PCIe bandwidth,
intra-node p2p. Non-scope: inter-node anything (phase 3), driver install.

## Build/runtime model
cudarc `dynamic-loading` + `cuda-12040` API baseline: libcuda/libcublas/
libnccl are dlopened at runtime. The `gpu` cargo feature (default on) gates
all of `src/agent/gpu/` and `nccl::imp`; a `--no-default-features` build
emits Skipped outcomes for the phase.

## Device availability (proto v14 / schema v18)
Driver init is classified, never just stringified. `gpu::device_availability`
runs `CudaContext::device_count` under `guard_typed` and reads the PCI bus
(`agent::gpu_driver::nvidia_pci_gpus`), giving a typed
`DeviceAvailability` (`src/agent/gpu_driver.rs`, GPU-independent):
- `Usable(NonZeroU32)` — run the tests.
- `Unavailable { reason }` — no usable GPU. Every phase-2 test
  (`gpu_gemm_correctness`, `gpu_gemm_perf`, `gpu_gemm_sdc`,
  `gpu_mem_bandwidth`, `gpu_p2p`) records **Skipped** at node scope with a
  reason naming the hardware: "no NVIDIA GPU (none on the PCI bus; ...)" or
  "no usable GPU (driver unavailable: CUDA cannot open the NVIDIA GPU on
  the PCI bus at 0000:01:00.0; <cuda error>); see gpu_driver". The driver
  itself is the inventory-derived `gpu_driver` finding
  (docs/features/phase0_inventory.md), so one root cause is one hard
  finding, and a CPU-only host is clean rather than failed.
- `Fault { reason }` — a genuine CUDA error on a working driver: one
  `gpu_gemm_correctness` infrastructure failure
  (`TestOutcome::infrastructure_failure`, hard, never in the SDC section).

Classification (`gpu::classify_driver_error`, pure over cudarc's
`DriverError`): `CUDA_ERROR_NO_DEVICE`, `NOT_INITIALIZED`,
`STUB_LIBRARY` and `SYSTEM_DRIVER_MISMATCH` are Unavailable; every other
code is a Fault. A cudarc loader panic (libcuda not loadable) is
Unavailable at init (`GuardError::init_class`) but a Fault after init
(`runtime_class`: a missing libcublas or symbol is a broken stack, not
an absent GPU). Zero devices is Unavailable. Errors are classified
through the anyhow chain, so context-wrapped `DriverError`s keep their
code.

After a successful init, a per-GPU failure (`gemm`, `bandwidth`, `p2p`)
goes through `runtime_failure_outcome`: Unavailable → Skipped (all three
GEMM tests for a gemm failure), otherwise an infrastructure failure on
the test that failed (`gpu_gemm_correctness` for the gemm run).

`guard_typed` renders a `DriverError` without calling into libcuda
when that would panic (its `Display` calls `cuGetErrorString`), so a
reason always survives.

## Control flow
`gpu::run` → `run_with(device_availability())`; with usable GPUs it
iterates them; per GPU: `gemm::run_on_gpu` → `bandwidth::run_on_gpu`;
then `p2p::run_all_pairs` once.

- gemm.rs: fixed-seed host PRNG fills A,B; per dtype in `gemm_dtypes` run
  cuBLAS GEMM; compare a deterministic ~4096-element subsample against f64
  CPU recomputation; max relative error → `gpu_gemm_correctness.residual`
  metric + Passed/Failed vs `residual_tolerance(dtype)`. Sustained loop for
  `gemm_secs` of *busy* time; sustained GFLOPS →
  `gpu_gemm_perf.gflops_<dtype>`. Sample clocks/temp (nvidia-smi poll
  thread) → `clock_mhz_start`, `clock_mhz_end`, `temp_c_max` per GPU
  (throttle detection = clock_end ≪ clock_start).
- Hot SDC (`gpu_gemm_sdc`, docs/features/hot_sdc.md): every
  `sdc_check_secs` of busy time during the sustained loop, C is downloaded
  and compared **bitwise** against a baseline captured after the warm-up
  GEMM (cuBLAS is bit-reproducible for identical calls on the same GPU).
  Checks run between timed windows so the reported GFLOPS is unpolluted.
  Metrics `checks_/mismatches_/max_abs_dev_<dtype>` per GPU; any mismatch
  ⇒ Failed outcome carrying the clock/temp at each failure (hard,
  exit-code relevant); zero completed checks ⇒ Skipped, never Passed.
- bandwidth.rs: best-of-N `bandwidth_bytes` copies: D2D, H2D pinned, D2H
  pinned → `gpu_mem_bandwidth.{d2d,h2d_pinned,d2h_pinned}` GiB/s.
- p2p.rs: for each ordered pair with p2p capability: enable access, large
  copy bandwidth + small copy latency → `gpu_p2p.{bandwidth,latency}` under
  GpuPair scope; no-p2p pairs ⇒ Skipped outcome.

## NCCL (phase 3 component, same owner)
Both sweep levels (fleet `agent nccl`, intra-node `agent run` network arm)
share `agent::sweep::run_plan`; see docs/features/phase3_network.md.
`agent nccl` reads `NcclDirective` from stdin. GenerateId (rank 0) prints
`NcclUniqueId` JSON on stdout. Participate: set NCCL_SOCKET_IFNAME if
given, init communicator (one process/node, one GPU/rank v1), warmup, then
per size in `sizes`: `iters_per_size` timed all-reduce + all-gather; rank 0
emits `nccl_all_reduce.elapsed_us` + `msg_bytes` + `bus_gib_per_sec`
metrics per size. Bus bandwidth formula: allreduce factor 2(n-1)/n.

## Files
`src/agent/gpu/{mod,gemm,bandwidth,p2p}.rs`, `src/agent/nccl/`;
specs/types in `src/proto/mod.rs` (`GpuTaskSpec`, `GemmDtype`, `NcclDirective`).
- `src/agent/gpu/mod.rs` — `run`, `run_with`, `guard`, `guard_typed`,
  `GuardError`, `CudaErrorClass`, `classify_driver_error`,
  `device_availability`, `availability_from`.
- `src/agent/gpu_driver.rs` — `DeviceAvailability`, `unavailable_reason`,
  `nvidia_pci_gpus` (shared with the inventory's driver probe).
- `src/proto/failure.rs` — `FailureKind`, `TestOutcome::infrastructure_failure`.

## Invariants
- A node without a usable GPU never produces a Failed phase-2 outcome;
  it produces exactly one Skipped outcome per phase-2 test.
- Genuine CUDA errors on a working driver stay Failed, as infrastructure
  failures (never SDC findings).

## Implementation notes

### cudarc surface used (0.19.8)
`CudaContext::{device_count,new,default_stream,alloc_pinned,cu_device,cu_ctx,
bind_to_thread}`, `CudaStream::{alloc_zeros,clone_htod,clone_dtoh,memcpy_htod,
memcpy_dtoh,memcpy_dtod,synchronize}`, `CudaSlice::{slice,slice_mut}`,
`DevicePtr::device_ptr` / `DevicePtrMut::device_ptr_mut`, `CudaBlas::new`,
`cublas::result::gemm_ex`, `nccl::{Comm::from_rank,Id::new,Id::uninit,
Id::internal,Comm::all_reduce,Comm::all_gather,ReduceOp}`. Peer access has no
safe wrapper, so `driver::sys::{cuDeviceCanAccessPeer,cuCtxEnablePeerAccess}`
are called directly.

`CudaStream::memcpy_dtod` already routes a cross-context copy through
`cuMemcpyPeerAsync`, which is what the p2p test relies on.

### Panic guard
With `dynamic-loading`, cudarc resolves symbols lazily and **panics**
(`panic_no_lib_found` / `Missing symbol ...`) when libcuda/libcublas/libnccl is
absent — it does not return `DriverError`. Every entry into cudarc therefore
goes through `gpu::guard` / `gpu::guard_typed`, which wrap `catch_unwind`.
`guard_typed` keeps the failure typed (`GuardFailure::{Error, Panicked}`)
for classification (see "Device availability"); `guard` is its string
form. `agent nccl` uses the same guard so a missing NCCL stack is an
`anyhow` error rather than an abort.

### GEMM dtype configuration
All four dtypes go through one `cublasGemmEx` call with C always `CUDA_R_32F`:

| dtype | A/B type      | compute type                    |
|-------|---------------|---------------------------------|
| f32   | `CUDA_R_32F`  | `CUBLAS_COMPUTE_32F`            |
| tf32  | `CUDA_R_32F`  | `CUBLAS_COMPUTE_32F_FAST_TF32`  |
| bf16  | `CUDA_R_16BF` | `CUBLAS_COMPUTE_32F`            |
| f16   | `CUDA_R_16F`  | `CUBLAS_COMPUTE_32F`            |

16-bit operands live in `CudaSlice<u16>` holding raw bit patterns (`half`
converts on the host), so cudarc's optional `f16` feature is not needed.
cuBLAS is column-major and the host matrices are row-major, so the operands are
passed swapped (B, A) with all leading dimensions `n`: column-major
Cᵀ = Bᵀ·Aᵀ is bit-for-bit row-major C = A·B.

### Residual normalization
bf16/f16 operands are rounded on the host *before* upload and the f64 reference
uses those rounded values, so the residual measures compute error, not
representation error. tf32 is not host-rounded (`FAST_TF32` is a permission, not
a guarantee), so its residual carries the in-kernel rounding.

The denominator is `max(|expected|, floor)` with
`floor = max(0.125 * rms(expected), 1e-12)`. Random centered dot products are
roughly Gaussian, so some sampled elements cancel to near zero; dividing their
normal absolute error by a near-zero reference would manufacture failures.
NaN is propagated explicitly (`f64::max` would swallow it) and reported as a
saturated metric value plus a `Failed` outcome, because serde_json renders
non-finite floats as `null`.

### Bandwidth conventions
- `d2d` counts `2 * bytes` per copy (read + write), matching NVIDIA
  `bandwidthTest`, so it is comparable to the HBM spec-sheet figure.
- `h2d_pinned` / `d2h_pinned` count `bytes` once.
- Each direction: one warmup transfer, then best of 5.
- p2p `latency` queues 1000 8-byte peer copies and synchronizes once, dividing
  the total by the count (same shape as `p2pBandwidthLatencyTest`).

### NCCL sweep
`rank 0` emits `Hello` first, then per size the three metrics for
`NcclAllReduce` and `NcclAllGather` under `Scope::Node`; other ranks emit
nothing. All-reduce uses `size/4` f32 elements; all-gather sends
`elements / world_size` and receives the gathered whole, and `msg_bytes` is the
gathered size so `busBW = algBW * (n-1)/n` holds. Sizes too small to shard
across the world skip the all-gather leg.

### Dependencies added
`half = "=2.7.1"` (f16/bf16 host conversion) and `base64 = "=0.22.1"` (NCCL
unique id transport), both optional and pulled in by the `gpu` feature.

## Invariants
- Reduced-precision correctness is tolerance-based, never bit-exact; the
  residual value is always emitted even when passing.
- One GPU failing must not stop tests on the remaining GPUs.
- All CUDA errors surface as structured Failed outcomes with the CUDA
  error string, not panics.
- No metric value is ever non-finite: the JSON-lines protocol cannot carry
  NaN/Inf.
- The host PRNG, the operand values and the verified subsample are identical
  on every node, so residuals are comparable fleet-wide.
