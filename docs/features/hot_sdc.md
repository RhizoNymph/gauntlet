# Hot SDC — silent-data-corruption screens under thermal load

## Scope
Correctness verification *while* the hardware is at maximum package/GPU
power and temperature, because SDC is strongly temperature- and
voltage-dependent and the start-of-phase screens run cold. Two screens:

- **GPU**: periodic bitwise verification of the sustained cuBLAS GEMM
  output during the loaded 30–60 s window (phase 2).
- **CPU**: the checksum correctness workloads interleaved with the
  power-heavy FMA workload on every core simultaneously (phase 1).

Non-scope: new math kernels (both screens reuse the existing workloads),
tolerance-based correctness (that remains the fixed-seed GEMM test's job),
soak testing, MAD outlier analysis (SDC hits are absolute hard failures).

## GPU screen — data/control flow

`gemm::run_on_gpu` → `sustained_pass` → per dtype `sustained_gflops`:

1. One warm-up GEMM, synchronize, then download C once as the **baseline**
   (skipped entirely when `sdc_check_secs = 0`).
2. Timed loop: batches of `SUSTAINED_BATCH` launches + one synchronize per
   batch. Wall time of each launch/sync window accumulates into **busy**
   time; both the `gemm_secs` budget and the reported GFLOPS run against
   busy time only.
3. `sdc::CheckScheduler` fires once per `sdc_check_secs` of busy time
   (missed checkpoints are skipped, never bursted). On fire — *between*
   timed windows, so the download/compare cost is invisible to the
   throughput figure — C is downloaded and compared bitwise against the
   baseline (`sdc::bitwise_mismatch`). On mismatch, nvidia-smi is queried
   for the clock/temperature at that moment and the failure is recorded
   with that context (`sdc::SdcStats`).
4. Per dtype, `gpu_gemm_sdc.{checks,mismatches,max_abs_dev}_<dtype>`
   metrics are emitted (Count/Count/Residual; deviation 0.0 when clean,
   saturated to f64::MAX when unquantifiable). After all dtypes,
   `sdc::summary_outcome` produces one `TestId::GpuGemmSdc` outcome per
   GPU: Failed on any mismatch (reason lists per-failure deviation and
   MHz/°C context), Skipped when disabled or when no check completed,
   Passed otherwise.

**Why bitwise**: cuBLAS guarantees bit-identical results for identical
calls (same routine, toolkit, GPU architecture/SM count), and the loop
re-launches the *same* GEMM on the same handle/stream/buffers. Any
deviation from the baseline is corruption or compute instability under
load — no tolerance involved. Tolerance-vs-f64-reference correctness stays
in the fixed-seed test.

## CPU screen — data/control flow

`cpu::run` → `run_hot_correctness`, deliberately ordered *after* the
all-core throughput run so the package is already at thermal steady state.

- One pinned worker per logical core, all concurrent (same
  pinning/Skipped rules as the isolated screen: enumeration-index scopes,
  no silent unpinned runs, panic ⇒ Failed).
- Each worker alternates for `sdc_hot_secs` wall seconds: ~5 ms FMA burn
  (`fma_gflops`, the power-heavy vector workload) → one integer + one
  float checksum round compared against the golden values computed on the
  main thread before any worker started.
- Unlike the isolated screen, mismatches are **counted** and the worker
  keeps going; the count is the signal. Emits per-core
  `cpu_sdc_hot.{mismatches,rounds}` (Count) and a per-core
  `TestId::CpuSdcHot` outcome — Failed with the first-mismatch description
  when the count is nonzero. `sdc_hot_secs = 0` ⇒ one Skipped outcome
  under Node scope. The isolated screen is unchanged; the hot screen is
  additional.

## Reporting

SDC hits are hard failures, never MAD outliers: Failed outcomes already
drive `Verdict::Stragglers` (exit code 1). Additionally
`fleet.sdc_failures` (schema v4) groups Failed outcomes of the four
correctness screens (`cpu_correctness`, `cpu_sdc_hot`,
`gpu_gemm_correctness`, `gpu_gemm_sdc`) as
`"host[:scope]: reason"`, and the terminal table renders them in a
dedicated "silent data corruption (hard failures)" section above the
fleet-relative sections. Metrics still flow through the normal aggregate
machinery; all-zero mismatch groups have zero MAD and are never flagged.

## Files
- `src/agent/gpu/sdc.rs` — pure scheduling/accounting: `CheckScheduler`
  (busy-time domain), `bitwise_mismatch`, `SdcStats`/`SdcFailure`,
  `summary_outcome`. No CUDA imports, fully unit-tested on GPU-less hosts.
- `src/agent/gpu/gemm.rs` — busy-time throughput loop, baseline capture,
  check execution, metric/outcome emission.
- `src/agent/cpu.rs` — `run_hot_correctness`, `hot_worker`,
  `HotCoreReport`.
- `src/proto.rs` — `TestId::{CpuSdcHot,GpuGemmSdc}`,
  `CpuTaskSpec::sdc_hot_secs`, `GpuTaskSpec::sdc_check_secs`
  (PROTO_VERSION 3).
- `src/config.rs` — `cpu_sdc_hot_secs` (default 10),
  `gemm_sdc_check_secs` (default 5).
- `src/report/mod.rs` — `FleetAnalysis::sdc_failures`, `render_sdc`,
  display names (SCHEMA_VERSION 4).
- Tests: `tests/gpu_tests.rs` (`mod sdc`), `tests/host_agent_tests.rs`,
  `tests/proto_tests.rs`, `tests/report_tests.rs`, unit tests in
  `src/agent/gpu/sdc.rs`.

## Invariants
- Verification cost never pollutes the sustained GFLOPS: the budget and
  the rate both run on accumulated busy time, and checks happen strictly
  between timed windows.
- The GPU comparison is bitwise; deviations are reported but no tolerance
  is applied. A NaN or length mismatch reports an infinite deviation
  internally and saturates to f64::MAX on the wire (metrics must stay
  finite).
- A run that completed zero checks is Skipped, never Passed.
- The hot screens are additive: the isolated per-core CPU screen and the
  fixed-seed GEMM correctness test are unchanged.
- Both spec fields are `#[serde(default)]`: absence means disabled, and
  v1 documents keep decoding.
- Any SDC mismatch anywhere ⇒ per-scope Failed outcome ⇒ exit code ≥ 1.
