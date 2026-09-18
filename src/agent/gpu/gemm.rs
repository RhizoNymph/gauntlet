//! GEMM correctness and sustained throughput per GPU, per dtype.
//!
//! Correctness: fixed-seed matrices (deterministic host-side PRNG), one
//! cuBLAS GEMM per dtype, result compared against an f64 CPU reference on a
//! deterministic element subsample (full f64 matmul at gemm_dim=8192 is too
//! slow; ~4096 sampled elements recomputed in f64 on the host suffice).
//! Report max relative error as `gpu_gemm_correctness.residual` per
//! GPU+dtype; outcome Failed if it exceeds the dtype tolerance:
//! f32 1e-5, tf32 5e-4, bf16 5e-2, f16 5e-3 (scaled for accumulation depth).
//! Reduced-precision results legitimately vary with tile order — tolerance
//! comparison, never bit-exactness. The residual VALUE is the straggler
//! signal fleet-wide even when it passes.
//!
//! Throughput: repeated GEMM for `gemm_secs` of loaded (busy) time; report
//! sustained (not first-iteration) GFLOPS per GPU+dtype under
//! `gpu_gemm_perf.gflops`; metric name carries the dtype: "gflops_f32",
//! "gflops_bf16", ...
//!
//! Hot SDC: during the sustained loop the output is verified bitwise against
//! a baseline every `sdc_check_secs` of busy time (`gpu_gemm_sdc`, see the
//! `sdc` module) — SDC is temperature/voltage dependent, so correctness must
//! be checked at load temperature, not just at start-of-phase. Verification
//! runs between timed windows so it never pollutes the GFLOPS figure.

use std::ffi::c_void;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use cudarc::cublas::{CudaBlas, result as cublas_result, sys as cublas_sys};
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, sys as driver_sys,
};
use half::{bf16, f16};

use crate::agent::EventSink;
use crate::agent::gpu::sdc::{self, CheckScheduler, SdcStats};
use crate::proto::{GemmDtype, GpuTaskSpec, MetricRecord, Scope, TestId, TestOutcome, Unit};

/// Per-dtype residual tolerance for correctness classification.
///
/// tf32 inputs are not host-rounded (FAST_TF32 is a permission, not a
/// guarantee), so a healthy tensor-core GEMM carries the in-kernel 10-bit
/// mantissa rounding: ~1e-3 at gemm_dim 2048. The tolerance leaves ~2x
/// headroom over that while staying an order of magnitude under f16.
pub fn residual_tolerance(dtype: GemmDtype) -> f64 {
    match dtype {
        // Accumulation error grows ~sqrt(K): a healthy GPU shows ~2.4e-5 at
        // gemm_dim 4096 (identical across nodes — it is deterministic
        // rounding, not hardware). 1e-4 keeps headroom to dim ~16k while
        // still catching real corruption, which is orders of magnitude off.
        GemmDtype::F32 => 1e-4,
        GemmDtype::Tf32 => 2e-3,
        GemmDtype::Bf16 => 5e-2,
        GemmDtype::F16 => 5e-3,
    }
}

/// Fixed seed: every node in the fleet multiplies bit-identical matrices, so
/// residuals are directly comparable across hosts.
const GEMM_SEED: u64 = 0x243F_6A88_85A3_08D3;
/// Independent fixed seed for choosing which elements of C to verify.
const SAMPLE_SEED: u64 = 0x1319_8A2E_0370_7344;
/// Elements of C recomputed in f64 on the host.
const RESIDUAL_SAMPLES: usize = 4096;
/// GEMM launches queued between stream synchronizations in the timed loop.
const SUSTAINED_BATCH: u64 = 8;
/// nvidia-smi polling period during the sustained run.
const TELEMETRY_PERIOD: Duration = Duration::from_millis(1000);

// ---------------------------------------------------------------------------
// Deterministic host PRNG
// ---------------------------------------------------------------------------

/// xorshift64* — three shifts plus a multiply. Deterministic and identical on
/// every host and target, which is the whole point: correctness is judged
/// against a CPU reference computed from these same numbers, and residuals
/// are compared across the fleet.
#[derive(Debug, Clone)]
pub struct Xorshift64(u64);

impl Xorshift64 {
    pub const fn new(seed: u64) -> Self {
        // The state must never be zero (zero is a fixed point of xorshift).
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    pub fn next_bits(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Next sample, uniformly distributed over [-0.5, 0.5).
    pub fn next_centered_unit(&mut self) -> f32 {
        // 24 bits is exactly the f32 significand, so the division is exact.
        const SCALE: f32 = 1.0 / 16_777_216.0;
        ((self.next_bits() >> 40) as u32) as f32 * SCALE - 0.5
    }
}

fn fill_matrix(rng: &mut Xorshift64, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.next_centered_unit()).collect()
}

/// Deterministic subsample of the flat indices of an `total`-element matrix.
/// A fixed stride would be a poor sample (with power-of-two dimensions it
/// visits only a couple of distinct columns), so the indices come from the
/// PRNG instead — still bit-identical on every node.
pub fn sample_indices(total: usize, wanted: usize) -> Vec<usize> {
    if total == 0 || wanted == 0 {
        return Vec::new();
    }
    let count = wanted.min(total);
    let mut rng = Xorshift64::new(SAMPLE_SEED);
    (0..count)
        .map(|_| (rng.next_bits() % total as u64) as usize)
        .collect()
}

// ---------------------------------------------------------------------------
// Residual
// ---------------------------------------------------------------------------

/// Fraction of the sample's RMS magnitude used as the denominator floor.
const RESIDUAL_FLOOR_FRACTION: f64 = 0.125;
/// Absolute backstop so an all-zero reference cannot divide by zero.
const RESIDUAL_ABS_FLOOR: f64 = 1e-12;

/// Max relative error of `actual` against the f64 reference `expected`.
///
/// The denominator is `max(|expected|, floor)`. The floor matters: dot
/// products of random centered inputs are roughly Gaussian, so a few sampled
/// elements always land near zero through cancellation, and dividing their
/// (perfectly normal, full-scale) absolute error by a near-zero reference
/// would manufacture failures on healthy hardware. Flooring at a fraction of
/// the sample's RMS magnitude keeps the metric a true relative error at full
/// scale while making it insensitive to cancellation noise.
///
/// Returns NaN if any element is NaN. `f64::max` would silently drop it (it
/// returns the non-NaN operand), and a GPU emitting NaN is exactly the failure
/// this test exists to catch, so the propagation is explicit.
pub fn max_relative_error(expected: &[f64], actual: &[f64]) -> f64 {
    debug_assert_eq!(expected.len(), actual.len());
    if expected.is_empty() {
        return 0.0;
    }
    let mean_square =
        expected.iter().map(|value| value * value).sum::<f64>() / expected.len() as f64;
    let floor = (mean_square.sqrt() * RESIDUAL_FLOOR_FRACTION).max(RESIDUAL_ABS_FLOOR);
    let mut worst = 0.0_f64;
    for (want, got) in expected.iter().zip(actual) {
        let error = (got - want).abs() / want.abs().max(floor);
        if error.is_nan() {
            return f64::NAN;
        }
        if error > worst {
            worst = error;
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// dtype plumbing
// ---------------------------------------------------------------------------

pub fn dtype_tag(dtype: GemmDtype) -> &'static str {
    match dtype {
        GemmDtype::F32 => "f32",
        GemmDtype::Tf32 => "tf32",
        GemmDtype::Bf16 => "bf16",
        GemmDtype::F16 => "f16",
    }
}

/// The value the GPU actually multiplies, given a host f32.
///
/// bf16/f16 operands are rounded on the host before upload, so the f64
/// reference uses exactly the same numbers and the residual isolates *compute*
/// error from unavoidable representation error.
///
/// tf32 is deliberately not rounded here: cuBLAS' `CUBLAS_COMPUTE_32F_FAST_TF32`
/// is a permission, not a guarantee — on pre-Ampere parts (or when cuBLAS picks
/// a non-tensor kernel) the inputs stay full f32. Rounding the reference would
/// then be wrong in the opposite direction, so the reference keeps the exact
/// f32 inputs and the wider tf32 tolerance absorbs the in-kernel rounding.
fn round_to_dtype(value: f32, dtype: GemmDtype) -> f32 {
    match dtype {
        GemmDtype::F32 | GemmDtype::Tf32 => value,
        GemmDtype::Bf16 => bf16::from_f32(value).to_f32(),
        GemmDtype::F16 => f16::from_f32(value).to_f32(),
    }
}

/// Device-side A and B. 16-bit operands live in `u16` buffers holding the raw
/// bit patterns; cuBLAS is told the element type through `cudaDataType_t`, so
/// there is no need for cudarc's optional `f16` feature.
enum Operands {
    Wide {
        a: CudaSlice<f32>,
        b: CudaSlice<f32>,
    },
    Narrow {
        a: CudaSlice<u16>,
        b: CudaSlice<u16>,
    },
}

impl Operands {
    fn device_ptrs(
        &self,
        stream: &CudaStream,
    ) -> (driver_sys::CUdeviceptr, driver_sys::CUdeviceptr) {
        match self {
            Operands::Wide { a, b } => (read_ptr(a, stream), read_ptr(b, stream)),
            Operands::Narrow { a, b } => (read_ptr(a, stream), read_ptr(b, stream)),
        }
    }
}

/// Raw device pointer for a buffer handed to cuBLAS.
///
/// All work in this module runs on the single stream the cuBLAS handle is
/// bound to, so stream ordering alone keeps the pointer valid for every launch
/// queued on it; the cross-stream sync guard cudarc hands back is therefore
/// dropped immediately.
fn read_ptr<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> driver_sys::CUdeviceptr {
    let (ptr, _sync) = slice.device_ptr(stream);
    ptr
}

fn write_ptr<T>(slice: &mut CudaSlice<T>, stream: &CudaStream) -> driver_sys::CUdeviceptr {
    let (ptr, _sync) = slice.device_ptr_mut(stream);
    ptr
}

fn upload_operands(
    stream: &Arc<CudaStream>,
    host_a: &[f32],
    host_b: &[f32],
    dtype: GemmDtype,
) -> Result<Operands> {
    let operands = match dtype {
        GemmDtype::F32 | GemmDtype::Tf32 => Operands::Wide {
            a: stream.clone_htod(host_a)?,
            b: stream.clone_htod(host_b)?,
        },
        GemmDtype::Bf16 => {
            let a: Vec<u16> = host_a
                .iter()
                .map(|&v| bf16::from_f32(v).to_bits())
                .collect();
            let b: Vec<u16> = host_b
                .iter()
                .map(|&v| bf16::from_f32(v).to_bits())
                .collect();
            Operands::Narrow {
                a: stream.clone_htod(&a)?,
                b: stream.clone_htod(&b)?,
            }
        }
        GemmDtype::F16 => {
            let a: Vec<u16> = host_a.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
            let b: Vec<u16> = host_b.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
            Operands::Narrow {
                a: stream.clone_htod(&a)?,
                b: stream.clone_htod(&b)?,
            }
        }
    };
    Ok(operands)
}

/// `(operand element type, compute type)` handed to `cublasGemmEx`.
///
/// - f32   : CUDA_R_32F operands, CUBLAS_COMPUTE_32F — true IEEE fp32, tensor
///   cores are *not* allowed to substitute tf32 for this compute type.
/// - tf32  : CUDA_R_32F operands, CUBLAS_COMPUTE_32F_FAST_TF32 — cuBLAS rounds
///   the operands to tf32 in-kernel and accumulates in fp32.
/// - bf16  : CUDA_R_16BF operands, CUBLAS_COMPUTE_32F — fp32 accumulation.
/// - f16   : CUDA_R_16F operands, CUBLAS_COMPUTE_32F — fp32 accumulation
///   (never CUBLAS_COMPUTE_16F, whose accumulator would overflow at these
///   accumulation depths and would be measuring the wrong thing).
///
/// C is always CUDA_R_32F so the download and the host comparison have a
/// single code path.
fn gemm_types(dtype: GemmDtype) -> (cublas_sys::cudaDataType_t, cublas_sys::cublasComputeType_t) {
    match dtype {
        GemmDtype::F32 => (
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        ),
        GemmDtype::Tf32 => (
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
        ),
        GemmDtype::Bf16 => (
            cublas_sys::cudaDataType_t::CUDA_R_16BF,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        ),
        GemmDtype::F16 => (
            cublas_sys::cudaDataType_t::CUDA_R_16F,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        ),
    }
}

/// Queue one `n x n` GEMM: C = A·B with row-major host semantics.
///
/// cuBLAS is column-major while the host matrices are row-major. Passing the
/// operands swapped (B first) with every leading dimension `n` computes, in
/// column-major terms, Cᵀ = Bᵀ·Aᵀ — which is bit-for-bit the row-major C = A·B
/// we want, with no transposes and no extra copies.
///
/// # Safety
/// `a_ptr` and `b_ptr` must each address at least `n*n` elements of the
/// operand type selected by `dtype`, and `c_ptr` at least `n*n` `f32`, all
/// allocated in the context bound to `blas`.
unsafe fn launch_gemm(
    blas: &CudaBlas,
    dtype: GemmDtype,
    n: i32,
    a_ptr: driver_sys::CUdeviceptr,
    b_ptr: driver_sys::CUdeviceptr,
    c_ptr: driver_sys::CUdeviceptr,
) -> Result<(), cublas_result::CublasError> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let (io_type, compute_type) = gemm_types(dtype);
    // SAFETY: the caller guarantees the three pointers are live device
    // allocations of at least n*n elements of the matching type in this
    // handle's context; alpha/beta are host f32 as required by a 32F compute
    // type with CUBLAS_POINTER_MODE_HOST (cudarc never changes the pointer
    // mode on a fresh handle).
    unsafe {
        cublas_result::gemm_ex(
            *blas.handle(),
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            n,
            n,
            n,
            std::ptr::from_ref(&alpha).cast::<c_void>(),
            b_ptr as *const c_void,
            io_type,
            n,
            a_ptr as *const c_void,
            io_type,
            n,
            std::ptr::from_ref(&beta).cast::<c_void>(),
            c_ptr as *mut c_void,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            n,
            compute_type,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
}

/// f64 recomputation of one element of C from the host operands.
fn reference_element(
    host_a: &[f32],
    host_b: &[f32],
    n: usize,
    row: usize,
    col: usize,
    dtype: GemmDtype,
) -> f64 {
    let a_row = &host_a[row * n..row * n + n];
    let mut acc = 0.0_f64;
    for (k, &a_value) in a_row.iter().enumerate() {
        let a = f64::from(round_to_dtype(a_value, dtype));
        let b = f64::from(round_to_dtype(host_b[k * n + col], dtype));
        acc += a * b;
    }
    acc
}

// ---------------------------------------------------------------------------
// Clock / temperature telemetry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct ClockSample {
    clock_mhz: f64,
    temp_c: f64,
}

/// Background nvidia-smi poller. Absent nvidia-smi means no telemetry, which
/// is a silent skip: the GEMM numbers are still the primary signal.
struct ClockSampler {
    stop: Arc<AtomicBool>,
    worker: JoinHandle<Vec<ClockSample>>,
}

impl ClockSampler {
    fn start(gpu_index: u32) -> Option<Self> {
        query_clocks(gpu_index)?;
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let worker = std::thread::spawn(move || {
            let mut samples = Vec::new();
            while !flag.load(Ordering::Relaxed) {
                if let Some(sample) = query_clocks(gpu_index) {
                    samples.push(sample);
                }
                // Wake often enough that stopping is prompt, sample ~1 Hz.
                let deadline = Instant::now() + TELEMETRY_PERIOD;
                while Instant::now() < deadline && !flag.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            samples
        });
        Some(Self { stop, worker })
    }

    fn finish(self) -> Vec<ClockSample> {
        self.stop.store(true, Ordering::Relaxed);
        self.worker.join().unwrap_or_default()
    }
}

fn query_clocks(gpu_index: u32) -> Option<ClockSample> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=clocks.sm,temperature.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            &gpu_index.to_string(),
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut fields = text.lines().next()?.split(',');
    let clock_mhz = fields.next()?.trim().parse::<f64>().ok()?;
    let temp_c = fields.next()?.trim().parse::<f64>().ok()?;
    Some(ClockSample { clock_mhz, temp_c })
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run_on_gpu(sink: &EventSink, gpu_index: u32, spec: &GpuTaskSpec) -> Result<()> {
    let n = spec.gemm_dim.max(1) as usize;
    let elements = n * n;
    let scope = Scope::Gpu { index: gpu_index };
    let dtypes: &[GemmDtype] = if spec.gemm_dtypes.is_empty() {
        &[GemmDtype::F32]
    } else {
        &spec.gemm_dtypes
    };

    let mut rng = Xorshift64::new(GEMM_SEED);
    let host_a = fill_matrix(&mut rng, elements);
    let host_b = fill_matrix(&mut rng, elements);

    let ctx = CudaContext::new(gpu_index as usize)
        .with_context(|| format!("creating cuda context for gpu {gpu_index}"))?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(Arc::clone(&stream)).context("creating cublas handle")?;
    let mut c_dev = stream.alloc_zeros::<f32>(elements)?;

    // Pass 1: correctness. Operands are re-uploaded per dtype rather than kept
    // resident so peak device memory stays at one dtype's worth.
    let indices = sample_indices(elements, RESIDUAL_SAMPLES);
    for &dtype in dtypes {
        let tag = dtype_tag(dtype);
        let operands = upload_operands(&stream, &host_a, &host_b, dtype)?;
        let (a_ptr, b_ptr) = operands.device_ptrs(&stream);
        let c_ptr = write_ptr(&mut c_dev, &stream);
        // SAFETY: a_ptr/b_ptr address n*n operands of `dtype` and c_ptr n*n
        // f32, all just allocated on this context's stream, which is also the
        // handle's stream.
        unsafe { launch_gemm(&blas, dtype, n as i32, a_ptr, b_ptr, c_ptr) }
            .with_context(|| format!("cublas gemm ({tag})"))?;
        stream.synchronize()?;
        let c_host = stream.clone_dtoh(&c_dev)?;
        drop(operands);

        let expected: Vec<f64> = indices
            .iter()
            .map(|&index| reference_element(&host_a, &host_b, n, index / n, index % n, dtype))
            .collect();
        let actual: Vec<f64> = indices
            .iter()
            .map(|&index| f64::from(c_host[index]))
            .collect();
        let residual = max_relative_error(&expected, &actual);
        let tolerance = residual_tolerance(dtype);
        // A non-finite residual means C contained NaN/Inf. It cannot be put on
        // the wire (serde_json renders non-finite floats as `null`, which will
        // not decode back into an f64), so it is reported as a saturated value
        // plus an explicit Failed reason.
        let (reported, outcome) = if !residual.is_finite() {
            (
                f64::MAX,
                TestOutcome::Failed {
                    reason: format!("{tag} gemm produced a non-finite result"),
                },
            )
        } else if residual <= tolerance {
            (residual, TestOutcome::Passed)
        } else {
            (
                residual,
                TestOutcome::Failed {
                    reason: format!(
                        "{tag} residual {residual:.3e} exceeds tolerance {tolerance:.3e}"
                    ),
                },
            )
        };

        // The residual value is emitted whether it passes or not: it is the
        // fleet-relative straggler signal.
        sink.metric(MetricRecord {
            test: TestId::GpuGemmCorrectness,
            scope: scope.clone(),
            name: format!("residual_{tag}"),
            value: reported,
            unit: Unit::Residual,
            repeat: 0,
        });
        sink.outcome(TestId::GpuGemmCorrectness, scope.clone(), outcome);
    }

    // Pass 2: sustained throughput, with clock/temperature sampling spanning
    // the whole loaded window so clock_mhz_end << clock_mhz_start reads as
    // thermal or power throttling. The sampler owns a thread, so it is always
    // stopped and joined before the result is propagated.
    let sampler = ClockSampler::start(gpu_index);
    let sustained = sustained_pass(
        sink, &scope, &blas, &stream, &mut c_dev, &host_a, &host_b, dtypes, n, spec, gpu_index,
    );
    let samples = sampler.map(ClockSampler::finish).unwrap_or_default();

    if let (Some(first), Some(last)) = (samples.first(), samples.last()) {
        let hottest = samples
            .iter()
            .map(|sample| sample.temp_c)
            .fold(f64::NEG_INFINITY, f64::max);
        for (name, value, unit) in [
            ("clock_mhz_start", first.clock_mhz, Unit::Mhz),
            ("clock_mhz_end", last.clock_mhz, Unit::Mhz),
            ("temp_c_max", hottest, Unit::Celsius),
        ] {
            sink.metric(MetricRecord {
                test: TestId::GpuGemmPerf,
                scope: scope.clone(),
                name: name.to_string(),
                value,
                unit,
                repeat: 0,
            });
        }
    }

    // Telemetry is emitted even when a dtype's sustained run failed: the
    // clocks are what explain the failure.
    let sdc_summary = sustained?;
    // Hot-SDC verdict: any bitwise mismatch during the loaded window is a
    // hard per-GPU failure, with the clock/temperature at each failure in
    // the reason.
    sink.outcome(
        TestId::GpuGemmSdc,
        scope.clone(),
        sdc::summary_outcome(spec.sdc_check_secs > 0, &sdc_summary),
    );
    sink.outcome(TestId::GpuGemmPerf, scope, TestOutcome::Passed);
    Ok(())
}

/// The sustained-throughput half of `run_on_gpu`, split out so the telemetry
/// thread is always joined regardless of how this returns. Returns the hot
/// SDC accounting per dtype for the per-GPU verdict.
#[allow(clippy::too_many_arguments)]
fn sustained_pass(
    sink: &EventSink,
    scope: &Scope,
    blas: &CudaBlas,
    stream: &Arc<CudaStream>,
    c_dev: &mut CudaSlice<f32>,
    host_a: &[f32],
    host_b: &[f32],
    dtypes: &[GemmDtype],
    n: usize,
    spec: &GpuTaskSpec,
    gpu_index: u32,
) -> Result<Vec<(&'static str, SdcStats)>> {
    let mut summary = Vec::with_capacity(dtypes.len());
    for &dtype in dtypes {
        let tag = dtype_tag(dtype);
        let operands = upload_operands(stream, host_a, host_b, dtype)?;
        let (gflops, stats) = sustained_gflops(
            blas,
            stream,
            dtype,
            n,
            spec.gemm_secs,
            &operands,
            c_dev,
            Duration::from_secs(spec.sdc_check_secs),
            gpu_index,
        )
        .with_context(|| format!("sustained gemm ({tag})"))?;
        drop(operands);
        sink.metric(MetricRecord {
            test: TestId::GpuGemmPerf,
            scope: scope.clone(),
            name: format!("gflops_{tag}"),
            value: gflops,
            unit: Unit::Gflops,
            repeat: 0,
        });
        // Hot-SDC accounting per dtype: check and mismatch counts plus the
        // worst absolute deviation (0.0 when clean, saturated when the
        // deviation was unquantifiable).
        for (name, value, unit) in [
            (format!("checks_{tag}"), stats.checks as f64, Unit::Count),
            (
                format!("mismatches_{tag}"),
                stats.mismatches as f64,
                Unit::Count,
            ),
            (
                format!("max_abs_dev_{tag}"),
                stats.reportable_max_abs_dev(),
                Unit::Residual,
            ),
        ] {
            sink.metric(MetricRecord {
                test: TestId::GpuGemmSdc,
                scope: scope.clone(),
                name,
                value,
                unit,
                repeat: 0,
            });
        }
        summary.push((tag, stats));
    }
    Ok(summary)
}

/// Repeat the GEMM for `secs` loaded seconds and report sustained GFLOPS,
/// verifying the hot output bitwise against a baseline every
/// `check_interval` of loaded time.
///
/// Launches are queued in batches so kernel-launch latency does not dominate,
/// and the stream is synchronized once per batch so the elapsed wall time
/// covers completed work only — the reported number is steady-state, not the
/// first (cold-clock) iteration.
///
/// Throughput accounting: the loop accumulates *busy* time (launch +
/// synchronize windows only) and both the budget and the reported GFLOPS
/// run against it, so verification overhead — the C download, the host
/// compare, the telemetry query on failure — never pollutes the number.
///
/// The baseline is C after the first (warm-up) GEMM: cuBLAS produces
/// bit-identical output for identical calls on the same GPU, so any later
/// deviation is corruption under load (see `sdc` module docs).
#[allow(clippy::too_many_arguments)]
fn sustained_gflops(
    blas: &CudaBlas,
    stream: &Arc<CudaStream>,
    dtype: GemmDtype,
    n: usize,
    secs: u64,
    operands: &Operands,
    c_dev: &mut CudaSlice<f32>,
    check_interval: Duration,
    gpu_index: u32,
) -> Result<(f64, SdcStats)> {
    let (a_ptr, b_ptr) = operands.device_ptrs(stream);
    let c_ptr = write_ptr(c_dev, stream);
    // SAFETY (all launches): pointers come from live allocations on this
    // stream's context, sized n*n as required by `launch_gemm`.
    unsafe { launch_gemm(blas, dtype, n as i32, a_ptr, b_ptr, c_ptr) }?;
    stream.synchronize()?;

    let mut scheduler = CheckScheduler::new(check_interval);
    let mut stats = SdcStats::default();
    // The baseline download is skipped entirely when checks are disabled,
    // so the disabled path costs nothing.
    let baseline: Option<Vec<f32>> = if scheduler.enabled() {
        Some(stream.clone_dtoh(c_dev)?)
    } else {
        None
    };

    let budget = Duration::from_secs(secs.max(1));
    let mut busy = Duration::ZERO;
    let mut iters = 0_u64;
    while busy < budget {
        let window = Instant::now();
        for _ in 0..SUSTAINED_BATCH {
            unsafe { launch_gemm(blas, dtype, n as i32, a_ptr, b_ptr, c_ptr) }?;
        }
        stream.synchronize()?;
        busy += window.elapsed();
        iters += SUSTAINED_BATCH;

        if let Some(baseline) = &baseline
            && scheduler.due(busy)
        {
            // Outside the timed window: this cost is excluded from `busy`.
            let current = stream.clone_dtoh(c_dev)?;
            let mismatch = sdc::bitwise_mismatch(baseline, &current);
            // Clock/temperature are captured at the moment of failure —
            // that context is the point of checking hot.
            let telemetry = if mismatch.is_some() {
                query_clocks(gpu_index)
            } else {
                None
            };
            stats.record(
                mismatch,
                telemetry.map(|sample| sample.clock_mhz),
                telemetry.map(|sample| sample.temp_c),
            );
        }
    }
    let elapsed = busy.as_secs_f64().max(1e-9);
    Ok((sustained_gflops_value(n, iters, elapsed), stats))
}

/// 2·n³ flops per GEMM (one multiply and one add per inner-product term).
fn sustained_gflops_value(n: usize, iters: u64, elapsed_secs: f64) -> f64 {
    2.0 * (n as f64).powi(3) * iters as f64 / elapsed_secs / 1e9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prng_is_deterministic_and_centered() {
        let first: Vec<f32> = {
            let mut rng = Xorshift64::new(GEMM_SEED);
            (0..4096).map(|_| rng.next_centered_unit()).collect()
        };
        let second: Vec<f32> = {
            let mut rng = Xorshift64::new(GEMM_SEED);
            (0..4096).map(|_| rng.next_centered_unit()).collect()
        };
        assert_eq!(first, second, "same seed must reproduce the same stream");

        assert!(
            first.iter().all(|value| (-0.5..=0.5).contains(value)),
            "samples must stay inside [-0.5, 0.5]"
        );
        let mean = first.iter().map(|value| f64::from(*value)).sum::<f64>() / first.len() as f64;
        assert!(mean.abs() < 0.02, "mean {mean} is not centered");
        let distinct: std::collections::BTreeSet<u32> =
            first.iter().map(|value| value.to_bits()).collect();
        assert!(distinct.len() > 4000, "stream is degenerate");
    }

    #[test]
    fn prng_survives_a_zero_seed() {
        let mut rng = Xorshift64::new(0);
        assert_ne!(rng.next_bits(), 0);
    }

    #[test]
    fn sample_indices_are_deterministic_and_in_range() {
        let total = 2048 * 2048;
        let first = sample_indices(total, RESIDUAL_SAMPLES);
        let second = sample_indices(total, RESIDUAL_SAMPLES);
        assert_eq!(first, second);
        assert_eq!(first.len(), RESIDUAL_SAMPLES);
        assert!(first.iter().all(|index| *index < total));

        // The sample must not collapse onto a handful of rows or columns.
        let rows: std::collections::BTreeSet<usize> =
            first.iter().map(|index| index / 2048).collect();
        let cols: std::collections::BTreeSet<usize> =
            first.iter().map(|index| index % 2048).collect();
        assert!(rows.len() > 1000, "only {} distinct rows", rows.len());
        assert!(cols.len() > 1000, "only {} distinct cols", cols.len());
    }

    #[test]
    fn sample_indices_handle_degenerate_sizes() {
        assert!(sample_indices(0, 16).is_empty());
        assert!(sample_indices(16, 0).is_empty());
        assert_eq!(sample_indices(4, 16).len(), 4);
    }

    #[test]
    fn residual_is_zero_for_an_exact_match() {
        let expected = vec![3.0, -2.0, 7.5, 0.0];
        assert_eq!(max_relative_error(&expected, &expected), 0.0);
        assert_eq!(max_relative_error(&[], &[]), 0.0);
    }

    #[test]
    fn residual_is_the_max_relative_error_at_full_scale() {
        let expected = vec![100.0, 100.0, 100.0, 100.0];
        let actual = vec![100.0, 101.0, 100.0, 99.5];
        // Floor is 0.125 * 100 = 12.5, so every |expected| dominates it.
        assert!((max_relative_error(&expected, &actual) - 0.01).abs() < 1e-12);
    }

    #[test]
    fn residual_floor_tames_cancellation_near_zero() {
        // One reference element cancels to ~0 while carrying a normal absolute
        // error. Without the floor this would be a relative error of 1e6.
        let expected = vec![8.0, -8.0, 1e-6, 8.0];
        let actual = vec![8.0, -8.0, 1e-6 + 1e-3, 8.0];
        let residual = max_relative_error(&expected, &actual);
        // rms ~ 6.93, floor ~ 0.866 => 1e-3 / 0.866
        assert!(residual < 2e-3, "residual {residual} was not floored");
    }

    #[test]
    fn residual_still_catches_a_broken_gpu() {
        let expected = vec![4.0, -4.0, 4.0, -4.0];
        let actual = vec![4.0, -4.0, 0.0, -4.0];
        assert!(max_relative_error(&expected, &actual) >= 1.0);
    }

    #[test]
    fn residual_propagates_non_finite_results() {
        // f64::max would swallow these; a NaN-producing GPU must not pass.
        assert!(max_relative_error(&[1.0, 1.0], &[1.0, f64::NAN]).is_nan());
        assert!(max_relative_error(&[1.0, 1.0], &[1.0, f64::INFINITY]).is_infinite());
        assert!(max_relative_error(&[1.0, f64::NAN], &[1.0, 1.0]).is_nan());
    }

    #[test]
    fn dtype_rounding_matches_the_target_format() {
        // 1/3 is not representable in any of these formats; the rounded value
        // must differ from f32 for the 16-bit ones and match it otherwise.
        let value = 1.0_f32 / 3.0;
        assert_eq!(round_to_dtype(value, GemmDtype::F32), value);
        assert_eq!(round_to_dtype(value, GemmDtype::Tf32), value);
        assert_ne!(round_to_dtype(value, GemmDtype::Bf16), value);
        assert_ne!(round_to_dtype(value, GemmDtype::F16), value);
        // bf16 keeps fewer mantissa bits than f16, so it is further off.
        let bf16_error =
            (f64::from(round_to_dtype(value, GemmDtype::Bf16)) - f64::from(value)).abs();
        let f16_error = (f64::from(round_to_dtype(value, GemmDtype::F16)) - f64::from(value)).abs();
        assert!(bf16_error > f16_error);
        // Exactly representable values round-trip in every format.
        for dtype in [
            GemmDtype::F32,
            GemmDtype::Tf32,
            GemmDtype::Bf16,
            GemmDtype::F16,
        ] {
            assert_eq!(round_to_dtype(0.5, dtype), 0.5);
            assert_eq!(round_to_dtype(-0.25, dtype), -0.25);
        }
    }

    #[test]
    fn reference_element_matches_a_hand_computed_product() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]] row-major.
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        assert_eq!(reference_element(&a, &b, 2, 0, 0, GemmDtype::F32), 19.0);
        assert_eq!(reference_element(&a, &b, 2, 0, 1, GemmDtype::F32), 22.0);
        assert_eq!(reference_element(&a, &b, 2, 1, 0, GemmDtype::F32), 43.0);
        assert_eq!(reference_element(&a, &b, 2, 1, 1, GemmDtype::F32), 50.0);
    }

    #[test]
    fn sustained_gflops_uses_two_flops_per_multiply_add() {
        // 1000^3 = 1e9 multiply-adds = 2e9 flops per GEMM; ten of them in 2 s
        // is 2e10 flops / 2 s = 1e10 flop/s = 10 GFLOPS.
        let gflops = sustained_gflops_value(1000, 10, 2.0);
        assert!((gflops - 10.0).abs() < 1e-9, "{gflops}");
        // Twice the work in the same time doubles the rate.
        assert!((sustained_gflops_value(1000, 20, 2.0) - 20.0).abs() < 1e-9);
        // A 2x larger dimension is 8x the work.
        assert!((sustained_gflops_value(2000, 10, 2.0) - 80.0).abs() < 1e-9);
    }
}
