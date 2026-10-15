//! Phase 2: per-GPU tests via cudarc (pure `dynamic-loading`: libcuda /
//! libcublas are dlopened at runtime, so the binary builds and runs on
//! GPU-less machines and only this phase needs the driver stack).
//!
//! Driver init is classified, never just stringified
//! (`device_availability`): a node without a usable GPU — no NVIDIA
//! hardware, or `CUDA_ERROR_NO_DEVICE` / `NOT_INITIALIZED` / an init-time
//! load failure — records every GPU test Skipped with a reason naming the
//! PCI GPUs, because the missing driver is already the inventory-derived
//! `gpu_driver` finding (one root cause, one hard finding). A genuine CUDA
//! error on a working driver stays Failed, as an infrastructure failure
//! (`TestOutcome::infrastructure_failure`) so it never lands in the SDC
//! section.

pub mod bandwidth;
pub mod gemm;
pub mod intranode;
pub(crate) mod node_comm;
pub mod overlap;
pub mod p2p;
pub mod sdc;
pub mod worker;

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use anyhow::Result;
use cudarc::driver::sys::CUresult;
use cudarc::driver::{CudaContext, DriverError};
use thiserror::Error;

use crate::agent::EventSink;
use crate::agent::gpu_driver::{self, DeviceAvailability, unavailable_reason};
use crate::agent::pci::PCI_DEVICES_ROOT;
use crate::proto::{GpuTaskSpec, LogLevel, PciBusId, Scope, TestId, TestOutcome};

/// Run a CUDA-touching closure, converting *any* failure into a string.
///
/// cudarc's `dynamic-loading` backend resolves symbols lazily and **panics**
/// (`Missing symbol ...` / `panic_no_lib_found`) when libcuda / libcublas is
/// not installed, instead of returning a `DriverError`. A node with a broken
/// or absent driver stack is a finding to report, never an agent crash, so
/// every entry into cudarc goes through here.
pub(crate) fn guard<T>(what: &str, body: impl FnOnce() -> Result<T>) -> Result<T, String> {
    guard_typed(what, body).map_err(|error| error.to_string())
}

/// How a guarded CUDA-touching closure failed.
#[derive(Debug)]
pub(crate) enum GuardFailure {
    /// It returned an error (a cudarc `DriverError` stays downcastable).
    Error(anyhow::Error),
    /// cudarc panicked: libcuda / a symbol could not be loaded.
    Panicked(String),
}

/// A guarded failure, still typed: `Display` is exactly `guard`'s string.
#[derive(Debug, Error)]
#[error("{what}: {}", render_failure(.failure))]
pub(crate) struct GuardError {
    what: String,
    failure: GuardFailure,
}

/// `{error:#}` (the cause chain joined by ": "), except that a cudarc
/// `DriverError` renders without risking a panic: its `Display` calls
/// `cuGetErrorString` through the dynamically loaded libcuda, which panics
/// when the library cannot be loaded. The code's name always survives.
fn render_failure(failure: &GuardFailure) -> String {
    match failure {
        GuardFailure::Error(error) => error
            .chain()
            .map(|cause| match cause.downcast_ref::<DriverError>() {
                Some(driver) => render_driver_error(driver),
                None => cause.to_string(),
            })
            .collect::<Vec<_>>()
            .join(": "),
        GuardFailure::Panicked(text) => format!("panicked: {text}"),
    }
}

fn render_driver_error(error: &DriverError) -> String {
    catch_unwind(AssertUnwindSafe(|| error.to_string()))
        .unwrap_or_else(|_| format!("DriverError({:?})", error.0))
}

impl GuardError {
    /// Classify a failure of the driver-init call (`cuInit` +
    /// `cuDeviceGetCount`): a loader panic there means libcuda (or the
    /// driver behind it) could not load.
    pub(crate) fn init_class(&self) -> CudaErrorClass {
        match &self.failure {
            GuardFailure::Panicked(_) => CudaErrorClass::Unavailable,
            GuardFailure::Error(error) => error_class(error),
        }
    }

    /// Classify a failure after init succeeded: only the driver's own
    /// "no device" codes mean unavailable; a loader panic now (libcublas
    /// missing, a symbol the driver lacks) is a fault of the stack.
    pub(crate) fn runtime_class(&self) -> CudaErrorClass {
        match &self.failure {
            GuardFailure::Panicked(_) => CudaErrorClass::Fault,
            GuardFailure::Error(error) => error_class(error),
        }
    }
}

/// `guard`, keeping the failure typed for classification.
pub(crate) fn guard_typed<T>(
    what: &str,
    body: impl FnOnce() -> Result<T>,
) -> Result<T, GuardError> {
    let failure = match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(value)) => return Ok(value),
        Ok(Err(error)) => GuardFailure::Error(error),
        Err(payload) => GuardFailure::Panicked(panic_text(payload)),
    };
    Err(GuardError {
        what: what.to_string(),
        failure,
    })
}

/// Whether a CUDA failure means "there is no usable GPU here" or a genuine
/// fault of a working driver stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaErrorClass {
    /// No device, or a driver that cannot initialize: GPU tests skip; the
    /// driver is the `gpu_driver` finding.
    Unavailable,
    /// Anything else: the test fails.
    Fault,
}

/// Pure classification of a cudarc driver error by its code.
/// `NO_DEVICE` / `NOT_INITIALIZED` / the init-time load failures
/// (`STUB_LIBRARY`: the toolkit's stub libcuda instead of the driver's;
/// `SYSTEM_DRIVER_MISMATCH`: libcuda and the kernel module disagree) mean
/// unavailable; every other code is a fault.
pub(crate) fn classify_driver_error(error: &DriverError) -> CudaErrorClass {
    match error.0 {
        CUresult::CUDA_ERROR_NO_DEVICE
        | CUresult::CUDA_ERROR_NOT_INITIALIZED
        | CUresult::CUDA_ERROR_STUB_LIBRARY
        | CUresult::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH => CudaErrorClass::Unavailable,
        _ => CudaErrorClass::Fault,
    }
}

/// An error that is (or wraps) a cudarc `DriverError` classifies by its
/// code; any other error (cuBLAS, NCCL, ours) is a fault.
fn error_class(error: &anyhow::Error) -> CudaErrorClass {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<DriverError>())
        .map_or(CudaErrorClass::Fault, classify_driver_error)
}

/// What the GPU-touching phases can do on this node: CUDA's device count,
/// classified, with the PCI bus naming the hardware in a Skipped reason.
pub(crate) fn device_availability() -> DeviceAvailability {
    let pci_gpus = gpu_driver::nvidia_pci_gpus(Path::new(PCI_DEVICES_ROOT));
    availability_from(
        guard_typed("cuda driver init", || Ok(CudaContext::device_count()?)),
        &pci_gpus,
    )
}

/// Pure core of `device_availability`.
pub(crate) fn availability_from(
    count: Result<i32, GuardError>,
    pci_gpus: &[PciBusId],
) -> DeviceAvailability {
    match count {
        Ok(count) => DeviceAvailability::from_count(u32::try_from(count).unwrap_or(0), pci_gpus),
        Err(error) => match error.init_class() {
            CudaErrorClass::Unavailable => DeviceAvailability::Unavailable {
                reason: unavailable_reason(pci_gpus, &error.to_string()),
            },
            CudaErrorClass::Fault => DeviceAvailability::Fault {
                reason: error.to_string(),
            },
        },
    }
}

/// The phase-2 tests, in emission order. All of them skip together on a
/// node without a usable GPU.
const GPU_PHASE_TESTS: [TestId; 5] = [
    TestId::GpuGemmCorrectness,
    TestId::GpuGemmPerf,
    TestId::GpuGemmSdc,
    TestId::GpuMemBandwidth,
    TestId::GpuP2p,
];

/// The per-GPU GEMM tests (`gemm::run_on_gpu` emits all three).
const GEMM_TESTS: [TestId; 3] = [
    TestId::GpuGemmCorrectness,
    TestId::GpuGemmPerf,
    TestId::GpuGemmSdc,
];

/// Outcome of a GPU test whose run failed after init: Skipped when the
/// GPU became unusable, otherwise an infrastructure failure (never an SDC
/// finding: the test could not produce a verdict).
pub(super) fn runtime_failure_outcome(error: &GuardError) -> TestOutcome {
    match error.runtime_class() {
        CudaErrorClass::Unavailable => TestOutcome::Skipped {
            reason: format!("no usable GPU (driver unavailable): {error}"),
        },
        CudaErrorClass::Fault => TestOutcome::infrastructure_failure(error),
    }
}

fn panic_text(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// Run GEMM correctness + sustained perf, memory bandwidth, and p2p tests
/// on every visible GPU. Clock/temperature sampling runs alongside the
/// sustained GEMM (via NVML or nvidia-smi polling) to expose thermal
/// throttling: emit `gpu_gemm_perf.clock_mhz_start` / `clock_mhz_end` /
/// `temp_c_max` per GPU.
pub fn run(sink: &EventSink, spec: &GpuTaskSpec) -> Result<()> {
    run_with(sink, spec, device_availability())
}

/// `run` on an already classified device probe.
pub(crate) fn run_with(
    sink: &EventSink,
    spec: &GpuTaskSpec,
    availability: DeviceAvailability,
) -> Result<()> {
    let device_count = match availability {
        DeviceAvailability::Usable(count) => count.get(),
        // No usable GPU (no NVIDIA hardware, or a driver that cannot drive
        // it): every GPU test skips. The driver is the inventory-derived
        // `gpu_driver` finding, so one root cause is one hard finding.
        DeviceAvailability::Unavailable { reason } => {
            for test in GPU_PHASE_TESTS {
                sink.outcome(
                    test,
                    Scope::Node,
                    TestOutcome::Skipped {
                        reason: reason.clone(),
                    },
                );
            }
            return Ok(());
        }
        // A working driver stack that still failed to initialize is a
        // finding, but an infrastructure one: the screens never ran.
        DeviceAvailability::Fault { reason } => {
            sink.outcome(
                TestId::GpuGemmCorrectness,
                Scope::Node,
                TestOutcome::infrastructure_failure(reason),
            );
            return Ok(());
        }
    };

    sink.log(
        LogLevel::Info,
        format!("gpu phase: {device_count} device(s) visible"),
    );

    // One GPU failing must not stop the rest: each test is caught, reported
    // under that GPU's scope, and the loop continues.
    for index in 0..device_count {
        let scope = Scope::Gpu { index };
        if let Err(error) = guard_typed("gemm", || gemm::run_on_gpu(sink, index, spec)) {
            let outcome = runtime_failure_outcome(&error);
            match outcome {
                TestOutcome::Skipped { .. } => {
                    for test in GEMM_TESTS {
                        sink.outcome(test, scope.clone(), outcome.clone());
                    }
                }
                _ => sink.outcome(TestId::GpuGemmCorrectness, scope.clone(), outcome),
            }
        }
        if let Err(error) = guard_typed("bandwidth", || {
            bandwidth::run_on_gpu(sink, index, spec.bandwidth_bytes)
        }) {
            sink.outcome(
                TestId::GpuMemBandwidth,
                scope,
                runtime_failure_outcome(&error),
            );
        }
    }

    if let Err(error) = guard_typed("p2p", || p2p::run_all_pairs(sink, spec.bandwidth_bytes)) {
        sink.outcome(TestId::GpuP2p, Scope::Node, runtime_failure_outcome(&error));
    }

    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn guard_reports_errors_without_propagating() {
        let error = guard("thing", || -> Result<()> { anyhow::bail!("boom") })
            .expect_err("error must be captured");
        assert!(error.contains("thing"), "{error}");
        assert!(error.contains("boom"), "{error}");
    }

    #[test]
    fn guard_converts_panics_into_findings() {
        let error = guard("thing", || -> Result<()> {
            panic!("missing symbol cuInit")
        })
        .expect_err("panic");
        assert!(error.contains("panicked"), "{error}");
        assert!(error.contains("cuInit"), "{error}");
    }

    #[test]
    fn guard_passes_values_through() {
        assert_eq!(guard("thing", || Ok(7u32)), Ok(7));
    }

    fn driver_error(code: CUresult) -> GuardError {
        guard_typed("cuda driver init", || -> Result<i32> {
            Err(DriverError(code).into())
        })
        .expect_err("error")
    }

    fn loader_panic() -> GuardError {
        guard_typed("cuda driver init", || -> Result<i32> {
            panic!("Unable to dynamically load the \"cuda\" shared library")
        })
        .expect_err("panic")
    }

    fn bus(raw: &str) -> PciBusId {
        raw.parse().expect("bus id")
    }

    #[test]
    fn driver_codes_classify_as_unavailable_or_fault() {
        for code in [
            CUresult::CUDA_ERROR_NO_DEVICE,
            CUresult::CUDA_ERROR_NOT_INITIALIZED,
            CUresult::CUDA_ERROR_STUB_LIBRARY,
            CUresult::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
        ] {
            assert_eq!(
                classify_driver_error(&DriverError(code)),
                CudaErrorClass::Unavailable,
                "{code:?}"
            );
        }
        for code in [
            CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
            CUresult::CUDA_ERROR_OUT_OF_MEMORY,
            CUresult::CUDA_ERROR_ECC_UNCORRECTABLE,
            CUresult::CUDA_ERROR_UNKNOWN,
            CUresult::CUDA_ERROR_LAUNCH_FAILED,
        ] {
            assert_eq!(
                classify_driver_error(&DriverError(code)),
                CudaErrorClass::Fault,
                "{code:?}"
            );
        }
    }

    #[test]
    fn wrapped_driver_errors_keep_their_class() {
        let wrapped = guard_typed("gemm", || -> Result<()> {
            Err(
                anyhow::Error::from(DriverError(CUresult::CUDA_ERROR_NO_DEVICE))
                    .context("creating cuda context for gpu 0"),
            )
        })
        .expect_err("error");
        assert_eq!(wrapped.runtime_class(), CudaErrorClass::Unavailable);
        let other = guard_typed("gemm", || -> Result<()> { anyhow::bail!("cublas: 13") })
            .expect_err("error");
        assert_eq!(other.runtime_class(), CudaErrorClass::Fault);
    }

    #[test]
    fn loader_panics_are_unavailable_only_at_init() {
        assert_eq!(loader_panic().init_class(), CudaErrorClass::Unavailable);
        assert_eq!(loader_panic().runtime_class(), CudaErrorClass::Fault);
        assert!(loader_panic().to_string().contains("panicked"));
    }

    #[test]
    fn the_rebooted_node_is_unavailable_not_failed() {
        // 10.1.0.67: RTX 3090 on the bus, kernel without the NVIDIA module.
        let availability = availability_from(
            Err(driver_error(CUresult::CUDA_ERROR_NO_DEVICE)),
            &[bus("0000:01:00.0")],
        );
        let DeviceAvailability::Unavailable { reason } = availability else {
            panic!("expected Unavailable, got {availability:?}");
        };
        assert!(
            reason.starts_with("no usable GPU (driver unavailable"),
            "{reason}"
        );
        assert!(reason.contains("0000:01:00.0"), "{reason}");
        assert!(reason.contains("CUDA_ERROR_NO_DEVICE"), "{reason}");
    }

    #[test]
    fn cpu_only_hosts_have_no_nvidia_gpu() {
        for count in [Err(loader_panic()), Ok(0)] {
            let DeviceAvailability::Unavailable { reason } = availability_from(count, &[]) else {
                panic!("expected Unavailable");
            };
            assert!(reason.starts_with("no NVIDIA GPU"), "{reason}");
        }
    }

    #[test]
    fn genuine_init_errors_stay_faults() {
        let availability = availability_from(
            Err(driver_error(CUresult::CUDA_ERROR_UNKNOWN)),
            &[bus("0000:01:00.0")],
        );
        assert!(
            matches!(&availability, DeviceAvailability::Fault { reason } if reason.contains("CUDA_ERROR_UNKNOWN")),
            "{availability:?}"
        );
        assert_eq!(
            availability_from(Ok(2), &[]),
            DeviceAvailability::Usable(std::num::NonZeroU32::new(2).expect("non-zero"))
        );
    }

    fn spec() -> GpuTaskSpec {
        GpuTaskSpec {
            gemm_secs: 1,
            gemm_dtypes: vec![],
            gemm_dim: 64,
            bandwidth_bytes: 1 << 20,
            sdc_check_secs: 0,
        }
    }

    pub(crate) fn outcomes_of(run: impl FnOnce(&EventSink)) -> Vec<(TestId, Scope, TestOutcome)> {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        struct Shared(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Shared {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0
                    .lock()
                    .map_err(|_| std::io::Error::other("poisoned"))?
                    .extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let sink = EventSink::new(Box::new(Shared(std::sync::Arc::clone(&buffer))));
        run(&sink);
        let bytes = buffer.lock().expect("buffer").clone();
        String::from_utf8(bytes)
            .expect("utf-8")
            .lines()
            .filter_map(
                |line| match crate::proto::decode_event(line).expect("event") {
                    crate::proto::AgentEvent::Outcome {
                        test,
                        scope,
                        outcome,
                    } => Some((test, scope, outcome)),
                    _ => None,
                },
            )
            .collect()
    }

    #[test]
    fn an_unusable_node_skips_every_gpu_test_once() {
        let outcomes = outcomes_of(|sink| {
            run_with(
                sink,
                &spec(),
                DeviceAvailability::Unavailable {
                    reason: "no usable GPU (driver unavailable: ...)".into(),
                },
            )
            .expect("gpu phase");
        });
        let tests: Vec<TestId> = outcomes.iter().map(|(test, ..)| *test).collect();
        assert_eq!(tests, GPU_PHASE_TESTS);
        for (_, scope, outcome) in outcomes {
            assert_eq!(scope, Scope::Node);
            assert!(
                matches!(&outcome, TestOutcome::Skipped { reason } if reason.contains("driver unavailable")),
                "{outcome:?}"
            );
        }
    }

    #[test]
    fn a_driver_fault_is_one_infrastructure_failure() {
        let outcomes = outcomes_of(|sink| {
            run_with(
                sink,
                &spec(),
                DeviceAvailability::Fault {
                    reason: "cuda driver init: CUDA_ERROR_UNKNOWN".into(),
                },
            )
            .expect("gpu phase");
        });
        assert_eq!(outcomes.len(), 1);
        let (test, _, outcome) = &outcomes[0];
        assert_eq!(*test, TestId::GpuGemmCorrectness);
        assert_eq!(
            outcome.failure_kind(),
            Some(crate::proto::FailureKind::Infrastructure)
        );
    }

    #[test]
    fn runtime_failures_skip_or_fail_by_class() {
        let skipped = runtime_failure_outcome(&driver_error(CUresult::CUDA_ERROR_NO_DEVICE));
        assert!(
            matches!(skipped, TestOutcome::Skipped { .. }),
            "{skipped:?}"
        );
        let failed = runtime_failure_outcome(&driver_error(CUresult::CUDA_ERROR_ILLEGAL_ADDRESS));
        assert_eq!(
            failed.failure_kind(),
            Some(crate::proto::FailureKind::Infrastructure)
        );
    }
}
