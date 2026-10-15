//! Per-host GPU driver state (proto v14), and the `gpu_driver` outcome
//! derived from it.
//!
//! The agent detects NVIDIA GPUs on the PCI bus independently of the
//! driver (sysfs vendor/class, `agent::gpu_driver`), so a node whose kernel
//! lost its NVIDIA module still reports "there is a GPU here and it cannot
//! be used" instead of looking like a CPU-only host. One root cause, one
//! finding: the orchestrator turns this state into exactly one `gpu_driver`
//! outcome per host (`orchestrator::derive`), and the GPU tests on such a
//! host record Skipped, not Failed.

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{PciBusId, TestOutcome};

/// NVIDIA GPUs found on the PCI bus: sorted, deduplicated, never empty.
/// A driver can only be "unavailable" for hardware that exists, so the
/// empty list is unrepresentable (and rejected on decode).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<PciBusId>", into = "Vec<PciBusId>")]
pub struct PciGpus(Vec<PciBusId>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("a PCI GPU list must name at least one GPU")]
pub struct EmptyPciGpus;

impl PciGpus {
    /// `None` when `addresses` is empty.
    pub fn new(mut addresses: Vec<PciBusId>) -> Option<Self> {
        addresses.sort();
        addresses.dedup();
        (!addresses.is_empty()).then_some(Self(addresses))
    }

    pub fn as_slice(&self) -> &[PciBusId] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Never true; present for clippy's `len_without_is_empty`.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl TryFrom<Vec<PciBusId>> for PciGpus {
    type Error = EmptyPciGpus;

    fn try_from(addresses: Vec<PciBusId>) -> Result<Self, Self::Error> {
        Self::new(addresses).ok_or(EmptyPciGpus)
    }
}

impl From<PciGpus> for Vec<PciBusId> {
    fn from(gpus: PciGpus) -> Self {
        gpus.0
    }
}

impl fmt::Display for PciGpus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let noun = if self.0.len() == 1 { "GPU" } else { "GPUs" };
        write!(f, "NVIDIA {noun} on the PCI bus at ")?;
        for (index, address) in self.0.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{address}")?;
        }
        Ok(())
    }
}

/// Whether the running kernel ships an `nvidia` module
/// (`/lib/modules/<kernel>/modules.dep`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModuleAvailability {
    Installed,
    /// The kernel was likely upgraded without rebuilding the NVIDIA module.
    MissingForKernel,
    /// `modules.dep` was unreadable.
    Unknown,
}

/// The `nvidia` kernel module, as `/sys/module/nvidia` shows it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum NvidiaModule {
    Loaded,
    NotLoaded {
        /// The running kernel release (`uname -r`).
        kernel: String,
        availability: ModuleAvailability,
    },
}

/// How `nvidia-smi` failed to list the GPUs the PCI bus has. (When it
/// lists any, the driver is `DriverOk`.)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NvidiaSmiFailure {
    /// Not installed / not on PATH.
    NotFound,
    /// Ran and exited non-zero; `message` is its first non-empty output
    /// line ("NVIDIA-SMI has failed because it couldn't communicate with
    /// the NVIDIA driver. ...").
    Failed { message: String },
    /// Did not finish within the probe budget.
    TimedOut,
    /// Exited zero but listed no GPU.
    ListedNone,
}

impl fmt::Display for NvidiaSmiFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NvidiaSmiFailure::NotFound => f.write_str("nvidia-smi not found"),
            NvidiaSmiFailure::Failed { message } => write!(f, "nvidia-smi: {message}"),
            NvidiaSmiFailure::TimedOut => f.write_str("nvidia-smi timed out"),
            NvidiaSmiFailure::ListedNone => f.write_str("nvidia-smi lists no GPU"),
        }
    }
}

/// Per-host GPU driver state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum GpuDriverState {
    /// No NVIDIA GPU on the PCI bus and none listed by nvidia-smi: a
    /// CPU-only host.
    NoNvidiaHardware,
    /// nvidia-smi lists the GPUs.
    DriverOk,
    /// NVIDIA GPUs are on the PCI bus but the driver cannot list them.
    DriverUnavailable {
        pci_gpus: PciGpus,
        module: NvidiaModule,
        nvidia_smi: NvidiaSmiFailure,
    },
}

impl GpuDriverState {
    /// Decide from the probes. nvidia-smi listing GPUs is authoritative
    /// (sysfs can be hidden in containers); otherwise the PCI bus decides
    /// between a CPU-only host and an unusable driver.
    pub fn classify(
        pci_gpus: Vec<PciBusId>,
        smi_listed_gpus: bool,
        smi_failure: impl FnOnce() -> NvidiaSmiFailure,
        module: impl FnOnce() -> NvidiaModule,
    ) -> Self {
        if smi_listed_gpus {
            return GpuDriverState::DriverOk;
        }
        match PciGpus::new(pci_gpus) {
            None => GpuDriverState::NoNvidiaHardware,
            Some(pci_gpus) => GpuDriverState::DriverUnavailable {
                pci_gpus,
                module: module(),
                nvidia_smi: smi_failure(),
            },
        }
    }

    /// The `gpu_driver` outcome: Failed for an unusable driver on NVIDIA
    /// hardware, Passed for a working driver, Skipped on a CPU-only host.
    pub fn outcome(&self) -> TestOutcome {
        match self {
            GpuDriverState::NoNvidiaHardware => TestOutcome::Skipped {
                reason: "no NVIDIA GPU on the PCI bus".into(),
            },
            GpuDriverState::DriverOk => TestOutcome::Passed,
            GpuDriverState::DriverUnavailable { .. } => TestOutcome::Failed {
                reason: self.to_string(),
            },
        }
    }

    pub fn is_unavailable(&self) -> bool {
        matches!(self, GpuDriverState::DriverUnavailable { .. })
    }
}

impl fmt::Display for GpuDriverState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuDriverState::NoNvidiaHardware => f.write_str("no NVIDIA GPU on the PCI bus"),
            GpuDriverState::DriverOk => f.write_str("NVIDIA driver loaded"),
            GpuDriverState::DriverUnavailable {
                pci_gpus,
                module,
                nvidia_smi,
            } => {
                write!(f, "{pci_gpus}, but the NVIDIA driver is unavailable: ")?;
                match module {
                    NvidiaModule::Loaded => f.write_str("the nvidia kernel module is loaded")?,
                    NvidiaModule::NotLoaded {
                        kernel,
                        availability,
                    } => {
                        f.write_str("the nvidia kernel module is not loaded")?;
                        match availability {
                            ModuleAvailability::MissingForKernel => write!(
                                f,
                                " and the running kernel {kernel} has no nvidia module \
                                 (kernel upgraded without rebuilding the driver?)"
                            )?,
                            ModuleAvailability::Installed => {
                                write!(f, " although the running kernel {kernel} ships one")?
                            }
                            ModuleAvailability::Unknown => write!(f, " (kernel {kernel})")?,
                        }
                    }
                }
                write!(f, "; {nvidia_smi}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bus(raw: &str) -> PciBusId {
        raw.parse().expect("bus id")
    }

    fn not_loaded(availability: ModuleAvailability) -> NvidiaModule {
        NvidiaModule::NotLoaded {
            kernel: "6.17.0-5-generic".into(),
            availability,
        }
    }

    fn smi_failed() -> NvidiaSmiFailure {
        NvidiaSmiFailure::Failed {
            message: "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA \
                      driver."
                .into(),
        }
    }

    #[test]
    fn classification_covers_the_three_states() {
        let module_unused = || -> NvidiaModule { panic!("module probe must not run") };
        let smi_unused = || -> NvidiaSmiFailure { panic!("smi failure must not be built") };
        assert_eq!(
            GpuDriverState::classify(vec![], false, smi_unused, module_unused),
            GpuDriverState::NoNvidiaHardware
        );
        // nvidia-smi listing GPUs wins even when sysfs shows nothing.
        assert_eq!(
            GpuDriverState::classify(vec![], true, smi_unused, module_unused),
            GpuDriverState::DriverOk
        );
        assert_eq!(
            GpuDriverState::classify(vec![bus("0000:01:00.0")], true, smi_unused, module_unused),
            GpuDriverState::DriverOk
        );
        let state = GpuDriverState::classify(
            vec![
                bus("0000:41:00.0"),
                bus("0000:01:00.0"),
                bus("0000:01:00.0"),
            ],
            false,
            smi_failed,
            || not_loaded(ModuleAvailability::MissingForKernel),
        );
        let GpuDriverState::DriverUnavailable { pci_gpus, .. } = &state else {
            panic!("expected DriverUnavailable, got {state:?}");
        };
        assert_eq!(
            pci_gpus.as_slice(),
            [bus("0000:01:00.0"), bus("0000:41:00.0")],
            "sorted and deduplicated"
        );
    }

    #[test]
    fn the_rebooted_node_fails_with_a_named_cause() {
        // 10.1.0.67: new kernel, no NVIDIA module, RTX 3090 on the bus.
        let state = GpuDriverState::classify(vec![bus("0000:01:00.0")], false, smi_failed, || {
            not_loaded(ModuleAvailability::MissingForKernel)
        });
        let TestOutcome::Failed { reason } = state.outcome() else {
            panic!("expected Failed");
        };
        for needle in [
            "NVIDIA GPU on the PCI bus at 0000:01:00.0",
            "not loaded",
            "6.17.0-5-generic has no nvidia module",
            "NVIDIA-SMI has failed",
        ] {
            assert!(reason.contains(needle), "{needle:?} missing from {reason}");
        }
    }

    #[test]
    fn outcomes_per_state() {
        assert_eq!(GpuDriverState::DriverOk.outcome(), TestOutcome::Passed);
        assert!(matches!(
            GpuDriverState::NoNvidiaHardware.outcome(),
            TestOutcome::Skipped { .. }
        ));
        let loaded = GpuDriverState::classify(
            vec![bus("0000:01:00.0"), bus("0000:02:00.0")],
            false,
            || NvidiaSmiFailure::ListedNone,
            || NvidiaModule::Loaded,
        );
        let TestOutcome::Failed { reason } = loaded.outcome() else {
            panic!("expected Failed");
        };
        assert!(reason.contains("GPUs on the PCI bus at 0000:01:00.0, 0000:02:00.0"));
        assert!(reason.contains("module is loaded"), "{reason}");
        assert!(reason.contains("nvidia-smi lists no GPU"), "{reason}");
    }

    #[test]
    fn states_round_trip_and_empty_gpu_lists_do_not_decode() {
        let state = GpuDriverState::classify(vec![bus("0000:01:00.0")], false, smi_failed, || {
            not_loaded(ModuleAvailability::Unknown)
        });
        let json = serde_json::to_string(&state).expect("serialize");
        assert!(json.contains(r#""state":"driver_unavailable""#), "{json}");
        assert!(json.contains(r#""pci_gpus":["0000:01:00.0"]"#), "{json}");
        assert_eq!(
            serde_json::from_str::<GpuDriverState>(&json).expect("decode"),
            state
        );
        let empty = json.replace(r#"["0000:01:00.0"]"#, "[]");
        assert!(serde_json::from_str::<GpuDriverState>(&empty).is_err());
        for simple in [GpuDriverState::NoNvidiaHardware, GpuDriverState::DriverOk] {
            let json = serde_json::to_string(&simple).expect("serialize");
            assert_eq!(
                serde_json::from_str::<GpuDriverState>(&json).expect("decode"),
                simple
            );
        }
    }
}
