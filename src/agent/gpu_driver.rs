//! Phase 0 GPU driver state: NVIDIA GPUs on the PCI bus, detected from
//! sysfs independently of the driver, plus the nvidia kernel module's state.
//!
//! Sources (roots injectable so tests run against fixture trees):
//! - `/sys/bus/pci/devices/*/{vendor,class}`: vendor `0x10de`, class
//!   `0x0300xx` (VGA) or `0x0302xx` (3D controller) is an NVIDIA GPU. The
//!   audio and USB functions of a consumer card are other classes.
//! - `/sys/module/nvidia`: present iff the module is loaded.
//! - `/lib/modules/<kernel>/modules.dep`: whether the running kernel ships
//!   an `nvidia` module at all (a kernel upgraded without rebuilding the
//!   DKMS module does not).
//!
//! The classification itself is `proto::GpuDriverState::classify`.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use crate::agent::sysfs::read_trimmed;
use crate::proto::{
    GpuDriverState, ModuleAvailability, NvidiaModule, NvidiaSmiFailure, PciBusId, PciGpus,
};

/// PCI vendor id of NVIDIA.
const NVIDIA_VENDOR: u32 = 0x10de;
/// PCI base class + subclass (`class >> 8`) of a VGA-compatible controller.
const CLASS_VGA: u32 = 0x0300;
/// PCI base class + subclass of a 3D controller (datacenter GPUs).
const CLASS_3D: u32 = 0x0302;

/// Where the probes look. `Default` is the real system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverProbeRoots {
    /// `/sys/bus/pci/devices`.
    pub pci_devices: PathBuf,
    /// `/sys/module`.
    pub sys_module: PathBuf,
    /// `/lib/modules`.
    pub lib_modules: PathBuf,
}

impl Default for DriverProbeRoots {
    fn default() -> Self {
        Self {
            pci_devices: PathBuf::from(crate::agent::pci::PCI_DEVICES_ROOT),
            sys_module: PathBuf::from("/sys/module"),
            lib_modules: PathBuf::from("/lib/modules"),
        }
    }
}

/// NVIDIA GPU functions under `pci_devices`, sorted. Unreadable entries
/// are skipped; an unreadable root is an empty bus.
pub fn nvidia_pci_gpus(pci_devices: &Path) -> Vec<PciBusId> {
    let Ok(entries) = std::fs::read_dir(pci_devices) else {
        return Vec::new();
    };
    let mut gpus: Vec<PciBusId> = entries
        .flatten()
        .filter_map(|entry| {
            let address: PciBusId = entry.file_name().to_str()?.parse().ok()?;
            let dir = entry.path();
            let vendor = parse_hex(&read_trimmed(dir.join("vendor"))?)?;
            let class = parse_hex(&read_trimmed(dir.join("class"))?)?;
            is_nvidia_gpu(vendor, class).then_some(address)
        })
        .collect();
    gpus.sort();
    gpus.dedup();
    gpus
}

/// `vendor` / `class` sysfs attributes are `0x`-prefixed hex.
fn parse_hex(raw: &str) -> Option<u32> {
    let digits = raw
        .strip_prefix("0x")
        .or_else(|| raw.strip_prefix("0X"))
        .unwrap_or(raw);
    u32::from_str_radix(digits, 16).ok()
}

/// NVIDIA vendor and a display-controller class: VGA or 3D controller.
pub fn is_nvidia_gpu(vendor: u32, class: u32) -> bool {
    vendor == NVIDIA_VENDOR && matches!(class >> 8, CLASS_VGA | CLASS_3D)
}

/// Whether `modules.dep` text lists a module named exactly `nvidia`
/// (`kernel/drivers/video/nvidia.ko`, `updates/dkms/nvidia.ko.zst`, ...).
/// `nvidia-drm`, `nvidia_uvm` and friends do not count: without the core
/// module none of them loads.
pub fn modules_dep_has_nvidia(modules_dep: &str) -> bool {
    modules_dep.lines().any(|line| {
        let Some((path, _)) = line.split_once(':') else {
            return false;
        };
        let file = path.rsplit('/').next().unwrap_or(path);
        file.split_once(".ko")
            .is_some_and(|(name, _)| name == "nvidia")
    })
}

/// The nvidia module's state on the running kernel.
pub fn nvidia_module(roots: &DriverProbeRoots, kernel: &str) -> NvidiaModule {
    if roots.sys_module.join("nvidia").is_dir() {
        return NvidiaModule::Loaded;
    }
    let availability =
        match std::fs::read_to_string(roots.lib_modules.join(kernel).join("modules.dep")) {
            Ok(text) if modules_dep_has_nvidia(&text) => ModuleAvailability::Installed,
            Ok(_) => ModuleAvailability::MissingForKernel,
            Err(_) => ModuleAvailability::Unknown,
        };
    NvidiaModule::NotLoaded {
        kernel: kernel.to_string(),
        availability,
    }
}

/// The host's driver state. `smi` is the GPU query's result: `Ok(n)` GPUs
/// listed, or how it failed. The module is only probed when the bus has
/// NVIDIA GPUs that nvidia-smi could not list.
pub fn probe(
    roots: &DriverProbeRoots,
    smi: Result<usize, NvidiaSmiFailure>,
    kernel: &str,
) -> GpuDriverState {
    let listed = matches!(smi, Ok(count) if count > 0);
    GpuDriverState::classify(
        nvidia_pci_gpus(&roots.pci_devices),
        listed,
        || match smi {
            Ok(_) => NvidiaSmiFailure::ListedNone,
            Err(failure) => failure,
        },
        || nvidia_module(roots, kernel),
    )
}

/// What the GPU-touching phases (gpu, overlap, the intra-node sweep) can
/// do on this node, decided once from CUDA's device count and the PCI bus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceAvailability {
    /// CUDA opened this many devices.
    Usable(NonZeroU32),
    /// No usable GPU: no NVIDIA hardware, or a driver that cannot drive it.
    /// GPU tests record Skipped with this reason; the driver itself is the
    /// `gpu_driver` finding, derived from the inventory.
    Unavailable { reason: String },
    /// The driver stack failed in a way that is not "no GPU" (a genuine
    /// CUDA error on a working driver): a Failed outcome.
    Fault { reason: String },
}

impl DeviceAvailability {
    /// From a successful device count: zero devices is unavailability.
    pub fn from_count(count: u32, pci_gpus: &[PciBusId]) -> Self {
        match NonZeroU32::new(count) {
            Some(count) => DeviceAvailability::Usable(count),
            None => DeviceAvailability::Unavailable {
                reason: unavailable_reason(pci_gpus, "CUDA reports no devices"),
            },
        }
    }
}

/// The Skipped reason for a GPU test on a node without a usable GPU,
/// naming whether NVIDIA hardware is there at all. `detail` is the CUDA
/// error (or "CUDA reports no devices").
pub fn unavailable_reason(pci_gpus: &[PciBusId], detail: &str) -> String {
    match PciGpus::new(pci_gpus.to_vec()) {
        None => format!("no NVIDIA GPU (none on the PCI bus; {detail})"),
        Some(gpus) => format!(
            "no usable GPU (driver unavailable: CUDA cannot open the {gpus}; {detail}); \
             see gpu_driver"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bus(raw: &str) -> PciBusId {
        raw.parse().expect("bus id")
    }

    #[test]
    fn unavailable_reasons_name_the_hardware() {
        let cpu_only = unavailable_reason(&[], "cuda driver init: no libcuda");
        assert!(cpu_only.starts_with("no NVIDIA GPU"), "{cpu_only}");
        let broken = unavailable_reason(&[bus("0000:01:00.0")], "CUDA_ERROR_NO_DEVICE");
        assert!(
            broken.starts_with("no usable GPU (driver unavailable"),
            "{broken}"
        );
        assert!(broken.contains("0000:01:00.0"), "{broken}");
        assert!(broken.contains("CUDA_ERROR_NO_DEVICE"), "{broken}");
    }

    #[test]
    fn zero_devices_are_unavailable() {
        assert!(matches!(
            DeviceAvailability::from_count(0, &[]),
            DeviceAvailability::Unavailable { .. }
        ));
        assert_eq!(
            DeviceAvailability::from_count(4, &[]),
            DeviceAvailability::Usable(NonZeroU32::new(4).expect("non-zero"))
        );
    }

    #[test]
    fn gpu_classes_and_vendors() {
        assert!(is_nvidia_gpu(0x10de, 0x030000), "VGA");
        assert!(is_nvidia_gpu(0x10de, 0x030200), "3D controller");
        assert!(!is_nvidia_gpu(0x10de, 0x040300), "HDMI audio function");
        assert!(!is_nvidia_gpu(0x10de, 0x0c0330), "USB controller");
        assert!(!is_nvidia_gpu(0x1002, 0x030000), "AMD VGA");
        assert!(!is_nvidia_gpu(0x15b3, 0x020700), "Mellanox");
    }

    #[test]
    fn hex_attributes() {
        assert_eq!(parse_hex("0x10de"), Some(0x10de));
        assert_eq!(parse_hex("0x030200"), Some(0x030200));
        assert_eq!(parse_hex("10de"), Some(0x10de));
        assert_eq!(parse_hex("0xzz"), None);
    }

    #[test]
    fn modules_dep_matches_the_core_module_only() {
        let with = "kernel/drivers/net/e1000e/e1000e.ko.zst:\n\
                    updates/dkms/nvidia.ko.zst:\n\
                    updates/dkms/nvidia-drm.ko.zst: updates/dkms/nvidia-modeset.ko.zst\n";
        assert!(modules_dep_has_nvidia(with));
        assert!(modules_dep_has_nvidia("kernel/nvidia-580/nvidia.ko: \n"));
        let without = "kernel/drivers/gpu/drm/nouveau/nouveau.ko.zst: kernel/x.ko\n\
                       updates/dkms/nvidia-drm.ko.zst:\n\
                       kernel/drivers/net/nvidia_uvm_like.ko:\n";
        assert!(!modules_dep_has_nvidia(without));
        assert!(!modules_dep_has_nvidia(""));
    }
}
