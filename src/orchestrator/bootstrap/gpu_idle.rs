//! Bootstrap's `gpu_idle` readiness column: the run's `gpu_idle` policy
//! (`proto::assess_gpu_idle`, same threshold) applied to the probe.

use super::ReadinessCheck;
use crate::proto::{GpuIdleAssessment, InventorySnapshot, assess_gpu_idle};

/// The `gpu_idle` policy of `gauntlet run` (same function, same
/// threshold), surfaced before the run. Advisory — a busy GPU does not stop
/// a run from starting, it makes its GPU numbers meaningless — so a busy or
/// unknown GPU warns, naming each foreign process.
pub(super) fn gpu_idle_check(inventory: &InventorySnapshot, max_used_mib: u64) -> ReadinessCheck {
    if inventory.gpus.is_empty() {
        return ReadinessCheck::ok("gpu_idle", "n/a (no GPUs)");
    }
    let mut busy = Vec::new();
    let mut unknown = Vec::new();
    for gpu in &inventory.gpus {
        match assess_gpu_idle(&gpu.occupancy, max_used_mib) {
            GpuIdleAssessment::Idle => {}
            assessment @ GpuIdleAssessment::Busy { .. } => {
                busy.push(format!("gpu{}: {}", gpu.index, assessment.detail()));
            }
            GpuIdleAssessment::Unknown { .. } => unknown.push(gpu.index.to_string()),
        }
    }
    if !busy.is_empty() {
        ReadinessCheck::warn("gpu_idle", busy.join("; "))
    } else if !unknown.is_empty() {
        ReadinessCheck::warn(
            "gpu_idle",
            format!("occupancy unknown on gpu {}", unknown.join(",")),
        )
    } else {
        ReadinessCheck::ok("gpu_idle", format!("{} idle gpu(s)", inventory.gpus.len()))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::orchestrator::bootstrap::CheckStatus;
    use crate::proto::{GpuInventory, GpuOccupancy, GpuProcess, ProcessOwner};

    const GPU_IDLE_MAX: u64 = 1024;

    fn gpu(index: u32, occupancy: GpuOccupancy) -> GpuInventory {
        GpuInventory {
            index,
            name: "NVIDIA GeForce RTX 3090".into(),
            uuid: format!("GPU-{index}"),
            vbios: "94.02".into(),
            mem_total_bytes: 24_576 << 20,
            ecc_volatile_errors: None,
            remapped_rows_pending: None,
            pcie_gen_current: Some(4),
            pcie_gen_max: Some(4),
            pcie_width_current: Some(16),
            pcie_width_max: Some(16),
            nvlinks_active: None,
            persistence_mode: Some(true),
            occupancy,
            pci: None,
        }
    }

    fn idle() -> GpuOccupancy {
        GpuOccupancy {
            memory_used_mib: Some(37),
            memory_total_mib: Some(24_576),
            compute_processes: Some(Vec::new()),
        }
    }

    fn node(gpus: Vec<GpuInventory>) -> InventorySnapshot {
        InventorySnapshot {
            hostname: "node0".into(),
            kernel: "6.8".into(),
            cpu_model: "x".into(),
            logical_cores: 8,
            numa_nodes: 1,
            mem_total_bytes: 1 << 34,
            cpu_governor: None,
            clock_offset_ms: None,
            nvidia_driver: Some("570.133.07".into()),
            cuda_version: None,
            gpus,
            nics: Vec::new(),
            ib_ports: Vec::new(),
            ib_devices: Vec::new(),
            xid_errors: Vec::new(),
            gpu_libs: BTreeMap::new(),
            cuda_visible_gpus: None,
        }
    }

    #[test]
    fn idle_gpus_are_ok_and_no_gpus_is_na() {
        let check = gpu_idle_check(&node(vec![gpu(0, idle()), gpu(1, idle())]), GPU_IDLE_MAX);
        assert_eq!(check.name, "gpu_idle");
        assert_eq!(check.status, CheckStatus::Ok);
        assert_eq!(check.detail, "2 idle gpu(s)");
        let none = gpu_idle_check(&node(Vec::new()), GPU_IDLE_MAX);
        assert_eq!(none.status, CheckStatus::Ok);
        assert!(none.detail.contains("n/a"), "{}", none.detail);
    }

    #[test]
    fn busy_gpus_warn_with_the_process_detail() {
        let busy = GpuOccupancy {
            memory_used_mib: Some(23_264),
            memory_total_mib: Some(24_576),
            compute_processes: Some(vec![GpuProcess {
                pid: 2_102_873,
                name: "VLLM::EngineCore".into(),
                used_mib: Some(23_232),
                owner: ProcessOwner::Foreign,
            }]),
        };
        let check = gpu_idle_check(&node(vec![gpu(0, idle()), gpu(1, busy)]), GPU_IDLE_MAX);
        assert_eq!(check.status, CheckStatus::Warn);
        assert!(
            check
                .detail
                .starts_with("gpu1: in use by VLLM::EngineCore (pid 2102873, 23232 MiB)"),
            "{}",
            check.detail
        );
        assert!(!check.detail.contains("gpu0"), "{}", check.detail);
    }

    #[test]
    fn unknown_occupancy_warns() {
        let check = gpu_idle_check(
            &node(vec![gpu(0, idle()), gpu(1, GpuOccupancy::default())]),
            GPU_IDLE_MAX,
        );
        assert_eq!(check.status, CheckStatus::Warn);
        assert_eq!(check.detail, "occupancy unknown on gpu 1");
    }

    #[test]
    fn threshold_boundary_matches_the_run_policy() {
        let mut at = idle();
        at.memory_used_mib = Some(GPU_IDLE_MAX);
        assert_eq!(
            gpu_idle_check(&node(vec![gpu(0, at.clone())]), GPU_IDLE_MAX).status,
            CheckStatus::Ok
        );
        at.memory_used_mib = Some(GPU_IDLE_MAX + 1);
        assert_eq!(
            gpu_idle_check(&node(vec![gpu(0, at)]), GPU_IDLE_MAX).status,
            CheckStatus::Warn
        );
    }
}
