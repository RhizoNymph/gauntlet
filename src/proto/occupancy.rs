//! GPU occupancy: who else is using a GPU at inventory time, and the
//! `gpu_idle` policy derived from it.
//!
//! The agent reports facts only (memory in use, the compute processes
//! nvidia-smi lists, with gauntlet's own agent excluded); the orchestrator
//! turns them into one `gpu_idle` outcome per GPU against the configured
//! `thresholds.gpu_idle_max_used_mib`. The same policy feeds the bootstrap
//! readiness matrix, so both surfaces agree on what "busy" means.

use serde::{Deserialize, Serialize};

use super::{InventorySnapshot, Scope, TestOutcome};

/// Memory and compute-process occupancy of one GPU at probe time.
///
/// Every field distinguishes "unknown" from "zero": `None` means nvidia-smi
/// did not say (field N/A, query failed, older snapshot), which the policy
/// reports as Skipped rather than guessing idle. The default is therefore
/// "nothing known", which is also what a pre-v10 inventory decodes to.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuOccupancy {
    /// `memory.used` in MiB; includes driver/graphics residue.
    #[serde(default)]
    pub memory_used_mib: Option<u64>,
    /// `memory.total` in MiB.
    #[serde(default)]
    pub memory_total_mib: Option<u64>,
    /// Compute processes on this GPU other than the reporting agent itself.
    /// `None`: the compute-apps query failed, so nothing is known;
    /// `Some(empty)`: verified none. Graphics-only processes (Xorg, display
    /// managers) are never listed — nvidia-smi's compute-apps query excludes
    /// them by construction.
    #[serde(default)]
    pub compute_processes: Option<Vec<GpuProcess>>,
}

/// One compute process holding a GPU context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuProcess {
    pub pid: u32,
    /// nvidia-smi's `process_name` (the process title, e.g.
    /// `VLLM::EngineCore` or a full executable path).
    pub name: String,
    /// `used_memory` in MiB; `None` when nvidia-smi reports N/A (e.g. no
    /// per-process accounting inside some containers).
    pub used_mib: Option<u64>,
    pub owner: ProcessOwner,
}

/// Whose process it is. The reporting agent (and its ancestry) is excluded
/// before this point, so every listed process is a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessOwner {
    /// Anything that is not gauntlet (an inference server, a training job).
    Foreign,
    /// A `gauntlet-agent` that is not the reporting agent: a leftover from a
    /// crashed earlier run, or a concurrent run on the same node.
    StaleGauntletAgent,
}

impl GpuProcess {
    /// "VLLM::EngineCore (pid 2102873, 23232 MiB)", prefixed with
    /// "stale gauntlet agent " for leftover agents.
    pub fn describe(&self) -> String {
        let memory = self
            .used_mib
            .map_or_else(|| "? MiB".to_string(), |mib| format!("{mib} MiB"));
        let prefix = match self.owner {
            ProcessOwner::Foreign => "",
            ProcessOwner::StaleGauntletAgent => "stale gauntlet agent ",
        };
        format!("{prefix}{} (pid {}, {memory})", self.name, self.pid)
    }
}

/// The `gpu_idle` verdict for one GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuIdleAssessment {
    /// No other compute process and memory use within the threshold.
    Idle,
    /// Something else is on the GPU. At least one of `processes` is
    /// non-empty or `memory_over` is set — `assess_gpu_idle` only builds
    /// this variant when there is evidence.
    Busy {
        processes: Vec<GpuProcess>,
        /// `Some` when memory use exceeds the threshold.
        memory_over: Option<MemoryOverage>,
    },
    /// Not enough information to tell; carries what was missing.
    Unknown { missing: &'static str },
}

/// Memory use beyond the `gpu_idle` threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryOverage {
    pub used_mib: u64,
    pub total_mib: Option<u64>,
    pub max_used_mib: u64,
}

/// Apply the `gpu_idle` policy to one GPU: Busy on any evidence (a listed
/// compute process, or memory used above `max_used_mib`), otherwise Idle
/// only when both signals are known, otherwise Unknown. Evidence wins over
/// ignorance: a known foreign process fails the GPU even when memory is N/A.
pub fn assess_gpu_idle(occupancy: &GpuOccupancy, max_used_mib: u64) -> GpuIdleAssessment {
    let memory_over = occupancy
        .memory_used_mib
        .filter(|used| *used > max_used_mib)
        .map(|used_mib| MemoryOverage {
            used_mib,
            total_mib: occupancy.memory_total_mib,
            max_used_mib,
        });
    let processes = occupancy.compute_processes.clone().unwrap_or_default();
    if !processes.is_empty() || memory_over.is_some() {
        return GpuIdleAssessment::Busy {
            processes,
            memory_over,
        };
    }
    match (&occupancy.compute_processes, occupancy.memory_used_mib) {
        (Some(_), Some(_)) => GpuIdleAssessment::Idle,
        (None, Some(_)) => GpuIdleAssessment::Unknown {
            missing: "compute process list unavailable",
        },
        (Some(_), None) => GpuIdleAssessment::Unknown {
            missing: "memory.used not reported",
        },
        (None, None) => GpuIdleAssessment::Unknown {
            missing: "occupancy not reported",
        },
    }
}

impl GpuIdleAssessment {
    /// One-line human detail: the process list for Busy, the missing signal
    /// for Unknown.
    pub fn detail(&self) -> String {
        match self {
            GpuIdleAssessment::Idle => "idle".to_string(),
            GpuIdleAssessment::Busy {
                processes,
                memory_over,
            } => {
                let mut parts = Vec::new();
                if !processes.is_empty() {
                    let listed: Vec<String> = processes.iter().map(GpuProcess::describe).collect();
                    parts.push(format!("in use by {}", listed.join(", ")));
                }
                if let Some(over) = memory_over {
                    let total = over
                        .total_mib
                        .map_or_else(String::new, |total| format!("/{total}"));
                    parts.push(format!(
                        "{}{total} MiB used (> {} MiB threshold)",
                        over.used_mib, over.max_used_mib
                    ));
                }
                parts.join("; ")
            }
            GpuIdleAssessment::Unknown { missing } => format!("occupancy unknown: {missing}"),
        }
    }

    pub fn outcome(&self) -> TestOutcome {
        match self {
            GpuIdleAssessment::Idle => TestOutcome::Passed,
            GpuIdleAssessment::Busy { .. } => TestOutcome::Failed {
                reason: self.detail(),
            },
            GpuIdleAssessment::Unknown { .. } => TestOutcome::Skipped {
                reason: self.detail(),
            },
        }
    }
}

/// One `gpu_idle` outcome per GPU in the snapshot, in GPU index order.
/// Empty for a host without GPUs.
pub fn gpu_idle_outcomes(
    inventory: &InventorySnapshot,
    max_used_mib: u64,
) -> Vec<(Scope, TestOutcome)> {
    inventory
        .gpus
        .iter()
        .map(|gpu| {
            (
                Scope::Gpu { index: gpu.index },
                assess_gpu_idle(&gpu.occupancy, max_used_mib).outcome(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLD: u64 = 1024;

    fn vllm() -> GpuProcess {
        GpuProcess {
            pid: 2_102_873,
            name: "VLLM::EngineCore".into(),
            used_mib: Some(23_232),
            owner: ProcessOwner::Foreign,
        }
    }

    fn occupancy(used: Option<u64>, processes: Option<Vec<GpuProcess>>) -> GpuOccupancy {
        GpuOccupancy {
            memory_used_mib: used,
            memory_total_mib: Some(24_576),
            compute_processes: processes,
        }
    }

    #[test]
    fn threshold_is_inclusive() {
        assert_eq!(
            assess_gpu_idle(&occupancy(Some(THRESHOLD), Some(vec![])), THRESHOLD),
            GpuIdleAssessment::Idle
        );
        let over = assess_gpu_idle(&occupancy(Some(THRESHOLD + 1), Some(vec![])), THRESHOLD);
        assert!(matches!(
            over,
            GpuIdleAssessment::Busy {
                memory_over: Some(MemoryOverage { used_mib: 1025, .. }),
                ..
            }
        ));
        let detail = over.detail();
        assert!(detail.contains("1025/24576 MiB used"), "{detail}");
        assert!(detail.contains("1024 MiB threshold"), "{detail}");
    }

    #[test]
    fn graphics_residue_below_threshold_is_idle() {
        // Xorg (12 MiB) + sddm-greeter (20 MiB) are graphics processes: not
        // in the compute list, and their memory is well inside the budget.
        let idle = occupancy(Some(37), Some(vec![]));
        assert_eq!(
            assess_gpu_idle(&idle, THRESHOLD).outcome(),
            TestOutcome::Passed
        );
    }

    #[test]
    fn a_foreign_process_fails_with_name_pid_and_memory() {
        let busy = occupancy(Some(23_264), Some(vec![vllm()]));
        let TestOutcome::Failed { reason } = assess_gpu_idle(&busy, THRESHOLD).outcome() else {
            panic!("expected Failed");
        };
        assert!(
            reason.contains("VLLM::EngineCore (pid 2102873, 23232 MiB)"),
            "{reason}"
        );
        assert!(reason.contains("23264/24576 MiB used"), "{reason}");
    }

    #[test]
    fn a_small_foreign_process_still_fails() {
        // Under the memory threshold, but a foreign compute context is a
        // shared GPU all the same.
        let mut small = vllm();
        small.used_mib = Some(300);
        let busy = occupancy(Some(310), Some(vec![small]));
        let assessment = assess_gpu_idle(&busy, THRESHOLD);
        assert!(matches!(
            assessment,
            GpuIdleAssessment::Busy {
                memory_over: None,
                ..
            }
        ));
        assert!(!assessment.detail().contains("threshold"));
    }

    #[test]
    fn stale_agents_are_named_distinctly() {
        let stale = GpuProcess {
            pid: 4242,
            name: "/home/u/.gauntlet/bin/gauntlet-agent".into(),
            used_mib: None,
            owner: ProcessOwner::StaleGauntletAgent,
        };
        let TestOutcome::Failed { reason } =
            assess_gpu_idle(&occupancy(Some(500), Some(vec![stale])), THRESHOLD).outcome()
        else {
            panic!("expected Failed");
        };
        assert!(
            reason.contains("stale gauntlet agent /home/u/.gauntlet/bin/gauntlet-agent"),
            "{reason}"
        );
        assert!(reason.contains("pid 4242, ? MiB"), "{reason}");
    }

    #[test]
    fn unknown_signals_skip_but_evidence_still_fails() {
        for occ in [
            GpuOccupancy::default(),
            occupancy(None, Some(vec![])),
            occupancy(Some(10), None),
        ] {
            assert!(
                matches!(
                    assess_gpu_idle(&occ, THRESHOLD).outcome(),
                    TestOutcome::Skipped { .. }
                ),
                "{occ:?}"
            );
        }
        // Memory N/A but a process is listed: that is evidence.
        assert!(matches!(
            assess_gpu_idle(&occupancy(None, Some(vec![vllm()])), THRESHOLD).outcome(),
            TestOutcome::Failed { .. }
        ));
        // Process list unknown but memory is way over: evidence too.
        assert!(matches!(
            assess_gpu_idle(&occupancy(Some(20_000), None), THRESHOLD).outcome(),
            TestOutcome::Failed { .. }
        ));
    }

    #[test]
    fn occupancy_round_trips_and_defaults_to_unknown() {
        let occ = occupancy(Some(23_264), Some(vec![vllm()]));
        let json = serde_json::to_string(&occ).expect("serialize");
        assert!(json.contains(r#""owner":"foreign""#), "{json}");
        let back: GpuOccupancy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, occ);
        let empty: GpuOccupancy = serde_json::from_str("{}").expect("empty object");
        assert_eq!(empty, GpuOccupancy::default());
    }
}
