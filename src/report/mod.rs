//! Results assembly and rendering.
//!
//! `RunResults` is the schema-versioned JSON document `gauntlet run` writes;
//! it is the machine interface for simulation calibration, so field renames
//! bump SCHEMA_VERSION. The terminal table is a projection of it, never a
//! second source of truth.

pub mod history;

use std::collections::BTreeMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::analysis::fit::AlphaBetaFit;
use crate::analysis::stats::Outlier;
use crate::cli::ReportArgs;
use crate::config::FleetConfig;
use crate::orchestrator::collect::HostObservations;
use crate::proto::TestId;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunResults {
    pub schema_version: u32,
    /// "<started epoch secs>-<6 hex chars>"; also the history filename stem.
    pub run_id: String,
    pub started_epoch_secs: u64,
    pub finished_epoch_secs: u64,
    pub hosts: BTreeMap<String, HostObservations>,
    pub fleet: FleetAnalysis,
    pub calibration: Calibration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct FleetAnalysis {
    /// MAD outliers, grouped by "<test>.<metric>".
    pub outliers: BTreeMap<String, Vec<Outlier>>,
    /// Hosts violating absolute thresholds, grouped the same way.
    pub threshold_violations: BTreeMap<String, Vec<String>>,
    /// Inventory consistency findings: field -> (majority value,
    /// dissenting host -> its value).
    pub consistency: BTreeMap<String, ConsistencyFinding>,
    /// Hosts that produced errors (Fatal, transport, timeout).
    pub failed_hosts: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsistencyFinding {
    pub majority_value: String,
    pub dissenters: BTreeMap<String, String>,
}

/// Simulator-facing constants extracted from the run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Calibration {
    /// Per-host sustained capability numbers.
    pub rooflines: BTreeMap<String, NodeRoofline>,
    /// Alpha-beta fits keyed by link class ("tcp_pairwise",
    /// "nccl_allreduce_fleet", "nccl_allreduce_pair", ...).
    pub links: BTreeMap<String, AlphaBetaFit>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct NodeRoofline {
    /// "gflops_f32", "gflops_bf16", ... -> sustained GFLOPS (min across
    /// that host's GPUs — the straggler defines the node).
    pub gpu_gflops: BTreeMap<String, f64>,
    pub cpu_gflops_allcore: Option<f64>,
    pub dram_gib_per_sec: Option<f64>,
    pub gpu_hbm_gib_per_sec: Option<f64>,
    pub pcie_h2d_gib_per_sec: Option<f64>,
    pub disk_read_gib_per_sec: Option<f64>,
    pub disk_write_gib_per_sec: Option<f64>,
}

/// Exit code contract for `gauntlet run`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// 0: all hosts completed, no outliers, no violations.
    Clean,
    /// 1: completed with outliers or threshold violations.
    Stragglers,
    /// 2: at least one host failed to complete.
    HostFailures,
}

impl Verdict {
    pub fn exit_code(self) -> i32 {
        match self {
            Verdict::Clean => 0,
            Verdict::Stragglers => 1,
            Verdict::HostFailures => 2,
        }
    }
}

/// Build the full results document from collected observations: run MAD
/// outlier detection per "<test>.<metric>" (grouped within comparable
/// scopes), apply absolute thresholds, compute consistency findings,
/// rooflines, and link fits.
pub fn build(
    config: &FleetConfig,
    observations: BTreeMap<String, HostObservations>,
    started_epoch_secs: u64,
    finished_epoch_secs: u64,
) -> RunResults {
    let _ = (
        config,
        observations,
        started_epoch_secs,
        finished_epoch_secs,
    );
    todo!("agent D: implement")
}

pub fn verdict(results: &RunResults) -> Verdict {
    let _ = results;
    todo!("agent D: implement")
}

/// Render the human table (per-host summary, outliers section, consistency
/// section, calibration digest) to the given writer.
pub fn render_table(results: &RunResults, out: &mut dyn std::io::Write) -> Result<()> {
    let _ = (results, out);
    todo!("agent D: implement")
}

/// `gauntlet report`: load a saved results JSON and re-render it.
pub fn render_saved(args: ReportArgs) -> Result<()> {
    let _ = args;
    todo!("agent D: implement")
}

/// Which tests a metric belongs to, for grouping/table sections.
pub fn test_display_name(test: TestId) -> &'static str {
    match test {
        TestId::Inventory => "inventory",
        TestId::CpuCorrectness => "cpu_correctness",
        TestId::CpuGflops => "cpu_gflops",
        TestId::MemBandwidth => "mem_bandwidth",
        TestId::DiskIo => "disk_io",
        TestId::GpuGemmCorrectness => "gpu_gemm_correctness",
        TestId::GpuGemmPerf => "gpu_gemm_perf",
        TestId::GpuMemBandwidth => "gpu_mem_bandwidth",
        TestId::GpuP2p => "gpu_p2p",
        TestId::NetLatency => "net_latency",
        TestId::NetBandwidth => "net_bandwidth",
        TestId::NcclAllReduce => "nccl_all_reduce",
        TestId::NcclAllGather => "nccl_all_gather",
    }
}
