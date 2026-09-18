//! Fleet configuration (TOML). See `gauntlet.example.toml` at the repo root.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::proto::{
    AgentTaskSpec, CpuTaskSpec, DiskTaskSpec, GemmDtype, GpuTaskSpec, MemTaskSpec, OverlapNcclSpec,
    OverlapTaskSpec, Phase,
};

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid TOML in {path}: {source}")]
    Parse {
        path: PathBuf,
        source: Box<toml::de::Error>,
    },
    #[error("no hosts configured")]
    NoHosts,
    #[error("duplicate host address: {addr}")]
    DuplicateHost { addr: String },
    #[error("thresholds.mad_k must be positive, got {got}")]
    BadMadK { got: f64 },
    #[error("thresholds.barrier_slowest_frac must be in (0, 1], got {got}")]
    BadBarrierFrac { got: f64 },
    #[error("unknown phase name: {name}")]
    UnknownPhase { name: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FleetConfig {
    #[serde(default)]
    pub ssh: SshConfig,
    pub hosts: Vec<HostEntry>,
    #[serde(default)]
    pub tests: TestConfig,
    #[serde(default)]
    pub thresholds: Thresholds,
    #[serde(default)]
    pub nccl: NcclConfig,
}

/// Hosts may be written as a bare address string or a full table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HostEntry {
    Addr(String),
    Full(HostConfig),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostConfig {
    pub addr: String,
    /// Address on the data-plane network, for clusters where ssh rides a
    /// management NIC but training traffic rides a faster fabric. Pairwise
    /// network tests target this when set; ssh always uses `addr`.
    #[serde(default)]
    pub data_addr: Option<String>,
    /// Free-form labels (rack, role, ...) surfaced in reports.
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SshConfig {
    /// Login user; defaults to whatever ~/.ssh/config resolves.
    pub user: Option<String>,
    /// Identity file; defaults to agent/ssh-config resolution.
    pub key: Option<PathBuf>,
    /// Directory on each node for the agent binary and scratch files.
    pub remote_dir: String,
    pub connect_timeout_secs: u64,
    /// Concurrent session-establishment limit (full sessions stay open after).
    pub max_concurrent: usize,
}

impl Default for SshConfig {
    fn default() -> Self {
        Self {
            user: None,
            key: None,
            remote_dir: "~/.gauntlet".into(),
            connect_timeout_secs: 10,
            max_concurrent: 32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TestConfig {
    /// Phases to run, in order. Defaults to all.
    pub phases: Vec<Phase>,
    pub phase_timeout_secs: u64,
    pub cpu_correctness_secs_per_core: u64,
    pub cpu_gflops_secs: u64,
    /// Wall seconds for the hot SDC screen (correctness interleaved with the
    /// all-core power workload). 0 disables.
    pub cpu_sdc_hot_secs: u64,
    pub mem_buffer_mib_per_numa: u64,
    pub mem_iters: u32,
    /// Paths exercised by the disk test (dataset dir, checkpoint dir).
    pub disk_paths: Vec<String>,
    pub disk_file_mib: u64,
    pub gemm_secs: u64,
    pub gemm_dim: u32,
    pub gemm_dtypes: Vec<GemmDtype>,
    /// Loaded seconds between bitwise output checks during the sustained
    /// GEMM (hot SDC screen). 0 disables.
    pub gemm_sdc_check_secs: u64,
    pub gpu_bandwidth_mib: u64,
    pub net_latency_secs: u64,
    pub net_bandwidth_secs: u64,
    /// Base port for peer listeners; each concurrent pair gets base+i.
    pub net_port_base: u16,
    /// NCCL sweep message sizes in bytes.
    pub nccl_sizes: Vec<u64>,
    pub nccl_iters_per_size: u32,
    /// Barrier-skew microbenchmark iterations (0 disables it).
    pub barrier_iters: u32,
    /// Payload of the tiny barrier all-reduce, in bytes.
    pub barrier_bytes: u64,
    /// Overlap phase: wall seconds of GEMM + all-reduce combined load.
    pub overlap_secs: u64,
    /// Overlap phase: isolated intra-node all-reduce baseline window.
    pub overlap_baseline_secs: u64,
    /// Overlap phase: all-reduce message size in MiB.
    pub overlap_msg_mib: u64,
    /// Overlap phase: also run the fleet-wide combined-load step (GEMM on
    /// every GPU under a cross-node all-reduce). Skipped quietly on fleets
    /// with fewer than two GPU-bearing hosts.
    pub overlap_fleet: bool,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            phases: Phase::ALL.to_vec(),
            phase_timeout_secs: 900,
            cpu_correctness_secs_per_core: 10,
            cpu_gflops_secs: 10,
            cpu_sdc_hot_secs: 10,
            mem_buffer_mib_per_numa: 1024,
            mem_iters: 20,
            disk_paths: vec!["/tmp".into()],
            disk_file_mib: 1024,
            gemm_secs: 30,
            gemm_dim: 8192,
            gemm_dtypes: vec![GemmDtype::F32, GemmDtype::Bf16, GemmDtype::F16],
            gemm_sdc_check_secs: 5,
            gpu_bandwidth_mib: 1024,
            net_latency_secs: 3,
            net_bandwidth_secs: 5,
            net_port_base: 29500,
            // 1 KiB .. 1 GiB, powers of 4.
            nccl_sizes: (0..=10).map(|i| 1024u64 * 4u64.pow(i)).collect(),
            nccl_iters_per_size: 20,
            barrier_iters: 2000,
            barrier_bytes: 8,
            overlap_secs: 30,
            overlap_baseline_secs: 5,
            overlap_msg_mib: 64,
            overlap_fleet: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Thresholds {
    /// Fleet outlier sensitivity: flag values more than `mad_k` median
    /// absolute deviations from the fleet median.
    pub mad_k: f64,
    /// Optional absolute bounds keyed by "<test>.<metric>", e.g.
    /// "gpu_gemm_perf.gflops" -> { min = 100000 }.
    pub absolute: BTreeMap<String, Bound>,
    /// Barrier-skew flag: a host is a straggler when it was the (unique,
    /// beyond-margin) late arriver in more than this fraction of the
    /// considered barrier iterations.
    pub barrier_slowest_frac: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            mad_k: 4.0,
            absolute: BTreeMap::new(),
            barrier_slowest_frac: 0.5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Bound {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields, default)]
pub struct NcclConfig {
    /// Value for NCCL_SOCKET_IFNAME on multi-homed nodes.
    pub socket_ifname: Option<String>,
}

impl FleetConfig {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let config: FleetConfig = toml::from_str(&text).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source: Box::new(source),
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.hosts.is_empty() {
            return Err(ConfigError::NoHosts);
        }
        let mut seen = std::collections::BTreeSet::new();
        for host in self.hosts() {
            if !seen.insert(host.addr.clone()) {
                return Err(ConfigError::DuplicateHost { addr: host.addr });
            }
        }
        if self.thresholds.mad_k.is_nan() || self.thresholds.mad_k <= 0.0 {
            return Err(ConfigError::BadMadK {
                got: self.thresholds.mad_k,
            });
        }
        let frac = self.thresholds.barrier_slowest_frac;
        if !(frac > 0.0 && frac <= 1.0) {
            return Err(ConfigError::BadBarrierFrac { got: frac });
        }
        Ok(())
    }

    /// Hosts normalized to full `HostConfig` form.
    pub fn hosts(&self) -> impl Iterator<Item = HostConfig> + '_ {
        self.hosts.iter().map(|entry| match entry {
            HostEntry::Addr(addr) => HostConfig {
                addr: addr.clone(),
                data_addr: None,
                labels: BTreeMap::new(),
            },
            HostEntry::Full(full) => full.clone(),
        })
    }

    /// The per-node task spec sent to `agent run` on stdin.
    pub fn task_spec(&self, phases: &[Phase]) -> AgentTaskSpec {
        let tests = &self.tests;
        AgentTaskSpec {
            phases: phases.to_vec(),
            cpu: CpuTaskSpec {
                correctness_secs_per_core: tests.cpu_correctness_secs_per_core,
                gflops_secs: tests.cpu_gflops_secs,
                sdc_hot_secs: tests.cpu_sdc_hot_secs,
            },
            mem: MemTaskSpec {
                buffer_bytes_per_numa: tests.mem_buffer_mib_per_numa * 1024 * 1024,
                iters: tests.mem_iters,
            },
            disk: DiskTaskSpec {
                paths: tests.disk_paths.clone(),
                file_bytes: tests.disk_file_mib * 1024 * 1024,
            },
            gpu: GpuTaskSpec {
                gemm_secs: tests.gemm_secs,
                gemm_dtypes: tests.gemm_dtypes.clone(),
                gemm_dim: tests.gemm_dim,
                bandwidth_bytes: tests.gpu_bandwidth_mib * 1024 * 1024,
                sdc_check_secs: tests.gemm_sdc_check_secs,
            },
            overlap: OverlapTaskSpec {
                duration_secs: tests.overlap_secs,
                baseline_secs: tests.overlap_baseline_secs,
                // Same dimension as phase 2 so the retention ratio divides
                // comparable numbers; the compute leg runs the first
                // configured dtype only.
                gemm_dim: tests.gemm_dim,
                gemm_dtype: tests.gemm_dtypes.first().copied().unwrap_or(GemmDtype::F32),
                msg_bytes: tests.overlap_msg_mib * 1024 * 1024,
            },
            // Counter passes are scheduled by the orchestrator as dedicated
            // invocations; a plain phase spec never carries one.
            counters: None,
        }
    }

    /// The fleet-overlap payload sent on the NCCL directives. Same knobs as
    /// the node-local overlap spec (`task_spec().overlap`): the fleet step
    /// measures the same contention, one topology level up.
    pub fn overlap_nccl_spec(&self) -> OverlapNcclSpec {
        let tests = &self.tests;
        OverlapNcclSpec {
            duration_secs: tests.overlap_secs,
            baseline_secs: tests.overlap_baseline_secs,
            gemm_dim: tests.gemm_dim,
            gemm_dtype: tests.gemm_dtypes.first().copied().unwrap_or(GemmDtype::F32),
            msg_bytes: tests.overlap_msg_mib * 1024 * 1024,
        }
    }

    /// Resolve `--phases` CLI strings against config, erroring on unknowns.
    pub fn resolve_phases(&self, cli_phases: &[String]) -> Result<Vec<Phase>, ConfigError> {
        if cli_phases.is_empty() {
            return Ok(self.tests.phases.clone());
        }
        cli_phases
            .iter()
            .map(|name| {
                Phase::parse(name).ok_or_else(|| ConfigError::UnknownPhase { name: name.clone() })
            })
            .collect()
    }
}
