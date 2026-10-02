//! Fleet configuration (TOML). See `gauntlet.example.toml` at the repo root.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::launch::{LaunchConfig, LaunchMode};
use crate::nccl_env::{NcclEnv, NcclEnvError, RawNcclEnvValue, stringify_raw};
use crate::proto::{
    AgentTaskSpec, CpuTaskSpec, DiskTaskSpec, GemmDtype, GpuTaskSpec, MemTaskSpec, NcclSweepSpec,
    OverlapSpec, Phase,
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
    /// Only srun launch can infer hosts (from the allocation); every other
    /// mode needs them listed.
    #[error("no hosts configured (launch mode {mode} needs `hosts`; only srun mode infers them)")]
    NoHosts { mode: LaunchMode },
    #[error("duplicate host address: {addr}")]
    DuplicateHost { addr: String },
    #[error("thresholds.mad_k must be positive, got {got}")]
    BadMadK { got: f64 },
    #[error("thresholds.barrier_slowest_frac must be in (0, 1], got {got}")]
    BadBarrierFrac { got: f64 },
    #[error("unknown phase name: {name}")]
    UnknownPhase { name: String },
    /// `[nccl]` violates the NCCL env policy (non-NCCL key, empty or
    /// NUL-bearing value, NCCL_SOCKET_IFNAME set twice).
    #[error("invalid [nccl] section: {source}")]
    Nccl { source: NcclEnvError },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FleetConfig {
    #[serde(default)]
    pub ssh: SshConfig,
    /// May be omitted in srun launch mode: the allocation's nodes are used.
    #[serde(default)]
    pub hosts: Vec<HostEntry>,
    /// `[launch]`: ssh (default) or srun inside a Slurm allocation.
    #[serde(default)]
    pub launch: LaunchConfig,
    #[serde(default)]
    pub tests: TestConfig,
    #[serde(default)]
    pub thresholds: Thresholds,
    #[serde(default)]
    pub nccl: NcclConfig,
    /// `nccl` resolved and validated, filled once by `validate` (or on
    /// first access). Private and write-once: the only value it can ever
    /// hold is one `NcclConfig::resolve` accepted.
    #[serde(skip)]
    resolved_nccl_env: OnceLock<NcclEnv>,
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
    /// Network phase: run the intra-node NCCL sweep (all local GPUs of each
    /// node, NVLink/PCIe) before the pairwise and fleet levels. Reuses
    /// `nccl_sizes` / `nccl_iters_per_size`.
    pub nccl_intranode: bool,
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
            nccl_intranode: true,
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
    /// `gpu_idle`: a GPU with more than this much memory in use (MiB) at
    /// inventory time fails the check even without a listed compute
    /// process. The default leaves headroom for a desktop's graphics
    /// clients and driver residue; any foreign compute process fails the
    /// check regardless.
    pub gpu_idle_max_used_mib: u64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            mad_k: 4.0,
            absolute: BTreeMap::new(),
            barrier_slowest_frac: 0.5,
            gpu_idle_max_used_mib: 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Bound {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

/// `[nccl]` as written in the file. Resolved into the one `NcclEnv` by
/// `FleetConfig::validate` (see `FleetConfig::nccl_env`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields, default)]
pub struct NcclConfig {
    /// Value for NCCL_SOCKET_IFNAME on multi-homed nodes.
    pub socket_ifname: Option<String>,
    /// Extra NCCL knobs, name -> value (keys `^NCCL_[A-Z0-9_]+$`; string,
    /// integer or boolean values).
    pub env: BTreeMap<String, RawNcclEnvValue>,
}

impl NcclConfig {
    /// Validate the section into the single resolved env: `env` stringified,
    /// plus `socket_ifname` folded in as NCCL_SOCKET_IFNAME.
    pub fn resolve(&self) -> Result<NcclEnv, NcclEnvError> {
        NcclEnv::resolve(self.socket_ifname.as_deref(), &stringify_raw(&self.env)?)
    }
}

impl FleetConfig {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        Self::load_for(path, None)
    }

    /// `load`, validated for the launch mode the command will actually use
    /// (`--launch` over `[launch] mode`): `gauntlet run --launch srun` on a
    /// config without hosts is valid, the allocation supplies them.
    pub fn load_for(path: &Path, cli_launch: Option<LaunchMode>) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let config = Self::parse(&text, path)?;
        config.validate_for(config.launch.effective_mode(cli_launch))?;
        Ok(config)
    }

    /// Parse and validate a config document; `path` only labels errors.
    pub fn from_toml_str(text: &str, path: &Path) -> Result<Self, ConfigError> {
        let config = Self::parse(text, path)?;
        config.validate()?;
        Ok(config)
    }

    fn parse(text: &str, path: &Path) -> Result<Self, ConfigError> {
        toml::from_str(text).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source: Box::new(source),
        })
    }

    /// Validate for the configured launch mode.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.validate_for(self.launch.mode)
    }

    /// Validate for an effective launch `mode` (which decides whether an
    /// empty host list is acceptable).
    pub fn validate_for(&self, mode: LaunchMode) -> Result<(), ConfigError> {
        self.require_hosts_for(mode)?;
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
        self.nccl_env()?;
        Ok(())
    }

    /// An empty host list is valid only where hosts can be inferred (srun
    /// launch). Checked against the *effective* mode, which a `--launch`
    /// override may change after `load`.
    pub fn require_hosts_for(&self, mode: LaunchMode) -> Result<(), ConfigError> {
        if self.hosts.is_empty() && mode != LaunchMode::Srun {
            return Err(ConfigError::NoHosts { mode });
        }
        Ok(())
    }

    /// The same config over a resolved host list (srun launch: the
    /// allocation, or the configured subset of it). Revalidated, so the
    /// result upholds every `validate` invariant, plus a non-empty fleet.
    pub fn with_hosts(&self, hosts: Vec<HostConfig>) -> Result<Self, ConfigError> {
        let mut config = self.clone();
        config.hosts = hosts.into_iter().map(HostEntry::Full).collect();
        if config.hosts.is_empty() {
            return Err(ConfigError::NoHosts {
                mode: config.launch.mode,
            });
        }
        config.validate_for(LaunchMode::Srun)?;
        Ok(config)
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
            overlap: self.overlap_spec(),
            // Counter passes are scheduled by the orchestrator as dedicated
            // invocations; a plain phase spec never carries one.
            counters: None,
            nccl_intranode: self.intranode_sweep_spec(),
        }
    }

    /// The resolved NCCL environment (`[nccl] env` plus `socket_ifname` as
    /// NCCL_SOCKET_IFNAME). Every agent spawn carries it on its remote
    /// command line (`HostSession`), and the results record it.
    ///
    /// Resolved once — by `validate`, which `load` always runs — and cached.
    /// On a config that skipped validation the first call resolves; either
    /// way every `NcclEnv` handed out passed `NcclConfig::resolve`, so an
    /// invalid env can never be observed. After `load` this cannot fail.
    pub fn nccl_env(&self) -> Result<&NcclEnv, ConfigError> {
        if let Some(env) = self.resolved_nccl_env.get() {
            return Ok(env);
        }
        let env = self
            .nccl
            .resolve()
            .map_err(|source| ConfigError::Nccl { source })?;
        Ok(self.resolved_nccl_env.get_or_init(|| env))
    }

    /// The intra-node sweep parameters, or `None` when the sweep is
    /// disabled. Same sizes and iteration count as the fleet sweep, so the
    /// hierarchy levels are directly comparable.
    pub fn intranode_sweep_spec(&self) -> Option<NcclSweepSpec> {
        let tests = &self.tests;
        tests.nccl_intranode.then(|| NcclSweepSpec {
            sizes: tests.nccl_sizes.clone(),
            iters_per_size: tests.nccl_iters_per_size,
        })
    }

    /// The overlap-step parameters, shared verbatim by the node-local
    /// phase (`AgentTaskSpec.overlap`) and the fleet step
    /// (`NcclWorkload::Overlap`): both measure the same contention, one
    /// topology level apart. The compute leg reuses the phase-2 dimension
    /// (so retention divides comparable numbers) and the *first* configured
    /// dtype only — it measures contention, not dtype coverage.
    pub fn overlap_spec(&self) -> OverlapSpec {
        let tests = &self.tests;
        OverlapSpec {
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
