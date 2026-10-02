//! Wire contract between orchestrator and agent.
//!
//! Direction agent -> orchestrator: newline-delimited `AgentEvent` JSON on
//! stdout. The first event MUST be `Hello`; the orchestrator refuses to
//! proceed on a `proto_version` mismatch (it re-deploys the agent instead).
//!
//! Direction orchestrator -> agent: a single JSON document on stdin
//! (`AgentTaskSpec` for `agent run`, `NcclDirective` for `agent nccl`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

mod occupancy;
mod ranks;

pub use occupancy::{
    GpuIdleAssessment, GpuOccupancy, GpuProcess, MemoryOverage, ProcessOwner, assess_gpu_idle,
    gpu_idle_outcomes,
};
pub use ranks::{RankAssignment, RankBlock, RankError};

// v2: hot silent-data-corruption screens — `TestId::{CpuSdcHot,GpuGemmSdc}`
// on the wire plus `CpuTaskSpec::sdc_hot_secs` / `GpuTaskSpec::sdc_check_secs`
// in the task spec (which is `deny_unknown_fields`, so a v1 agent would
// reject a v2 spec; the version handshake forces a re-deploy instead).
// v3: error-counter snapshot/delta events (`counter_baseline`,
// `counter_deltas`) and the `counters` request on `AgentTaskSpec`.
// v4: overlap phase (`Phase::Overlap`, `AgentTaskSpec.overlap`, overlap test
// ids).
// v5: barrier-skew microbenchmark — `NcclDirective` variants carry an
// optional `barrier` spec and every rank reports `NcclBarrierTimings`.
// v6: fleet overlap — the NCCL directives carry a `NcclWorkload` (the
// message-size sweep or the combined GEMM + fleet all-reduce protocol,
// mutually exclusive by construction), every rank reports
// `OverlapFleetReport`, and the overlap_fleet test ids ride the wire.
// v7: rank-per-GPU fleet NCCL world — the directives carry a validated
// `RankAssignment` (the host's contiguous `RankBlock` plus world size)
// instead of a single rank, one process drives every local rank, and each
// `OverlapFleetReport` covers exactly one rank (= one GPU); inventories
// carry `cuda_visible_gpus` (rank blocks are sized from it), `agent nccl`
// sends Hello before anything can fail, and exits with
// `AGENT_EXIT_CASCADE` when it stopped because the fleet stopped.
// v8: intra-node NCCL sweep — `AgentTaskSpec.nccl_intranode` and the
// `nccl_intra_all_reduce` / `nccl_intra_all_gather` test ids.
// v9: NCCL env passthrough — `socket_ifname` leaves the NCCL directives
// (now `deny_unknown_fields`). The resolved NCCL env (socket_ifname folded
// in as NCCL_SOCKET_IFNAME) is set on the remote `env` command line at
// spawn, never on the wire.
// v10: GPU occupancy — every `GpuInventory` carries a `GpuOccupancy`
// (memory used/total plus the compute processes other than the reporting
// agent, stale gauntlet agents marked as such), and the `gpu_idle` test id
// rides the wire. Serde-defaulted, so older inventories decode as
// "occupancy unknown".
// v11: NCCL world shapes — `RankBlock` carries the local GPU its ranks
// start on (`first_gpu`, serde-defaulted to 0 and omitted when 0), the
// sweep workload carries its `SweepSeries` (serde-defaulted to
// rank-per-GPU), `NcclWorkload::BarrierOnly` runs the barrier probe alone,
// and the `nccl_inter_all_reduce` / `nccl_inter_all_gather` test ids ride
// the wire. The lead of every fleet sweep also emits a fleet-level
// `bus_gib_per_sec_peak` (or `bus_gib_per_sec_peak_rail<r>`) headline.
pub const PROTO_VERSION: u32 = 11;

/// Exit status of an `agent nccl` process that stopped *itself* because
/// the fleet stopped, not because of a fault on its own host: the fleet
/// overlap hard-deadline watchdog, or a follower whose lead never signalled
/// a window close. The orchestrator classifies it like a timeout — a
/// secondary failure attributed to whichever host failed first. (124 is
/// coreutils `timeout`'s status, for the same reason.)
pub const AGENT_EXIT_CASCADE: i32 = 124;

#[derive(Debug, Error)]
pub enum ProtoError {
    #[error("malformed event line: {source}")]
    Malformed {
        #[from]
        source: serde_json::Error,
    },
    #[error("agent speaks proto v{agent}, orchestrator requires v{required}")]
    VersionMismatch { agent: u32, required: u32 },
    #[error("first event was not hello: {got}")]
    MissingHello { got: String },
}

// ---------------------------------------------------------------------------
// Events (agent -> orchestrator)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum AgentEvent {
    Hello {
        proto_version: u32,
        hostname: String,
    },
    PhaseStart {
        phase: Phase,
    },
    Inventory {
        /// Boxed: the snapshot dwarfs every other variant, and events move
        /// through channels by value.
        snapshot: Box<InventorySnapshot>,
    },
    Metric {
        #[serde(flatten)]
        record: MetricRecord,
    },
    Outcome {
        test: TestId,
        scope: Scope,
        outcome: TestOutcome,
    },
    Log {
        level: LogLevel,
        message: String,
    },
    PhaseEnd {
        phase: Phase,
    },
    /// NCCL lead rank only: the freshly minted rendezvous id, emitted before
    /// communicator init. The orchestrator relays it to the other ranks. It
    /// must come from the process that stays alive as rank 0:
    /// ncclGetUniqueId opens the bootstrap listen socket in the calling
    /// process, so a mint-and-exit helper leaves every rank connecting to a
    /// dead port.
    NcclId {
        unique_id_b64: String,
    },
    /// Barrier-skew microbenchmark: one event per NCCL rank carrying that
    /// rank's local per-iteration completion times (microseconds) of the
    /// tiny all-reduce. Emitted by *every* rank (this is the one place
    /// participants speak); the phase-3 driver intercepts and merges them
    /// before running `analysis::skew`, so one reaching the collector is a
    /// stray.
    NcclBarrierTimings {
        rank: u32,
        elapsed_us: Vec<f64>,
    },
    /// Fleet overlap step: one event per NCCL rank carrying that rank's
    /// local results — isolated and overlapped all-reduce bus bandwidth
    /// plus the overlapped GEMM throughput of every local GPU. Emitted by
    /// *every* rank (the other place participants speak); the overlap
    /// driver intercepts and merges them, so one reaching the collector is
    /// a stray.
    OverlapFleetReport {
        report: Box<OverlapFleetReport>,
    },
    /// Error-counter snapshot taken before the load phases. Boxed for the
    /// same reason as `Inventory`. The orchestrator intercepts and holds it;
    /// it never reaches the collector on the happy path.
    CounterBaseline {
        snapshot: Box<CounterSnapshot>,
    },
    /// Per-node counter deltas across the load phases, computed on the agent
    /// from the baseline the orchestrator handed back in the task spec.
    CounterDeltas {
        deltas: Box<CounterDeltas>,
    },
    /// Unrecoverable agent-side failure; always the last event if emitted.
    Fatal {
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Inventory,
    CpuMem,
    Gpu,
    Network,
    /// Sustained GEMM concurrent with an intra-node NCCL all-reduce. Runs
    /// last: its straggler signal is *retention* against the isolated
    /// phase-2 GEMM baselines from the same run.
    Overlap,
}

impl Phase {
    pub const ALL: [Phase; 5] = [
        Phase::Inventory,
        Phase::CpuMem,
        Phase::Gpu,
        Phase::Network,
        Phase::Overlap,
    ];

    pub fn parse(s: &str) -> Option<Phase> {
        match s {
            "inventory" => Some(Phase::Inventory),
            "cpu_mem" | "cpu" => Some(Phase::CpuMem),
            "gpu" => Some(Phase::Gpu),
            "network" | "net" => Some(Phase::Network),
            "overlap" => Some(Phase::Overlap),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestId {
    Inventory,
    /// Per-GPU occupancy check derived from the inventory: Failed when a
    /// foreign compute process holds the GPU or memory use exceeds
    /// `thresholds.gpu_idle_max_used_mib`. Derived orchestrator-side from
    /// the snapshot (the threshold is orchestrator config); agents never
    /// emit this test id.
    GpuIdle,
    CpuCorrectness,
    CpuGflops,
    /// Correctness screen re-run while every core burns power: silent data
    /// corruption is temperature/voltage dependent, so the cold screen alone
    /// is not sufficient.
    CpuSdcHot,
    MemBandwidth,
    DiskIo,
    GpuGemmCorrectness,
    GpuGemmPerf,
    /// Periodic bitwise output verification during the sustained (hot) GEMM.
    GpuGemmSdc,
    GpuMemBandwidth,
    GpuP2p,
    NetLatency,
    NetBandwidth,
    NcclAllReduce,
    NcclAllGather,
    /// Intra-node all-reduce sweep: one rank per local GPU, single process
    /// (`ncclCommInitAll`), NVLink/PCIe only.
    NcclIntraAllReduce,
    /// Intra-node all-gather sweep, same communicator as the all-reduce.
    NcclIntraAllGather,
    /// Pure inter-node all-reduce sweep (`tests.nccl_world` =
    /// `rank_per_node` or `per_rail`): one rank per host, so every peer is
    /// on another node and all collective traffic crosses the NIC.
    NcclInterAllReduce,
    /// Pure inter-node all-gather sweep, same world as the all-reduce.
    NcclInterAllGather,
    /// Barrier-skew microbenchmark over the NCCL group (tiny all-reduce).
    NcclBarrier,
    /// Barrier-skew microbenchmark over a TCP star (CPU-only fallback).
    TcpBarrier,
    /// GEMM throughput measured while the intra-node all-reduce runs.
    OverlapGemm,
    /// Intra-node all-reduce bandwidth: isolated baseline and under GEMM load.
    OverlapAllReduce,
    /// GEMM throughput measured while the fleet-wide all-reduce runs.
    OverlapFleetGemm,
    /// Fleet all-reduce bus bandwidth, per rank: isolated baseline and under
    /// fleet-wide GEMM load.
    OverlapFleetAllReduce,
    /// Derived orchestrator-side (`report::build`): overlapped/isolated
    /// ratios. Agents never emit this test id.
    OverlapRetention,
}

/// What a metric is *about*. Per-core / per-GPU granularity is the point:
/// a single bad core or downtrained link must not hide in a node average.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Scope {
    Node,
    Core {
        id: u32,
    },
    Numa {
        node: u32,
    },
    Gpu {
        index: u32,
    },
    GpuPair {
        a: u32,
        b: u32,
    },
    Disk {
        path: String,
    },
    /// Set by the orchestrator when attributing pairwise results; agents
    /// never emit it (they only know their own end).
    HostPair {
        peer: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricRecord {
    pub test: TestId,
    pub scope: Scope,
    /// Metric name within the test, e.g. "gflops", "rtt_p99", "residual".
    pub name: String,
    pub value: f64,
    pub unit: Unit,
    /// Which `--repeat` iteration produced this sample. Agents always emit
    /// 0; the orchestrator stamps the real index on ingestion, so old wire
    /// output (field absent) still decodes.
    #[serde(default)]
    pub repeat: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    Gflops,
    GibPerSec,
    Micros,
    Millis,
    Celsius,
    Mhz,
    Bytes,
    Count,
    /// Error vs a reference result: relative error against a
    /// higher-precision recomputation, or absolute deviation from a
    /// baseline output of the identical computation.
    Residual,
    Ratio,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum TestOutcome {
    Passed,
    Failed { reason: String },
    Skipped { reason: String },
}

// ---------------------------------------------------------------------------
// Inventory
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventorySnapshot {
    pub hostname: String,
    pub kernel: String,
    pub cpu_model: String,
    pub logical_cores: u32,
    pub numa_nodes: u32,
    pub mem_total_bytes: u64,
    pub cpu_governor: Option<String>,
    /// Offset reported by chrony/ntp, if available.
    pub clock_offset_ms: Option<f64>,
    pub nvidia_driver: Option<String>,
    pub cuda_version: Option<String>,
    pub gpus: Vec<GpuInventory>,
    pub nics: Vec<NicInventory>,
    pub ib_ports: Vec<IbPortInventory>,
    /// Xid error codes seen in the kernel log since boot.
    pub xid_errors: Vec<u32>,
    /// Runtime-loadability of the GPU library stack ("cuda", "cublas",
    /// "nccl" -> dlopen succeeded). This is what the GPU/NCCL phases will
    /// actually experience, unlike ldconfig or nvidia-smi presence.
    #[serde(default)]
    pub gpu_libs: BTreeMap<String, bool>,
    /// GPUs the CUDA driver can actually open (`cuDeviceGetCount` in the
    /// agent), as opposed to the nvidia-smi listing in `gpus`. The fleet
    /// NCCL world sizes each host's rank block from this; `None` when the
    /// agent could not ask CUDA (no driver, or built without the gpu
    /// feature). See [`gpu_visibility_mismatch`].
    #[serde(default)]
    pub cuda_visible_gpus: Option<u32>,
}

/// The per-host inventory finding for a GPU-visibility mismatch: nvidia-smi
/// lists GPUs that CUDA does not expose (a GPU fell off the bus, MIG is on,
/// or CUDA_VISIBLE_DEVICES is set in the agent's environment). Such a host
/// runs fewer NCCL ranks than it has GPUs. `None` when the counts agree or
/// the host lists no GPUs at all.
pub fn gpu_visibility_mismatch(inventory: &InventorySnapshot) -> Option<String> {
    let listed = inventory.gpus.len();
    if listed == 0 {
        return None;
    }
    match inventory.cuda_visible_gpus {
        Some(visible) if visible as usize == listed => None,
        Some(visible) => Some(format!(
            "nvidia-smi lists {listed} GPUs but CUDA can open {visible} \
             (GPU off the bus, MIG enabled, or CUDA_VISIBLE_DEVICES set?)"
        )),
        None => Some(format!(
            "nvidia-smi lists {listed} GPUs but CUDA could not enumerate devices"
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuInventory {
    pub index: u32,
    pub name: String,
    pub uuid: String,
    pub vbios: String,
    pub mem_total_bytes: u64,
    pub ecc_volatile_errors: Option<u64>,
    pub remapped_rows_pending: Option<bool>,
    pub pcie_gen_current: Option<u32>,
    pub pcie_gen_max: Option<u32>,
    pub pcie_width_current: Option<u32>,
    pub pcie_width_max: Option<u32>,
    pub nvlinks_active: Option<u32>,
    pub persistence_mode: Option<bool>,
    /// Who else is on this GPU at probe time (proto v10). Defaults to
    /// "nothing known" for older inventories.
    #[serde(default)]
    pub occupancy: GpuOccupancy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NicInventory {
    pub name: String,
    pub mtu: u32,
    pub speed_mbps: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IbPortInventory {
    pub device: String,
    pub port: u32,
    pub state: String,
    pub rate_gbps: Option<f64>,
    pub link_downed_count: Option<u64>,
}

// ---------------------------------------------------------------------------
// Error counters
// ---------------------------------------------------------------------------

/// Which hardware subsystem an error counter belongs to. Order matters only
/// for deterministic rendering (deltas sort by (domain, device, counter)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CounterDomain {
    PcieAer,
    GpuEcc,
    GpuXid,
    Nvlink,
    Edac,
    IbPort,
    Nvme,
}

impl CounterDomain {
    pub fn label(self) -> &'static str {
        match self {
            CounterDomain::PcieAer => "pcie_aer",
            CounterDomain::GpuEcc => "gpu_ecc",
            CounterDomain::GpuXid => "gpu_xid",
            CounterDomain::Nvlink => "nvlink",
            CounterDomain::Edac => "edac",
            CounterDomain::IbPort => "ib_port",
            CounterDomain::Nvme => "nvme",
        }
    }
}

/// One monotonic error counter at one instant. Identity is
/// (domain, device, counter); a snapshot never carries the same identity
/// twice.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterReading {
    pub domain: CounterDomain,
    /// The hardware unit the counter belongs to: a PCI address
    /// ("0000:65:00.0"), "gpu0", "mc0/dimm1", "mlx5_0/1" (device/port),
    /// "gpu0/link1", "nvme0", or "dmesg" for the kernel-log Xid tally.
    pub device: String,
    pub counter: String,
    pub value: u64,
}

/// Everything counted on a node at one instant. Absence of a subsystem
/// (no IB, no NVMe, ...) is simply absence of its readings, never an error.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterSnapshot {
    pub readings: Vec<CounterReading>,
}

/// Before/after values of one counter across the load phases.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterDelta {
    pub domain: CounterDomain,
    pub device: String,
    pub counter: String,
    pub before: u64,
    pub after: u64,
}

impl CounterDelta {
    /// Signed change. Negative means the counter reset between snapshots
    /// (driver reload, log rotation) — recorded, but not an error finding.
    pub fn increment(&self) -> i128 {
        i128::from(self.after) - i128::from(self.before)
    }
}

/// Full delta list for a node, zero deltas included: the JSON document keeps
/// everything; only nonzero increments become report findings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterDeltas {
    pub deltas: Vec<CounterDelta>,
}

/// What the orchestrator wants from a counter pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum CounterRequest {
    /// Snapshot now and emit `CounterBaseline`.
    Baseline,
    /// Snapshot now, diff against `baseline`, emit `CounterDeltas`.
    Delta { baseline: CounterSnapshot },
}

// ---------------------------------------------------------------------------
// Task specs (orchestrator -> agent)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentTaskSpec {
    pub phases: Vec<Phase>,
    pub cpu: CpuTaskSpec,
    pub mem: MemTaskSpec,
    pub disk: DiskTaskSpec,
    pub gpu: GpuTaskSpec,
    pub overlap: OverlapSpec,
    /// Error-counter pass to run after the listed phases; the orchestrator
    /// sends this in dedicated invocations with an empty phase list.
    #[serde(default)]
    pub counters: Option<CounterRequest>,
    /// Intra-node NCCL message-size sweep, run from the network phase.
    /// `None` disables it (`tests.nccl_intranode = false`).
    #[serde(default)]
    pub nccl_intranode: Option<NcclSweepSpec>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuTaskSpec {
    pub correctness_secs_per_core: u64,
    pub gflops_secs: u64,
    /// Wall seconds for the hot SDC screen: correctness rounds interleaved
    /// with the power-heavy FMA workload on every core simultaneously, so
    /// correctness is exercised at max package power/temperature. 0 disables
    /// (the phase emits a Skipped outcome). Additional to — never a
    /// replacement for — the isolated per-core screen.
    #[serde(default)]
    pub sdc_hot_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemTaskSpec {
    pub buffer_bytes_per_numa: u64,
    pub iters: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiskTaskSpec {
    pub paths: Vec<String>,
    pub file_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuTaskSpec {
    pub gemm_secs: u64,
    pub gemm_dtypes: Vec<GemmDtype>,
    /// Square GEMM dimension for the sustained-perf run.
    pub gemm_dim: u32,
    pub bandwidth_bytes: u64,
    /// Loaded seconds between bitwise output checks during the sustained
    /// GEMM (the hot SDC screen). Verification happens outside the timed
    /// throughput windows, so it never pollutes the reported GFLOPS.
    /// 0 disables (a Skipped outcome is emitted).
    #[serde(default)]
    pub sdc_check_secs: u64,
}

/// Message-size sweep parameters for the node-local (intra-node) NCCL
/// sweep. The same knobs as the fleet sweep (`tests.nccl_sizes`,
/// `tests.nccl_iters_per_size`), so the two hierarchy levels measure the
/// same sizes with the same iteration counts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NcclSweepSpec {
    /// Message sizes in bytes.
    pub sizes: Vec<u64>,
    pub iters_per_size: u32,
}

/// Barrier-skew metric names shared by the emitter
/// (`orchestrator::barrier`) and the report's fleet-level rule.
pub mod barrier_metric {
    /// Prefix of the fleet-level barrier-span distribution
    /// (`fleet_span_{p50,p90,p99,max}_us`): one series per run, attributed
    /// to the lead host, never MAD-compared (`report::fleet_nccl`).
    pub const FLEET_SPAN_PREFIX: &str = "fleet_span";
    pub const FLEET_SPAN_P50_US: &str = "fleet_span_p50_us";
    pub const FLEET_SPAN_P90_US: &str = "fleet_span_p90_us";
    pub const FLEET_SPAN_P99_US: &str = "fleet_span_p99_us";
    pub const FLEET_SPAN_MAX_US: &str = "fleet_span_max_us";
}

/// Metric names shared by the NCCL sweep emitters (fleet and intra-node)
/// and the report's link-fit extraction, so a renamed string cannot
/// silently break the join.
pub mod nccl_metric {
    /// Mean microseconds per collective at one message size.
    pub const ELAPSED_US: &str = "elapsed_us";
    /// Message size of the sweep point (the gathered size for all-gather).
    pub const MSG_BYTES: &str = "msg_bytes";
    /// Bus bandwidth at one message size.
    pub const BUS: &str = "bus_gib_per_sec";
    /// Prefix of every sweep headline: the best bus bandwidth across the
    /// sweep's sizes. The intra-node headline is always keyed by
    /// communicator size (`bus_peak`); the fleet-level headlines are
    /// `BUS_PEAK` (bare) and `bus_peak_rail`.
    pub const BUS_PEAK_PREFIX: &str = "bus_gib_per_sec_peak";
    /// Fleet-level headline of a fleet or inter-node sweep (proto v11):
    /// one value per run, attributed to the world's lead host at node
    /// scope. Not fleet-comparable within a run (it has no peers); the
    /// report aggregates it across repeats but keeps it out of MAD.
    pub const BUS_PEAK: &str = BUS_PEAK_PREFIX;
    /// Intra-node only: ranks (local GPUs) in the communicator; keys the
    /// intra-node calibration link class.
    pub const RANKS: &str = "ranks";

    /// Rail suffix of the per-rail headlines: `rail<r>`.
    pub fn rail_suffix(rail: u32) -> String {
        format!("rail{rail}")
    }

    /// Per-rail inter-node headline: `bus_gib_per_sec_peak_rail<r>`, the
    /// best bus bandwidth of rail `r`'s world (GPU `r` of every host that
    /// has one), attributed to that rail's lead host.
    pub fn bus_peak_rail(rail: u32) -> String {
        format!("{BUS_PEAK_PREFIX}_{}", rail_suffix(rail))
    }

    /// Topology suffix shared by the intra-node headline and the intra-node
    /// calibration link classes: `<n>gpu`.
    pub fn gpu_class_suffix(gpus: u32) -> String {
        format!("{gpus}gpu")
    }

    /// Intra-node headline name for a communicator of `gpus` ranks:
    /// `bus_gib_per_sec_peak_<n>gpu`. Different GPU counts are different
    /// links (8-GPU NVLink vs 4-GPU PCIe), so each gets its own metric —
    /// and therefore its own MAD comparison group — just like its own
    /// calibration link class.
    pub fn bus_peak(gpus: u32) -> String {
        format!("{BUS_PEAK_PREFIX}_{}", gpu_class_suffix(gpus))
    }
}

/// Parameters of an overlap step: sustained GEMM on every GPU concurrently
/// with an NCCL all-reduce. One shared shape for both steps — the
/// intra-node phase (`AgentTaskSpec.overlap`) and the fleet step
/// (`NcclWorkload::Overlap`) measure the same contention, one topology
/// level apart. An isolated all-reduce baseline (`baseline_secs`) precedes
/// the combined window so the collective retention ratio compares like
/// against like — same communicator, same topology, seconds apart. The
/// GEMM retention baseline is the phase-2 sustained number, joined
/// orchestrator-side.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlapSpec {
    /// Wall seconds of combined GEMM + all-reduce load. In the fleet step
    /// this is rank 0's clock; the window consensus propagates the
    /// boundary.
    pub duration_secs: u64,
    /// Wall seconds of the isolated all-reduce baseline.
    pub baseline_secs: u64,
    /// Square GEMM dimension for the compute leg (same as the phase-2 dim,
    /// so retention divides comparable numbers).
    pub gemm_dim: u32,
    /// Single dtype for the compute leg: the phase measures contention,
    /// not dtype coverage.
    pub gemm_dtype: GemmDtype,
    /// All-reduce message size in bytes.
    pub msg_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmDtype {
    F32,
    Tf32,
    Bf16,
    F16,
}

impl GemmDtype {
    /// Metric-name suffix ("gflops_<tag>", "residual_<tag>"), shared by the
    /// agent emitters and the orchestrator/report side so both spell the
    /// same group keys.
    pub fn tag(self) -> &'static str {
        match self {
            GemmDtype::F32 => "f32",
            GemmDtype::Tf32 => "tf32",
            GemmDtype::Bf16 => "bf16",
            GemmDtype::F16 => "f16",
        }
    }
}

/// Metric names shared between the overlap emitters (agent and
/// orchestrator) and the report's retention derivation, so a renamed
/// string cannot silently break the join.
pub mod overlap_metric {
    /// Actual all-reduce payload bytes per iteration.
    pub const MSG_BYTES: &str = "msg_bytes";
    /// Bus bandwidth of the isolated (quiet-GPU) baseline window.
    pub const ISOLATED_BUS: &str = "isolated_bus_gib_per_sec";
    /// Bus bandwidth of the combined-load window.
    pub const OVERLAP_BUS: &str = "overlap_bus_gib_per_sec";
}

/// One rank's fleet-overlap results; the fleet world runs one rank per
/// GPU, so this is one GPU's report. Bus bandwidths are the rank's *local*
/// timings (arrival skew makes them differ across ranks — that spread is
/// part of the signal); `gemm` is the overlapped compute leg on the same
/// GPU. The orchestrator maps `rank` to (host, local GPU) through its rank
/// layout, so the report carries no separate GPU index that could
/// disagree with it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlapFleetReport {
    pub rank: u32,
    /// Actual payload bytes moved per all-reduce (the requested size
    /// rounded to whole f32 elements, never zero).
    pub msg_bytes: u64,
    pub isolated_bus_gib_per_sec: f64,
    pub overlap_bus_gib_per_sec: f64,
    pub gemm: OverlapGemmLeg,
}

/// Outcome of one GPU's fleet-overlap compute leg. A GPU whose GEMM worker
/// failed still ran its collective rank, so failure is a value here, not a
/// dead rank.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum OverlapGemmLeg {
    Ok { gflops: f64 },
    Failed { reason: String },
}

/// Barrier-skew microbenchmark parameters, appended to the NCCL sweep when
/// present: many iterations of a tiny all-reduce with per-iteration local
/// timing on every rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BarrierSpec {
    pub iters: u32,
    /// Payload in bytes; rounded up to one f32 element by the agent.
    pub bytes: u64,
}

/// What a fleet NCCL group runs once the communicator is up. One workload
/// per invocation, mutually exclusive by construction — no sentinel values
/// and no precedence rules between co-resident options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NcclWorkload {
    /// Message-size sweep (rank 0 emits the measurements), optionally
    /// followed by the barrier-skew microbenchmark.
    Sweep {
        /// Message sizes in bytes.
        sizes: Vec<u64>,
        iters_per_size: u32,
        #[serde(default)]
        barrier: Option<BarrierSpec>,
        /// Which world shape this sweep's world was laid out in, and so
        /// which test ids and headline name the lead emits under (proto
        /// v11; absent = rank-per-GPU).
        #[serde(default)]
        series: SweepSeries,
    },
    /// Fleet overlap protocol: isolated all-reduce baseline, then the same
    /// all-reduce under GEMM load on every local GPU; every rank reports
    /// an `OverlapFleetReport`.
    Overlap(OverlapSpec),
    /// The barrier-skew microbenchmark alone (proto v11): run on the
    /// rank-per-GPU world when the sweep itself ran in a different world
    /// shape, so barrier subjects stay `host:gpuN` whatever the shape.
    BarrierOnly(BarrierSpec),
}

/// Which fleet sweep a `NcclWorkload::Sweep` is: the world shape it was
/// laid out in (`tests.nccl_world`). Decides the test ids of the lead's
/// per-size series and the name of its headline:
///
/// - `RankPerGpu` (default): `nccl_all_*`, headline
///   `bus_gib_per_sec_peak`.
/// - `RankPerNode`: one rank per host on its GPU 0, every peer on another
///   node — `nccl_inter_all_*`, headline `bus_gib_per_sec_peak`.
/// - `Rail { rail }`: one rank per host on its GPU `rail` —
///   `nccl_inter_all_*`, headline `bus_gib_per_sec_peak_rail<r>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case", deny_unknown_fields)]
pub enum SweepSeries {
    #[default]
    RankPerGpu,
    RankPerNode,
    Rail {
        rail: u32,
    },
}

impl SweepSeries {
    /// Whether every peer of the world is on another node, so the sweep
    /// measures the NIC path alone.
    pub fn is_inter_node(self) -> bool {
        !matches!(self, SweepSeries::RankPerGpu)
    }

    /// Name of the headline the world's lead emits for this sweep.
    pub fn headline(self) -> String {
        match self {
            SweepSeries::RankPerGpu | SweepSeries::RankPerNode => nccl_metric::BUS_PEAK.to_string(),
            SweepSeries::Rail { rail } => nccl_metric::bus_peak_rail(rail),
        }
    }

    /// The two test ids this sweep's records land under (all-reduce,
    /// all-gather).
    pub fn tests(self) -> [TestId; 2] {
        if self.is_inter_node() {
            [TestId::NcclInterAllReduce, TestId::NcclInterAllGather]
        } else {
            [TestId::NcclAllReduce, TestId::NcclAllGather]
        }
    }
}

impl std::fmt::Display for SweepSeries {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SweepSeries::RankPerGpu => f.write_str("rank-per-gpu"),
            SweepSeries::RankPerNode => f.write_str("rank-per-node"),
            SweepSeries::Rail { rail } => write!(f, "rail {rail}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "directive", rename_all = "snake_case", deny_unknown_fields)]
pub enum NcclDirective {
    /// The host holding global rank 0 (its block starts at 0): mint the
    /// rendezvous id, announce it as an `NcclId` event, and stay alive
    /// through the whole workload (the id's bootstrap listen socket lives
    /// in this process). Drives every rank of its block.
    Lead {
        assignment: RankAssignment,
        workload: NcclWorkload,
    },
    /// Every other host: join the lead's communicator with every rank of
    /// its block and run the workload silently (barrier timings and
    /// fleet-overlap reports are the two things participants report).
    Participate {
        unique_id_b64: String,
        assignment: RankAssignment,
        workload: NcclWorkload,
    },
}

// ---------------------------------------------------------------------------
// Encoding / decoding
// ---------------------------------------------------------------------------

pub fn encode_event(event: &AgentEvent) -> String {
    // Serialization of these enums cannot fail: no non-string map keys, no
    // non-finite float rejection at this layer.
    serde_json::to_string(event).expect("AgentEvent serialization is infallible")
}

pub fn decode_event(line: &str) -> Result<AgentEvent, ProtoError> {
    Ok(serde_json::from_str(line)?)
}

/// Validate the first event of a stream and extract the hostname.
pub fn expect_hello(first: &AgentEvent) -> Result<String, ProtoError> {
    match first {
        AgentEvent::Hello {
            proto_version,
            hostname,
        } => {
            if *proto_version != PROTO_VERSION {
                Err(ProtoError::VersionMismatch {
                    agent: *proto_version,
                    required: PROTO_VERSION,
                })
            } else {
                Ok(hostname.clone())
            }
        }
        other => Err(ProtoError::MissingHello {
            got: format!("{other:?}"),
        }),
    }
}

fn absent() -> String {
    "(absent)".to_string()
}

/// Consistency-relevant fields extracted from an inventory, keyed by field
/// name. Fleet analysis flags any host whose value differs from the majority.
pub fn consistency_fields(inv: &InventorySnapshot) -> BTreeMap<String, String> {
    let mut fields = BTreeMap::new();
    fields.insert("kernel".into(), inv.kernel.clone());
    fields.insert("cpu_model".into(), inv.cpu_model.clone());
    // Present-vs-absent is itself a skew signal (one node with a driver the
    // others lack), so Option fields dissent through a sentinel instead of
    // being skipped.
    fields.insert(
        "nvidia_driver".into(),
        inv.nvidia_driver.clone().unwrap_or_else(absent),
    );
    fields.insert(
        "cuda_version".into(),
        inv.cuda_version.clone().unwrap_or_else(absent),
    );
    fields.insert("gpu_count".into(), inv.gpus.len().to_string());
    if !inv.gpus.is_empty() {
        for (lib, available) in &inv.gpu_libs {
            fields.insert(
                format!("lib:{lib}"),
                if *available { "present" } else { "absent" }.to_string(),
            );
        }
    }
    if let Some(gpu) = inv.gpus.first() {
        fields.insert("gpu_model".into(), gpu.name.clone());
        fields.insert("vbios".into(), gpu.vbios.clone());
    }
    for nic in &inv.nics {
        fields.insert(format!("mtu:{}", nic.name), nic.mtu.to_string());
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn barrier_timings_events_round_trip() {
        let event = AgentEvent::NcclBarrierTimings {
            rank: 3,
            elapsed_us: vec![12.5, 240.0, 11.75],
        };
        let line = encode_event(&event);
        assert!(line.contains("nccl_barrier_timings"), "{line}");
        assert_eq!(decode_event(&line).expect("decode"), event);
    }

    fn assignment(base: u32, count: u32, world_size: u32) -> RankAssignment {
        RankAssignment::new(RankBlock::new(base, count).expect("block"), world_size)
            .expect("assignment")
    }

    #[test]
    fn sweep_workloads_default_the_barrier_absent() {
        // A sweep directive written without the optional barrier probe.
        let json = r#"{"directive":"lead",
            "assignment":{"block":{"base":0,"count":8},"world_size":16},
            "workload":{"kind":"sweep","sizes":[1024],"iters_per_size":20}}"#;
        let directive: NcclDirective = serde_json::from_str(json).expect("decode sweep lead");
        let NcclDirective::Lead {
            assignment: decoded,
            workload: NcclWorkload::Sweep { barrier, .. },
            ..
        } = directive
        else {
            panic!("expected a Lead sweep directive");
        };
        assert_eq!(decoded, assignment(0, 8, 16));
        assert_eq!(barrier, None);
    }

    #[test]
    fn sweep_workloads_default_to_the_rank_per_gpu_series() {
        // A pre-v11 sweep workload has no series: it is the rank-per-GPU
        // sweep.
        let json = r#"{"kind":"sweep","sizes":[1024],"iters_per_size":20}"#;
        let workload: NcclWorkload = serde_json::from_str(json).expect("decode");
        let NcclWorkload::Sweep { series, .. } = workload else {
            panic!("expected a sweep");
        };
        assert_eq!(series, SweepSeries::RankPerGpu);
    }

    #[test]
    fn sweep_series_round_trip_and_name_their_headline() {
        for (series, json, headline, tests) in [
            (
                SweepSeries::RankPerGpu,
                r#"{"shape":"rank_per_gpu"}"#,
                "bus_gib_per_sec_peak",
                [TestId::NcclAllReduce, TestId::NcclAllGather],
            ),
            (
                SweepSeries::RankPerNode,
                r#"{"shape":"rank_per_node"}"#,
                "bus_gib_per_sec_peak",
                [TestId::NcclInterAllReduce, TestId::NcclInterAllGather],
            ),
            (
                SweepSeries::Rail { rail: 7 },
                r#"{"shape":"rail","rail":7}"#,
                "bus_gib_per_sec_peak_rail7",
                [TestId::NcclInterAllReduce, TestId::NcclInterAllGather],
            ),
        ] {
            assert_eq!(serde_json::to_string(&series).expect("serialize"), json);
            let back: SweepSeries = serde_json::from_str(json).expect("deserialize");
            assert_eq!(back, series);
            assert_eq!(series.headline(), headline);
            assert_eq!(series.tests(), tests);
            assert_eq!(series.is_inter_node(), series != SweepSeries::RankPerGpu);
        }
        assert!(serde_json::from_str::<SweepSeries>(r#"{"shape":"rail"}"#).is_err());
        for (test, wire) in [
            (TestId::NcclInterAllReduce, "\"nccl_inter_all_reduce\""),
            (TestId::NcclInterAllGather, "\"nccl_inter_all_gather\""),
        ] {
            assert_eq!(serde_json::to_string(&test).expect("serialize"), wire);
        }
    }

    #[test]
    fn barrier_only_workloads_round_trip() {
        let directive = NcclDirective::Lead {
            assignment: assignment(0, 4, 8),
            workload: NcclWorkload::BarrierOnly(BarrierSpec {
                iters: 2000,
                bytes: 8,
            }),
        };
        let json = serde_json::to_string(&directive).expect("serialize");
        assert!(json.contains(r#""kind":"barrier_only""#), "{json}");
        let back: NcclDirective = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, directive);
    }

    #[test]
    fn rail_headline_names_carry_the_rail() {
        assert_eq!(nccl_metric::bus_peak_rail(0), "bus_gib_per_sec_peak_rail0");
        assert_eq!(nccl_metric::rail_suffix(3), "rail3");
        assert!(nccl_metric::bus_peak_rail(2).starts_with(nccl_metric::BUS_PEAK_PREFIX));
        assert_eq!(nccl_metric::BUS_PEAK, "bus_gib_per_sec_peak");
    }

    #[test]
    fn barrier_specs_ride_the_sweep_workload() {
        let directive = NcclDirective::Participate {
            unique_id_b64: "abc".into(),
            assignment: assignment(8, 8, 24),
            workload: NcclWorkload::Sweep {
                sizes: vec![1024],
                iters_per_size: 20,
                barrier: Some(BarrierSpec {
                    iters: 2000,
                    bytes: 8,
                }),
                series: SweepSeries::RankPerGpu,
            },
        };
        let json = serde_json::to_string(&directive).expect("serialize");
        let back: NcclDirective = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, directive);
    }

    #[test]
    fn overlap_workloads_ride_the_directive() {
        let directive = NcclDirective::Lead {
            assignment: assignment(0, 4, 12),
            workload: NcclWorkload::Overlap(OverlapSpec {
                duration_secs: 30,
                baseline_secs: 5,
                gemm_dim: 8192,
                gemm_dtype: GemmDtype::Bf16,
                msg_bytes: 64 << 20,
            }),
        };
        let json = serde_json::to_string(&directive).expect("serialize");
        assert!(json.contains(r#""kind":"overlap""#), "{json}");
        let back: NcclDirective = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, directive);
    }

    #[test]
    fn stale_directive_fields_fail_loudly() {
        // socket_ifname (now set on the spawn command line) must not be
        // silently ignored, for either variant.
        for json in [
            r#"{"directive":"lead",
                "assignment":{"block":{"base":0,"count":2},"world_size":4},
                "socket_ifname":"bond0",
                "workload":{"kind":"sweep","sizes":[1024],"iters_per_size":20}}"#,
            r#"{"directive":"participate","unique_id_b64":"abc",
                "assignment":{"block":{"base":2,"count":2},"world_size":4},
                "socket_ifname":null,
                "workload":{"kind":"sweep","sizes":[1024],"iters_per_size":20}}"#,
        ] {
            let error = serde_json::from_str::<NcclDirective>(json).expect_err("stale field");
            assert!(error.to_string().contains("socket_ifname"), "{error}");
        }
        // The same documents without it decode.
        let clean = r#"{"directive":"participate","unique_id_b64":"abc",
            "assignment":{"block":{"base":2,"count":2},"world_size":4},
            "workload":{"kind":"sweep","sizes":[1024],"iters_per_size":20}}"#;
        assert!(serde_json::from_str::<NcclDirective>(clean).is_ok());
    }

    #[test]
    fn directives_with_an_invalid_rank_block_do_not_decode() {
        // The v6 single-rank shape is gone, and a block reaching past the
        // world is rejected at decode time rather than inside the agent.
        let v6 = r#"{"directive":"participate","unique_id_b64":"abc","rank":1,
            "world_size":2,
            "workload":{"kind":"sweep","sizes":[1024],"iters_per_size":20}}"#;
        assert!(serde_json::from_str::<NcclDirective>(v6).is_err());
        let outside = r#"{"directive":"participate","unique_id_b64":"abc",
            "assignment":{"block":{"base":4,"count":8},"world_size":8},
            "workload":{"kind":"sweep","sizes":[1024],"iters_per_size":20}}"#;
        assert!(serde_json::from_str::<NcclDirective>(outside).is_err());
    }

    #[test]
    fn overlap_fleet_reports_round_trip() {
        for gemm in [
            OverlapGemmLeg::Ok { gflops: 91_000.0 },
            OverlapGemmLeg::Failed {
                reason: "worker panicked".into(),
            },
        ] {
            let event = AgentEvent::OverlapFleetReport {
                report: Box::new(OverlapFleetReport {
                    rank: 9,
                    msg_bytes: 64 << 20,
                    isolated_bus_gib_per_sec: 42.5,
                    overlap_bus_gib_per_sec: 31.25,
                    gemm,
                }),
            };
            let line = encode_event(&event);
            assert!(line.contains("overlap_fleet_report"), "{line}");
            assert_eq!(decode_event(&line).expect("decode"), event);
        }
    }

    fn inventory_with(listed: usize, visible: Option<u32>) -> InventorySnapshot {
        let gpu = GpuInventory {
            index: 0,
            name: "H100".into(),
            uuid: "u".into(),
            vbios: "v".into(),
            mem_total_bytes: 1,
            ecc_volatile_errors: None,
            remapped_rows_pending: None,
            pcie_gen_current: None,
            pcie_gen_max: None,
            pcie_width_current: None,
            pcie_width_max: None,
            nvlinks_active: None,
            persistence_mode: None,
            occupancy: GpuOccupancy::default(),
        };
        InventorySnapshot {
            hostname: "n1".into(),
            kernel: "6.8".into(),
            cpu_model: "x".into(),
            logical_cores: 8,
            numa_nodes: 1,
            mem_total_bytes: 1,
            cpu_governor: None,
            clock_offset_ms: None,
            nvidia_driver: None,
            cuda_version: None,
            gpus: vec![gpu; listed],
            nics: vec![],
            ib_ports: vec![],
            xid_errors: vec![],
            gpu_libs: BTreeMap::new(),
            cuda_visible_gpus: visible,
        }
    }

    #[test]
    fn gpu_visibility_mismatches_are_findings() {
        assert_eq!(gpu_visibility_mismatch(&inventory_with(8, Some(8))), None);
        assert_eq!(gpu_visibility_mismatch(&inventory_with(0, None)), None);
        let fewer = gpu_visibility_mismatch(&inventory_with(8, Some(7))).expect("finding");
        assert!(
            fewer.contains("lists 8") && fewer.contains("open 7"),
            "{fewer}"
        );
        assert!(gpu_visibility_mismatch(&inventory_with(8, None)).is_some());
    }

    #[test]
    fn inventories_without_a_cuda_count_still_decode() {
        let mut value = serde_json::to_value(inventory_with(2, Some(2))).expect("to value");
        value
            .as_object_mut()
            .expect("object")
            .remove("cuda_visible_gpus");
        let back: InventorySnapshot = serde_json::from_value(value).expect("decode");
        assert_eq!(back.cuda_visible_gpus, None);
    }

    #[test]
    fn dtype_tags_are_stable_metric_suffixes() {
        assert_eq!(GemmDtype::F32.tag(), "f32");
        assert_eq!(GemmDtype::Tf32.tag(), "tf32");
        assert_eq!(GemmDtype::Bf16.tag(), "bf16");
        assert_eq!(GemmDtype::F16.tag(), "f16");
    }
}
