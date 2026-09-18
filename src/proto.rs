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
pub const PROTO_VERSION: u32 = 5;

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
    /// Barrier-skew microbenchmark over the NCCL group (tiny all-reduce).
    NcclBarrier,
    /// Barrier-skew microbenchmark over a TCP star (CPU-only fallback).
    TcpBarrier,
    /// GEMM throughput measured while the intra-node all-reduce runs.
    OverlapGemm,
    /// Intra-node all-reduce bandwidth: isolated baseline and under GEMM load.
    OverlapAllReduce,
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
    pub overlap: OverlapTaskSpec,
    /// Error-counter pass to run after the listed phases; the orchestrator
    /// sends this in dedicated invocations with an empty phase list.
    #[serde(default)]
    pub counters: Option<CounterRequest>,
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

/// Phase "overlap": sustained GEMM on every GPU concurrently with an
/// intra-node NCCL all-reduce across those same GPUs. An isolated all-reduce
/// baseline (`baseline_secs`) precedes the combined window so the collective
/// retention ratio compares like against like — same communicator, same
/// topology, seconds apart. The GEMM retention baseline is the phase-2
/// sustained number, joined orchestrator-side.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlapTaskSpec {
    /// Wall seconds of combined GEMM + all-reduce load.
    pub duration_secs: u64,
    /// Wall seconds of the isolated intra-node all-reduce baseline.
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "directive", rename_all = "snake_case")]
pub enum NcclDirective {
    /// Rank 0: mint the rendezvous id, announce it as an `NcclId` event, and
    /// stay alive through the whole sweep (the id's bootstrap listen socket
    /// lives in this process).
    Lead {
        world_size: u32,
        /// Message sizes in bytes for the sweep.
        sizes: Vec<u64>,
        iters_per_size: u32,
        /// Value for NCCL_SOCKET_IFNAME, if the cluster needs it.
        socket_ifname: Option<String>,
        /// Run the barrier-skew microbenchmark after the sweep.
        #[serde(default)]
        barrier: Option<BarrierSpec>,
    },
    /// Ranks 1..n: join the lead's communicator and run the sweep silently
    /// (barrier timings are the one thing participants report).
    Participate {
        unique_id_b64: String,
        rank: u32,
        world_size: u32,
        sizes: Vec<u64>,
        iters_per_size: u32,
        socket_ifname: Option<String>,
        #[serde(default)]
        barrier: Option<BarrierSpec>,
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

    #[test]
    fn directives_without_a_barrier_field_still_decode() {
        // Wire output from a pre-v2 orchestrator build.
        let old = r#"{"directive":"lead","world_size":4,"sizes":[1024],"iters_per_size":20,"socket_ifname":null}"#;
        let directive: NcclDirective = serde_json::from_str(old).expect("decode old lead");
        let NcclDirective::Lead { barrier, .. } = directive else {
            panic!("expected a Lead directive");
        };
        assert_eq!(barrier, None);
    }

    #[test]
    fn barrier_specs_ride_the_directive() {
        let directive = NcclDirective::Participate {
            unique_id_b64: "abc".into(),
            rank: 2,
            world_size: 4,
            sizes: vec![1024],
            iters_per_size: 20,
            socket_ifname: Some("bond0".into()),
            barrier: Some(BarrierSpec {
                iters: 2000,
                bytes: 8,
            }),
        };
        let json = serde_json::to_string(&directive).expect("serialize");
        let back: NcclDirective = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, directive);
    }
}
