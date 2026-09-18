//! Results assembly and rendering.
//!
//! `RunResults` is the schema-versioned JSON document `gauntlet run` writes;
//! it is the machine interface for simulation calibration, so field renames
//! bump SCHEMA_VERSION. The terminal table is a projection of it, never a
//! second source of truth.

pub mod history;

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

use anyhow::{Context, Result};
use comfy_table::{ContentArrangement, Table, presets::UTF8_FULL};
use serde::{Deserialize, Serialize};

use crate::analysis::fit::{AlphaBetaFit, fit_alpha_beta};
use crate::analysis::stats::{self, Moments, Outlier, Sample};
use crate::cli::ReportArgs;
use crate::config::{Bound, FleetConfig};
use crate::orchestrator::collect::HostObservations;
use crate::proto::{
    CounterDeltas, CounterDomain, Scope, TestId, TestOutcome, Unit, consistency_fields,
};

// v2: metric `aggregates` (per-subject Moments), `fleet.jitter_outliers`,
// and `repeat` on raw metric records.
// v3: hot SDC screens — `fleet.sdc_failures` plus the `cpu_sdc_hot` /
// `gpu_gemm_sdc` test groups appearing in outcomes and metrics.
// v4: per-host error-counter deltas (`hosts.*.counter_deltas`) and
// `fleet.counter_findings`.
pub const SCHEMA_VERSION: u32 = 4;

/// Bytes per GiB, for turning a GiB/s reading into microseconds per byte.
const BYTES_PER_GIB: f64 = (1u64 << 30) as f64;
/// Microseconds per second.
const MICROS_PER_SEC: f64 = 1_000_000.0;

/// Per-subject distribution summary of one metric group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricAggregate {
    pub unit: Unit,
    pub moments: Moments,
}

/// "<test>.<metric>" group -> sample key -> aggregate. Sweep series (a
/// subject emitting several values within one repeat, e.g. the NCCL
/// message-size sweeps) are excluded; they feed `calibration.links`.
pub type Aggregates = BTreeMap<String, BTreeMap<String, MetricAggregate>>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunResults {
    pub schema_version: u32,
    /// "<started epoch secs>-<6 hex chars>"; also the history filename stem.
    pub run_id: String,
    pub started_epoch_secs: u64,
    pub finished_epoch_secs: u64,
    /// True when the orchestrator (and therefore the deployed agent) was
    /// built without optimizations; such numbers are not comparable.
    #[serde(default)]
    pub debug_build: bool,
    pub hosts: BTreeMap<String, HostObservations>,
    pub fleet: FleetAnalysis,
    /// Per-subject distributions; n == 1 everywhere unless the run used
    /// `--repeat`.
    #[serde(default)]
    pub aggregates: Aggregates,
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
    /// Subjects whose run-to-run spread (MAD across repeats) is a fleet
    /// outlier on the high side: jitter, not slowness. Informational —
    /// does not affect the verdict. Empty unless the run used `--repeat`.
    #[serde(default)]
    pub jitter_outliers: BTreeMap<String, Vec<Outlier>>,
    /// Silent-data-corruption findings: Failed outcomes from the
    /// correctness screens (isolated and hot), grouped by test display
    /// name; entries are "host[:scope]: reason". Hard failures — these are
    /// absolute findings on a node, never fleet-relative outliers, and any
    /// entry makes the verdict at least `Stragglers`.
    #[serde(default)]
    pub sdc_failures: BTreeMap<String, Vec<String>>,
    /// Error counters that incremented across the load phases, per host.
    /// Any positive increment is a finding (marginal hardware accumulating
    /// errors under load); zero and negative deltas stay in
    /// `hosts.*.counter_deltas` only.
    #[serde(default)]
    pub counter_findings: BTreeMap<String, Vec<CounterFinding>>,
}

/// One error counter that went up between the pre-load and post-load
/// snapshots on a node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterFinding {
    pub domain: CounterDomain,
    /// The hardware unit: PCI address, "gpu0", "mc0/dimm1", "mlx5_0/1", ...
    pub device: String,
    pub counter: String,
    pub before: u64,
    pub after: u64,
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
    /// 1: completed with failed tests, outliers, or threshold violations.
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

// ---------------------------------------------------------------------------
// Keys: the naming contract shared with the orchestrator
// ---------------------------------------------------------------------------

/// Which tests a metric belongs to, for grouping/table sections.
pub fn test_display_name(test: TestId) -> &'static str {
    match test {
        TestId::Inventory => "inventory",
        TestId::CpuCorrectness => "cpu_correctness",
        TestId::CpuGflops => "cpu_gflops",
        TestId::CpuSdcHot => "cpu_sdc_hot",
        TestId::MemBandwidth => "mem_bandwidth",
        TestId::DiskIo => "disk_io",
        TestId::GpuGemmCorrectness => "gpu_gemm_correctness",
        TestId::GpuGemmPerf => "gpu_gemm_perf",
        TestId::GpuGemmSdc => "gpu_gemm_sdc",
        TestId::GpuMemBandwidth => "gpu_mem_bandwidth",
        TestId::GpuP2p => "gpu_p2p",
        TestId::NetLatency => "net_latency",
        TestId::NetBandwidth => "net_bandwidth",
        TestId::NcclAllReduce => "nccl_all_reduce",
        TestId::NcclAllGather => "nccl_all_gather",
    }
}

/// Comparison-group key for a metric: "<test>.<metric name>". Unit is a
/// function of (test, name), so a group never mixes units.
pub fn metric_key(test: TestId, name: &str) -> String {
    format!("{}.{}", test_display_name(test), name)
}

/// Stable rendering of a scope for use inside sample keys. `Node` has no
/// label because the host name already identifies it.
pub fn scope_label(scope: &Scope) -> Option<String> {
    match scope {
        Scope::Node => None,
        Scope::Core { id } => Some(format!("core{id}")),
        Scope::Numa { node } => Some(format!("numa{node}")),
        Scope::Gpu { index } => Some(format!("gpu{index}")),
        Scope::GpuPair { a, b } => Some(format!("gpupair{a}-{b}")),
        Scope::Disk { path } => Some(format!("disk:{path}")),
        Scope::HostPair { peer } => Some(format!("pair:{peer}")),
    }
}

/// Sample key within a comparison group: "host" for node-wide metrics,
/// "host:<scope>" for anything finer-grained.
pub fn sample_key(host: &str, scope: &Scope) -> String {
    match scope_label(scope) {
        None => host.to_string(),
        Some(label) => format!("{host}:{label}"),
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

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
    let aggregates = aggregate_metrics(&observations);
    let raw_groups = group_samples(&observations);

    // Fleet-relative straggler detection over per-subject medians, so
    // run-to-run noise inside one subject cannot masquerade as slowness.
    let mut outliers = BTreeMap::new();
    for (group, subjects) in &aggregates {
        let samples: Vec<Sample> = subjects
            .iter()
            .map(|(key, aggregate)| Sample {
                key: key.clone(),
                value: aggregate.moments.median,
            })
            .collect();
        let flagged = stats::flag_outliers(&samples, config.thresholds.mad_k);
        if !flagged.is_empty() {
            outliers.insert(group.clone(), flagged);
        }
    }

    // Jitter: a subject whose spread is a high-side fleet outlier. Only
    // meaningful with repeats (n >= 2).
    let mut jitter_outliers = BTreeMap::new();
    for (group, subjects) in &aggregates {
        let spreads: Vec<Sample> = subjects
            .iter()
            .filter(|(_, aggregate)| aggregate.moments.n >= 2)
            .map(|(key, aggregate)| Sample {
                key: key.clone(),
                value: aggregate.moments.mad,
            })
            .collect();
        let flagged: Vec<Outlier> = stats::flag_outliers(&spreads, config.thresholds.mad_k)
            .into_iter()
            .filter(|outlier| outlier.deviation_mads > 0.0)
            .collect();
        if !flagged.is_empty() {
            jitter_outliers.insert(group.clone(), flagged);
        }
    }

    // Absolute bounds check the per-subject median; sweep series (absent
    // from aggregates) stay a per-value opt-in as before.
    let mut threshold_violations = BTreeMap::new();
    for (group, bound) in &config.thresholds.absolute {
        let violators: Vec<String> = if let Some(subjects) = aggregates.get(group) {
            subjects
                .iter()
                .filter(|(_, aggregate)| violates(bound, aggregate.moments.median))
                .map(|(key, _)| key.clone())
                .collect()
        } else if let Some(samples) = raw_groups.get(group) {
            samples
                .iter()
                .filter(|sample| violates(bound, sample.value))
                .map(|sample| sample.key.clone())
                .collect()
        } else {
            continue;
        };
        if !violators.is_empty() {
            threshold_violations.insert(group.clone(), violators);
        }
    }

    let failed_hosts: BTreeMap<String, Vec<String>> = observations
        .iter()
        .filter(|(_, obs)| !obs.errors.is_empty())
        .map(|(host, obs)| (host.clone(), obs.errors.clone()))
        .collect();

    let fleet = FleetAnalysis {
        outliers,
        threshold_violations,
        consistency: consistency_findings(&observations),
        failed_hosts,
        jitter_outliers,
        sdc_failures: sdc_failures(&observations),
        counter_findings: counter_findings(&observations),
    };
    let calibration = Calibration {
        rooflines: rooflines(&observations),
        links: link_fits(&observations),
    };
    let run_id = format!(
        "{started_epoch_secs}-{}",
        run_suffix(finished_epoch_secs, &observations)
    );

    RunResults {
        schema_version: SCHEMA_VERSION,
        run_id,
        started_epoch_secs,
        finished_epoch_secs,
        debug_build: false,
        hosts: observations,
        fleet,
        aggregates,
        calibration,
    }
}

/// Reduce raw (possibly repeated) metric records into per-subject
/// `Moments`. A group where any (subject, repeat) pair occurs twice is a
/// per-host series (the sweeps) and is skipped wholesale.
pub fn aggregate_metrics(observations: &BTreeMap<String, HostObservations>) -> Aggregates {
    let mut raw: BTreeMap<String, Vec<(String, u32, f64, Unit)>> = BTreeMap::new();
    for (host, obs) in observations {
        for record in &obs.metrics {
            raw.entry(metric_key(record.test, &record.name))
                .or_default()
                .push((
                    sample_key(host, &record.scope),
                    record.repeat,
                    record.value,
                    record.unit,
                ));
        }
    }

    let mut aggregates = Aggregates::new();
    for (group, entries) in raw {
        let mut seen = BTreeSet::new();
        if !entries
            .iter()
            .all(|(subject, repeat, _, _)| seen.insert((subject.clone(), *repeat)))
        {
            continue;
        }
        let mut per_subject: BTreeMap<String, (Unit, Vec<f64>)> = BTreeMap::new();
        for (subject, _, value, unit) in entries {
            per_subject
                .entry(subject)
                .or_insert_with(|| (unit, Vec::new()))
                .1
                .push(value);
        }
        let subjects: BTreeMap<String, MetricAggregate> = per_subject
            .into_iter()
            .filter_map(|(subject, (unit, values))| {
                stats::moments(&values).map(|moments| (subject, MetricAggregate { unit, moments }))
            })
            .collect();
        if !subjects.is_empty() {
            aggregates.insert(group, subjects);
        }
    }
    aggregates
}

pub fn verdict(results: &RunResults) -> Verdict {
    if !results.fleet.failed_hosts.is_empty() {
        return Verdict::HostFailures;
    }
    let has_failed_tests = results.hosts.values().any(|obs| {
        obs.outcomes
            .iter()
            .any(|(_, _, outcome)| matches!(outcome, crate::proto::TestOutcome::Failed { .. }))
    });
    let has_outliers = results
        .fleet
        .outliers
        .values()
        .any(|flagged| !flagged.is_empty());
    let has_violations = results
        .fleet
        .threshold_violations
        .values()
        .any(|violators| !violators.is_empty());
    let has_counter_findings = results
        .fleet
        .counter_findings
        .values()
        .any(|findings| !findings.is_empty());
    if has_failed_tests || has_outliers || has_violations || has_counter_findings {
        Verdict::Stragglers
    } else {
        Verdict::Clean
    }
}

/// Error counters that went up under load, per host. Zero deltas stay in
/// the raw `counter_deltas`; negative deltas are resets, not errors.
fn counter_findings(
    observations: &BTreeMap<String, HostObservations>,
) -> BTreeMap<String, Vec<CounterFinding>> {
    let mut findings = BTreeMap::new();
    for (host, obs) in observations {
        let Some(CounterDeltas { deltas }) = &obs.counter_deltas else {
            continue;
        };
        let increments: Vec<CounterFinding> = deltas
            .iter()
            .filter(|delta| delta.after > delta.before)
            .map(|delta| CounterFinding {
                domain: delta.domain,
                device: delta.device.clone(),
                counter: delta.counter.clone(),
                before: delta.before,
                after: delta.after,
            })
            .collect();
        if !increments.is_empty() {
            findings.insert(host.clone(), increments);
        }
    }
    findings
}

/// Every metric in the fleet, bucketed into comparison groups.
fn group_samples(
    observations: &BTreeMap<String, HostObservations>,
) -> BTreeMap<String, Vec<Sample>> {
    let mut groups: BTreeMap<String, Vec<Sample>> = BTreeMap::new();
    for (host, obs) in observations {
        for record in &obs.metrics {
            groups
                .entry(metric_key(record.test, &record.name))
                .or_default()
                .push(Sample {
                    key: sample_key(host, &record.scope),
                    value: record.value,
                });
        }
    }
    groups
}

/// Tests whose Failed outcomes mean data corruption rather than degraded
/// performance. They share a dedicated report section because they are the
/// findings that silently poison training runs.
fn is_sdc_test(test: TestId) -> bool {
    matches!(
        test,
        TestId::CpuCorrectness
            | TestId::CpuSdcHot
            | TestId::GpuGemmCorrectness
            | TestId::GpuGemmSdc
    )
}

/// Failed correctness-screen outcomes, grouped by test display name.
/// Derived from `hosts[].outcomes` (never a second source of truth) so JSON
/// consumers get the hard findings without re-scanning every outcome.
fn sdc_failures(
    observations: &BTreeMap<String, HostObservations>,
) -> BTreeMap<String, Vec<String>> {
    let mut failures: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (host, obs) in observations {
        for (test, scope, outcome) in &obs.outcomes {
            if !is_sdc_test(*test) {
                continue;
            }
            if let TestOutcome::Failed { reason } = outcome {
                failures
                    .entry(test_display_name(*test).to_string())
                    .or_default()
                    .push(format!("{}: {reason}", sample_key(host, scope)));
            }
        }
    }
    failures
}

fn violates(bound: &Bound, value: f64) -> bool {
    if !value.is_finite() {
        return false;
    }
    bound.min.is_some_and(|min| value < min) || bound.max.is_some_and(|max| value > max)
}

/// Majority vote over `proto::consistency_fields`; only fields with at least
/// one dissenter are reported.
fn consistency_findings(
    observations: &BTreeMap<String, HostObservations>,
) -> BTreeMap<String, ConsistencyFinding> {
    let mut by_field: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for (host, obs) in observations {
        let Some(inventory) = &obs.inventory else {
            continue;
        };
        for (field, value) in consistency_fields(inventory) {
            by_field
                .entry(field)
                .or_default()
                .insert(host.clone(), value);
        }
    }

    let mut findings = BTreeMap::new();
    for (field, values) in by_field {
        let Some(majority_value) = majority(&values) else {
            continue;
        };
        let dissenters: BTreeMap<String, String> = values
            .into_iter()
            .filter(|(_, value)| *value != majority_value)
            .collect();
        if !dissenters.is_empty() {
            findings.insert(
                field,
                ConsistencyFinding {
                    majority_value,
                    dissenters,
                },
            );
        }
    }
    findings
}

/// Most common value; ties broken by the lexicographically smallest value so
/// the result never depends on map iteration luck.
fn majority(values: &BTreeMap<String, String>) -> Option<String> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for value in values.values() {
        *counts.entry(value.as_str()).or_default() += 1;
    }
    counts
        .into_iter()
        .min_by(|(a_value, a_count), (b_value, b_count)| {
            b_count.cmp(a_count).then_with(|| a_value.cmp(b_value))
        })
        .map(|(value, _)| value.to_string())
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

fn rooflines(observations: &BTreeMap<String, HostObservations>) -> BTreeMap<String, NodeRoofline> {
    observations
        .iter()
        .map(|(host, obs)| (host.clone(), roofline_for(obs)))
        .collect()
}

fn roofline_for(obs: &HostObservations) -> NodeRoofline {
    // Median per (metric, GPU) across repeats, then the slowest GPU
    // defines what the node can sustain.
    let mut per_gpu: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    for record in &obs.metrics {
        if record.test == TestId::GpuGemmPerf
            && record.name.starts_with("gflops_")
            && record.value.is_finite()
        {
            per_gpu
                .entry((
                    record.name.clone(),
                    scope_label(&record.scope).unwrap_or_default(),
                ))
                .or_default()
                .push(record.value);
        }
    }
    let mut gpu_gflops: BTreeMap<String, f64> = BTreeMap::new();
    for ((name, _scope), values) in per_gpu {
        if let Some(median) = stats::median(&values) {
            let slot = gpu_gflops.entry(name).or_insert(median);
            *slot = slot.min(median);
        }
    }

    NodeRoofline {
        gpu_gflops,
        cpu_gflops_allcore: reduce(obs, TestId::CpuGflops, "gflops_allcore", Reduce::Min),
        dram_gib_per_sec: reduce(obs, TestId::MemBandwidth, "triad_allnode", Reduce::Max)
            // Agents that only measured per-NUMA triad: the best node is the
            // closest stand-in for the all-node figure.
            .or_else(|| reduce(obs, TestId::MemBandwidth, "triad", Reduce::Max)),
        gpu_hbm_gib_per_sec: reduce(obs, TestId::GpuMemBandwidth, "d2d", Reduce::Min),
        pcie_h2d_gib_per_sec: reduce(obs, TestId::GpuMemBandwidth, "h2d_pinned", Reduce::Min),
        disk_read_gib_per_sec: reduce(obs, TestId::DiskIo, "seq_read", Reduce::Max),
        disk_write_gib_per_sec: reduce(obs, TestId::DiskIo, "seq_write", Reduce::Max),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reduce {
    /// Worst-case capability: the straggling GPU/core defines the node.
    Min,
    /// Best-case capability: distinct disks/NUMA nodes are not comparable,
    /// so the fastest one is the meaningful headline number.
    Max,
}

/// Reduce across subjects (cores, GPUs, disks, NUMA nodes) after taking
/// each subject's median across repeats.
fn reduce(obs: &HostObservations, test: TestId, name: &str, how: Reduce) -> Option<f64> {
    let mut per_subject: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for record in &obs.metrics {
        if record.test == test && record.name == name && record.value.is_finite() {
            per_subject
                .entry(scope_label(&record.scope).unwrap_or_default())
                .or_default()
                .push(record.value);
        }
    }
    per_subject
        .into_values()
        .filter_map(|values| stats::median(&values))
        .reduce(|a, b| match how {
            Reduce::Min => a.min(b),
            Reduce::Max => a.max(b),
        })
}

fn link_fits(observations: &BTreeMap<String, HostObservations>) -> BTreeMap<String, AlphaBetaFit> {
    let mut links = BTreeMap::new();
    for (test, key) in [
        (TestId::NcclAllReduce, "nccl_allreduce_fleet"),
        (TestId::NcclAllGather, "nccl_allgather_fleet"),
    ] {
        let points = sweep_points(observations, test);
        if let Ok(fit) = fit_alpha_beta(&points) {
            links.insert(key.to_string(), fit);
        }
    }
    if let Some(fit) = tcp_pairwise_fit(observations) {
        links.insert("tcp_pairwise".to_string(), fit);
    }
    links
}

/// Pair the `msg_bytes` and `elapsed_us` metrics of a collective sweep.
/// They are emitted as parallel metrics, one of each per message size, so
/// emission order within a host's metric list is the join key.
fn sweep_points(
    observations: &BTreeMap<String, HostObservations>,
    test: TestId,
) -> Vec<(u64, f64)> {
    let mut points = Vec::new();
    for obs in observations.values() {
        let mut sizes = Vec::new();
        let mut timings = Vec::new();
        for record in &obs.metrics {
            if record.test != test {
                continue;
            }
            match record.name.as_str() {
                "msg_bytes" => sizes.push(record.value),
                "elapsed_us" => timings.push(record.value),
                _ => {}
            }
        }
        for (bytes, elapsed_us) in sizes.into_iter().zip(timings) {
            if bytes.is_finite() && bytes >= 0.0 {
                points.push((bytes as u64, elapsed_us));
            }
        }
    }
    points
}

/// TCP pairwise is a two-point synthesis, not a regression: the peer tests
/// measure latency and streaming bandwidth separately (no size sweep), so
/// alpha comes from the median pairwise RTT and beta from the median
/// pairwise throughput. `r_squared` is 1.0 because the model is constructed
/// from exactly two measurements, not fitted — do not read it as evidence.
fn tcp_pairwise_fit(observations: &BTreeMap<String, HostObservations>) -> Option<AlphaBetaFit> {
    let rtts = collect_values(observations, TestId::NetLatency, "rtt_p50");
    let bandwidths = collect_values(observations, TestId::NetBandwidth, "gib_per_sec");
    let alpha_us = stats::median(&rtts)?;
    let gib_per_sec = stats::median(&bandwidths)?;
    if gib_per_sec <= 0.0 {
        return None;
    }
    Some(AlphaBetaFit {
        alpha_us,
        beta_us_per_byte: MICROS_PER_SEC / (gib_per_sec * BYTES_PER_GIB),
        r_squared: 1.0,
    })
}

fn collect_values(
    observations: &BTreeMap<String, HostObservations>,
    test: TestId,
    name: &str,
) -> Vec<f64> {
    observations
        .values()
        .flat_map(|obs| obs.metrics.iter())
        .filter(|record| record.test == test && record.name == name)
        .map(|record| record.value)
        .collect()
}

/// Six hex characters that distinguish two runs started in the same second.
/// FNV-1a over the finish time and host set: deterministic, dependency-free,
/// and not required to be unpredictable.
fn run_suffix(
    finished_epoch_secs: u64,
    observations: &BTreeMap<String, HostObservations>,
) -> String {
    id_suffix(finished_epoch_secs, observations.keys().map(String::as_str))
}

/// The run id a run will carry, known before any observation arrives: same
/// shape as `build`'s, seeded from the *start* time and the configured host
/// list rather than the finish time and the hosts heard from. In-flight
/// partial snapshots and the final document therefore share one id.
pub fn make_run_id(started_epoch_secs: u64, hosts: &[String]) -> String {
    format!(
        "{started_epoch_secs}-{}",
        id_suffix(started_epoch_secs, hosts.iter().map(String::as_str))
    )
}

/// FNV-1a over a timestamp then the host set (count first, so two host lists
/// cannot alias by concatenation), folded to six hex characters.
fn id_suffix<'a>(seed_epoch_secs: u64, hosts: impl ExactSizeIterator<Item = &'a str>) -> String {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn mix(hash: u64, bytes: &[u8]) -> u64 {
        bytes.iter().fold(hash, |acc, byte| {
            (acc ^ u64::from(*byte)).wrapping_mul(PRIME)
        })
    }

    let mut hash = mix(OFFSET, &seed_epoch_secs.to_le_bytes());
    hash = mix(hash, &(hosts.len() as u64).to_le_bytes());
    for host in hosts {
        hash = mix(hash, host.as_bytes());
    }
    format!("{:06x}", hash & 0xff_ffff)
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render the human table (per-host summary, outliers section, consistency
/// section, calibration digest) to the given writer.
pub fn render_table(results: &RunResults, out: &mut dyn Write) -> Result<()> {
    writeln!(
        out,
        "gauntlet run {} ({} hosts, {}s wall, verdict: {:?})",
        results.run_id,
        results.hosts.len(),
        results
            .finished_epoch_secs
            .saturating_sub(results.started_epoch_secs),
        verdict(results),
    )?;

    render_hosts(results, out)?;
    render_sdc(results, out)?;
    render_outliers(results, out)?;
    render_jitter(results, out)?;
    render_counter_findings(results, out)?;
    render_violations(results, out)?;
    render_consistency(results, out)?;
    render_failures(results, out)?;
    render_links(results, out)?;
    Ok(())
}

fn new_table(header: &[&str]) -> Table {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(header.to_vec());
    table
}

fn section(out: &mut dyn Write, title: &str, table: &Table) -> Result<()> {
    writeln!(out, "\n{title}")?;
    writeln!(out, "{table}")?;
    Ok(())
}

fn render_hosts(results: &RunResults, out: &mut dyn Write) -> Result<()> {
    let mut table = new_table(&[
        "host",
        "pass",
        "fail",
        "skip",
        "cpu gflops",
        "dram gib/s",
        "gpu hbm gib/s",
        "gpu gflops (min)",
    ]);
    for (host, obs) in &results.hosts {
        let (passed, failed, skipped) = outcome_counts(obs);
        let roofline = results.calibration.rooflines.get(host);
        table.add_row(vec![
            host.clone(),
            passed.to_string(),
            failed.to_string(),
            skipped.to_string(),
            optional(roofline.and_then(|r| r.cpu_gflops_allcore)),
            optional(roofline.and_then(|r| r.dram_gib_per_sec)),
            optional(roofline.and_then(|r| r.gpu_hbm_gib_per_sec)),
            roofline
                .map(|r| {
                    r.gpu_gflops
                        .iter()
                        .map(|(name, value)| format!("{name}={value:.0}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .filter(|text| !text.is_empty())
                .unwrap_or_else(|| "-".into()),
        ]);
    }
    section(out, "hosts", &table)
}

fn outcome_counts(obs: &HostObservations) -> (usize, usize, usize) {
    let mut counts = (0, 0, 0);
    for (_, _, outcome) in &obs.outcomes {
        match outcome {
            TestOutcome::Passed => counts.0 += 1,
            TestOutcome::Failed { .. } => counts.1 += 1,
            TestOutcome::Skipped { .. } => counts.2 += 1,
        }
    }
    counts
}

/// Hard correctness findings get their own section, above the
/// fleet-relative noise: a node computing wrong answers is never "just an
/// outlier".
fn render_sdc(results: &RunResults, out: &mut dyn Write) -> Result<()> {
    if results.fleet.sdc_failures.is_empty() {
        return Ok(());
    }
    let mut table = new_table(&["test", "subject", "reason"]);
    for (group, entries) in &results.fleet.sdc_failures {
        for entry in entries {
            let (subject, reason) = entry.split_once(": ").unwrap_or((entry.as_str(), ""));
            table.add_row(vec![group.clone(), subject.to_string(), reason.to_string()]);
        }
    }
    section(out, "silent data corruption (hard failures)", &table)
}

fn render_outliers(results: &RunResults, out: &mut dyn Write) -> Result<()> {
    if results.fleet.outliers.is_empty() {
        writeln!(out, "\noutliers: none")?;
        return Ok(());
    }
    let mut table = new_table(&["subject", "metric", "value", "fleet median", "mads"]);
    for (group, flagged) in &results.fleet.outliers {
        for outlier in flagged {
            table.add_row(vec![
                outlier.key.clone(),
                group.clone(),
                format!("{:.3}", outlier.value),
                format!("{:.3}", outlier.fleet_median),
                format!("{:+.1}", outlier.deviation_mads),
            ]);
        }
    }
    section(out, "outliers (fleet-relative)", &table)
}

fn render_jitter(results: &RunResults, out: &mut dyn Write) -> Result<()> {
    if results.fleet.jitter_outliers.is_empty() {
        return Ok(());
    }
    let mut table = new_table(&["subject", "metric", "spread (MAD)", "fleet spread", "mads"]);
    for (group, flagged) in &results.fleet.jitter_outliers {
        for outlier in flagged {
            table.add_row(vec![
                outlier.key.clone(),
                group.clone(),
                format!("{:.3}", outlier.value),
                format!("{:.3}", outlier.fleet_median),
                format!("{:+.1}", outlier.deviation_mads),
            ]);
        }
    }
    section(
        out,
        "jitter outliers (fleet-relative run-to-run spread)",
        &table,
    )
}

/// Only counters that actually incremented appear; a run with no findings
/// omits the section entirely (the full delta list lives in the JSON).
fn render_counter_findings(results: &RunResults, out: &mut dyn Write) -> Result<()> {
    if results.fleet.counter_findings.is_empty() {
        return Ok(());
    }
    let mut table = new_table(&[
        "host", "domain", "device", "counter", "before", "after", "+",
    ]);
    for (host, findings) in &results.fleet.counter_findings {
        for finding in findings {
            table.add_row(vec![
                host.clone(),
                finding.domain.label().to_string(),
                finding.device.clone(),
                finding.counter.clone(),
                finding.before.to_string(),
                finding.after.to_string(),
                format!("+{}", finding.after.saturating_sub(finding.before)),
            ]);
        }
    }
    section(out, "error-counter deltas (across load phases)", &table)
}

fn render_violations(results: &RunResults, out: &mut dyn Write) -> Result<()> {
    if results.fleet.threshold_violations.is_empty() {
        return Ok(());
    }
    let mut table = new_table(&["metric", "subjects"]);
    for (group, violators) in &results.fleet.threshold_violations {
        table.add_row(vec![group.clone(), violators.join(", ")]);
    }
    section(out, "absolute threshold violations", &table)
}

fn render_consistency(results: &RunResults, out: &mut dyn Write) -> Result<()> {
    if results.fleet.consistency.is_empty() {
        writeln!(out, "\nconsistency: fleet agrees on every inventory field")?;
        return Ok(());
    }
    let mut table = new_table(&["field", "majority", "dissenter", "value"]);
    for (field, finding) in &results.fleet.consistency {
        for (host, value) in &finding.dissenters {
            table.add_row(vec![
                field.clone(),
                finding.majority_value.clone(),
                host.clone(),
                value.clone(),
            ]);
        }
    }
    section(out, "inventory consistency", &table)
}

fn render_failures(results: &RunResults, out: &mut dyn Write) -> Result<()> {
    if results.fleet.failed_hosts.is_empty() {
        return Ok(());
    }
    let mut table = new_table(&["host", "errors"]);
    for (host, errors) in &results.fleet.failed_hosts {
        table.add_row(vec![host.clone(), errors.join("; ")]);
    }
    section(out, "failed hosts", &table)
}

fn render_links(results: &RunResults, out: &mut dyn Write) -> Result<()> {
    if results.calibration.links.is_empty() {
        return Ok(());
    }
    let mut table = new_table(&["link class", "alpha (us)", "beta (us/B)", "gib/s", "r^2"]);
    for (class, fit) in &results.calibration.links {
        table.add_row(vec![
            class.clone(),
            format!("{:.2}", fit.alpha_us),
            format!("{:.3e}", fit.beta_us_per_byte),
            format!("{:.2}", fit.bandwidth_gib_per_sec()),
            format!("{:.4}", fit.r_squared),
        ]);
    }
    section(out, "calibration: link alpha-beta", &table)
}

fn optional(value: Option<f64>) -> String {
    value.map_or_else(|| "-".into(), |value| format!("{value:.1}"))
}

/// `gauntlet report`: load a saved results JSON and re-render it.
pub fn render_saved(args: ReportArgs) -> Result<()> {
    let results = history::load(&args.input)?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    if args.json {
        serde_json::to_writer_pretty(&mut out, &results).context("writing results JSON")?;
        writeln!(out)?;
        Ok(())
    } else {
        render_table(&results, &mut out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{MetricRecord, Unit};

    fn metric(test: TestId, scope: Scope, name: &str, value: f64) -> MetricRecord {
        MetricRecord {
            test,
            scope,
            name: name.into(),
            value,
            unit: Unit::GibPerSec,
            repeat: 0,
        }
    }

    #[test]
    fn scope_labels_are_stable() {
        assert_eq!(scope_label(&Scope::Node), None);
        assert_eq!(
            scope_label(&Scope::Core { id: 3 }).as_deref(),
            Some("core3")
        );
        assert_eq!(
            scope_label(&Scope::Numa { node: 0 }).as_deref(),
            Some("numa0")
        );
        assert_eq!(
            scope_label(&Scope::Gpu { index: 1 }).as_deref(),
            Some("gpu1")
        );
        assert_eq!(
            scope_label(&Scope::GpuPair { a: 0, b: 2 }).as_deref(),
            Some("gpupair0-2")
        );
        assert_eq!(
            scope_label(&Scope::Disk {
                path: "/tmp".into()
            })
            .as_deref(),
            Some("disk:/tmp")
        );
        assert_eq!(
            scope_label(&Scope::HostPair { peer: "n7".into() }).as_deref(),
            Some("pair:n7")
        );
    }

    #[test]
    fn sample_keys_distinguish_scopes_within_a_host() {
        assert_eq!(sample_key("n1", &Scope::Node), "n1");
        assert_eq!(sample_key("n1", &Scope::Gpu { index: 2 }), "n1:gpu2");
        assert_eq!(
            sample_key("n1", &Scope::HostPair { peer: "n2".into() }),
            "n1:pair:n2"
        );
    }

    #[test]
    fn metric_keys_join_test_and_name() {
        assert_eq!(
            metric_key(TestId::MemBandwidth, "triad"),
            "mem_bandwidth.triad"
        );
        assert_eq!(
            metric_key(TestId::NcclAllReduce, "elapsed_us"),
            "nccl_all_reduce.elapsed_us"
        );
    }

    #[test]
    fn majority_breaks_ties_by_smallest_value() {
        let values = BTreeMap::from([
            ("n1".to_string(), "b".to_string()),
            ("n2".to_string(), "a".to_string()),
        ]);
        assert_eq!(majority(&values).as_deref(), Some("a"));
        // Reversed host order must not change the answer.
        let reversed = BTreeMap::from([
            ("n1".to_string(), "a".to_string()),
            ("n2".to_string(), "b".to_string()),
        ]);
        assert_eq!(majority(&reversed).as_deref(), Some("a"));
        assert_eq!(majority(&BTreeMap::new()), None);
    }

    #[test]
    fn majority_prefers_count_over_value_order() {
        let values = BTreeMap::from([
            ("n1".to_string(), "z".to_string()),
            ("n2".to_string(), "z".to_string()),
            ("n3".to_string(), "a".to_string()),
        ]);
        assert_eq!(majority(&values).as_deref(), Some("z"));
    }

    #[test]
    fn bounds_only_flag_finite_out_of_range_values() {
        let bound = Bound {
            min: Some(10.0),
            max: Some(20.0),
        };
        assert!(violates(&bound, 9.9));
        assert!(violates(&bound, 20.1));
        assert!(!violates(&bound, 10.0));
        assert!(!violates(&bound, 20.0));
        assert!(!violates(&bound, f64::NAN));
    }

    #[test]
    fn run_suffix_is_six_lowercase_hex_and_deterministic() {
        let hosts = BTreeMap::from([("n1".to_string(), HostObservations::default())]);
        let first = run_suffix(1_700_000_600, &hosts);
        assert_eq!(first.len(), 6);
        assert!(
            first
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
        assert_eq!(first, run_suffix(1_700_000_600, &hosts));
        assert_ne!(first, run_suffix(1_700_000_601, &hosts));
    }

    #[test]
    fn disk_and_gpu_rooflines_reduce_in_opposite_directions() {
        let mut obs = HostObservations::default();
        obs.metrics.push(metric(
            TestId::GpuMemBandwidth,
            Scope::Gpu { index: 0 },
            "d2d",
            3000.0,
        ));
        obs.metrics.push(metric(
            TestId::GpuMemBandwidth,
            Scope::Gpu { index: 1 },
            "d2d",
            2400.0,
        ));
        obs.metrics.push(metric(
            TestId::DiskIo,
            Scope::Disk {
                path: "/slow".into(),
            },
            "seq_read",
            0.5,
        ));
        obs.metrics.push(metric(
            TestId::DiskIo,
            Scope::Disk {
                path: "/fast".into(),
            },
            "seq_read",
            6.5,
        ));
        let roofline = roofline_for(&obs);
        assert_eq!(roofline.gpu_hbm_gib_per_sec, Some(2400.0));
        assert_eq!(roofline.disk_read_gib_per_sec, Some(6.5));
    }

    #[test]
    fn dram_falls_back_to_per_numa_triad() {
        let mut obs = HostObservations::default();
        obs.metrics.push(metric(
            TestId::MemBandwidth,
            Scope::Numa { node: 0 },
            "triad",
            190.0,
        ));
        obs.metrics.push(metric(
            TestId::MemBandwidth,
            Scope::Numa { node: 1 },
            "triad",
            210.0,
        ));
        assert_eq!(roofline_for(&obs).dram_gib_per_sec, Some(210.0));

        obs.metrics.push(metric(
            TestId::MemBandwidth,
            Scope::Node,
            "triad_allnode",
            380.0,
        ));
        assert_eq!(roofline_for(&obs).dram_gib_per_sec, Some(380.0));
    }

    #[test]
    fn sweep_points_pair_sizes_with_timings_in_emission_order() {
        let mut obs = HostObservations::default();
        for (bytes, us) in [(1024.0, 30.0), (4096.0, 45.0)] {
            obs.metrics.push(metric(
                TestId::NcclAllReduce,
                Scope::Node,
                "msg_bytes",
                bytes,
            ));
            obs.metrics
                .push(metric(TestId::NcclAllReduce, Scope::Node, "elapsed_us", us));
            obs.metrics.push(metric(
                TestId::NcclAllReduce,
                Scope::Node,
                "bus_gib_per_sec",
                12.0,
            ));
        }
        let observations = BTreeMap::from([("n1".to_string(), obs)]);
        assert_eq!(
            sweep_points(&observations, TestId::NcclAllReduce),
            vec![(1024, 30.0), (4096, 45.0)]
        );
        let links = link_fits(&observations);
        assert!(links.contains_key("nccl_allreduce_fleet"));
        assert!(!links.contains_key("tcp_pairwise"));
    }

    #[test]
    fn nccl_sweeps_never_produce_outliers() {
        let config: FleetConfig =
            toml::from_str(r#"hosts = ["n1", "n2", "n3", "n4"]"#).expect("config");
        let mut observations = BTreeMap::new();
        for host in ["n1", "n2", "n3", "n4"] {
            let mut obs = HostObservations::default();
            for (bytes, us) in [(1024.0, 30.0), (65_536.0, 60.0), (1_048_576.0, 900.0)] {
                obs.metrics.push(metric(
                    TestId::NcclAllReduce,
                    Scope::Node,
                    "msg_bytes",
                    bytes,
                ));
                obs.metrics
                    .push(metric(TestId::NcclAllReduce, Scope::Node, "elapsed_us", us));
            }
            observations.insert(host.to_string(), obs);
        }
        let results = build(&config, observations, 1, 2);
        assert!(results.fleet.outliers.is_empty(), "{:?}", results.fleet);
        assert_eq!(verdict(&results), Verdict::Clean);
        assert!(
            results
                .calibration
                .links
                .contains_key("nccl_allreduce_fleet")
        );
    }

    #[test]
    fn tcp_synthesis_uses_median_latency_and_bandwidth() {
        let mut obs = HostObservations::default();
        obs.metrics.push(metric(
            TestId::NetLatency,
            Scope::HostPair { peer: "n2".into() },
            "rtt_p50",
            40.0,
        ));
        obs.metrics.push(metric(
            TestId::NetBandwidth,
            Scope::HostPair { peer: "n2".into() },
            "gib_per_sec",
            10.0,
        ));
        let observations = BTreeMap::from([("n1".to_string(), obs)]);
        let fit = tcp_pairwise_fit(&observations).expect("fit");
        assert_eq!(fit.alpha_us, 40.0);
        assert!((fit.bandwidth_gib_per_sec() - 10.0).abs() < 1e-6, "{fit:?}");
    }
}
