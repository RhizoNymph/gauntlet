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
use crate::analysis::stats::{self, Outlier, Sample};
use crate::cli::ReportArgs;
use crate::config::{Bound, FleetConfig};
use crate::orchestrator::collect::HostObservations;
use crate::proto::{Scope, TestId, TestOutcome, consistency_fields};

pub const SCHEMA_VERSION: u32 = 1;

/// Bytes per GiB, for turning a GiB/s reading into microseconds per byte.
const BYTES_PER_GIB: f64 = (1u64 << 30) as f64;
/// Microseconds per second.
const MICROS_PER_SEC: f64 = 1_000_000.0;

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

// ---------------------------------------------------------------------------
// Keys: the naming contract shared with the orchestrator
// ---------------------------------------------------------------------------

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
    let groups = group_samples(&observations);

    let mut outliers = BTreeMap::new();
    for (group, samples) in &groups {
        if !is_fleet_comparable(samples) {
            continue;
        }
        let flagged = stats::flag_outliers(samples, config.thresholds.mad_k);
        if !flagged.is_empty() {
            outliers.insert(group.clone(), flagged);
        }
    }

    let mut threshold_violations = BTreeMap::new();
    for (group, bound) in &config.thresholds.absolute {
        let Some(samples) = groups.get(group) else {
            continue;
        };
        let violators: Vec<String> = samples
            .iter()
            .filter(|sample| violates(bound, sample.value))
            .map(|sample| sample.key.clone())
            .collect();
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
        hosts: observations,
        fleet,
        calibration,
    }
}

pub fn verdict(results: &RunResults) -> Verdict {
    if !results.fleet.failed_hosts.is_empty() {
        return Verdict::HostFailures;
    }
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
    if has_outliers || has_violations {
        Verdict::Stragglers
    } else {
        Verdict::Clean
    }
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

/// A group is a fleet comparison only when every sample key is distinct.
/// A repeated key means the metric is a per-host series rather than one
/// reading per subject — the NCCL sweeps emit `msg_bytes`/`elapsed_us` once
/// per message size, where spread across the series is the design, not a
/// straggler signal. Those feed `calibration.links` instead. Absolute
/// thresholds still apply to them: a bound is an explicit per-value opt-in.
fn is_fleet_comparable(samples: &[Sample]) -> bool {
    let mut seen = BTreeSet::new();
    samples
        .iter()
        .all(|sample| seen.insert(sample.key.as_str()))
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
    let mut gpu_gflops: BTreeMap<String, f64> = BTreeMap::new();
    for record in &obs.metrics {
        if record.test == TestId::GpuGemmPerf
            && record.name.starts_with("gflops_")
            && record.value.is_finite()
        {
            // The slowest GPU defines what the node can sustain.
            let slot = gpu_gflops
                .entry(record.name.clone())
                .or_insert(record.value);
            *slot = slot.min(record.value);
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

fn reduce(obs: &HostObservations, test: TestId, name: &str, how: Reduce) -> Option<f64> {
    obs.metrics
        .iter()
        .filter(|record| record.test == test && record.name == name && record.value.is_finite())
        .map(|record| record.value)
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
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn mix(hash: u64, bytes: &[u8]) -> u64 {
        bytes.iter().fold(hash, |acc, byte| {
            (acc ^ u64::from(*byte)).wrapping_mul(PRIME)
        })
    }

    let mut hash = mix(OFFSET, &finished_epoch_secs.to_le_bytes());
    hash = mix(hash, &(observations.len() as u64).to_le_bytes());
    for host in observations.keys() {
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
    render_outliers(results, out)?;
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
    fn sweep_series_are_not_treated_as_a_fleet_comparison() {
        let unique = vec![
            Sample {
                key: "n1".into(),
                value: 1.0,
            },
            Sample {
                key: "n2".into(),
                value: 2.0,
            },
        ];
        assert!(is_fleet_comparable(&unique));
        let series = vec![
            Sample {
                key: "n1".into(),
                value: 1.0,
            },
            Sample {
                key: "n1".into(),
                value: 2.0,
            },
        ];
        assert!(!is_fleet_comparable(&series));
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
