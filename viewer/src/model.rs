//! Pure projection of a `RunResults` document into what the UI draws:
//! colored nodes and edges of the fully connected fleet graph, plus a flat
//! metric table. Deliberately gpui-free so all of it unit tests headlessly;
//! the `ui` layer renders these shapes without doing further analysis.

use std::collections::{BTreeMap, BTreeSet};

use gauntlet::analysis::stats;
use gauntlet::proto::{Scope, TestId, TestOutcome, Unit};
use gauntlet::report::{
    NodeRoofline, RunResults, Verdict, metric_key, sample_key, scope_label, test_display_name,
    verdict,
};

/// Health of a subject, ordered so `Ord::max` picks the worse of two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Ok,
    Warn,
    Bad,
}

/// A finding attached to a node or edge, with its own severity so mixed
/// findings render in the right color.
pub type Issue = (Severity, String);

#[derive(Debug, Clone, PartialEq)]
pub struct NodeView {
    /// Host key as configured (the management address).
    pub host: String,
    /// Hostname reported by inventory, when the host got that far.
    pub hostname: Option<String>,
    pub severity: Severity,
    pub issues: Vec<Issue>,
    /// Roofline digest: (label, formatted value) pairs.
    pub stats: Vec<(String, String)>,
}

/// A per-direction reading on an undirected edge; `from_a` was measured by
/// endpoint `a` (the lexicographically smaller host).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Directional {
    pub from_a: Option<f64>,
    pub from_b: Option<f64>,
}

impl Directional {
    /// Worst-case (smallest) reading across directions, for edge labels.
    pub fn min(&self) -> Option<f64> {
        match (self.from_a, self.from_b) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeView {
    /// Lexicographically smaller endpoint.
    pub a: String,
    pub b: String,
    pub severity: Severity,
    pub issues: Vec<Issue>,
    pub bandwidth_gib: Directional,
    pub rtt_p50_us: Directional,
    pub rtt_p99_us: Directional,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricRow {
    /// "<test>.<metric>" comparison-group key.
    pub group: String,
    /// Sample key: "host" or "host:<scope>".
    pub subject: String,
    pub value: f64,
    pub unit: Unit,
    /// Signed distance from the group median in MADs; `None` when the
    /// group's spread is degenerate or the value is not finite.
    pub deviation_mads: Option<f64>,
    /// Flagged by fleet-relative outlier detection.
    pub flagged: bool,
    /// Violates an absolute threshold from the run's config.
    pub violated: bool,
}

/// One alpha-beta link fit from `calibration.links`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkRow {
    pub class: String,
    pub alpha_us: f64,
    pub gib_per_sec: f64,
    pub r_squared: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ViewModel {
    pub run_id: String,
    pub wall_secs: u64,
    pub verdict: Verdict,
    /// Sorted by host.
    pub nodes: Vec<NodeView>,
    /// Sorted by (a, b).
    pub edges: Vec<EdgeView>,
    /// Sorted by (group, subject); sweep series excluded.
    pub rows: Vec<MetricRow>,
    pub links: Vec<LinkRow>,
}

impl ViewModel {
    pub fn new(results: &RunResults) -> Self {
        let host_keys: Vec<String> = results.hosts.keys().cloned().collect();

        let mut nodes: BTreeMap<String, NodeView> = results
            .hosts
            .iter()
            .map(|(host, obs)| {
                (
                    host.clone(),
                    NodeView {
                        host: host.clone(),
                        hostname: obs.inventory.as_ref().map(|inv| inv.hostname.clone()),
                        severity: Severity::Ok,
                        issues: Vec::new(),
                        stats: results
                            .calibration
                            .rooflines
                            .get(host)
                            .map(roofline_stats)
                            .unwrap_or_default(),
                    },
                )
            })
            .collect();

        let mut edges: BTreeMap<(String, String), EdgeView> = BTreeMap::new();

        // Pairwise measurements define the graph's edges.
        for (host, obs) in &results.hosts {
            for record in &obs.metrics {
                let Scope::HostPair { peer } = &record.scope else {
                    continue;
                };
                let edge = edge_entry(&mut edges, host, peer);
                let measured_by_a = *host == edge.a;
                let slot = match (record.test, record.name.as_str()) {
                    (TestId::NetBandwidth, "gib_per_sec") => Some(&mut edge.bandwidth_gib),
                    (TestId::NetLatency, "rtt_p50") => Some(&mut edge.rtt_p50_us),
                    (TestId::NetLatency, "rtt_p99") => Some(&mut edge.rtt_p99_us),
                    _ => None,
                };
                if let Some(direction) = slot {
                    if measured_by_a {
                        direction.from_a = Some(record.value);
                    } else {
                        direction.from_b = Some(record.value);
                    }
                }
            }
        }

        // Findings, worst category first so issue lists read top-down.
        for (host, errors) in &results.fleet.failed_hosts {
            if let Some(node) = nodes.get_mut(host) {
                for error in errors {
                    note(node, Severity::Bad, format!("error: {error}"));
                }
            }
        }

        for (host, obs) in &results.hosts {
            for (test, scope, outcome) in &obs.outcomes {
                let TestOutcome::Failed { reason } = outcome else {
                    continue;
                };
                let name = test_display_name(*test);
                if let Scope::HostPair { peer } = scope {
                    let edge = edge_entry(&mut edges, host, peer);
                    bump(
                        edge,
                        Severity::Bad,
                        format!("{name} {host} -> {peer} failed: {reason}"),
                    );
                } else if let Some(node) = nodes.get_mut(host) {
                    let label = scope_label(scope)
                        .map(|label| format!(" {label}"))
                        .unwrap_or_default();
                    note(
                        node,
                        Severity::Bad,
                        format!("{name}{label} failed: {reason}"),
                    );
                }
            }
        }

        for (group, outliers) in &results.fleet.outliers {
            for outlier in outliers {
                let issue = format!(
                    "{group}: {:.3} vs fleet median {:.3} ({:+.1} MADs)",
                    outlier.value, outlier.fleet_median, outlier.deviation_mads
                );
                apply_finding(
                    &mut nodes,
                    &mut edges,
                    &host_keys,
                    &outlier.key,
                    Severity::Warn,
                    issue,
                );
            }
        }

        for (group, violators) in &results.fleet.threshold_violations {
            for key in violators {
                apply_finding(
                    &mut nodes,
                    &mut edges,
                    &host_keys,
                    key,
                    Severity::Warn,
                    format!("{group}: violates configured threshold"),
                );
            }
        }

        for (field, finding) in &results.fleet.consistency {
            for (host, value) in &finding.dissenters {
                if let Some(node) = nodes.get_mut(host) {
                    note(
                        node,
                        Severity::Warn,
                        format!(
                            "inventory {field}: {value:?} (majority {:?})",
                            finding.majority_value
                        ),
                    );
                }
            }
        }

        let links = results
            .calibration
            .links
            .iter()
            .map(|(class, fit)| LinkRow {
                class: class.clone(),
                alpha_us: fit.alpha_us,
                gib_per_sec: fit.bandwidth_gib_per_sec(),
                r_squared: fit.r_squared,
            })
            .collect();

        ViewModel {
            run_id: results.run_id.clone(),
            wall_secs: results
                .finished_epoch_secs
                .saturating_sub(results.started_epoch_secs),
            verdict: verdict(results),
            nodes: nodes.into_values().collect(),
            edges: edges.into_values().collect(),
            rows: metric_rows(results),
            links,
        }
    }
}

fn note(node: &mut NodeView, severity: Severity, text: String) {
    node.severity = node.severity.max(severity);
    node.issues.push((severity, text));
}

fn bump(edge: &mut EdgeView, severity: Severity, text: String) {
    edge.severity = edge.severity.max(severity);
    edge.issues.push((severity, text));
}

fn edge_entry<'a>(
    edges: &'a mut BTreeMap<(String, String), EdgeView>,
    x: &str,
    y: &str,
) -> &'a mut EdgeView {
    let (a, b) = if x <= y { (x, y) } else { (y, x) };
    edges
        .entry((a.to_string(), b.to_string()))
        .or_insert_with(|| EdgeView {
            a: a.to_string(),
            b: b.to_string(),
            severity: Severity::Ok,
            issues: Vec::new(),
            bandwidth_gib: Directional::default(),
            rtt_p50_us: Directional::default(),
            rtt_p99_us: Directional::default(),
        })
}

/// Where a fleet-analysis sample key points: a node's own metric or a
/// directed pair measurement.
pub(crate) enum Attribution {
    Node { host: String },
    Pair { host: String, peer: String },
    Unmatched,
}

/// Resolve a sample key ("host", "host:<scope>", "host:pair:<peer>")
/// against the known hosts. Longest host prefix wins, so scope labels that
/// themselves contain ':' (e.g. "disk:/tmp") never confuse attribution.
pub(crate) fn attribute(hosts: &[String], key: &str) -> Attribution {
    let mut best: Option<&String> = None;
    for host in hosts {
        let matches = key == host.as_str()
            || (key.len() > host.len()
                && key.starts_with(host.as_str())
                && key.as_bytes()[host.len()] == b':');
        if matches && best.is_none_or(|current| host.len() > current.len()) {
            best = Some(host);
        }
    }
    let Some(host) = best else {
        return Attribution::Unmatched;
    };
    if key.len() > host.len() {
        let label = &key[host.len() + 1..];
        if let Some(peer) = label.strip_prefix("pair:") {
            return Attribution::Pair {
                host: host.clone(),
                peer: peer.to_string(),
            };
        }
    }
    Attribution::Node { host: host.clone() }
}

fn apply_finding(
    nodes: &mut BTreeMap<String, NodeView>,
    edges: &mut BTreeMap<(String, String), EdgeView>,
    hosts: &[String],
    key: &str,
    severity: Severity,
    issue: String,
) {
    match attribute(hosts, key) {
        Attribution::Pair { host, peer } => {
            let edge = edge_entry(edges, &host, &peer);
            bump(edge, severity, issue);
        }
        Attribution::Node { host } => {
            if let Some(node) = nodes.get_mut(&host) {
                note(node, severity, issue);
            }
        }
        Attribution::Unmatched => {}
    }
}

/// Flatten every fleet-comparable metric group into table rows with
/// per-sample deviations. Groups with repeated sample keys are per-host
/// series (the NCCL sweeps), which render as link fits instead.
fn metric_rows(results: &RunResults) -> Vec<MetricRow> {
    let mut groups: BTreeMap<String, Vec<(String, f64, Unit)>> = BTreeMap::new();
    for (host, obs) in &results.hosts {
        for record in &obs.metrics {
            groups
                .entry(metric_key(record.test, &record.name))
                .or_default()
                .push((sample_key(host, &record.scope), record.value, record.unit));
        }
    }

    let mut rows = Vec::new();
    for (group, samples) in groups {
        let mut seen = BTreeSet::new();
        if !samples.iter().all(|(key, _, _)| seen.insert(key.clone())) {
            continue;
        }
        let values: Vec<f64> = samples.iter().map(|(_, value, _)| *value).collect();
        let median = stats::median(&values);
        let mad = stats::mad(&values).filter(|mad| *mad > f64::EPSILON);
        let flagged: BTreeSet<&str> = results
            .fleet
            .outliers
            .get(&group)
            .map(|outliers| outliers.iter().map(|o| o.key.as_str()).collect())
            .unwrap_or_default();
        let violated: BTreeSet<&str> = results
            .fleet
            .threshold_violations
            .get(&group)
            .map(|keys| keys.iter().map(String::as_str).collect())
            .unwrap_or_default();

        let mut group_rows: Vec<MetricRow> = samples
            .iter()
            .map(|(subject, value, unit)| MetricRow {
                group: group.clone(),
                subject: subject.clone(),
                value: *value,
                unit: *unit,
                deviation_mads: match (median, mad) {
                    (Some(median), Some(mad)) if value.is_finite() => Some((value - median) / mad),
                    _ => None,
                },
                flagged: flagged.contains(subject.as_str()),
                violated: violated.contains(subject.as_str()),
            })
            .collect();
        group_rows.sort_by(|left, right| left.subject.cmp(&right.subject));
        rows.extend(group_rows);
    }
    rows
}

fn roofline_stats(roofline: &NodeRoofline) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut push = |label: &str, unit: Unit, value: Option<f64>| {
        if let Some(value) = value {
            out.push((label.to_string(), format_value(unit, value)));
        }
    };
    push("cpu all-core", Unit::Gflops, roofline.cpu_gflops_allcore);
    push("dram", Unit::GibPerSec, roofline.dram_gib_per_sec);
    push("gpu hbm", Unit::GibPerSec, roofline.gpu_hbm_gib_per_sec);
    push("pcie h2d", Unit::GibPerSec, roofline.pcie_h2d_gib_per_sec);
    push("disk read", Unit::GibPerSec, roofline.disk_read_gib_per_sec);
    push(
        "disk write",
        Unit::GibPerSec,
        roofline.disk_write_gib_per_sec,
    );
    for (name, value) in &roofline.gpu_gflops {
        out.push((
            format!("gpu {}", name.trim_start_matches("gflops_")),
            format_value(Unit::Gflops, *value),
        ));
    }
    out
}

/// Unit-aware value rendering shared by the table and detail cards.
pub fn format_value(unit: Unit, value: f64) -> String {
    match unit {
        Unit::Gflops => format!("{value:.0} GFLOPS"),
        Unit::GibPerSec => format!("{value:.2} GiB/s"),
        Unit::Micros => format!("{value:.1} µs"),
        Unit::Millis => format!("{value:.2} ms"),
        Unit::Celsius => format!("{value:.0} °C"),
        Unit::Mhz => format!("{value:.0} MHz"),
        Unit::Bytes => format_bytes(value),
        Unit::Count => format!("{value:.0}"),
        Unit::Residual => format!("{value:.2e}"),
        Unit::Ratio => format!("{value:.3}"),
    }
}

fn format_bytes(value: f64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut scaled = value;
    let mut index = 0;
    while scaled.abs() >= 1024.0 && index < UNITS.len() - 1 {
        scaled /= 1024.0;
        index += 1;
    }
    if index == 0 {
        format!("{scaled:.0} B")
    } else {
        format!("{scaled:.1} {}", UNITS[index])
    }
}
