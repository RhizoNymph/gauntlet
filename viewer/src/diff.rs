//! Run-to-run diffing: project a current run against a baseline run and
//! flag regressions. Pure and gpui-free, like `model`.
//!
//! Direction of goodness is unit-derived (throughput up = good, latency /
//! residual / temperature up = bad); units without a defensible direction
//! (bytes, counts, ratios) get deltas but never severities.

use std::collections::BTreeMap;

use gauntlet::proto::Unit;

use crate::model::{Attribution, Issue, MetricRow, Severity, ViewModel, attribute, format_value};

/// Regression fraction (in the bad direction) that colors a subject amber.
pub const REGRESS_WARN: f64 = 0.05;
/// Regression fraction that colors a subject red.
pub const REGRESS_BAD: f64 = 0.15;
/// When both runs carry repeat spreads, a delta must also exceed this many
/// units of pooled spread before it is judged — the % floors alone cannot
/// tell regression from run-to-run noise.
pub const NOISE_GATE_MADS: f64 = 2.0;
/// Issue-list cap per subject; the rest collapse into a summary line.
const MAX_ISSUES: usize = 6;

/// Whether a larger value of this unit is better, or `None` when the unit
/// carries no direction (deltas shown, never judged).
pub fn higher_is_better(unit: Unit) -> Option<bool> {
    match unit {
        Unit::Gflops | Unit::GibPerSec | Unit::Mhz => Some(true),
        Unit::Micros | Unit::Millis | Unit::Celsius | Unit::Residual => Some(false),
        Unit::Bytes | Unit::Count | Unit::Ratio => None,
    }
}

/// Delta of one metric row against the baseline run.
#[derive(Debug, Clone, PartialEq)]
pub struct RowDelta {
    pub baseline_value: f64,
    /// (current - baseline) / |baseline|.
    pub delta_fraction: Option<f64>,
    /// Regression severity; `Ok` for improvements and no-direction units.
    pub severity: Severity,
    /// Moved at least `REGRESS_WARN` in the good direction.
    pub improved: bool,
}

/// Everything the UI needs to render diff mode. Keys mirror `ViewModel`:
/// rows by (group, subject), edges by the ordered host pair.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DiffView {
    pub baseline_run_id: String,
    pub rows: BTreeMap<(String, String), RowDelta>,
    /// Only subjects with at least one regression appear here.
    pub node_severity: BTreeMap<String, Severity>,
    pub node_issues: BTreeMap<String, Vec<Issue>>,
    pub edge_severity: BTreeMap<(String, String), Severity>,
    pub edge_issues: BTreeMap<(String, String), Vec<Issue>>,
}

struct Finding {
    severity: Severity,
    /// Signed regression fraction (positive = worse), for ranking.
    regression: f64,
    text: String,
}

impl DiffView {
    pub fn new(current: &ViewModel, baseline: &ViewModel) -> Self {
        let hosts: Vec<String> = current.nodes.iter().map(|n| n.host.clone()).collect();
        let baseline_rows: BTreeMap<(&str, &str), &MetricRow> = baseline
            .rows
            .iter()
            .map(|row| ((row.group.as_str(), row.subject.as_str()), row))
            .collect();

        let mut rows = BTreeMap::new();
        let mut node_findings: BTreeMap<String, Vec<Finding>> = BTreeMap::new();
        let mut edge_findings: BTreeMap<(String, String), Vec<Finding>> = BTreeMap::new();

        for row in &current.rows {
            let Some(base) = baseline_rows.get(&(row.group.as_str(), row.subject.as_str())) else {
                continue;
            };
            if !row.value.is_finite() || !base.value.is_finite() || base.value == 0.0 {
                continue;
            }
            let delta = (row.value - base.value) / base.value.abs();
            let direction = higher_is_better(row.unit);
            let regression = match direction {
                Some(true) => -delta,
                Some(false) => delta,
                None => 0.0,
            };
            // Noise gate: with spreads on both sides, the shift must clear
            // ~2x the pooled run-to-run spread to be judged at all.
            let clears_noise = match (row.spread_mad, base.spread_mad) {
                (Some(current_spread), Some(baseline_spread)) if row.n >= 2 && base.n >= 2 => {
                    let pooled = (current_spread.powi(2) + baseline_spread.powi(2)).sqrt();
                    pooled <= f64::EPSILON
                        || (row.value - base.value).abs() > NOISE_GATE_MADS * pooled
                }
                _ => true,
            };
            let severity = if !clears_noise {
                Severity::Ok
            } else if direction.is_some() && regression >= REGRESS_BAD {
                Severity::Bad
            } else if direction.is_some() && regression >= REGRESS_WARN {
                Severity::Warn
            } else {
                Severity::Ok
            };
            let improved = clears_noise && direction.is_some() && regression <= -REGRESS_WARN;

            rows.insert(
                (row.group.clone(), row.subject.clone()),
                RowDelta {
                    baseline_value: base.value,
                    delta_fraction: Some(delta),
                    severity,
                    improved,
                },
            );

            if severity > Severity::Ok {
                let finding = Finding {
                    severity,
                    regression,
                    text: format!(
                        "{} [{}]: {} → {} ({:+.1}%)",
                        row.group,
                        row.subject,
                        format_value(row.unit, base.value),
                        format_value(row.unit, row.value),
                        delta * 100.0
                    ),
                };
                match attribute(&hosts, &row.subject) {
                    Attribution::Pair { host, peer } => {
                        edge_findings
                            .entry(ordered(&host, &peer))
                            .or_default()
                            .push(finding);
                    }
                    Attribution::Node { host } => {
                        node_findings.entry(host).or_default().push(finding);
                    }
                    Attribution::Unmatched => {}
                }
            }
        }

        let mut view = DiffView {
            baseline_run_id: baseline.run_id.clone(),
            rows,
            ..DiffView::default()
        };
        for (host, findings) in node_findings {
            let (severity, issues) = finalize(findings);
            view.node_severity.insert(host.clone(), severity);
            view.node_issues.insert(host, issues);
        }
        for (pair, findings) in edge_findings {
            let (severity, issues) = finalize(findings);
            view.edge_severity.insert(pair.clone(), severity);
            view.edge_issues.insert(pair, issues);
        }
        view
    }
}

fn ordered(x: &str, y: &str) -> (String, String) {
    if x <= y {
        (x.to_string(), y.to_string())
    } else {
        (y.to_string(), x.to_string())
    }
}

/// Worst severity plus a ranked, capped issue list.
fn finalize(mut findings: Vec<Finding>) -> (Severity, Vec<Issue>) {
    let severity = findings
        .iter()
        .map(|f| f.severity)
        .max()
        .unwrap_or(Severity::Ok);
    findings.sort_by(|left, right| {
        right
            .severity
            .cmp(&left.severity)
            .then(right.regression.total_cmp(&left.regression))
    });
    let extra = findings.len().saturating_sub(MAX_ISSUES);
    let mut issues: Vec<Issue> = findings
        .into_iter()
        .take(MAX_ISSUES)
        .map(|f| (f.severity, f.text))
        .collect();
    if extra > 0 {
        issues.push((Severity::Warn, format!("… and {extra} more regressions")));
    }
    (severity, issues)
}
