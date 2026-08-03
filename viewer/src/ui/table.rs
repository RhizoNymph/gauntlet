//! Side panel: selection detail card plus the quantitative metric table.
//! In diff mode the deviation column becomes a Δ% column against the
//! baseline run and cards show regressions instead of absolute findings.

use gpui::prelude::FluentBuilder as _;
use gpui::{
    Context, Div, InteractiveElement, MouseButton, MouseDownEvent, ParentElement,
    StatefulInteractiveElement as _, Styled, div, px, rgb,
};

use gauntlet::proto::Unit;

use super::{
    BAD, BG, MUTED, OK, PANEL, PANEL_BORDER, RootView, SELECT, Selection, TEXT, WARN,
    severity_color,
};
use crate::diff::RowDelta;
use crate::model::{EdgeView, Issue, MetricRow, NodeView, Severity, format_number, format_value};

impl RootView {
    pub(super) fn render_side(&self, cx: &mut Context<Self>) -> Div {
        div()
            .w(px(560.0))
            .flex()
            .flex_col()
            .border_l_1()
            .border_color(rgb(PANEL_BORDER))
            .bg(rgb(PANEL))
            .child(self.render_details(cx))
            .child(self.render_rows())
    }

    fn render_details(&self, cx: &mut Context<Self>) -> Div {
        let card = match &self.selection {
            Some(Selection::Node(host)) => {
                match self
                    .current
                    .as_ref()
                    .and_then(|run| run.vm.nodes.iter().find(|n| &n.host == host))
                {
                    Some(node) => self.node_card(node),
                    None => self.fleet_card(),
                }
            }
            Some(Selection::Edge(a, b)) => {
                match self
                    .current
                    .as_ref()
                    .and_then(|run| run.vm.edges.iter().find(|e| &e.a == a && &e.b == b))
                {
                    Some(edge) => self.edge_card(edge),
                    None => self.fleet_card(),
                }
            }
            None => self.fleet_card(),
        };
        div()
            .flex()
            .flex_col()
            .child(card)
            .when(self.selection.is_some(), |details| {
                details.child(
                    div()
                        .id("clear-selection")
                        .px_3()
                        .pb_2()
                        .text_size(px(11.0))
                        .text_color(rgb(SELECT))
                        .cursor_pointer()
                        .child("✕ clear selection")
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|this, _: &MouseDownEvent, _window, cx| {
                                this.selection = None;
                                cx.notify();
                            }),
                        ),
                )
            })
    }

    fn fleet_card(&self) -> Div {
        let Some(run) = &self.current else {
            return div();
        };
        let vm = &run.vm;

        let mut card = div().flex().flex_col().gap_2().p_3();
        if self.diff_active() {
            let Some(diff) = &self.diff else {
                return card;
            };
            let count = |severity: Severity| {
                diff.node_severity
                    .values()
                    .filter(|s| **s == severity)
                    .count()
            };
            let (warned, failed) = (count(Severity::Warn), count(Severity::Bad));
            let regressed_links = diff.edge_severity.len();
            card = card
                .child(title(format!("diff vs {}", diff.baseline_run_id)))
                .child(
                    div()
                        .flex()
                        .gap_2()
                        .when(warned + failed == 0 && regressed_links == 0, |row| {
                            row.child(chip(OK, "no regressions".into()))
                        })
                        .when(warned > 0, |row| {
                            row.child(chip(WARN, format!("{} >5% worse", nodes_label(warned))))
                        })
                        .when(failed > 0, |row| {
                            row.child(chip(BAD, format!("{} >15% worse", nodes_label(failed))))
                        })
                        .when(regressed_links > 0, |row| {
                            row.child(chip(
                                WARN,
                                format!(
                                    "{regressed_links} link{} regressed",
                                    if regressed_links == 1 { "" } else { "s" }
                                ),
                            ))
                        }),
                );
        } else {
            // Chips split by cause: performance findings vs version skew,
            // so "outliers" never means "merely unpatched".
            let healthy = vm
                .nodes
                .iter()
                .filter(|n| n.severity == Severity::Ok)
                .count();
            let perf_warned = vm
                .nodes
                .iter()
                .filter(|n| n.perf_severity == Severity::Warn)
                .count();
            let failed = vm
                .nodes
                .iter()
                .filter(|n| n.perf_severity == Severity::Bad)
                .count();
            let skewed = vm.nodes.iter().filter(|n| n.skew).count();
            let flagged_links = vm
                .edges
                .iter()
                .filter(|e| e.severity > Severity::Ok)
                .count();
            let any_bad_link = vm.edges.iter().any(|e| e.severity == Severity::Bad);
            card = card.child(title("fleet overview")).child(
                div()
                    .flex()
                    .gap_2()
                    .child(chip(OK, format!("{healthy} healthy")))
                    .when(perf_warned > 0, |row| {
                        row.child(chip(WARN, format!("{perf_warned} perf outliers")))
                    })
                    .when(failed > 0, |row| {
                        row.child(chip(BAD, format!("{failed} failed")))
                    })
                    .when(flagged_links > 0, |row| {
                        row.child(chip(
                            if any_bad_link { BAD } else { WARN },
                            format!(
                                "{flagged_links} link{} flagged",
                                if flagged_links == 1 { "" } else { "s" }
                            ),
                        ))
                    })
                    .when(skewed > 0, |row| {
                        row.child(chip(WARN, format!("{skewed} version skew")))
                    }),
            );
        }

        if !vm.links.is_empty() {
            let mut links = div().flex().flex_col().gap_1().child(
                div()
                    .flex()
                    .gap_2()
                    .text_size(px(10.0))
                    .text_color(rgb(MUTED))
                    .child(cell_grow("link class"))
                    .child(cell_num("α (µs)"))
                    .child(cell_num("GiB/s"))
                    .child(cell_num("r²")),
            );
            for link in &vm.links {
                links = links.child(
                    div()
                        .flex()
                        .gap_2()
                        .text_size(px(11.0))
                        .child(cell_grow(link.class.clone()).text_color(rgb(TEXT)))
                        .child(cell_num(format!("{:.1}", link.alpha_us)))
                        .child(cell_num(format!("{:.2}", link.gib_per_sec)))
                        .child(cell_num(format!("{:.4}", link.r_squared))),
                );
            }
            card = card.child(links);
        }

        card.child(
            div()
                .text_size(px(11.0))
                .text_color(rgb(MUTED))
                .child("click a node or edge to filter the metrics below"),
        )
    }

    fn node_card(&self, node: &NodeView) -> Div {
        let severity = self.display_node_severity(node);
        let diff_issues: Vec<Issue>;
        let (issues, empty_text): (&[Issue], &str) = if self.diff_active() {
            diff_issues = self
                .diff
                .as_ref()
                .and_then(|d| d.node_issues.get(&node.host))
                .cloned()
                .unwrap_or_default();
            (&diff_issues, "no regressions vs baseline")
        } else {
            (&node.issues, "no findings")
        };

        let hollow = !self.diff_active() && node.skew_only();
        let dot = if hollow {
            div()
                .w(px(12.0))
                .h(px(12.0))
                .rounded_full()
                .bg(rgb(BG))
                .border_1()
                .border_color(rgb(severity_color(severity)))
        } else {
            div()
                .w(px(12.0))
                .h(px(12.0))
                .rounded_full()
                .bg(rgb(severity_color(severity)))
        };
        let mut card = div().flex().flex_col().gap_2().p_3().child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(dot)
                .child(title(
                    node.hostname.clone().unwrap_or_else(|| node.host.clone()),
                ))
                .child(
                    div()
                        .text_size(px(11.0))
                        .text_color(rgb(MUTED))
                        .child(node.host.clone()),
                ),
        );

        if !node.stats.is_empty() {
            let mut stats = div().flex().flex_wrap().gap_x_3().gap_y_1();
            for (label, value) in &node.stats {
                stats = stats.child(
                    div()
                        .flex()
                        .gap_1()
                        .text_size(px(11.0))
                        .child(div().text_color(rgb(MUTED)).child(label.clone()))
                        .child(div().text_color(rgb(TEXT)).child(value.clone())),
                );
            }
            card = card.child(stats);
        }

        card.child(issues_list(issues, empty_text))
    }

    fn edge_card(&self, edge: &EdgeView) -> Div {
        let severity = self.display_edge_severity(edge);
        let diff_issues: Vec<Issue>;
        let (issues, empty_text): (&[Issue], &str) = if self.diff_active() {
            diff_issues = self
                .diff
                .as_ref()
                .and_then(|d| d.edge_issues.get(&(edge.a.clone(), edge.b.clone())))
                .cloned()
                .unwrap_or_default();
            (&diff_issues, "no regressions vs baseline")
        } else {
            (&edge.issues, "no findings")
        };

        let direction_row =
            |label: String, bandwidth: Option<f64>, p50: Option<f64>, p99: Option<f64>| {
                div()
                    .flex()
                    .gap_2()
                    .text_size(px(11.0))
                    .child(cell_grow(label).text_color(rgb(TEXT)))
                    .child(cell_num(opt(Unit::GibPerSec, bandwidth)))
                    .child(cell_num(opt(Unit::Micros, p50)))
                    .child(cell_num(opt(Unit::Micros, p99)))
            };
        div()
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .w(px(12.0))
                            .h(px(12.0))
                            .rounded_full()
                            .bg(rgb(severity_color(severity))),
                    )
                    .child(title(format!("{} ⟷ {}", edge.a, edge.b))),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .text_size(px(10.0))
                    .text_color(rgb(MUTED))
                    .child(cell_grow("direction"))
                    .child(cell_num("bandwidth"))
                    .child(cell_num("rtt p50"))
                    .child(cell_num("rtt p99")),
            )
            .child(direction_row(
                format!("{} → {}", edge.a, edge.b),
                edge.bandwidth_gib.from_a,
                edge.rtt_p50_us.from_a,
                edge.rtt_p99_us.from_a,
            ))
            .child(direction_row(
                format!("{} → {}", edge.b, edge.a),
                edge.bandwidth_gib.from_b,
                edge.rtt_p50_us.from_b,
                edge.rtt_p99_us.from_b,
            ))
            .child(issues_list(issues, empty_text))
    }

    fn render_rows(&self) -> Div {
        let rows = self.visible_rows();
        let diff_mode = self.diff_active();
        let filter = match &self.selection {
            None => "all subjects".to_string(),
            Some(Selection::Node(host)) => host.clone(),
            Some(Selection::Edge(a, b)) => format!("{a} ⟷ {b}"),
        };

        let mut list = div()
            .id("metric-rows")
            .flex_1()
            .min_h(px(0.0))
            .overflow_y_scroll()
            .flex()
            .flex_col();
        for row in &rows {
            let delta = self
                .diff
                .as_ref()
                .filter(|_| diff_mode)
                .and_then(|d| d.rows.get(&(row.group.clone(), row.subject.clone())));
            list = list.child(render_row(row, delta, diff_mode));
        }

        div()
            .flex_1()
            .min_h(px(0.0))
            .flex()
            .flex_col()
            .border_t_1()
            .border_color(rgb(PANEL_BORDER))
            .child(
                div()
                    .px_3()
                    .pt_1()
                    .text_size(px(10.0))
                    .text_color(rgb(MUTED))
                    .child(format!("{} rows · {filter}", rows.len())),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .px_3()
                    .py_1()
                    .text_size(px(10.0))
                    .text_color(rgb(MUTED))
                    .child(cell_metric("metric"))
                    .child(cell_subject("subject"))
                    .child(cell_value("value"))
                    .child(cell_dev(if diff_mode {
                        "Δ% vs base"
                    } else {
                        "dev (MADs)"
                    })),
            )
            .child(list)
    }

    fn visible_rows(&self) -> Vec<&MetricRow> {
        let Some(run) = &self.current else {
            return Vec::new();
        };
        let rows = run.vm.rows.iter();
        match &self.selection {
            None => rows.collect(),
            Some(Selection::Node(host)) => {
                let prefix = format!("{host}:");
                rows.filter(|row| &row.subject == host || row.subject.starts_with(&prefix))
                    .collect()
            }
            Some(Selection::Edge(a, b)) => {
                let forward = format!("{a}:pair:{b}");
                let backward = format!("{b}:pair:{a}");
                rows.filter(|row| row.subject == forward || row.subject == backward)
                    .collect()
            }
        }
    }
}

fn render_row(row: &MetricRow, delta: Option<&RowDelta>, diff_mode: bool) -> Div {
    let value_color = if row.violated {
        BAD
    } else if row.flagged {
        WARN
    } else {
        TEXT
    };

    let (last_text, last_color) = if diff_mode {
        match delta.and_then(|d| d.delta_fraction.map(|f| (d, f))) {
            Some((delta, fraction)) => {
                let color = match delta.severity {
                    Severity::Bad => BAD,
                    Severity::Warn => WARN,
                    Severity::Ok if delta.improved => OK,
                    Severity::Ok => MUTED,
                };
                (format!("{:+.1}%", fraction * 100.0), color)
            }
            None => ("—".to_string(), MUTED),
        }
    } else {
        let text = row
            .deviation_mads
            .map(|d| format!("{d:+.1}"))
            .unwrap_or_else(|| "—".to_string());
        let color = if row.flagged || row.violated {
            value_color
        } else if row.deviation_mads.is_some_and(|d| d.abs() > 2.0) {
            WARN
        } else {
            MUTED
        };
        (text, color)
    };

    let mut value_cell = div()
        .w(px(150.0))
        .flex()
        .justify_end()
        .items_center()
        .gap_1()
        .child(
            div()
                .text_color(rgb(value_color))
                .child(format_value(row.unit, row.value)),
        );
    if let Some(spread) = row.spread_mad {
        value_cell = value_cell.child(
            div()
                .text_size(px(10.0))
                .text_color(rgb(MUTED))
                .child(format!("±{}", format_number(row.unit, spread))),
        );
    }
    div()
        .flex()
        .items_center()
        .gap_2()
        .px_3()
        .py(px(2.0))
        .text_size(px(12.0))
        .child(cell_metric(row.group.clone()).text_color(rgb(MUTED)))
        .child(cell_subject(row.subject.clone()))
        .child(value_cell)
        .child(cell_dev(last_text).text_color(rgb(last_color)))
}

fn nodes_label(count: usize) -> String {
    format!("{count} node{}", if count == 1 { "" } else { "s" })
}

fn issues_list(issues: &[Issue], empty_text: &str) -> Div {
    if issues.is_empty() {
        return div()
            .text_size(px(11.0))
            .text_color(rgb(OK))
            .child(empty_text.to_string());
    }
    let mut list = div().flex().flex_col().gap_1();
    for (severity, text) in issues {
        list = list.child(
            div()
                .text_size(px(11.0))
                .text_color(rgb(severity_color(*severity)))
                .child(text.clone()),
        );
    }
    list
}

fn title(text: impl Into<String>) -> Div {
    div()
        .text_size(px(13.0))
        .text_color(rgb(TEXT))
        .child(text.into())
}

fn chip(color: u32, label: String) -> Div {
    div()
        .px_2()
        .py(px(1.0))
        .rounded_md()
        .border_1()
        .border_color(rgb(color))
        .text_size(px(11.0))
        .text_color(rgb(color))
        .child(label)
}

fn opt(unit: Unit, value: Option<f64>) -> String {
    value
        .map(|value| format_value(unit, value))
        .unwrap_or_else(|| "—".to_string())
}

fn cell_grow(text: impl Into<String>) -> Div {
    div().flex_1().min_w(px(0.0)).truncate().child(text.into())
}

fn cell_num(text: impl Into<String>) -> Div {
    div().w(px(80.0)).flex().justify_end().child(text.into())
}

fn cell_metric(text: impl Into<String>) -> Div {
    div().flex_1().min_w(px(0.0)).truncate().child(text.into())
}

fn cell_subject(text: impl Into<String>) -> Div {
    div().w(px(140.0)).truncate().child(text.into())
}

fn cell_value(text: impl Into<String>) -> Div {
    div().w(px(150.0)).flex().justify_end().child(text.into())
}

fn cell_dev(text: impl Into<String>) -> Div {
    div().w(px(70.0)).flex().justify_end().child(text.into())
}
