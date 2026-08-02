//! Side panel: selection detail card plus the quantitative metric table.

use gpui::prelude::FluentBuilder as _;
use gpui::{
    Context, Div, InteractiveElement, MouseButton, MouseDownEvent, ParentElement,
    StatefulInteractiveElement as _, Styled, div, px, rgb,
};

use super::{
    BAD, MUTED, OK, PANEL, PANEL_BORDER, RootView, SELECT, Selection, TEXT, WARN, severity_color,
};
use crate::model::{EdgeView, MetricRow, NodeView, Severity, format_value};
use gauntlet::proto::Unit;

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
        let card = match self.selection {
            Some(Selection::Node(i)) => match self.vm.nodes.get(i) {
                Some(node) => node_card(node),
                None => self.fleet_card(),
            },
            Some(Selection::Edge(i)) => match self.vm.edges.get(i) {
                Some(edge) => edge_card(edge),
                None => self.fleet_card(),
            },
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
        let vm = &self.vm;
        let count = |severity: Severity| vm.nodes.iter().filter(|n| n.severity == severity).count();
        let (healthy, warned, failed) = (
            count(Severity::Ok),
            count(Severity::Warn),
            count(Severity::Bad),
        );

        let mut card = div()
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .child(title("fleet overview"))
            .child(
                div()
                    .flex()
                    .gap_2()
                    .child(chip(OK, format!("{healthy} healthy")))
                    .when(warned > 0, |row| {
                        row.child(chip(WARN, format!("{warned} outliers")))
                    })
                    .when(failed > 0, |row| {
                        row.child(chip(BAD, format!("{failed} failed")))
                    }),
            );

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

    fn render_rows(&self) -> Div {
        let rows = self.visible_rows();
        let filter = match self.selection {
            None => "all subjects".to_string(),
            Some(Selection::Node(i)) => self
                .vm
                .nodes
                .get(i)
                .map(|n| n.host.clone())
                .unwrap_or_default(),
            Some(Selection::Edge(i)) => self
                .vm
                .edges
                .get(i)
                .map(|e| format!("{} ⟷ {}", e.a, e.b))
                .unwrap_or_default(),
        };

        let mut list = div()
            .id("metric-rows")
            .flex_1()
            .min_h(px(0.0))
            .overflow_y_scroll()
            .flex()
            .flex_col();
        for row in &rows {
            list = list.child(render_row(row));
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
                    .child(cell_dev("dev (MADs)")),
            )
            .child(list)
    }

    fn visible_rows(&self) -> Vec<&MetricRow> {
        let rows = self.vm.rows.iter();
        match self.selection {
            None => rows.collect(),
            Some(Selection::Node(i)) => {
                let Some(node) = self.vm.nodes.get(i) else {
                    return self.vm.rows.iter().collect();
                };
                let prefix = format!("{}:", node.host);
                rows.filter(|row| row.subject == node.host || row.subject.starts_with(&prefix))
                    .collect()
            }
            Some(Selection::Edge(i)) => {
                let Some(edge) = self.vm.edges.get(i) else {
                    return self.vm.rows.iter().collect();
                };
                let forward = format!("{}:pair:{}", edge.a, edge.b);
                let backward = format!("{}:pair:{}", edge.b, edge.a);
                rows.filter(|row| row.subject == forward || row.subject == backward)
                    .collect()
            }
        }
    }
}

fn render_row(row: &MetricRow) -> Div {
    let value_color = if row.violated {
        BAD
    } else if row.flagged {
        WARN
    } else {
        TEXT
    };
    let deviation = row
        .deviation_mads
        .map(|d| format!("{d:+.1}"))
        .unwrap_or_else(|| "—".to_string());
    let deviation_color = if row.flagged || row.violated {
        value_color
    } else if row.deviation_mads.is_some_and(|d| d.abs() > 2.0) {
        WARN
    } else {
        MUTED
    };
    div()
        .flex()
        .items_center()
        .gap_2()
        .px_3()
        .py(px(2.0))
        .text_size(px(12.0))
        .child(cell_metric(row.group.clone()).text_color(rgb(MUTED)))
        .child(cell_subject(row.subject.clone()))
        .child(cell_value(format_value(row.unit, row.value)).text_color(rgb(value_color)))
        .child(cell_dev(deviation).text_color(rgb(deviation_color)))
}

fn node_card(node: &NodeView) -> Div {
    let mut card = div().flex().flex_col().gap_2().p_3().child(
        div()
            .flex()
            .items_center()
            .gap_2()
            .child(
                div()
                    .w(px(12.0))
                    .h(px(12.0))
                    .rounded_full()
                    .bg(rgb(severity_color(node.severity))),
            )
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

    card.child(issues_list(&node.issues))
}

fn edge_card(edge: &EdgeView) -> Div {
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
                        .bg(rgb(severity_color(edge.severity))),
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
        .child(issues_list(&edge.issues))
}

fn issues_list(issues: &[(Severity, String)]) -> Div {
    if issues.is_empty() {
        return div()
            .text_size(px(11.0))
            .text_color(rgb(OK))
            .child("no findings");
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
    div().w(px(105.0)).flex().justify_end().child(text.into())
}

fn cell_dev(text: impl Into<String>) -> Div {
    div().w(px(70.0)).flex().justify_end().child(text.into())
}
