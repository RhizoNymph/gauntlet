//! The fully connected fleet graph: canvas-painted edges underneath
//! absolutely positioned, clickable node chips.

use std::collections::BTreeMap;

use gpui::prelude::FluentBuilder as _;
use gpui::{
    Context, Div, InteractiveElement, MouseButton, MouseDownEvent, ParentElement, PathBuilder,
    Pixels, Size, Styled, canvas, div, point, px, rgb,
};

use super::{
    BAD, EDGE_OK, MUTED, OK, PANEL_BORDER, RootView, SELECT, Selection, TEXT, WARN, severity_color,
};
use crate::layout;
use crate::model::{Severity, ViewModel};

const NODE_RADIUS: f32 = 16.0;
const EDGE_HIT_TOLERANCE: f32 = 8.0;
/// Above this many edges the midpoint bandwidth labels become clutter.
const MAX_EDGE_LABELS: usize = 24;

fn positions(count: usize, size: Size<Pixels>) -> Vec<(f32, f32)> {
    layout::circle_positions(count)
        .into_iter()
        .map(|unit| layout::to_px(unit, f32::from(size.width), f32::from(size.height)))
        .collect()
}

/// Edge endpoints as node indices, aligned with `vm.edges`. Unknown hosts
/// map to `usize::MAX`, which hit-testing and painting skip safely.
fn edge_indices(vm: &ViewModel) -> Vec<(usize, usize)> {
    let index: BTreeMap<&str, usize> = vm
        .nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (node.host.as_str(), i))
        .collect();
    vm.edges
        .iter()
        .map(|edge| {
            (
                index.get(edge.a.as_str()).copied().unwrap_or(usize::MAX),
                index.get(edge.b.as_str()).copied().unwrap_or(usize::MAX),
            )
        })
        .collect()
}

impl RootView {
    fn edge_selected(&self, a: &str, b: &str) -> bool {
        matches!(&self.selection, Some(Selection::Edge(sa, sb)) if sa == a && sb == b)
    }

    fn node_selected(&self, host: &str) -> bool {
        matches!(&self.selection, Some(Selection::Node(h)) if h == host)
    }

    pub(super) fn render_graph(&self, cx: &mut Context<Self>) -> Div {
        let Some(run) = &self.current else {
            return div().flex_1();
        };
        let vm = run.vm.clone();
        let known_bounds = self.graph_bounds;
        let entity = cx.entity();
        let indices = edge_indices(&vm);

        // Segments in draw order (selected last so its highlight sits on
        // top), fully precomputed so the paint closure owns plain data.
        let mut segments: Vec<(usize, usize, u32, f32)> = Vec::new();
        let mut selected_segment: Option<(usize, usize, u32, f32)> = None;
        for (i, edge) in vm.edges.iter().enumerate() {
            let Some(&(ia, ib)) = indices.get(i) else {
                continue;
            };
            let selected = self.edge_selected(&edge.a, &edge.b);
            let severity = self.display_edge_severity(edge);
            let (color, width) = if selected {
                (SELECT, 4.0)
            } else if severity == Severity::Ok {
                (EDGE_OK, 2.0)
            } else {
                (severity_color(severity), 3.0)
            };
            if selected {
                selected_segment = Some((ia, ib, color, width));
            } else {
                segments.push((ia, ib, color, width));
            }
        }
        segments.extend(selected_segment);
        let node_count = vm.nodes.len();

        let canvas_element = canvas(
            move |bounds, _window, cx| {
                if known_bounds != Some(bounds) {
                    cx.defer(move |cx| {
                        entity.update(cx, |this, cx| {
                            this.graph_bounds = Some(bounds);
                            cx.notify();
                        });
                    });
                }
            },
            move |bounds, (), window, _cx| {
                if node_count == 0 {
                    return;
                }
                let centers = positions(node_count, bounds.size);
                let origin = (f32::from(bounds.origin.x), f32::from(bounds.origin.y));
                for (ia, ib, color, width) in segments {
                    let (Some(a), Some(b)) = (centers.get(ia), centers.get(ib)) else {
                        continue;
                    };
                    let mut builder = PathBuilder::stroke(px(width));
                    builder.move_to(point(px(origin.0 + a.0), px(origin.1 + a.1)));
                    builder.line_to(point(px(origin.0 + b.0), px(origin.1 + b.1)));
                    if let Ok(path) = builder.build() {
                        window.paint_path(path, rgb(color));
                    }
                }
            },
        )
        .absolute()
        .size_full();

        let mut pane = div()
            .relative()
            .flex_1()
            .min_w(px(0.0))
            .overflow_hidden()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &MouseDownEvent, _window, cx| {
                    let Some(bounds) = this.graph_bounds else {
                        return;
                    };
                    let Some(run) = &this.current else {
                        return;
                    };
                    let vm = run.vm.clone();
                    let local = (
                        f32::from(event.position.x) - f32::from(bounds.origin.x),
                        f32::from(event.position.y) - f32::from(bounds.origin.y),
                    );
                    let centers = positions(vm.nodes.len(), bounds.size);
                    let mut hit = layout::hit_node(local, &centers, NODE_RADIUS + 2.0)
                        .and_then(|i| vm.nodes.get(i))
                        .map(|node| Selection::Node(node.host.clone()));
                    if hit.is_none() {
                        let indices = edge_indices(&vm);
                        hit = layout::hit_edge(local, &centers, &indices, EDGE_HIT_TOLERANCE)
                            .and_then(|i| vm.edges.get(i))
                            .map(|edge| Selection::Edge(edge.a.clone(), edge.b.clone()));
                    }
                    this.selection = hit;
                    cx.notify();
                }),
            )
            .child(canvas_element);

        if let Some(bounds) = self.graph_bounds {
            let centers = positions(vm.nodes.len(), bounds.size);

            if vm.edges.len() <= MAX_EDGE_LABELS {
                for (i, edge) in vm.edges.iter().enumerate() {
                    let Some(&(ia, ib)) = indices.get(i) else {
                        continue;
                    };
                    let (Some(a), Some(b)) = (centers.get(ia), centers.get(ib)) else {
                        continue;
                    };
                    let Some(bandwidth) = edge.bandwidth_gib.min() else {
                        continue;
                    };
                    let mid = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
                    let severity = self.display_edge_severity(edge);
                    let color = if self.edge_selected(&edge.a, &edge.b) {
                        SELECT
                    } else if severity == Severity::Ok {
                        MUTED
                    } else {
                        severity_color(severity)
                    };
                    pane = pane.child(
                        div()
                            .absolute()
                            .left(px(mid.0 - 45.0))
                            .top(px(mid.1 - 9.0))
                            .w(px(90.0))
                            .flex()
                            .justify_center()
                            .text_size(px(10.0))
                            .text_color(rgb(color))
                            .child(format!("{bandwidth:.2} GiB/s")),
                    );
                }
            }

            for (i, node) in vm.nodes.iter().enumerate() {
                let Some(&(x, y)) = centers.get(i) else {
                    continue;
                };
                let selected = self.node_selected(&node.host);
                let color = severity_color(self.display_node_severity(node));
                let host = node.host.clone();
                pane = pane
                    .child(
                        div()
                            .id(("node", i))
                            .absolute()
                            .left(px(x - NODE_RADIUS))
                            .top(px(y - NODE_RADIUS))
                            .w(px(NODE_RADIUS * 2.0))
                            .h(px(NODE_RADIUS * 2.0))
                            .rounded_full()
                            .bg(rgb(color))
                            .border_2()
                            .border_color(if selected {
                                rgb(SELECT)
                            } else {
                                rgb(PANEL_BORDER)
                            })
                            .cursor_pointer()
                            .hover(|style| style.border_color(rgb(SELECT)))
                            .on_mouse_down(
                                MouseButton::Left,
                                cx.listener(move |this, _: &MouseDownEvent, _window, cx| {
                                    this.selection = Some(Selection::Node(host.clone()));
                                    cx.stop_propagation();
                                    cx.notify();
                                }),
                            ),
                    )
                    .child(
                        div()
                            .absolute()
                            .left(px(x - 70.0))
                            .top(px(y + NODE_RADIUS + 4.0))
                            .w(px(140.0))
                            .flex()
                            .flex_col()
                            .items_center()
                            .child(
                                div().text_size(px(11.0)).text_color(rgb(TEXT)).child(
                                    node.hostname.clone().unwrap_or_else(|| node.host.clone()),
                                ),
                            )
                            .when(node.hostname.is_some(), |label| {
                                label.child(
                                    div()
                                        .text_size(px(10.0))
                                        .text_color(rgb(MUTED))
                                        .child(node.host.clone()),
                                )
                            }),
                    );
            }

            pane = pane.child(legend(self.diff_active()));
        }

        pane
    }
}

fn legend(diff: bool) -> Div {
    let item = |color: u32, label: &'static str| {
        div()
            .flex()
            .items_center()
            .gap_1()
            .child(div().w(px(10.0)).h(px(10.0)).rounded_full().bg(rgb(color)))
            .child(
                div()
                    .text_size(px(10.0))
                    .text_color(rgb(MUTED))
                    .child(label),
            )
    };
    let (ok_label, warn_label, bad_label) = if diff {
        ("no change", "regressed >5%", "regressed >15%")
    } else {
        ("healthy", "outlier", "failed")
    };
    div()
        .absolute()
        .left(px(12.0))
        .bottom(px(10.0))
        .flex()
        .gap_3()
        .child(item(OK, ok_label))
        .child(item(WARN, warn_label))
        .child(item(BAD, bad_label))
}
