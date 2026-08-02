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
use crate::model::Severity;

const NODE_RADIUS: f32 = 16.0;
const EDGE_HIT_TOLERANCE: f32 = 8.0;
/// Above this many edges the midpoint bandwidth labels become clutter.
const MAX_EDGE_LABELS: usize = 24;

impl RootView {
    fn positions(&self, size: Size<Pixels>) -> Vec<(f32, f32)> {
        layout::circle_positions(self.vm.nodes.len())
            .into_iter()
            .map(|unit| layout::to_px(unit, f32::from(size.width), f32::from(size.height)))
            .collect()
    }

    /// Edge endpoints as node indices, aligned with `vm.edges`. Unknown
    /// hosts map to `usize::MAX`, which the hit-testing skips safely.
    fn edge_indices(&self) -> Vec<(usize, usize)> {
        let index: BTreeMap<&str, usize> = self
            .vm
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.host.as_str(), i))
            .collect();
        self.vm
            .edges
            .iter()
            .map(|edge| {
                (
                    index.get(edge.a.as_str()).copied().unwrap_or(usize::MAX),
                    index.get(edge.b.as_str()).copied().unwrap_or(usize::MAX),
                )
            })
            .collect()
    }

    pub(super) fn render_graph(&self, cx: &mut Context<Self>) -> Div {
        let vm = self.vm.clone();
        let selection = self.selection;
        let known_bounds = self.graph_bounds;
        let entity = cx.entity();

        // The canvas records its window bounds for the next frame (node
        // chips and hit-testing need them) and paints the edges.
        let paint_vm = vm.clone();
        let edge_indices = self.edge_indices();
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
                let node_count = paint_vm.nodes.len();
                if node_count == 0 {
                    return;
                }
                let centers: Vec<(f32, f32)> = layout::circle_positions(node_count)
                    .into_iter()
                    .map(|unit| {
                        layout::to_px(
                            unit,
                            f32::from(bounds.size.width),
                            f32::from(bounds.size.height),
                        )
                    })
                    .collect();

                // Draw the selected edge last so its highlight sits on top.
                let mut order: Vec<usize> = (0..paint_vm.edges.len()).collect();
                if let Some(Selection::Edge(selected)) = selection {
                    order.retain(|i| *i != selected);
                    order.push(selected);
                }
                for i in order {
                    let (Some(edge), Some((ia, ib))) =
                        (paint_vm.edges.get(i), edge_indices.get(i).copied())
                    else {
                        continue;
                    };
                    let (Some(a), Some(b)) = (centers.get(ia), centers.get(ib)) else {
                        continue;
                    };
                    let selected = selection == Some(Selection::Edge(i));
                    let (color, width) = match (selected, edge.severity) {
                        (true, _) => (SELECT, 4.0),
                        (false, Severity::Ok) => (EDGE_OK, 2.0),
                        (false, severity) => (severity_color(severity), 3.0),
                    };
                    let mut builder = PathBuilder::stroke(px(width));
                    builder.move_to(point(
                        px(f32::from(bounds.origin.x) + a.0),
                        px(f32::from(bounds.origin.y) + a.1),
                    ));
                    builder.line_to(point(
                        px(f32::from(bounds.origin.x) + b.0),
                        px(f32::from(bounds.origin.y) + b.1),
                    ));
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
                    let local = (
                        f32::from(event.position.x) - f32::from(bounds.origin.x),
                        f32::from(event.position.y) - f32::from(bounds.origin.y),
                    );
                    let centers = this.positions(bounds.size);
                    let mut hit =
                        layout::hit_node(local, &centers, NODE_RADIUS + 2.0).map(Selection::Node);
                    if hit.is_none() {
                        let edges = this.edge_indices();
                        hit = layout::hit_edge(local, &centers, &edges, EDGE_HIT_TOLERANCE)
                            .map(Selection::Edge);
                    }
                    this.selection = hit;
                    cx.notify();
                }),
            )
            .child(canvas_element);

        if let Some(bounds) = self.graph_bounds {
            let centers = self.positions(bounds.size);
            let edge_indices = self.edge_indices();

            if vm.edges.len() <= MAX_EDGE_LABELS {
                for (i, edge) in vm.edges.iter().enumerate() {
                    let Some((ia, ib)) = edge_indices.get(i).copied() else {
                        continue;
                    };
                    let (Some(a), Some(b)) = (centers.get(ia), centers.get(ib)) else {
                        continue;
                    };
                    let Some(bandwidth) = edge.bandwidth_gib.min() else {
                        continue;
                    };
                    let mid = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
                    let selected = self.selection == Some(Selection::Edge(i));
                    let color = if selected {
                        SELECT
                    } else if edge.severity == Severity::Ok {
                        MUTED
                    } else {
                        severity_color(edge.severity)
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
                let selected = self.selection == Some(Selection::Node(i));
                let color = severity_color(node.severity);
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
                                    this.selection = Some(Selection::Node(i));
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

            pane = pane.child(legend());
        }

        pane
    }
}

fn legend() -> Div {
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
    div()
        .absolute()
        .left(px(12.0))
        .bottom(px(10.0))
        .flex()
        .gap_3()
        .child(item(OK, "healthy"))
        .child(item(WARN, "outlier"))
        .child(item(BAD, "failed"))
}
