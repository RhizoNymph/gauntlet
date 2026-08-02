//! gpui shell: theme constants, selection state, and the root view.

pub mod graph;
pub mod table;

use std::sync::Arc;

use gauntlet::report::Verdict;
use gpui::{
    Bounds, Context, Div, FontWeight, IntoElement, ParentElement, Pixels, Render, Styled, Window,
    div, px, rgb,
};

use crate::model::{Severity, ViewModel};

// Catppuccin-mocha-ish palette.
pub const BG: u32 = 0x11111b;
pub const PANEL: u32 = 0x181825;
pub const PANEL_BORDER: u32 = 0x313244;
pub const TEXT: u32 = 0xcdd6f4;
pub const MUTED: u32 = 0x7f849c;
pub const OK: u32 = 0xa6e3a1;
pub const WARN: u32 = 0xf9e2af;
pub const BAD: u32 = 0xf38ba8;
pub const EDGE_OK: u32 = 0x45475a;
pub const SELECT: u32 = 0xcba6f7;

pub fn severity_color(severity: Severity) -> u32 {
    match severity {
        Severity::Ok => OK,
        Severity::Warn => WARN,
        Severity::Bad => BAD,
    }
}

/// What the user clicked; indexes into `ViewModel::nodes` / `edges`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    Node(usize),
    Edge(usize),
}

pub struct RootView {
    pub vm: Arc<ViewModel>,
    pub selection: Option<Selection>,
    /// Window bounds of the graph pane, recorded from the previous frame's
    /// canvas prepaint; node chips and hit-testing need it.
    pub graph_bounds: Option<Bounds<Pixels>>,
}

impl RootView {
    pub fn new(vm: Arc<ViewModel>) -> Self {
        Self {
            vm,
            selection: None,
            graph_bounds: None,
        }
    }

    fn render_header(&self) -> Div {
        let vm = &self.vm;
        let (verdict_text, verdict_color) = match vm.verdict {
            Verdict::Clean => ("clean", OK),
            Verdict::Stragglers => ("stragglers", WARN),
            Verdict::HostFailures => ("host failures", BAD),
        };
        div()
            .flex()
            .items_center()
            .gap_3()
            .px_4()
            .py_2()
            .bg(rgb(PANEL))
            .border_b_1()
            .border_color(rgb(PANEL_BORDER))
            .child(div().font_weight(FontWeight::BOLD).child("gauntlet"))
            .child(
                div()
                    .text_color(rgb(MUTED))
                    .text_size(px(12.0))
                    .child(format!("run {}", vm.run_id)),
            )
            .child(
                div()
                    .px_2()
                    .py(px(1.0))
                    .rounded_md()
                    .text_size(px(11.0))
                    .text_color(rgb(verdict_color))
                    .border_1()
                    .border_color(rgb(verdict_color))
                    .child(verdict_text),
            )
            .child(
                div()
                    .text_color(rgb(MUTED))
                    .text_size(px(12.0))
                    .child(format!(
                        "{} hosts · {} links · {}s wall",
                        vm.nodes.len(),
                        vm.edges.len(),
                        vm.wall_secs
                    )),
            )
    }
}

impl Render for RootView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(BG))
            .text_color(rgb(TEXT))
            .text_size(px(13.0))
            .child(self.render_header())
            .child(
                div()
                    .flex_1()
                    .min_h(px(0.0))
                    .flex()
                    .child(self.render_graph(cx))
                    .child(self.render_side(cx)),
            )
    }
}
