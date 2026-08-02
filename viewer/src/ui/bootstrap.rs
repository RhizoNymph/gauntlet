//! Center-pane readiness matrix for `gauntlet bootstrap --json` results:
//! rows are checks, columns are hosts, details below for anything non-ok.

use gauntlet::orchestrator::bootstrap::CheckStatus;
use gpui::{
    Context, Div, InteractiveElement, MouseButton, MouseDownEvent, ParentElement,
    StatefulInteractiveElement as _, Styled, div, px, rgb,
};

use super::{BAD, MUTED, OK, PANEL_BORDER, RootView, SELECT, TEXT, WARN};
use crate::bootstrap::{check_columns, status_of, worst_of};
use crate::runs::format_epoch_utc;

fn status_color(status: CheckStatus) -> u32 {
    match status {
        CheckStatus::Ok => OK,
        CheckStatus::Warn => WARN,
        CheckStatus::Fail => BAD,
    }
}

fn status_glyph(status: Option<CheckStatus>) -> (&'static str, u32) {
    match status {
        Some(CheckStatus::Ok) => ("✓", OK),
        Some(CheckStatus::Warn) => ("!", WARN),
        Some(CheckStatus::Fail) => ("✗", BAD),
        None => ("–", MUTED),
    }
}

const CHECK_COLUMN_WIDTH: f32 = 150.0;
const HOST_COLUMN_WIDTH: f32 = 130.0;

impl RootView {
    pub(super) fn render_bootstrap_pane(&self, cx: &mut Context<Self>) -> Div {
        let Some(report) = &self.bootstrap_report else {
            return div().flex_1();
        };
        let columns = check_columns(&report.hosts);

        let header_bar = div()
            .flex()
            .items_center()
            .gap_3()
            .px_4()
            .py_2()
            .border_b_1()
            .border_color(rgb(PANEL_BORDER))
            .child(
                div()
                    .text_size(px(13.0))
                    .text_color(rgb(TEXT))
                    .child("bootstrap readiness"),
            )
            .child(
                div()
                    .text_size(px(11.0))
                    .text_color(rgb(MUTED))
                    .child(format_epoch_utc(report.finished_epoch_secs)),
            )
            .child(
                div()
                    .id("close-bootstrap")
                    .text_size(px(11.0))
                    .text_color(rgb(SELECT))
                    .cursor_pointer()
                    .child("✕ back to graph")
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, _: &MouseDownEvent, _window, cx| {
                            this.show_graph();
                            cx.notify();
                        }),
                    ),
            );

        // Host header row: hostnames colored by their worst status.
        let mut host_header = div()
            .flex()
            .gap_2()
            .px_4()
            .py_1()
            .child(div().w(px(CHECK_COLUMN_WIDTH)));
        for host in &report.hosts {
            host_header = host_header.child(
                div()
                    .w(px(HOST_COLUMN_WIDTH))
                    .flex()
                    .justify_center()
                    .text_size(px(11.0))
                    .text_color(rgb(status_color(worst_of(host))))
                    .child(host.host.clone()),
            );
        }

        let mut matrix = div().flex().flex_col().gap_1();
        for check_name in &columns {
            let mut row = div().flex().items_center().gap_2().px_4().child(
                div()
                    .w(px(CHECK_COLUMN_WIDTH))
                    .text_size(px(11.0))
                    .text_color(rgb(MUTED))
                    .child(check_name.clone()),
            );
            for host in &report.hosts {
                let (glyph, color) = status_glyph(status_of(host, check_name));
                row = row.child(
                    div()
                        .w(px(HOST_COLUMN_WIDTH))
                        .flex()
                        .justify_center()
                        .text_size(px(12.0))
                        .text_color(rgb(color))
                        .child(glyph),
                );
            }
            matrix = matrix.child(row);
        }

        // Everything non-ok, spelled out.
        let mut details = div().flex().flex_col().gap_1().px_4().pt_3();
        let mut any_finding = false;
        for host in &report.hosts {
            for check in &host.checks {
                if check.status == CheckStatus::Ok {
                    continue;
                }
                any_finding = true;
                details = details.child(
                    div()
                        .text_size(px(11.0))
                        .text_color(rgb(status_color(check.status)))
                        .child(format!("{} · {}: {}", host.host, check.name, check.detail)),
                );
            }
        }
        if !any_finding {
            details = details.child(
                div()
                    .text_size(px(11.0))
                    .text_color(rgb(OK))
                    .child("every check passed on every host"),
            );
        }

        div()
            .flex_1()
            .min_w(px(0.0))
            .flex()
            .flex_col()
            .child(header_bar)
            .child(
                div()
                    .id("bootstrap-pane")
                    .flex_1()
                    .min_h(px(0.0))
                    .overflow_y_scroll()
                    .flex()
                    .flex_col()
                    .py_2()
                    .child(host_header)
                    .child(matrix)
                    .child(details),
            )
    }
}
