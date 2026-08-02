//! Left rail: launch runs and pick which run (and baseline) to view.

use gpui::prelude::FluentBuilder as _;
use gpui::{
    Context, Div, InteractiveElement, MouseButton, MouseDownEvent, ParentElement, Stateful,
    StatefulInteractiveElement as _, Styled, div, px, rgb, rgba,
};

use super::{
    BG, MUTED, PANEL, PANEL_BORDER, RootView, SELECT, TEXT, severity_color, verdict_color,
};
use crate::runs::{RunEntry, format_epoch_utc};

impl RootView {
    pub(super) fn render_sidebar(&self, cx: &mut Context<Self>) -> Div {
        let mut list = div()
            .id("runs-list")
            .flex_1()
            .min_h(px(0.0))
            .overflow_y_scroll()
            .flex()
            .flex_col()
            .gap_1()
            .p_2();
        for (ix, entry) in self.runs.iter().enumerate() {
            list = list.child(self.run_row(ix, entry, cx));
        }

        let mut rail = div()
            .w(px(250.0))
            .flex()
            .flex_col()
            .border_r_1()
            .border_color(rgb(PANEL_BORDER))
            .bg(rgb(PANEL))
            .child(self.run_button(cx));
        if let Some((severity, note)) = &self.note {
            rail = rail.child(
                div()
                    .px_3()
                    .pb_1()
                    .text_size(px(10.0))
                    .text_color(rgb(severity_color(*severity)))
                    .child(note.clone()),
            );
        }
        rail.child(
            div()
                .px_3()
                .py_1()
                .text_size(px(10.0))
                .text_color(rgb(MUTED))
                .child(format!("{} runs", self.runs.len())),
        )
        .child(list)
    }

    fn run_button(&self, cx: &mut Context<Self>) -> Stateful<Div> {
        let running = self.run_in_flight();
        let (label, color) = if running {
            ("running…", MUTED)
        } else {
            ("▶ run gauntlet", SELECT)
        };
        let button = div()
            .id("start-run")
            .m_2()
            .px_3()
            .py_1()
            .rounded_md()
            .border_1()
            .border_color(rgb(color))
            .text_color(rgb(color))
            .flex()
            .justify_center()
            .child(label);
        if running {
            button
        } else {
            button.cursor_pointer().on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _: &MouseDownEvent, _window, cx| {
                    this.start_run(cx);
                }),
            )
        }
    }

    fn run_row(&self, ix: usize, entry: &RunEntry, cx: &mut Context<Self>) -> Stateful<Div> {
        let selected = self
            .current
            .as_ref()
            .is_some_and(|c| c.entry.run_id == entry.run_id);
        let pinned = self.pinned_baseline.as_deref() == Some(entry.run_id.as_str());
        let dot = verdict_color(entry.verdict);

        let open = {
            let entry = entry.clone();
            cx.listener(
                move |this: &mut RootView, _: &MouseDownEvent, _window, cx| {
                    this.open_entry(&entry, cx);
                },
            )
        };
        let pin = {
            let run_id = entry.run_id.clone();
            cx.listener(
                move |this: &mut RootView, _: &MouseDownEvent, _window, cx| {
                    cx.stop_propagation();
                    this.toggle_pin(&run_id, cx);
                },
            )
        };

        div()
            .id(("run", ix))
            .p_2()
            .rounded_md()
            .cursor_pointer()
            .bg(if selected { rgb(BG) } else { rgba(0x0000_0000) })
            .border_1()
            .border_color(if selected {
                rgb(SELECT)
            } else {
                rgba(0x0000_0000)
            })
            .hover(|style| style.bg(rgb(BG)))
            .on_mouse_down(MouseButton::Left, open)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().w(px(8.0)).h(px(8.0)).rounded_full().bg(rgb(dot)))
                    .child(
                        div()
                            .text_size(px(11.0))
                            .text_color(rgb(TEXT))
                            .child(format_epoch_utc(entry.started_epoch_secs)),
                    )
                    .when(entry.live, |row| {
                        row.child(
                            div()
                                .text_size(px(9.0))
                                .text_color(rgb(SELECT))
                                .child("live"),
                        )
                    }),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.0))
                            .truncate()
                            .text_size(px(9.0))
                            .text_color(rgb(MUTED))
                            .child(format!("{} · {} hosts", entry.run_id, entry.hosts)),
                    )
                    .when(!entry.live, |row| {
                        row.child(
                            div()
                                .id(("pin", ix))
                                .text_size(px(9.0))
                                .text_color(if pinned { rgb(SELECT) } else { rgb(MUTED) })
                                .cursor_pointer()
                                .hover(|style| style.text_color(rgb(SELECT)))
                                .on_mouse_down(MouseButton::Left, pin)
                                .child(if pinned {
                                    "baseline ✕"
                                } else {
                                    "set baseline"
                                }),
                        )
                    }),
            )
    }
}
