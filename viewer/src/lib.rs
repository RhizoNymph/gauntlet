//! gauntlet-view: native GPUI viewer for gauntlet run results.
//!
//! `model` and `layout` are pure and headless-testable; `ui` is the thin
//! gpui shell that renders them.

pub mod layout;
pub mod model;
pub mod ui;
