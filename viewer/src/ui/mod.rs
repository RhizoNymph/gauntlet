//! gpui shell: theme constants, selection and run state, the root view,
//! and the 1s poll loop that powers live tailing.

pub mod graph;
pub mod sidebar;
pub mod table;

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use gauntlet::report::{self, Verdict};
use gpui::{
    Bounds, Context, Div, FontWeight, IntoElement, ParentElement, Pixels, Render, Styled, Task,
    Window, div, px, rgb,
};

use crate::diff::DiffView;
use crate::model::{EdgeView, NodeView, Severity, ViewModel};
use crate::runs::{self, RunEntry, ScanCache};

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

const POLL_INTERVAL: Duration = Duration::from_secs(1);
/// Basename of the log file GUI-launched runs write into the runs dir.
pub const RUN_LOG_NAME: &str = "gauntlet-run.log";

pub fn severity_color(severity: Severity) -> u32 {
    match severity {
        Severity::Ok => OK,
        Severity::Warn => WARN,
        Severity::Bad => BAD,
    }
}

pub fn verdict_color(verdict: Option<Verdict>) -> u32 {
    match verdict {
        Some(Verdict::Clean) => OK,
        Some(Verdict::Stragglers) => WARN,
        Some(Verdict::HostFailures) => BAD,
        None => MUTED,
    }
}

/// What the user clicked, keyed by identity so it survives live reloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    Node(String),
    /// Ordered pair (a < b), matching `EdgeView`.
    Edge(String, String),
}

/// The run currently on screen.
pub struct LoadedRun {
    pub entry: RunEntry,
    pub vm: Arc<ViewModel>,
}

pub struct RootView {
    runs_dir: PathBuf,
    config_path: PathBuf,
    scan_cache: ScanCache,
    pub runs: Vec<RunEntry>,
    pub current: Option<LoadedRun>,
    /// Fingerprint of the current run's file at load time; a change means
    /// the partial grew and needs a reload.
    current_fingerprint: Option<(SystemTime, u64)>,
    pub pinned_baseline: Option<String>,
    /// Cached baseline (run_id, view model) for diff mode.
    baseline: Option<(String, Arc<ViewModel>)>,
    pub diff_enabled: bool,
    pub diff: Option<Arc<DiffView>>,
    pub selection: Option<Selection>,
    /// Window bounds of the graph pane, recorded from the previous frame's
    /// canvas prepaint; node chips and hit-testing need it.
    pub graph_bounds: Option<Bounds<Pixels>>,
    child: Option<Child>,
    /// Status line under the run button (spawn results, load errors).
    pub note: Option<(Severity, String)>,
    /// Jump to the next live run that appears (set when a run is launched).
    auto_follow: bool,
    _poll: Task<()>,
}

impl RootView {
    pub fn new(
        runs_dir: PathBuf,
        config_path: PathBuf,
        initial_file: Option<PathBuf>,
        cx: &mut Context<Self>,
    ) -> Self {
        let poll = cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(POLL_INTERVAL).await;
                if this.update(cx, |this, cx| this.tick(cx)).is_err() {
                    break;
                }
            }
        });
        let mut view = Self {
            runs_dir,
            config_path,
            scan_cache: ScanCache::default(),
            runs: Vec::new(),
            current: None,
            current_fingerprint: None,
            pinned_baseline: None,
            baseline: None,
            diff_enabled: false,
            diff: None,
            selection: None,
            graph_bounds: None,
            child: None,
            note: None,
            auto_follow: false,
            _poll: poll,
        };
        view.runs = runs::scan(&view.runs_dir, &mut view.scan_cache);
        match initial_file {
            Some(path) => view.load_external(&path),
            None => {
                if let Some(newest) = view.runs.first().cloned() {
                    view.load_run(&newest);
                }
            }
        }
        view
    }

    // -- state transitions -------------------------------------------------

    fn set_current(&mut self, entry: RunEntry, results: &report::RunResults) {
        self.current_fingerprint = self.scan_cache.fingerprint(&entry.path);
        self.current = Some(LoadedRun {
            entry,
            vm: Arc::new(ViewModel::new(results)),
        });
    }

    pub fn load_run(&mut self, entry: &RunEntry) {
        match report::history::load(&entry.path) {
            Ok(results) => self.set_current(entry.clone(), &results),
            // A partial can vanish between scan and load (the run just
            // finished); the next tick picks up the final file.
            Err(_) if entry.live => {}
            Err(error) => {
                self.note = Some((
                    Severity::Bad,
                    format!("failed to load {}: {error:#}", entry.path.display()),
                ));
            }
        }
    }

    /// Open a file passed on the command line (possibly outside runs_dir).
    fn load_external(&mut self, path: &std::path::Path) {
        match report::history::load(path) {
            Ok(results) => {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                let key = runs::classify_file_name(name);
                let entry = RunEntry {
                    run_id: results.run_id.clone(),
                    live: key.map(|k| k.live).unwrap_or(false),
                    path: path.to_path_buf(),
                    verdict: Some(report::verdict(&results)),
                    hosts: results.hosts.len(),
                    started_epoch_secs: results.started_epoch_secs,
                };
                self.set_current(entry, &results);
            }
            Err(error) => {
                self.note = Some((
                    Severity::Bad,
                    format!("failed to load {}: {error:#}", path.display()),
                ));
            }
        }
    }

    pub fn open_entry(&mut self, entry: &RunEntry, cx: &mut Context<Self>) {
        self.load_run(entry);
        self.refresh_diff();
        cx.notify();
    }

    pub fn toggle_diff(&mut self, cx: &mut Context<Self>) {
        self.diff_enabled = !self.diff_enabled;
        self.refresh_diff();
        cx.notify();
    }

    pub fn toggle_pin(&mut self, run_id: &str, cx: &mut Context<Self>) {
        self.pinned_baseline = match self.pinned_baseline.as_deref() {
            Some(pinned) if pinned == run_id => None,
            _ => Some(run_id.to_string()),
        };
        self.refresh_diff();
        cx.notify();
    }

    /// Rebuild `self.diff` from the current run and the effective baseline.
    fn refresh_diff(&mut self) {
        self.diff = None;
        if !self.diff_enabled {
            return;
        }
        let Some(current) = &self.current else {
            return;
        };
        let Some(base_entry) = runs::effective_baseline(
            &self.runs,
            self.pinned_baseline.as_deref(),
            &current.entry.run_id,
        ) else {
            self.baseline = None;
            return;
        };
        let cached = self
            .baseline
            .as_ref()
            .is_some_and(|(id, _)| *id == base_entry.run_id);
        if !cached {
            self.baseline = match report::history::load(&base_entry.path) {
                Ok(results) => Some((
                    base_entry.run_id.clone(),
                    Arc::new(ViewModel::new(&results)),
                )),
                Err(_) => None,
            };
        }
        if let Some((_, baseline_vm)) = &self.baseline {
            self.diff = Some(Arc::new(DiffView::new(&current.vm, baseline_vm)));
        }
    }

    // -- live polling ------------------------------------------------------

    fn tick(&mut self, cx: &mut Context<Self>) {
        let mut changed = self.poll_child();

        let fresh = runs::scan(&self.runs_dir, &mut self.scan_cache);
        if fresh != self.runs {
            self.runs = fresh;
            changed = true;
        }

        if let Some(current) = &self.current
            && current.entry.live
        {
            let run_id = current.entry.run_id.clone();
            let path = current.entry.path.clone();
            if let Some(final_entry) = self
                .runs
                .iter()
                .find(|e| e.run_id == run_id && !e.live)
                .cloned()
            {
                self.load_run(&final_entry);
                changed = true;
            } else if self.scan_cache.fingerprint(&path).is_some()
                && self.scan_cache.fingerprint(&path) != self.current_fingerprint
            {
                if let Some(live_entry) = self.runs.iter().find(|e| e.run_id == run_id).cloned() {
                    self.load_run(&live_entry);
                }
                changed = true;
            }
        }

        if self.auto_follow
            && let Some(live) = self.runs.iter().find(|e| e.live).cloned()
        {
            let already_there = self
                .current
                .as_ref()
                .is_some_and(|c| c.entry.run_id == live.run_id);
            if !already_there {
                self.load_run(&live);
                changed = true;
            }
            self.auto_follow = false;
        }

        if changed {
            self.refresh_diff();
            cx.notify();
        }
    }

    // -- launching runs ----------------------------------------------------

    pub fn run_in_flight(&self) -> bool {
        self.child.is_some()
    }

    pub fn start_run(&mut self, cx: &mut Context<Self>) {
        if self.child.is_some() {
            return;
        }
        match self.spawn_gauntlet() {
            Ok(child) => {
                self.child = Some(child);
                self.auto_follow = true;
                self.note = Some((Severity::Ok, "run started".into()));
            }
            Err(error) => {
                self.note = Some((Severity::Bad, format!("failed to start run: {error:#}")));
            }
        }
        cx.notify();
    }

    fn spawn_gauntlet(&self) -> anyhow::Result<Child> {
        use anyhow::Context as _;
        // Prefer the gauntlet binary next to this one; fall back to PATH.
        let sibling = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|dir| dir.join("gauntlet")));
        let program = match sibling {
            Some(path) if path.exists() => path,
            _ => PathBuf::from("gauntlet"),
        };
        std::fs::create_dir_all(&self.runs_dir)
            .with_context(|| format!("creating {}", self.runs_dir.display()))?;
        let log_path = self.runs_dir.join(RUN_LOG_NAME);
        let log = std::fs::File::create(&log_path)
            .with_context(|| format!("creating {}", log_path.display()))?;
        let log_err = log.try_clone().context("cloning log handle")?;
        Command::new(&program)
            .arg("run")
            .arg("--config")
            .arg(&self.config_path)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(log_err))
            .spawn()
            .with_context(|| format!("spawning {}", program.display()))
    }

    fn poll_child(&mut self) -> bool {
        let Some(child) = &mut self.child else {
            return false;
        };
        match child.try_wait() {
            Ok(None) => false,
            Ok(Some(status)) => {
                self.child = None;
                self.note = Some(match status.code() {
                    Some(0) => (Severity::Ok, "run finished: clean".into()),
                    Some(1) => (Severity::Warn, "run finished: stragglers".into()),
                    Some(2) => (Severity::Bad, "run finished: host failures".into()),
                    code => (
                        Severity::Bad,
                        format!("run exited abnormally ({code:?}) — see {RUN_LOG_NAME}"),
                    ),
                });
                true
            }
            Err(error) => {
                self.child = None;
                self.note = Some((Severity::Bad, format!("lost track of run: {error}")));
                true
            }
        }
    }

    // -- diff-aware display helpers ---------------------------------------

    /// Diff mode is only *active* once a baseline actually resolved.
    pub fn diff_active(&self) -> bool {
        self.diff_enabled && self.diff.is_some()
    }

    pub(super) fn display_node_severity(&self, node: &NodeView) -> Severity {
        match &self.diff {
            Some(diff) if self.diff_enabled => diff
                .node_severity
                .get(&node.host)
                .copied()
                .unwrap_or(Severity::Ok),
            _ => node.severity,
        }
    }

    pub(super) fn display_edge_severity(&self, edge: &EdgeView) -> Severity {
        match &self.diff {
            Some(diff) if self.diff_enabled => diff
                .edge_severity
                .get(&(edge.a.clone(), edge.b.clone()))
                .copied()
                .unwrap_or(Severity::Ok),
            _ => edge.severity,
        }
    }

    // -- header ------------------------------------------------------------

    fn render_header(&self, cx: &mut Context<Self>) -> Div {
        use gpui::{InteractiveElement as _, MouseButton, MouseDownEvent};

        let mut header = div()
            .flex()
            .items_center()
            .gap_3()
            .px_4()
            .py_2()
            .bg(rgb(PANEL))
            .border_b_1()
            .border_color(rgb(PANEL_BORDER))
            .child(div().font_weight(FontWeight::BOLD).child("gauntlet"));

        match &self.current {
            Some(run) => {
                let vm = &run.vm;
                let (verdict_text, color) = match vm.verdict {
                    Verdict::Clean => ("clean", OK),
                    Verdict::Stragglers => ("stragglers", WARN),
                    Verdict::HostFailures => ("host failures", BAD),
                };
                header = header
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
                            .text_color(rgb(color))
                            .border_1()
                            .border_color(rgb(color))
                            .child(verdict_text),
                    );
                if run.entry.live {
                    header = header.child(
                        div()
                            .px_2()
                            .py(px(1.0))
                            .rounded_md()
                            .text_size(px(11.0))
                            .text_color(rgb(SELECT))
                            .border_1()
                            .border_color(rgb(SELECT))
                            .child("● live"),
                    );
                }
                header = header.child(div().text_color(rgb(MUTED)).text_size(px(12.0)).child(
                    format!(
                        "{} hosts · {} links · {}s wall",
                        vm.nodes.len(),
                        vm.edges.len(),
                        vm.wall_secs
                    ),
                ));
            }
            None => {
                header = header.child(
                    div()
                        .text_color(rgb(MUTED))
                        .text_size(px(12.0))
                        .child("no run loaded"),
                );
            }
        }

        // Diff toggle.
        let (diff_label, diff_color) = if !self.diff_enabled {
            ("diff: off".to_string(), MUTED)
        } else {
            match &self.diff {
                Some(diff) => (format!("diff vs {}", diff.baseline_run_id), SELECT),
                None => ("diff: no baseline".to_string(), WARN),
            }
        };
        header.child(
            div()
                .id("diff-toggle")
                .px_2()
                .py(px(1.0))
                .rounded_md()
                .text_size(px(11.0))
                .text_color(rgb(diff_color))
                .border_1()
                .border_color(rgb(diff_color))
                .cursor_pointer()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|this, _: &MouseDownEvent, _window, cx| {
                        this.toggle_diff(cx);
                    }),
                )
                .child(diff_label),
        )
    }
}

impl Render for RootView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let mut row = div()
            .flex_1()
            .min_h(px(0.0))
            .flex()
            .child(self.render_sidebar(cx));
        if self.current.is_some() {
            row = row.child(self.render_graph(cx)).child(self.render_side(cx));
        } else {
            row = row.child(
                div()
                    .flex_1()
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_color(rgb(MUTED))
                    .child("no runs yet — press ▶ run gauntlet"),
            );
        }

        div()
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(BG))
            .text_color(rgb(TEXT))
            .text_size(px(13.0))
            .child(self.render_header(cx))
            .child(row)
    }
}
