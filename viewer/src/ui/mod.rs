//! gpui shell: theme constants, selection and run state, the root view,
//! and the 1s poll loop that powers live tailing.

pub mod bootstrap;
pub mod graph;
pub mod sidebar;
pub mod table;

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use gauntlet::orchestrator::bootstrap::BootstrapReport;
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
/// Dot-prefixed so the run scanner ignores them.
pub const BOOTSTRAP_JSON_NAME: &str = ".bootstrap.json";
pub const BOOTSTRAP_LOG_NAME: &str = ".bootstrap.log";

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

/// What kind of gauntlet child process the GUI is managing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildKind {
    Run,
    Bootstrap,
}

/// What occupies the center pane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CenterView {
    Graph,
    Bootstrap,
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
    child: Option<(ChildKind, Child)>,
    /// The run id the GUI's own launch produced, once its partial appears;
    /// cancel uses it to clean up the orphaned partial file.
    launched_run_id: Option<String>,
    /// Status line under the run button (spawn results, load errors).
    pub note: Option<(Severity, String)>,
    /// Jump to the next live run that appears (set when a run is launched).
    auto_follow: bool,
    /// Last bootstrap readiness report, shown in the center pane.
    pub bootstrap_report: Option<BootstrapReport>,
    center: CenterView,
    _poll: Task<()>,
}

impl RootView {
    pub fn new(
        runs_dir: PathBuf,
        config_path: PathBuf,
        initial_file: Option<PathBuf>,
        start_in_diff: bool,
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
            diff_enabled: start_in_diff,
            diff: None,
            selection: None,
            graph_bounds: None,
            child: None,
            launched_run_id: None,
            note: None,
            auto_follow: false,
            bootstrap_report: None,
            center: CenterView::Graph,
            _poll: poll,
        };
        view.runs = runs::scan(&view.runs_dir, &mut view.scan_cache);
        // A previous bootstrap outlives viewer restarts; land on it when
        // there is nothing else to show yet.
        view.bootstrap_report = std::fs::read_to_string(view.runs_dir.join(BOOTSTRAP_JSON_NAME))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok());
        if view.bootstrap_report.is_some() && view.runs.is_empty() {
            view.center = CenterView::Bootstrap;
        }
        match initial_file {
            Some(path) => view.load_external(&path),
            None => {
                if let Some(newest) = view.runs.first().cloned() {
                    view.load_run(&newest);
                }
            }
        }
        view.refresh_diff();
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
                    modified: None,
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
            if matches!(self.child, Some((ChildKind::Run, _))) {
                self.launched_run_id = Some(live.run_id.clone());
            }
            self.auto_follow = false;
        }

        if changed {
            self.refresh_diff();
            cx.notify();
        }
    }

    // -- launching runs ----------------------------------------------------

    pub fn child_kind(&self) -> Option<ChildKind> {
        self.child.as_ref().map(|(kind, _)| *kind)
    }

    pub fn start_run(&mut self, cx: &mut Context<Self>) {
        if self.child.is_some() {
            return;
        }
        match self.spawn_run() {
            Ok(child) => {
                self.child = Some((ChildKind::Run, child));
                self.auto_follow = true;
                self.center = CenterView::Graph;
                self.note = Some((Severity::Ok, "run started".into()));
            }
            Err(error) => {
                self.note = Some((Severity::Bad, format!("failed to start run: {error:#}")));
            }
        }
        cx.notify();
    }

    pub fn start_bootstrap(&mut self, cx: &mut Context<Self>) {
        if self.child.is_some() {
            return;
        }
        match self.spawn_bootstrap() {
            Ok(child) => {
                self.child = Some((ChildKind::Bootstrap, child));
                self.note = Some((Severity::Ok, "bootstrap started".into()));
            }
            Err(error) => {
                self.note = Some((
                    Severity::Bad,
                    format!("failed to start bootstrap: {error:#}"),
                ));
            }
        }
        cx.notify();
    }

    /// Kill the child process. A cancelled run leaves an orphaned partial
    /// snapshot behind (the orchestrator never got to clean it up), so
    /// remove it and fall back to the newest finished run.
    pub fn cancel_child(&mut self, cx: &mut Context<Self>) {
        let Some((kind, mut child)) = self.child.take() else {
            return;
        };
        let _ = child.kill();
        let _ = child.wait();
        match kind {
            ChildKind::Run => {
                self.auto_follow = false;
                if let Some(run_id) = self.launched_run_id.take() {
                    let partial = self.runs_dir.join(format!("{run_id}.partial.json"));
                    let _ = std::fs::remove_file(&partial);
                    let was_viewing = self
                        .current
                        .as_ref()
                        .is_some_and(|c| c.entry.run_id == run_id && c.entry.live);
                    self.runs = runs::scan(&self.runs_dir, &mut self.scan_cache);
                    if was_viewing {
                        if let Some(next) = self.runs.iter().find(|e| !e.live).cloned() {
                            self.load_run(&next);
                        } else {
                            self.current = None;
                        }
                    }
                }
                self.note = Some((Severity::Warn, "run cancelled".into()));
            }
            ChildKind::Bootstrap => {
                self.note = Some((Severity::Warn, "bootstrap cancelled".into()));
            }
        }
        self.refresh_diff();
        cx.notify();
    }

    /// The gauntlet binary next to this one, falling back to PATH.
    fn gauntlet_program() -> PathBuf {
        let sibling = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|dir| dir.join("gauntlet")));
        match sibling {
            Some(path) if path.exists() => path,
            _ => PathBuf::from("gauntlet"),
        }
    }

    fn spawn_run(&self) -> anyhow::Result<Child> {
        use anyhow::Context as _;
        let program = Self::gauntlet_program();
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

    fn spawn_bootstrap(&self) -> anyhow::Result<Child> {
        use anyhow::Context as _;
        let program = Self::gauntlet_program();
        std::fs::create_dir_all(&self.runs_dir)
            .with_context(|| format!("creating {}", self.runs_dir.display()))?;
        let json_path = self.runs_dir.join(BOOTSTRAP_JSON_NAME);
        let json = std::fs::File::create(&json_path)
            .with_context(|| format!("creating {}", json_path.display()))?;
        let log_path = self.runs_dir.join(BOOTSTRAP_LOG_NAME);
        let log = std::fs::File::create(&log_path)
            .with_context(|| format!("creating {}", log_path.display()))?;
        Command::new(&program)
            .arg("bootstrap")
            .arg("--config")
            .arg(&self.config_path)
            .arg("--json")
            .stdin(Stdio::null())
            .stdout(Stdio::from(json))
            .stderr(Stdio::from(log))
            .spawn()
            .with_context(|| format!("spawning {}", program.display()))
    }

    fn poll_child(&mut self) -> bool {
        let Some((kind, child)) = &mut self.child else {
            return false;
        };
        let kind = *kind;
        match child.try_wait() {
            Ok(None) => false,
            Ok(Some(status)) => {
                self.child = None;
                match kind {
                    ChildKind::Run => {
                        self.launched_run_id = None;
                        self.note = Some(match status.code() {
                            Some(0) => (Severity::Ok, "run finished: clean".into()),
                            Some(1) => (Severity::Warn, "run finished: stragglers".into()),
                            Some(2) => (Severity::Bad, "run finished: host failures".into()),
                            code => (
                                Severity::Bad,
                                format!("run exited abnormally ({code:?}) — see {RUN_LOG_NAME}"),
                            ),
                        });
                    }
                    ChildKind::Bootstrap => self.finish_bootstrap(status.success()),
                }
                true
            }
            Err(error) => {
                self.child = None;
                self.note = Some((Severity::Bad, format!("lost track of child: {error}")));
                true
            }
        }
    }

    fn finish_bootstrap(&mut self, all_ready: bool) {
        let path = self.runs_dir.join(BOOTSTRAP_JSON_NAME);
        let report = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<BootstrapReport>(&text).ok());
        match report {
            Some(report) => {
                self.note = Some(if all_ready {
                    (Severity::Ok, "bootstrap: all hosts ready".into())
                } else {
                    (Severity::Warn, "bootstrap: some hosts not ready".into())
                });
                self.bootstrap_report = Some(report);
                self.center = CenterView::Bootstrap;
            }
            None => {
                self.note = Some((
                    Severity::Bad,
                    format!("bootstrap produced no readiness report — see {BOOTSTRAP_LOG_NAME}"),
                ));
            }
        }
    }

    pub(super) fn show_graph(&mut self) {
        self.center = CenterView::Graph;
    }

    pub(super) fn show_bootstrap(&mut self) {
        if self.bootstrap_report.is_some() {
            self.center = CenterView::Bootstrap;
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
                if vm.debug_build {
                    header = header.child(
                        div()
                            .px_2()
                            .py(px(1.0))
                            .rounded_md()
                            .text_size(px(11.0))
                            .text_color(rgb(BAD))
                            .border_1()
                            .border_color(rgb(BAD))
                            .child("debug build"),
                    );
                }
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
        if self.center == CenterView::Bootstrap && self.bootstrap_report.is_some() {
            row = row.child(self.render_bootstrap_pane(cx));
        } else if self.current.is_some() {
            row = row.child(self.render_graph(cx)).child(self.render_side(cx));
        } else {
            row = row.child(
                div()
                    .flex_1()
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_color(rgb(MUTED))
                    .child("no runs yet — press ▶ run gauntlet or ⚙ bootstrap"),
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
