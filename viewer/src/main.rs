use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use gpui::{
    App, AppContext as _, Application, Bounds, TitlebarOptions, WindowBounds, WindowOptions, px,
    size,
};

use gauntlet_view::ui::RootView;

/// Native viewer for gauntlet run results: browse runs, watch live runs
/// fill in, and diff runs against a baseline.
#[derive(Debug, Parser)]
#[command(name = "gauntlet-view", version)]
struct Args {
    /// Optional: a specific results JSON to open (a directory here is
    /// treated as --runs-dir).
    path: Option<PathBuf>,
    /// Directory watched for runs; also where GUI-launched runs write.
    #[arg(long, default_value = "runs")]
    runs_dir: PathBuf,
    /// Fleet config passed to `gauntlet run` for GUI-launched runs.
    #[arg(long, default_value = "gauntlet.toml")]
    config: PathBuf,
    /// Start in diff mode (vs the previous run, until a baseline is pinned).
    #[arg(long)]
    diff: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut runs_dir = args.runs_dir;
    let mut initial_file = None;
    if let Some(path) = args.path {
        if path.is_dir() {
            runs_dir = path;
        } else {
            initial_file = Some(path);
        }
    }
    let config_path = args.config;
    let start_in_diff = args.diff;

    Application::new().run(move |cx: &mut App| {
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        let options = WindowOptions {
            titlebar: Some(TitlebarOptions {
                title: Some("gauntlet".into()),
                ..Default::default()
            }),
            window_bounds: Some(WindowBounds::Windowed(Bounds::centered(
                None,
                size(px(1500.0), px(880.0)),
                cx,
            ))),
            ..Default::default()
        };
        let opened = cx.open_window(options, |_window, cx| {
            cx.new(|cx| {
                RootView::new(
                    runs_dir.clone(),
                    config_path.clone(),
                    initial_file.clone(),
                    start_in_diff,
                    cx,
                )
            })
        });
        if let Err(error) = opened {
            eprintln!("gauntlet-view: failed to open window: {error:#}");
            cx.quit();
        }
    });
    Ok(())
}
