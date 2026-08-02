use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context as _, Result};
use clap::Parser;
use gpui::{
    App, AppContext as _, Application, Bounds, TitlebarOptions, WindowBounds, WindowOptions, px,
    size,
};

use gauntlet_view::model::ViewModel;
use gauntlet_view::ui::RootView;

/// Native viewer for gauntlet run results.
#[derive(Debug, Parser)]
#[command(name = "gauntlet-view", version)]
struct Args {
    /// A run results JSON file, or a directory of them (the newest run is
    /// shown). Defaults to gauntlet's `runs/` history directory.
    #[arg(default_value = "runs")]
    path: PathBuf,
}

fn resolve_input(path: &Path) -> Result<PathBuf> {
    let metadata =
        std::fs::metadata(path).with_context(|| format!("reading {}", path.display()))?;
    if !metadata.is_dir() {
        return Ok(path.to_path_buf());
    }
    let runs = gauntlet::report::history::list(path)?;
    runs.into_iter()
        .next_back()
        .with_context(|| format!("no run results (*.json) in {}", path.display()))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input = resolve_input(&args.path)?;
    let results = gauntlet::report::history::load(&input)?;
    let vm = Arc::new(ViewModel::new(&results));

    Application::new().run(move |cx: &mut App| {
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        let title = format!("gauntlet — {}", vm.run_id);
        let options = WindowOptions {
            titlebar: Some(TitlebarOptions {
                title: Some(title.into()),
                ..Default::default()
            }),
            window_bounds: Some(WindowBounds::Windowed(Bounds::centered(
                None,
                size(px(1400.0), px(860.0)),
                cx,
            ))),
            ..Default::default()
        };
        let opened = cx.open_window(options, |_window, cx| cx.new(|_| RootView::new(vm.clone())));
        if let Err(error) = opened {
            eprintln!("gauntlet-view: failed to open window: {error:#}");
            cx.quit();
        }
    });
    Ok(())
}
