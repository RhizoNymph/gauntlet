//! Run history: every run is written to `runs/<run_id>.json` (or --out).
//! History enables "diff against last known-good" later; v1 only lists and
//! loads.
//!
//! A run in flight also drops periodic snapshots at
//! `runs/<run_id>.partial.json`, written atomically (temp file + rename) so a
//! viewer tailing the directory never observes a half-written document. The
//! partial is removed once the final `<run_id>.json` lands.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::RunResults;

pub const DEFAULT_DIR: &str = "runs";

/// File-name suffix of an in-flight snapshot: `<run_id>.partial.json`.
pub const PARTIAL_SUFFIX: &str = ".partial.json";

/// Write `results` under `dir`, creating it; returns the path written.
pub fn save(results: &RunResults, dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating run history directory {}", dir.display()))?;
    let path = dir.join(format!("{}.json", results.run_id));
    let document =
        serde_json::to_string_pretty(results).context("serializing run results to JSON")?;
    std::fs::write(&path, document)
        .with_context(|| format!("writing run results to {}", path.display()))?;
    Ok(path)
}

/// Write an in-flight snapshot of `results` to `<dir>/<run_id>.partial.json`,
/// creating `dir`; returns the path written.
///
/// The document is staged in a dot-prefixed temporary beside its destination
/// and moved into place with a rename, so a reader either sees the previous
/// snapshot or this one, never a torn document. The temporary is hidden and
/// carries a `.tmp` extension so `list`/`list_live` never surface it.
pub fn save_partial(results: &RunResults, dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating run history directory {}", dir.display()))?;
    let path = dir.join(format!("{}{PARTIAL_SUFFIX}", results.run_id));
    let staging = dir.join(format!(".{}{PARTIAL_SUFFIX}.tmp", results.run_id));
    let document =
        serde_json::to_string_pretty(results).context("serializing partial run results to JSON")?;
    std::fs::write(&staging, document)
        .with_context(|| format!("writing partial run results to {}", staging.display()))?;
    std::fs::rename(&staging, &path).with_context(|| {
        format!(
            "publishing partial run results {} -> {}",
            staging.display(),
            path.display()
        )
    })?;
    Ok(path)
}

/// Drop the in-flight snapshot for `run_id`; an already-absent one is success.
pub fn remove_partial(run_id: &str, dir: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(dir.join(format!("{run_id}{PARTIAL_SUFFIX}"))) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

pub fn load(path: &Path) -> Result<RunResults> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading run results from {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing run results {}", path.display()))
}

/// Paths of *completed* runs in `dir`, oldest first (run_id sorts
/// chronologically). In-flight snapshots (`*.partial.json`) and hidden or
/// staging files (leading `.`) are skipped. Missing dir yields an empty list.
pub fn list(dir: &Path) -> Result<Vec<PathBuf>> {
    entries_matching(dir, |name| {
        !name.starts_with('.') && name.ends_with(".json") && !name.ends_with(PARTIAL_SUFFIX)
    })
}

/// Paths of in-flight snapshots in `dir`, sorted the same way. Missing dir
/// yields an empty list.
pub fn list_live(dir: &Path) -> Result<Vec<PathBuf>> {
    entries_matching(dir, |name| {
        !name.starts_with('.') && name.ends_with(PARTIAL_SUFFIX)
    })
}

fn entries_matching(dir: &Path, keep: impl Fn(&str) -> bool) -> Result<Vec<PathBuf>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("listing run history directory {}", dir.display()));
        }
    };
    let mut paths = Vec::new();
    for entry in entries {
        let entry =
            entry.with_context(|| format!("reading run history directory {}", dir.display()))?;
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(&keep)
        {
            paths.push(path);
        }
    }
    // run_id starts with a zero-padded-by-width epoch, so lexical order over
    // file names is chronological for any run in this era.
    paths.sort();
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{Calibration, FleetAnalysis, RunResults, SCHEMA_VERSION};
    use std::collections::BTreeMap;

    fn results(run_id: &str) -> RunResults {
        RunResults {
            schema_version: SCHEMA_VERSION,
            run_id: run_id.into(),
            started_epoch_secs: 1,
            debug_build: false,
            aggregates: Default::default(),
            finished_epoch_secs: 2,
            hosts: BTreeMap::new(),
            fleet: FleetAnalysis::default(),
            calibration: Calibration::default(),
        }
    }

    fn scratch(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("gauntlet-unit-{tag}-{nanos}"))
    }

    #[test]
    fn list_is_chronological_and_ignores_foreign_files() {
        let dir = scratch("history-order");
        let first = save(&results("1700000000-aaaaaa"), &dir).expect("save");
        let second = save(&results("1700000900-000000"), &dir).expect("save");
        std::fs::write(dir.join("notes.txt"), "ignored").expect("write");
        assert_eq!(list(&dir).expect("list"), vec![first, second]);
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn load_of_garbage_is_an_error_not_a_panic() {
        let dir = scratch("history-garbage");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("broken.json");
        std::fs::write(&path, "{ not json").expect("write");
        assert!(load(&path).is_err());
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}
