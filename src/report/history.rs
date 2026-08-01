//! Run history: every run is written to `runs/<run_id>.json` (or --out).
//! History enables "diff against last known-good" later; v1 only lists and
//! loads.

use std::path::{Path, PathBuf};

use anyhow::Result;

use super::RunResults;

pub const DEFAULT_DIR: &str = "runs";

/// Write `results` under `dir`, creating it; returns the path written.
pub fn save(results: &RunResults, dir: &Path) -> Result<PathBuf> {
    let _ = (results, dir);
    todo!("agent D: implement")
}

pub fn load(path: &Path) -> Result<RunResults> {
    let _ = path;
    todo!("agent D: implement")
}

/// Paths of saved runs in `dir`, oldest first (run_id sorts
/// chronologically). Missing dir yields an empty list.
pub fn list(dir: &Path) -> Result<Vec<PathBuf>> {
    let _ = dir;
    todo!("agent D: implement")
}
