//! Run-list plumbing for the sidebar: classify files in the runs
//! directory, keep entries ordered and deduplicated, resolve the diff
//! baseline, and format timestamps. Everything except `scan`'s file IO is
//! pure; `scan` caches parses by (mtime, len) so polling every second stays
//! cheap.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use gauntlet::report::{self, Verdict};

/// What a runs-directory file name means.
pub struct RunKey {
    pub run_id: String,
    pub live: bool,
}

/// One run visible in the sidebar.
#[derive(Debug, Clone, PartialEq)]
pub struct RunEntry {
    pub run_id: String,
    /// True for `<run_id>.partial.json` — a run still in flight.
    pub live: bool,
    pub path: PathBuf,
    pub verdict: Option<Verdict>,
    pub hosts: usize,
    pub started_epoch_secs: u64,
}

/// Classify a runs-directory file name. Hidden files (leading '.') and
/// non-JSON files are ignored; `<id>.partial.json` marks a live run.
pub fn classify_file_name(name: &str) -> Option<RunKey> {
    if name.starts_with('.') {
        return None;
    }
    if let Some(id) = name.strip_suffix(".partial.json") {
        return (!id.is_empty()).then(|| RunKey {
            run_id: id.to_string(),
            live: true,
        });
    }
    if let Some(id) = name.strip_suffix(".json") {
        return (!id.is_empty()).then(|| RunKey {
            run_id: id.to_string(),
            live: false,
        });
    }
    None
}

/// Newest first (run ids sort chronologically); when a run has both a
/// final and a stale partial file, the final wins.
pub fn order_and_dedupe(mut entries: Vec<RunEntry>) -> Vec<RunEntry> {
    // Final (live == false) sorts before partial within a run id.
    entries.sort_by(|a, b| a.run_id.cmp(&b.run_id).then(a.live.cmp(&b.live)));
    entries.dedup_by(|b, a| a.run_id == b.run_id);
    entries.reverse();
    entries
}

/// The baseline a diff compares against: the pinned run when it exists,
/// is finished, and is not the run being viewed; otherwise the newest
/// finished run older than the current one.
pub fn effective_baseline<'a>(
    entries: &'a [RunEntry],
    pinned: Option<&str>,
    current_run_id: &str,
) -> Option<&'a RunEntry> {
    if let Some(pinned) = pinned
        && let Some(entry) = entries
            .iter()
            .find(|e| e.run_id == pinned && !e.live && e.run_id != current_run_id)
    {
        return Some(entry);
    }
    entries
        .iter()
        .find(|e| !e.live && e.run_id.as_str() < current_run_id)
}

type Fingerprint = (SystemTime, u64);

/// Parse cache for `scan`: fingerprint plus the parsed entry, or `None`
/// for files that failed to parse (so junk is not re-parsed every tick).
#[derive(Debug, Default)]
pub struct ScanCache {
    files: BTreeMap<PathBuf, (Fingerprint, Option<RunEntry>)>,
}

impl ScanCache {
    /// Fingerprint recorded for `path` by the latest `scan`.
    pub fn fingerprint(&self, path: &Path) -> Option<(SystemTime, u64)> {
        self.files.get(path).map(|(fingerprint, _)| *fingerprint)
    }
}

/// List the runs in `dir`, newest first. Unreadable or unparsable files
/// are skipped. A missing directory is an empty list.
pub fn scan(dir: &Path, cache: &mut ScanCache) -> Vec<RunEntry> {
    let Ok(read) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut seen: Vec<PathBuf> = Vec::new();
    let mut entries: Vec<RunEntry> = Vec::new();
    for dir_entry in read.flatten() {
        let path = dir_entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(key) = classify_file_name(name) else {
            continue;
        };
        let Ok(meta) = dir_entry.metadata() else {
            continue;
        };
        let fingerprint = (
            meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            meta.len(),
        );
        seen.push(path.clone());

        let cached = cache
            .files
            .get(&path)
            .filter(|(cached_fp, _)| *cached_fp == fingerprint);
        let entry = match cached {
            Some((_, entry)) => entry.clone(),
            None => {
                let parsed = report::history::load(&path).ok().map(|results| RunEntry {
                    // The file name is authoritative for identity/liveness;
                    // the document supplies display metadata.
                    run_id: key.run_id.clone(),
                    live: key.live,
                    path: path.clone(),
                    verdict: Some(report::verdict(&results)),
                    hosts: results.hosts.len(),
                    started_epoch_secs: results.started_epoch_secs,
                });
                cache
                    .files
                    .insert(path.clone(), (fingerprint, parsed.clone()));
                parsed
            }
        };
        if let Some(entry) = entry {
            entries.push(entry);
        }
    }
    cache.files.retain(|path, _| seen.contains(path));
    order_and_dedupe(entries)
}

/// "YYYY-MM-DD HH:MM:SS" in UTC.
pub fn format_epoch_utc(epoch_secs: u64) -> String {
    let days = (epoch_secs / 86_400) as i64;
    let rem = epoch_secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02}",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Howard Hinnant's civil-from-days algorithm (days since 1970-01-01).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = (z - era * 146_097) as u64;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era as i64 + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}
