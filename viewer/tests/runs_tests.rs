//! Tests for run-list scanning, ordering, baseline resolution, and
//! timestamp formatting.

use std::collections::BTreeMap;
use std::path::PathBuf;

use gauntlet::report::{Calibration, FleetAnalysis, RunResults, SCHEMA_VERSION, Verdict};
use gauntlet_view::runs::{
    RunEntry, ScanCache, classify_file_name, effective_baseline, format_epoch_utc,
    order_and_dedupe, scan,
};

fn entry(run_id: &str, live: bool) -> RunEntry {
    RunEntry {
        run_id: run_id.into(),
        live,
        path: PathBuf::from(format!(
            "runs/{run_id}{}",
            if live { ".partial.json" } else { ".json" }
        )),
        verdict: Some(Verdict::Clean),
        hosts: 3,
        started_epoch_secs: 0,
        modified: None,
    }
}

#[test]
fn file_names_classify_finals_partials_and_junk() {
    let final_run = classify_file_name("1700000000-abc123.json").expect("final");
    assert_eq!(final_run.run_id, "1700000000-abc123");
    assert!(!final_run.live);

    let live_run = classify_file_name("1700000000-abc123.partial.json").expect("live");
    assert_eq!(live_run.run_id, "1700000000-abc123");
    assert!(live_run.live);

    assert!(classify_file_name(".1700000000-abc123.partial.json.tmp").is_none());
    assert!(classify_file_name(".hidden.json").is_none());
    assert!(classify_file_name("notes.txt").is_none());
    assert!(classify_file_name(".json").is_none());
}

#[test]
fn ordering_is_newest_first_and_finals_shadow_partials() {
    let entries = vec![
        entry("1700000100-bbb", false),
        entry("1700000200-ccc", true),
        entry("1700000100-bbb", true), // stale partial of a finished run
        entry("1700000000-aaa", false),
    ];
    let ordered = order_and_dedupe(entries);
    let ids: Vec<(&str, bool)> = ordered
        .iter()
        .map(|e| (e.run_id.as_str(), e.live))
        .collect();
    assert_eq!(
        ids,
        vec![
            ("1700000200-ccc", true),
            ("1700000100-bbb", false),
            ("1700000000-aaa", false),
        ]
    );
}

#[test]
fn baseline_defaults_to_the_previous_finished_run() {
    let runs = vec![
        entry("1700000300-ddd", true),
        entry("1700000200-ccc", false),
        entry("1700000100-bbb", false),
        entry("1700000000-aaa", false),
    ];
    // Current is the newest finished run; baseline is the one before it.
    let baseline = effective_baseline(&runs, None, "1700000200-ccc").expect("baseline");
    assert_eq!(baseline.run_id, "1700000100-bbb");
    // A live run never becomes the implicit baseline.
    let baseline = effective_baseline(&runs, None, "1700000400-eee").expect("baseline");
    assert_eq!(baseline.run_id, "1700000200-ccc");
}

#[test]
fn pinned_baseline_wins_unless_it_is_the_current_run() {
    let runs = vec![
        entry("1700000200-ccc", false),
        entry("1700000100-bbb", false),
        entry("1700000000-aaa", false),
    ];
    let baseline =
        effective_baseline(&runs, Some("1700000000-aaa"), "1700000200-ccc").expect("baseline");
    assert_eq!(baseline.run_id, "1700000000-aaa");
    // Pinning the run being viewed falls back to the previous run.
    let baseline =
        effective_baseline(&runs, Some("1700000200-ccc"), "1700000200-ccc").expect("baseline");
    assert_eq!(baseline.run_id, "1700000100-bbb");
    // Unknown pin falls back too.
    let baseline = effective_baseline(&runs, Some("gone"), "1700000200-ccc").expect("baseline");
    assert_eq!(baseline.run_id, "1700000100-bbb");
}

#[test]
fn no_baseline_when_current_is_the_oldest() {
    let runs = vec![entry("1700000000-aaa", false)];
    assert!(effective_baseline(&runs, None, "1700000000-aaa").is_none());
}

#[test]
fn epoch_formatting_is_utc_civil_time() {
    assert_eq!(format_epoch_utc(0), "1970-01-01 00:00:00");
    assert_eq!(format_epoch_utc(86_399), "1970-01-01 23:59:59");
    assert_eq!(format_epoch_utc(86_400), "1970-01-02 00:00:00");
    assert_eq!(format_epoch_utc(1_704_067_200), "2024-01-01 00:00:00");
    assert_eq!(format_epoch_utc(1_785_637_899), "2026-08-02 02:31:39");
}

fn results(run_id: &str, started: u64) -> RunResults {
    RunResults {
        schema_version: SCHEMA_VERSION,
        run_id: run_id.into(),
        started_epoch_secs: started,
        debug_build: false,
        aggregates: Default::default(),
        finished_epoch_secs: started + 60,
        hosts: BTreeMap::new(),
        fleet: FleetAnalysis::default(),
        calibration: Calibration::default(),
    }
}

#[test]
fn scan_reads_finals_and_partials_and_skips_junk() {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("gauntlet-view-scan-{nanos}"));
    std::fs::create_dir_all(&dir).expect("mkdir");

    let write = |name: &str, results: &RunResults| {
        let json = serde_json::to_string(results).expect("serialize");
        std::fs::write(dir.join(name), json).expect("write");
    };
    write(
        "1700000000-aaa.json",
        &results("1700000000-aaa", 1_700_000_000),
    );
    write(
        "1700000900-bbb.partial.json",
        &results("1700000900-bbb", 1_700_000_900),
    );
    std::fs::write(dir.join("garbage.json"), "{ not json").expect("write junk");
    std::fs::write(dir.join("notes.txt"), "ignored").expect("write notes");

    let mut cache = ScanCache::default();
    let entries = scan(&dir, &mut cache);
    let ids: Vec<(&str, bool)> = entries
        .iter()
        .map(|e| (e.run_id.as_str(), e.live))
        .collect();
    assert_eq!(
        ids,
        vec![("1700000900-bbb", true), ("1700000000-aaa", false)]
    );
    assert_eq!(entries[1].started_epoch_secs, 1_700_000_000);

    // A second scan with the same cache parses nothing new and agrees.
    let again = scan(&dir, &mut cache);
    assert_eq!(again, entries);

    // Missing directory is an empty list, not an error.
    assert!(scan(&dir.join("nope"), &mut cache).is_empty());

    std::fs::remove_dir_all(&dir).expect("cleanup");
}
