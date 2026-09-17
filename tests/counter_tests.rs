//! Error-counter snapshot/delta tests: parsing from fixture sysfs trees and
//! tool output, delta logic, event round-trips, and report integration.
//! Nothing here touches real GPUs/IB/NVMe hardware.

mod common;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use gauntlet::agent::counters;
use gauntlet::orchestrator::collect::{Collector, HostObservations};
use gauntlet::proto::{
    AgentEvent, CounterDelta, CounterDeltas, CounterDomain, CounterReading, CounterRequest,
    CounterSnapshot, decode_event, encode_event,
};
use gauntlet::report::{self, Verdict};

fn reading(domain: CounterDomain, device: &str, counter: &str, value: u64) -> CounterReading {
    CounterReading {
        domain,
        device: device.into(),
        counter: counter.into(),
        value,
    }
}

fn write(path: &Path, content: &str) {
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    fs::write(path, content).expect("write fixture");
}

// ---------------------------------------------------------------------------
// PCIe AER
// ---------------------------------------------------------------------------

#[test]
fn pcie_aer_counters_parse_per_device_files() {
    let root = common::scratch_dir("aer");
    let dev = root.join("0000:65:00.0");
    write(
        &dev.join("aer_dev_correctable"),
        "RxErr 0\nBadTLP 3\nBadDLLP 0\nRollover 0\nTimeout 1\nNonFatalErr 0\nCorrIntErr 0\nHeaderOF 0\nTOTAL_ERR_COR 4\n",
    );
    write(
        &dev.join("aer_dev_fatal"),
        "Undefined 0\nDLP 0\nTOTAL_ERR_FATAL 0\n",
    );
    // A device without AER capability contributes nothing (and must not
    // error the scan).
    fs::create_dir_all(root.join("0000:00:1f.2")).expect("mkdir");

    let readings = counters::pcie_aer_counters(&root);
    assert!(!readings.is_empty());
    assert!(readings.iter().all(|r| r.domain == CounterDomain::PcieAer));
    assert!(readings.iter().all(|r| r.device == "0000:65:00.0"));
    let by_counter: BTreeMap<&str, u64> = readings
        .iter()
        .map(|r| (r.counter.as_str(), r.value))
        .collect();
    assert_eq!(by_counter.get("aer_dev_correctable.BadTLP"), Some(&3));
    assert_eq!(by_counter.get("aer_dev_correctable.Timeout"), Some(&1));
    assert_eq!(
        by_counter.get("aer_dev_correctable.TOTAL_ERR_COR"),
        Some(&4)
    );
    assert_eq!(by_counter.get("aer_dev_fatal.TOTAL_ERR_FATAL"), Some(&0));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn pcie_aer_missing_root_is_empty() {
    assert!(counters::pcie_aer_counters(Path::new("/no/such/dir")).is_empty());
}

// ---------------------------------------------------------------------------
// EDAC
// ---------------------------------------------------------------------------

#[test]
fn edac_counters_cover_mc_and_dimm_levels() {
    let root = common::scratch_dir("edac");
    write(&root.join("mc0/ce_count"), "7\n");
    write(&root.join("mc0/ue_count"), "0\n");
    write(&root.join("mc0/ce_noinfo_count"), "2\n");
    write(&root.join("mc0/dimm0/dimm_ce_count"), "5\n");
    write(&root.join("mc0/dimm0/dimm_ue_count"), "0\n");
    write(&root.join("mc1/ce_count"), "0\n");
    // Non-mc entries are ignored.
    write(&root.join("mc_symlink_junk/ce_count"), "9\n");

    let readings = counters::edac_counters(&root);
    assert!(readings.iter().all(|r| r.domain == CounterDomain::Edac));
    let by_key: BTreeMap<(String, String), u64> = readings
        .iter()
        .map(|r| ((r.device.clone(), r.counter.clone()), r.value))
        .collect();
    assert_eq!(by_key.get(&("mc0".into(), "ce_count".into())), Some(&7));
    assert_eq!(
        by_key.get(&("mc0".into(), "ce_noinfo_count".into())),
        Some(&2)
    );
    assert_eq!(
        by_key.get(&("mc0/dimm0".into(), "dimm_ce_count".into())),
        Some(&5)
    );
    assert_eq!(by_key.get(&("mc1".into(), "ce_count".into())), Some(&0));
    assert!(!by_key.keys().any(|(device, _)| device.contains("junk")));
    fs::remove_dir_all(&root).ok();
}

// ---------------------------------------------------------------------------
// InfiniBand
// ---------------------------------------------------------------------------

#[test]
fn ib_port_counters_read_the_error_counter_set() {
    let root = common::scratch_dir("ib");
    let counters_dir = root.join("mlx5_0/ports/1/counters");
    write(&counters_dir.join("symbol_error"), "2\n");
    write(&counters_dir.join("link_error_recovery"), "1\n");
    write(&counters_dir.join("port_rcv_errors"), "0\n");
    write(&counters_dir.join("port_xmit_discards"), "4\n");
    write(&counters_dir.join("port_xmit_wait"), "12345\n");
    write(&counters_dir.join("link_downed"), "0\n");
    // A second port missing most counters degrades to whatever exists.
    write(&root.join("mlx5_1/ports/1/counters/symbol_error"), "9\n");

    let readings = counters::ib_port_counters(&root);
    assert!(readings.iter().all(|r| r.domain == CounterDomain::IbPort));
    let by_key: BTreeMap<(String, String), u64> = readings
        .iter()
        .map(|r| ((r.device.clone(), r.counter.clone()), r.value))
        .collect();
    assert_eq!(
        by_key.get(&("mlx5_0/1".into(), "symbol_error".into())),
        Some(&2)
    );
    assert_eq!(
        by_key.get(&("mlx5_0/1".into(), "port_xmit_wait".into())),
        Some(&12345)
    );
    assert_eq!(
        by_key.get(&("mlx5_1/1".into(), "symbol_error".into())),
        Some(&9)
    );
    assert!(
        !by_key.contains_key(&("mlx5_1/1".into(), "port_rcv_errors".into())),
        "absent counters are skipped, not zeroed"
    );
    fs::remove_dir_all(&root).ok();
}

#[test]
fn ib_missing_root_is_empty() {
    assert!(counters::ib_port_counters(Path::new("/no/such/dir")).is_empty());
}

// ---------------------------------------------------------------------------
// NVMe
// ---------------------------------------------------------------------------

#[test]
fn nvme_device_names_enumerate_controllers() {
    let root = common::scratch_dir("nvme");
    fs::create_dir_all(root.join("nvme0")).expect("mkdir");
    fs::create_dir_all(root.join("nvme1")).expect("mkdir");
    let names = counters::nvme_device_names(&root);
    assert_eq!(names, vec!["nvme0".to_string(), "nvme1".to_string()]);
    assert!(counters::nvme_device_names(Path::new("/no/such/dir")).is_empty());
    fs::remove_dir_all(&root).ok();
}

#[test]
fn nvme_smart_json_parses_smartctl_shape() {
    let text = r#"{
        "json_format_version": [1, 0],
        "nvme_smart_health_information_log": {
            "critical_warning": 0,
            "media_errors": 3,
            "num_err_log_entries": 17
        }
    }"#;
    let readings = counters::parse_nvme_smart_json("nvme0", text);
    let by_counter: BTreeMap<&str, u64> = readings
        .iter()
        .map(|r| (r.counter.as_str(), r.value))
        .collect();
    assert_eq!(by_counter.get("media_errors"), Some(&3));
    assert_eq!(by_counter.get("num_err_log_entries"), Some(&17));
    assert!(readings.iter().all(|r| r.device == "nvme0"));
    assert!(readings.iter().all(|r| r.domain == CounterDomain::Nvme));
}

#[test]
fn nvme_smart_json_parses_nvme_cli_shape() {
    // `nvme smart-log -o json` puts the fields at the top level.
    let text = r#"{"critical_warning":0,"media_errors":0,"num_err_log_entries":2}"#;
    let readings = counters::parse_nvme_smart_json("nvme1", text);
    let by_counter: BTreeMap<&str, u64> = readings
        .iter()
        .map(|r| (r.counter.as_str(), r.value))
        .collect();
    assert_eq!(by_counter.get("media_errors"), Some(&0));
    assert_eq!(by_counter.get("num_err_log_entries"), Some(&2));
}

#[test]
fn nvme_smart_json_garbage_is_empty() {
    assert!(counters::parse_nvme_smart_json("nvme0", "not json").is_empty());
    assert!(counters::parse_nvme_smart_json("nvme0", "{}").is_empty());
}

// ---------------------------------------------------------------------------
// GPU ECC / row remap (nvidia-smi CSV, same conventions as the inventory)
// ---------------------------------------------------------------------------

#[test]
fn gpu_ecc_csv_parses_counters_per_gpu() {
    let text = "0, 0, 1, 5, 0, 2, 0\n1, 0, 0, 0, 0, 0, 0\n";
    let readings = counters::parse_gpu_ecc_csv(text);
    assert!(readings.iter().all(|r| r.domain == CounterDomain::GpuEcc));
    let by_key: BTreeMap<(String, String), u64> = readings
        .iter()
        .map(|r| ((r.device.clone(), r.counter.clone()), r.value))
        .collect();
    assert_eq!(
        by_key.get(&("gpu0".into(), "ecc_uncorrected_volatile".into())),
        Some(&1)
    );
    assert_eq!(
        by_key.get(&("gpu0".into(), "ecc_corrected_aggregate".into())),
        Some(&5)
    );
    assert_eq!(
        by_key.get(&("gpu0".into(), "remapped_rows_correctable".into())),
        Some(&2)
    );
    assert_eq!(
        by_key.get(&("gpu1".into(), "ecc_corrected_volatile".into())),
        Some(&0)
    );
}

#[test]
fn gpu_ecc_csv_skips_na_fields_not_whole_gpus() {
    let text = "0, [N/A], N/A, 12, 0, [Not Supported], 0\n";
    let readings = counters::parse_gpu_ecc_csv(text);
    let counters_seen: Vec<&str> = readings.iter().map(|r| r.counter.as_str()).collect();
    assert!(counters_seen.contains(&"ecc_corrected_aggregate"));
    assert!(!counters_seen.contains(&"ecc_corrected_volatile"));
    assert!(!counters_seen.contains(&"remapped_rows_correctable"));
}

#[test]
fn gpu_ecc_csv_empty_or_malformed_is_empty() {
    assert!(counters::parse_gpu_ecc_csv("").is_empty());
    assert!(counters::parse_gpu_ecc_csv("garbage line\n").is_empty());
}

// ---------------------------------------------------------------------------
// NVLink (nvidia-smi nvlink -e)
// ---------------------------------------------------------------------------

#[test]
fn nvlink_errors_parse_gpu_and_link_structure() {
    let text = "GPU 0: NVIDIA H100 (UUID: GPU-aaa)\n\
         Link 0: Replay Errors: 0\n\
         Link 0: Recovery Errors: 1\n\
         Link 0: CRC Errors: 0\n\
         Link 1: Replay Errors: 7\n\
GPU 1: NVIDIA H100 (UUID: GPU-bbb)\n\
         Link 0: CRC Errors: 2\n";
    let readings = counters::parse_nvlink_errors(text);
    assert!(readings.iter().all(|r| r.domain == CounterDomain::Nvlink));
    let by_key: BTreeMap<(String, String), u64> = readings
        .iter()
        .map(|r| ((r.device.clone(), r.counter.clone()), r.value))
        .collect();
    assert_eq!(
        by_key.get(&("gpu0/link0".into(), "recovery_errors".into())),
        Some(&1)
    );
    assert_eq!(
        by_key.get(&("gpu0/link1".into(), "replay_errors".into())),
        Some(&7)
    );
    assert_eq!(
        by_key.get(&("gpu1/link0".into(), "crc_errors".into())),
        Some(&2)
    );
}

#[test]
fn nvlink_noise_lines_are_ignored() {
    assert!(counters::parse_nvlink_errors("NVLink is not supported\n").is_empty());
    assert!(counters::parse_nvlink_errors("").is_empty());
}

// ---------------------------------------------------------------------------
// dmesg Xid lines
// ---------------------------------------------------------------------------

#[test]
fn xid_lines_are_counted_not_deduplicated() {
    let log = "boot ok\n\
NVRM: Xid (PCI:0000:65:00): 13, pid=1234, Graphics Exception\n\
noise\n\
NVRM: Xid (PCI:0000:65:00): 13, pid=1235, Graphics Exception\n\
kernel: NVRM: Xid (0000:01:00): 79, GPU has fallen off the bus.\n";
    assert_eq!(counters::count_xid_lines(log), 3);
    assert_eq!(counters::count_xid_lines(""), 0);
    assert_eq!(counters::count_xid_lines("no xids here"), 0);
}

// ---------------------------------------------------------------------------
// Delta logic
// ---------------------------------------------------------------------------

#[test]
fn diff_matches_counters_by_domain_device_and_name() {
    let before = CounterSnapshot {
        readings: vec![
            reading(CounterDomain::Edac, "mc0", "ce_count", 5),
            reading(CounterDomain::IbPort, "mlx5_0/1", "symbol_error", 0),
            reading(
                CounterDomain::PcieAer,
                "0000:65:00.0",
                "aer_dev_correctable.BadTLP",
                2,
            ),
        ],
    };
    let after = CounterSnapshot {
        readings: vec![
            reading(CounterDomain::Edac, "mc0", "ce_count", 9),
            reading(CounterDomain::IbPort, "mlx5_0/1", "symbol_error", 0),
            reading(
                CounterDomain::PcieAer,
                "0000:65:00.0",
                "aer_dev_correctable.BadTLP",
                2,
            ),
        ],
    };
    let deltas = counters::diff_snapshots(&before, &after);
    assert_eq!(deltas.deltas.len(), 3, "every matched counter is reported");
    let edac = deltas
        .deltas
        .iter()
        .find(|delta| delta.domain == CounterDomain::Edac)
        .expect("edac delta");
    assert_eq!(edac.before, 5);
    assert_eq!(edac.after, 9);
    assert_eq!(edac.increment(), 4);
    let ib = deltas
        .deltas
        .iter()
        .find(|delta| delta.domain == CounterDomain::IbPort)
        .expect("ib delta");
    assert_eq!(ib.increment(), 0);
}

#[test]
fn diff_ignores_counters_present_on_only_one_side() {
    let before = CounterSnapshot {
        readings: vec![reading(CounterDomain::Edac, "mc0", "ce_count", 1)],
    };
    let after = CounterSnapshot {
        readings: vec![
            reading(CounterDomain::Edac, "mc0", "ce_count", 1),
            reading(CounterDomain::Nvme, "nvme0", "media_errors", 0),
        ],
    };
    let deltas = counters::diff_snapshots(&before, &after);
    assert_eq!(deltas.deltas.len(), 1);
    assert_eq!(deltas.deltas[0].domain, CounterDomain::Edac);
}

#[test]
fn diff_records_counter_resets_as_negative_increments() {
    let before = CounterSnapshot {
        readings: vec![reading(CounterDomain::GpuXid, "dmesg", "xid_lines", 40)],
    };
    let after = CounterSnapshot {
        readings: vec![reading(CounterDomain::GpuXid, "dmesg", "xid_lines", 2)],
    };
    let deltas = counters::diff_snapshots(&before, &after);
    assert_eq!(deltas.deltas[0].increment(), -38);
}

#[test]
fn diff_output_is_sorted_and_deterministic() {
    let before = CounterSnapshot {
        readings: vec![
            reading(CounterDomain::Nvme, "nvme0", "media_errors", 0),
            reading(CounterDomain::Edac, "mc0", "ce_count", 0),
        ],
    };
    let mut after = before.clone();
    after.readings.reverse();
    let deltas = counters::diff_snapshots(&before, &after);
    let keys: Vec<(CounterDomain, &str, &str)> = deltas
        .deltas
        .iter()
        .map(|d| (d.domain, d.device.as_str(), d.counter.as_str()))
        .collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted);
}

// ---------------------------------------------------------------------------
// Local collection on this (exotic-hardware-free) machine
// ---------------------------------------------------------------------------

#[test]
fn collect_snapshot_never_errors_without_exotic_hardware() {
    // The dev machine has no IB or datacenter GPUs; collection must still
    // produce a snapshot (possibly with only PCIe/EDAC/NVMe entries, or
    // nothing at all) rather than erroring.
    let snapshot = counters::collect_snapshot();
    let mut keys: Vec<(CounterDomain, &String, &String)> = snapshot
        .readings
        .iter()
        .map(|r| (r.domain, &r.device, &r.counter))
        .collect();
    let len_before = keys.len();
    keys.dedup();
    assert_eq!(keys.len(), len_before, "no duplicate counter identities");
}

// ---------------------------------------------------------------------------
// Agent event emission
// ---------------------------------------------------------------------------

#[test]
fn baseline_request_emits_one_snapshot_event() {
    let (sink, buf) = common::capturing_sink();
    counters::run(&sink, &CounterRequest::Baseline);
    let events = common::decode_events(&buf);
    let count = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::CounterBaseline { .. }))
        .count();
    assert_eq!(count, 1);
}

#[test]
fn delta_request_emits_one_deltas_event() {
    let (sink, buf) = common::capturing_sink();
    counters::run(
        &sink,
        &CounterRequest::Delta {
            baseline: CounterSnapshot::default(),
        },
    );
    let events = common::decode_events(&buf);
    let count = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::CounterDeltas { .. }))
        .count();
    assert_eq!(count, 1);
}

#[test]
fn counter_events_round_trip() {
    let events = vec![
        AgentEvent::CounterBaseline {
            snapshot: Box::new(CounterSnapshot {
                readings: vec![reading(CounterDomain::Edac, "mc0", "ce_count", 3)],
            }),
        },
        AgentEvent::CounterDeltas {
            deltas: Box::new(CounterDeltas {
                deltas: vec![CounterDelta {
                    domain: CounterDomain::IbPort,
                    device: "mlx5_0/1".into(),
                    counter: "symbol_error".into(),
                    before: 0,
                    after: 2,
                }],
            }),
        },
    ];
    for event in events {
        let line = encode_event(&event);
        assert!(!line.contains('\n'));
        assert_eq!(decode_event(&line).expect("round trip"), event);
    }
}

// ---------------------------------------------------------------------------
// Collector and report integration
// ---------------------------------------------------------------------------

fn delta(
    domain: CounterDomain,
    device: &str,
    counter: &str,
    before: u64,
    after: u64,
) -> CounterDelta {
    CounterDelta {
        domain,
        device: device.into(),
        counter: counter.into(),
        before,
        after,
    }
}

#[test]
fn collector_stores_counter_deltas_per_host() {
    let mut collector = Collector::new();
    collector.ingest(
        "n1",
        AgentEvent::CounterDeltas {
            deltas: Box::new(CounterDeltas {
                deltas: vec![delta(CounterDomain::Edac, "mc0", "ce_count", 1, 4)],
            }),
        },
    );
    let observations = collector.into_observations();
    let stored = observations["n1"]
        .counter_deltas
        .as_ref()
        .expect("deltas stored");
    assert_eq!(stored.deltas.len(), 1);
}

fn config_for(hosts: &str) -> gauntlet::config::FleetConfig {
    toml::from_str(&format!("hosts = {hosts}")).expect("config")
}

fn observations_with_deltas(
    deltas_by_host: &[(&str, Vec<CounterDelta>)],
) -> BTreeMap<String, HostObservations> {
    deltas_by_host
        .iter()
        .map(|(host, deltas)| {
            let obs = HostObservations {
                counter_deltas: Some(CounterDeltas {
                    deltas: deltas.clone(),
                }),
                ..Default::default()
            };
            (host.to_string(), obs)
        })
        .collect()
}

#[test]
fn nonzero_increments_become_findings_and_zero_stays_quiet() {
    let config = config_for(r#"["n1", "n2"]"#);
    let observations = observations_with_deltas(&[
        (
            "n1",
            vec![
                delta(CounterDomain::Edac, "mc0", "ce_count", 1, 4),
                delta(CounterDomain::IbPort, "mlx5_0/1", "symbol_error", 0, 0),
            ],
        ),
        (
            "n2",
            vec![delta(CounterDomain::Edac, "mc0", "ce_count", 2, 2)],
        ),
    ]);
    let results = report::build(&config, observations, 1, 2);

    let findings = &results.fleet.counter_findings;
    assert_eq!(findings.len(), 1, "only n1 has an increment: {findings:?}");
    let n1 = &findings["n1"];
    assert_eq!(n1.len(), 1);
    assert_eq!(n1[0].counter, "ce_count");
    assert_eq!(n1[0].device, "mc0");
    assert_eq!(n1[0].before, 1);
    assert_eq!(n1[0].after, 4);

    // The full delta list (zeros included) still lands in the JSON.
    assert_eq!(
        results.hosts["n1"]
            .counter_deltas
            .as_ref()
            .expect("deltas kept")
            .deltas
            .len(),
        2
    );
    assert_eq!(report::verdict(&results), Verdict::Stragglers);
}

#[test]
fn counter_resets_are_not_findings() {
    let config = config_for(r#"["n1"]"#);
    let observations = observations_with_deltas(&[(
        "n1",
        vec![delta(CounterDomain::GpuXid, "dmesg", "xid_lines", 40, 2)],
    )]);
    let results = report::build(&config, observations, 1, 2);
    assert!(results.fleet.counter_findings.is_empty());
    assert_eq!(report::verdict(&results), Verdict::Clean);
}

#[test]
fn counter_findings_render_in_the_table() {
    let config = config_for(r#"["n1"]"#);
    let observations = observations_with_deltas(&[(
        "n1",
        vec![
            delta(
                CounterDomain::PcieAer,
                "0000:65:00.0",
                "aer_dev_correctable.BadTLP",
                2,
                9,
            ),
            delta(CounterDomain::Edac, "mc0", "ue_count", 0, 0),
        ],
    )]);
    let results = report::build(&config, observations, 1, 2);
    let mut rendered = Vec::new();
    report::render_table(&results, &mut rendered).expect("render");
    let text = String::from_utf8(rendered).expect("utf-8 table");
    assert!(text.contains("error-counter deltas"), "{text}");
    assert!(text.contains("aer_dev_correctable.BadTLP"), "{text}");
    assert!(text.contains("0000:65:00.0"), "{text}");
    assert!(!text.contains("ue_count"), "zero deltas stay quiet: {text}");
}

#[test]
fn clean_runs_omit_the_counter_section() {
    let config = config_for(r#"["n1"]"#);
    let observations = observations_with_deltas(&[(
        "n1",
        vec![delta(CounterDomain::Edac, "mc0", "ce_count", 3, 3)],
    )]);
    let results = report::build(&config, observations, 1, 2);
    let mut rendered = Vec::new();
    report::render_table(&results, &mut rendered).expect("render");
    let text = String::from_utf8(rendered).expect("utf-8 table");
    assert!(!text.contains("error-counter deltas"), "{text}");
    assert_eq!(report::verdict(&results), Verdict::Clean);
}

#[test]
fn old_results_json_without_counter_fields_still_decodes() {
    let json = r#"{
        "schema_version": 2,
        "run_id": "1700000000-abcdef",
        "started_epoch_secs": 1700000000,
        "finished_epoch_secs": 1700000100,
        "hosts": {
            "n1": {
                "inventory": null,
                "metrics": [],
                "outcomes": [],
                "errors": []
            }
        },
        "fleet": {
            "outliers": {},
            "threshold_violations": {},
            "consistency": {},
            "failed_hosts": {}
        },
        "calibration": { "rooflines": {}, "links": {} }
    }"#;
    let results: report::RunResults = serde_json::from_str(json).expect("old schema decodes");
    assert!(results.hosts["n1"].counter_deltas.is_none());
    assert!(results.fleet.counter_findings.is_empty());
}
