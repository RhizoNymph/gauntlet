//! Error-counter snapshots and deltas across the load phases.
//!
//! Marginal hardware often passes throughput tests while silently
//! accumulating errors; the signal is any counter increment between a
//! snapshot taken before the load phases (baseline) and one taken after.
//! Collection is strictly best effort — a node without IB, NVMe, GPUs or
//! EDAC simply contributes no readings for that domain, never an error —
//! and every parser takes injectable paths/text so tests run on machines
//! with none of the hardware.
//!
//! Reuses the phase-0 probe machinery (`inventory::run_capture`,
//! `inventory::csv_field`, `inventory::parse_xid_line`) rather than
//! duplicating it.

use std::collections::BTreeMap;
use std::path::Path;

use crate::agent::EventSink;
use crate::agent::inventory::{PROBE_TIMEOUT, csv_field, parse_xid_line, run_capture};
use crate::proto::{
    AgentEvent, CounterDelta, CounterDeltas, CounterDomain, CounterReading, CounterRequest,
    CounterSnapshot,
};

// Real sysfs roots. Tests point the collectors at fixture trees instead.
const PCI_DEVICES_ROOT: &str = "/sys/bus/pci/devices";
const EDAC_MC_ROOT: &str = "/sys/devices/system/edac/mc";
const INFINIBAND_ROOT: &str = "/sys/class/infiniband";
const NVME_CLASS_ROOT: &str = "/sys/class/nvme";

/// Per-device AER tally files exposed by the kernel when the device has the
/// AER capability. Each holds "<name> <count>" lines.
const AER_FILES: [&str; 3] = ["aer_dev_correctable", "aer_dev_nonfatal", "aer_dev_fatal"];

/// Memory-controller-level EDAC counters.
const EDAC_MC_FILES: [&str; 4] = ["ce_count", "ue_count", "ce_noinfo_count", "ue_noinfo_count"];
/// Per-DIMM EDAC counters.
const EDAC_DIMM_FILES: [&str; 2] = ["dimm_ce_count", "dimm_ue_count"];

/// The IB port error/back-pressure counter set worth diffing. `link_downed`
/// rides along because a flap under load is exactly the kind of silent
/// failure this pass exists to catch.
const IB_COUNTER_FILES: [&str; 6] = [
    "symbol_error",
    "link_error_recovery",
    "port_rcv_errors",
    "port_xmit_discards",
    "port_xmit_wait",
    "link_downed",
];

/// ECC/row-remap counters queried from `nvidia-smi` in one shot; the parse
/// follows the same CSV conventions as the phase-0 GPU inventory.
const GPU_ECC_QUERY: &str = concat!(
    "--query-gpu=index,",
    "ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,",
    "ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total,",
    "remapped_rows.correctable,remapped_rows.uncorrectable"
);
/// Counter names for the query above, in field order after `index`.
const GPU_ECC_COUNTERS: [&str; 6] = [
    "ecc_corrected_volatile",
    "ecc_uncorrected_volatile",
    "ecc_corrected_aggregate",
    "ecc_uncorrected_aggregate",
    "remapped_rows_correctable",
    "remapped_rows_uncorrectable",
];

/// Execute a counter pass: snapshot now, and either announce it as the
/// baseline or diff it against the baseline the orchestrator handed back.
/// Collection is infallible by design; this emits exactly one event.
pub fn run(sink: &EventSink, request: &CounterRequest) {
    let snapshot = collect_snapshot();
    match request {
        CounterRequest::Baseline => sink.emit(&AgentEvent::CounterBaseline {
            snapshot: Box::new(snapshot),
        }),
        CounterRequest::Delta { baseline } => sink.emit(&AgentEvent::CounterDeltas {
            deltas: Box::new(diff_snapshots(baseline, &snapshot)),
        }),
    }
}

/// Snapshot every available error counter on this node. Missing subsystems,
/// tools, or permissions contribute nothing; the result is sorted and
/// duplicate-free by (domain, device, counter).
pub fn collect_snapshot() -> CounterSnapshot {
    let mut readings = Vec::new();
    readings.extend(pcie_aer_counters(Path::new(PCI_DEVICES_ROOT)));
    readings.extend(edac_counters(Path::new(EDAC_MC_ROOT)));
    readings.extend(ib_port_counters(Path::new(INFINIBAND_ROOT)));
    readings.extend(nvme_counters());
    readings.extend(gpu_ecc_counters());
    readings.extend(nvlink_counters());
    if let Some(count) = probe_xid_line_count() {
        readings.push(CounterReading {
            domain: CounterDomain::GpuXid,
            device: "dmesg".into(),
            counter: "xid_lines".into(),
            value: count,
        });
    }
    readings.sort();
    readings
        .dedup_by(|a, b| (a.domain, &a.device, &a.counter) == (b.domain, &b.device, &b.counter));
    CounterSnapshot { readings }
}

/// Diff two snapshots by counter identity. Counters present on only one
/// side are dropped: a device that appeared or vanished mid-run is a
/// presence question for the inventory, not a delta. Output is sorted.
pub fn diff_snapshots(before: &CounterSnapshot, after: &CounterSnapshot) -> CounterDeltas {
    let earlier: BTreeMap<(CounterDomain, &str, &str), u64> = before
        .readings
        .iter()
        .map(|r| ((r.domain, r.device.as_str(), r.counter.as_str()), r.value))
        .collect();
    let mut deltas: Vec<CounterDelta> = after
        .readings
        .iter()
        .filter_map(|reading| {
            let key = (
                reading.domain,
                reading.device.as_str(),
                reading.counter.as_str(),
            );
            earlier.get(&key).map(|&value| CounterDelta {
                domain: reading.domain,
                device: reading.device.clone(),
                counter: reading.counter.clone(),
                before: value,
                after: reading.value,
            })
        })
        .collect();
    deltas
        .sort_by(|a, b| (a.domain, &a.device, &a.counter).cmp(&(b.domain, &b.device, &b.counter)));
    CounterDeltas { deltas }
}

// ---------------------------------------------------------------------------
// sysfs collectors (injectable roots)
// ---------------------------------------------------------------------------

/// PCIe AER tallies: for every device under `pci_root`, parse whichever of
/// the `aer_dev_*` files exist. Counter names are "<file>.<field>", e.g.
/// "aer_dev_correctable.BadTLP".
pub fn pcie_aer_counters(pci_root: &Path) -> Vec<CounterReading> {
    let Ok(devices) = std::fs::read_dir(pci_root) else {
        return Vec::new();
    };
    let mut readings = Vec::new();
    for device in devices.flatten() {
        let address = device.file_name().to_string_lossy().into_owned();
        for file in AER_FILES {
            let Some(text) = read_text(&device.path().join(file)) else {
                continue;
            };
            for line in text.lines() {
                let mut parts = line.split_whitespace();
                let (Some(name), Some(raw)) = (parts.next(), parts.next()) else {
                    continue;
                };
                let Ok(value) = raw.parse::<u64>() else {
                    continue;
                };
                readings.push(CounterReading {
                    domain: CounterDomain::PcieAer,
                    device: address.clone(),
                    counter: format!("{file}.{name}"),
                    value,
                });
            }
        }
    }
    readings
}

/// EDAC DRAM error counters: mc*/{ce,ue}_count (+noinfo variants) and any
/// per-DIMM dimm_{ce,ue}_count beneath them.
pub fn edac_counters(edac_root: &Path) -> Vec<CounterReading> {
    let Ok(controllers) = std::fs::read_dir(edac_root) else {
        return Vec::new();
    };
    let mut readings = Vec::new();
    for controller in controllers.flatten() {
        let name = controller.file_name().to_string_lossy().into_owned();
        // Only mc<N> entries are memory controllers.
        if !name
            .strip_prefix("mc")
            .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
        {
            continue;
        }
        for file in EDAC_MC_FILES {
            if let Some(value) = read_u64(&controller.path().join(file)) {
                readings.push(CounterReading {
                    domain: CounterDomain::Edac,
                    device: name.clone(),
                    counter: file.to_string(),
                    value,
                });
            }
        }
        let Ok(entries) = std::fs::read_dir(controller.path()) else {
            continue;
        };
        for entry in entries.flatten() {
            let child = entry.file_name().to_string_lossy().into_owned();
            if !child.starts_with("dimm")
                && !child.starts_with("rank")
                && !child.starts_with("csrow")
            {
                continue;
            }
            for file in EDAC_DIMM_FILES {
                if let Some(value) = read_u64(&entry.path().join(file)) {
                    readings.push(CounterReading {
                        domain: CounterDomain::Edac,
                        device: format!("{name}/{child}"),
                        counter: file.to_string(),
                        value,
                    });
                }
            }
        }
    }
    readings
}

/// InfiniBand per-port error counters from
/// `<ib_root>/<device>/ports/<port>/counters/`. Device is "<hca>/<port>".
pub fn ib_port_counters(ib_root: &Path) -> Vec<CounterReading> {
    let Ok(devices) = std::fs::read_dir(ib_root) else {
        return Vec::new();
    };
    let mut readings = Vec::new();
    for device in devices.flatten() {
        let hca = device.file_name().to_string_lossy().into_owned();
        let Ok(ports) = std::fs::read_dir(device.path().join("ports")) else {
            continue;
        };
        for port in ports.flatten() {
            let port_name = port.file_name().to_string_lossy().into_owned();
            if port_name.parse::<u32>().is_err() {
                continue;
            }
            let counters_dir = port.path().join("counters");
            for file in IB_COUNTER_FILES {
                if let Some(value) = read_u64(&counters_dir.join(file)) {
                    readings.push(CounterReading {
                        domain: CounterDomain::IbPort,
                        device: format!("{hca}/{port_name}"),
                        counter: file.to_string(),
                        value,
                    });
                }
            }
        }
    }
    readings
}

/// NVMe controller names ("nvme0", ...) under the class directory.
pub fn nvme_device_names(nvme_root: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(nvme_root) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("nvme"))
        .collect();
    names.sort();
    names
}

/// Parse the SMART health JSON for one controller. Accepts both shapes in
/// the field: `smartctl -j -A` nests the log under
/// `nvme_smart_health_information_log`; `nvme smart-log -o json` puts the
/// fields at the top level. Anything else degrades to empty.
pub fn parse_nvme_smart_json(device: &str, text: &str) -> Vec<CounterReading> {
    let Ok(doc) = serde_json::from_str::<serde_json::Value>(text) else {
        return Vec::new();
    };
    let log = doc.get("nvme_smart_health_information_log").unwrap_or(&doc);
    ["media_errors", "num_err_log_entries"]
        .into_iter()
        .filter_map(|counter| {
            log.get(counter)
                .and_then(|v| v.as_u64())
                .map(|value| CounterReading {
                    domain: CounterDomain::Nvme,
                    device: device.to_string(),
                    counter: counter.to_string(),
                    value,
                })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// tool-output parsers (injectable text)
// ---------------------------------------------------------------------------

/// Parse the `nvidia-smi --query-gpu=index,<ecc fields>` CSV. Per-field
/// "N/A" skips that reading only, mirroring the inventory's conventions.
pub fn parse_gpu_ecc_csv(text: &str) -> Vec<CounterReading> {
    let mut readings = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 1 + GPU_ECC_COUNTERS.len() {
            continue;
        }
        let Some(index) = csv_field(fields[0]).and_then(|raw| raw.parse::<u32>().ok()) else {
            continue;
        };
        for (offset, counter) in GPU_ECC_COUNTERS.iter().enumerate() {
            let Some(value) = csv_field(fields[1 + offset]).and_then(|raw| raw.parse::<u64>().ok())
            else {
                continue;
            };
            readings.push(CounterReading {
                domain: CounterDomain::GpuEcc,
                device: format!("gpu{index}"),
                counter: (*counter).to_string(),
                value,
            });
        }
    }
    readings
}

/// Parse `nvidia-smi nvlink -e` output:
///
/// ```text
/// GPU 0: NVIDIA H100 (UUID: GPU-...)
///          Link 0: Replay Errors: 0
///          Link 0: Recovery Errors: 0
///          Link 0: CRC Errors: 0
/// ```
///
/// Counter names vary across driver generations, so any
/// "Link N: <Name>: <count>" line is kept with the name normalized to
/// lowercase underscores. Device is "gpu<G>/link<L>".
pub fn parse_nvlink_errors(text: &str) -> Vec<CounterReading> {
    let mut readings = Vec::new();
    let mut gpu: Option<u32> = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("GPU ") {
            gpu = rest
                .split([':', ' '])
                .next()
                .and_then(|raw| raw.parse().ok());
            continue;
        }
        let Some(gpu) = gpu else { continue };
        let Some(rest) = trimmed.strip_prefix("Link ") else {
            continue;
        };
        let Some((link_raw, rest)) = rest.split_once(':') else {
            continue;
        };
        let Ok(link) = link_raw.trim().parse::<u32>() else {
            continue;
        };
        let Some((name, value_raw)) = rest.rsplit_once(':') else {
            continue;
        };
        let Ok(value) = value_raw.trim().parse::<u64>() else {
            continue;
        };
        let counter = name.trim().to_ascii_lowercase().replace([' ', '-'], "_");
        if counter.is_empty() {
            continue;
        }
        readings.push(CounterReading {
            domain: CounterDomain::Nvlink,
            device: format!("gpu{gpu}/link{link}"),
            counter,
            value,
        });
    }
    readings
}

/// Total count of kernel-log Xid lines (not deduplicated: two occurrences
/// of the same code are two events, and the delta is what matters).
pub fn count_xid_lines(text: &str) -> u64 {
    text.lines()
        .filter(|line| parse_xid_line(line).is_some())
        .count() as u64
}

// ---------------------------------------------------------------------------
// live probes (real machine only; everything above stays testable)
// ---------------------------------------------------------------------------

fn nvme_counters() -> Vec<CounterReading> {
    let mut readings = Vec::new();
    for name in nvme_device_names(Path::new(NVME_CLASS_ROOT)) {
        let node = format!("/dev/{name}");
        // smartctl first (present on most fleets), nvme-cli as fallback;
        // both speak JSON. Neither present -> no NVMe readings.
        let text = run_capture("smartctl", &["-j", "-A", &node], PROBE_TIMEOUT)
            .or_else(|| run_capture("nvme", &["smart-log", "-o", "json", &node], PROBE_TIMEOUT));
        if let Some(text) = text {
            readings.extend(parse_nvme_smart_json(&name, &text));
        }
    }
    readings
}

fn gpu_ecc_counters() -> Vec<CounterReading> {
    run_capture(
        "nvidia-smi",
        &[GPU_ECC_QUERY, "--format=csv,noheader,nounits"],
        PROBE_TIMEOUT,
    )
    .map(|text| parse_gpu_ecc_csv(&text))
    .unwrap_or_default()
}

fn nvlink_counters() -> Vec<CounterReading> {
    run_capture("nvidia-smi", &["nvlink", "-e"], PROBE_TIMEOUT)
        .map(|text| parse_nvlink_errors(&text))
        .unwrap_or_default()
}

/// Same sources as the inventory Xid scan, but counting lines rather than
/// deduplicating codes. `None` when no kernel log is readable at all, so a
/// permission problem cannot masquerade as "zero Xids" on one side of the
/// diff.
fn probe_xid_line_count() -> Option<u64> {
    let text = run_capture(
        "journalctl",
        &["-k", "--no-pager", "-o", "cat"],
        PROBE_TIMEOUT,
    )
    .or_else(|| std::fs::read_to_string("/var/log/kern.log").ok())?;
    Some(count_xid_lines(&text))
}

// ---------------------------------------------------------------------------
// small IO helpers
// ---------------------------------------------------------------------------

fn read_text(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn read_u64(path: &Path) -> Option<u64> {
    read_text(path)?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fixture-tree parsing and delta logic are covered by the integration
    // suite (tests/counter_tests.rs); these are quick sanity checks on the
    // pure text parsers.

    #[test]
    fn ecc_csv_and_nvlink_parsers_ignore_garbage() {
        assert!(parse_gpu_ecc_csv("nvidia-smi: command banner\n").is_empty());
        assert!(parse_nvlink_errors("Unable to determine NVLink status\n").is_empty());
    }

    #[test]
    fn xid_counting_reuses_the_inventory_matcher() {
        assert_eq!(
            count_xid_lines("NVRM: Xid (PCI:0000:65:00): 13, pid=1\nnope\n"),
            1
        );
    }

    #[test]
    fn diff_of_identical_snapshots_is_all_zero() {
        let snapshot = CounterSnapshot {
            readings: vec![CounterReading {
                domain: CounterDomain::Edac,
                device: "mc0".into(),
                counter: "ce_count".into(),
                value: 3,
            }],
        };
        let deltas = diff_snapshots(&snapshot, &snapshot);
        assert_eq!(deltas.deltas.len(), 1);
        assert_eq!(deltas.deltas[0].increment(), 0);
    }
}
