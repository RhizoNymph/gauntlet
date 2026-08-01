//! Phase 0/1 agent tests that run for real on the build machine.

mod common;

use std::collections::BTreeSet;

use gauntlet::agent::{cpu, disk, inventory, mem};
use gauntlet::proto::{AgentEvent, CpuTaskSpec, DiskTaskSpec, MemTaskSpec, Scope, TestId, Unit};

// ---------------------------------------------------------------------------
// inventory
// ---------------------------------------------------------------------------

#[test]
fn inventory_reflects_this_machine() {
    let snapshot = inventory::collect().expect("inventory on a healthy machine");
    assert!(!snapshot.hostname.is_empty());
    assert!(!snapshot.kernel.is_empty());
    assert!(!snapshot.cpu_model.is_empty());
    assert!(snapshot.logical_cores >= 1);
    assert!(snapshot.numa_nodes >= 1);
    assert!(
        snapshot.mem_total_bytes > 1 << 28,
        "at least 256 MiB of RAM"
    );
    // NICs: loopback excluded, but the machine reaches the network somehow.
    assert!(snapshot.nics.iter().all(|nic| nic.name != "lo"));
    for nic in &snapshot.nics {
        assert!(nic.mtu >= 68, "impossible MTU {} on {}", nic.mtu, nic.name);
    }
}

#[test]
fn inventory_emits_exactly_one_event() {
    let (sink, buf) = common::capturing_sink();
    inventory::run(&sink).expect("inventory run");
    let events = common::decode_events(&buf);
    let count = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::Inventory { .. }))
        .count();
    assert_eq!(count, 1);
}

// ---------------------------------------------------------------------------
// cpu
// ---------------------------------------------------------------------------

#[test]
fn checksum_workloads_are_deterministic() {
    assert_eq!(
        cpu::checksum_round(42, 10_000),
        cpu::checksum_round(42, 10_000)
    );
    assert_ne!(
        cpu::checksum_round(42, 10_000),
        cpu::checksum_round(43, 10_000)
    );
    assert_ne!(
        cpu::checksum_round(42, 10_000),
        cpu::checksum_round(42, 10_001)
    );

    assert_eq!(
        cpu::float_checksum_round(7, 10_000),
        cpu::float_checksum_round(7, 10_000)
    );
    assert_ne!(
        cpu::float_checksum_round(7, 10_000),
        cpu::float_checksum_round(8, 10_000)
    );
    assert_ne!(cpu::float_checksum_round(7, 10_000), 0);
}

#[test]
fn cpu_run_covers_every_core() {
    let (sink, buf) = common::capturing_sink();
    let spec = CpuTaskSpec {
        correctness_secs_per_core: 1,
        gflops_secs: 1,
    };
    cpu::run(&sink, &spec).expect("cpu phase");
    let events = common::decode_events(&buf);

    let cores = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .expect("parallelism");

    let correctness_cores: BTreeSet<u32> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::Outcome {
                test: TestId::CpuCorrectness,
                scope: Scope::Core { id },
                ..
            } => Some(*id),
            _ => None,
        })
        .collect();
    assert_eq!(
        correctness_cores.len() as u32,
        cores,
        "one correctness outcome per logical core"
    );

    let gflops: Vec<f64> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::Metric { record }
                if record.test == TestId::CpuGflops
                    && matches!(record.scope, Scope::Core { .. }) =>
            {
                assert_eq!(record.unit, Unit::Gflops);
                Some(record.value)
            }
            _ => None,
        })
        .collect();
    assert_eq!(gflops.len() as u32, cores, "one gflops metric per core");
    for value in &gflops {
        assert!(
            value.is_finite() && *value > 0.0,
            "implausible gflops {value}"
        );
    }

    let allcore = events.iter().any(|event| {
        matches!(event, AgentEvent::Metric { record }
            if record.test == TestId::CpuGflops
                && record.scope == Scope::Node
                && record.name == "gflops_allcore")
    });
    assert!(allcore, "all-core aggregate metric missing");
}

// ---------------------------------------------------------------------------
// mem
// ---------------------------------------------------------------------------

#[test]
fn triad_reports_plausible_bandwidth() {
    // 64 MiB buffers, 3 iters: fast but bigger than any LLC prefetch fantasy.
    let gib_per_sec = mem::triad_gib_per_sec(64 << 20, 3).expect("triad");
    assert!(
        gib_per_sec.is_finite() && gib_per_sec > 0.5 && gib_per_sec < 10_000.0,
        "implausible DRAM bandwidth: {gib_per_sec} GiB/s"
    );
}

#[test]
fn mem_run_reports_every_numa_node() {
    let (sink, buf) = common::capturing_sink();
    let spec = MemTaskSpec {
        buffer_bytes_per_numa: 64 << 20,
        iters: 3,
    };
    mem::run(&sink, &spec).expect("mem phase");
    let events = common::decode_events(&buf);
    let numa_metrics: BTreeSet<u32> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::Metric { record }
                if record.test == TestId::MemBandwidth && record.name == "triad" =>
            {
                match record.scope {
                    Scope::Numa { node } => Some(node),
                    _ => None,
                }
            }
            _ => None,
        })
        .collect();
    assert!(!numa_metrics.is_empty(), "at least NUMA node 0 measured");
}

// ---------------------------------------------------------------------------
// disk
// ---------------------------------------------------------------------------

#[test]
fn disk_measure_cleans_up_and_reports() {
    let dir = common::scratch_dir("disk");
    let before: BTreeSet<_> = std::fs::read_dir(&dir)
        .expect("readdir")
        .map(|e| e.expect("entry").file_name())
        .collect();

    let throughput =
        disk::measure(dir.to_str().expect("utf8 path"), 32 << 20).expect("disk measure");
    assert!(throughput.write_gib_per_sec.is_finite() && throughput.write_gib_per_sec > 0.0);
    assert!(throughput.read_gib_per_sec.is_finite() && throughput.read_gib_per_sec > 0.0);

    let after: BTreeSet<_> = std::fs::read_dir(&dir)
        .expect("readdir")
        .map(|e| e.expect("entry").file_name())
        .collect();
    assert_eq!(before, after, "temp files must be removed");
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn disk_run_scopes_by_path() {
    let dir = common::scratch_dir("disk-run");
    let (sink, buf) = common::capturing_sink();
    let spec = DiskTaskSpec {
        paths: vec![dir.to_str().expect("utf8").to_string()],
        file_bytes: 16 << 20,
    };
    disk::run(&sink, &spec).expect("disk phase");
    let events = common::decode_events(&buf);
    let mut saw_read = false;
    let mut saw_write = false;
    for event in &events {
        if let AgentEvent::Metric { record } = event
            && record.test == TestId::DiskIo
        {
            assert!(matches!(&record.scope, Scope::Disk { path } if !path.is_empty()));
            assert_eq!(record.unit, Unit::GibPerSec);
            saw_read |= record.name == "seq_read";
            saw_write |= record.name == "seq_write";
        }
    }
    assert!(saw_read && saw_write, "both directions reported");
    std::fs::remove_dir_all(&dir).expect("cleanup");
}
