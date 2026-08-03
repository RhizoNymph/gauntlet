//! Phase 1b: DRAM bandwidth, STREAM-triad style, NUMA-aware.
//!
//! For each NUMA node: pin the measuring threads to that node's cores,
//! touch-allocate the buffers there (first-touch policy), then time
//! a[i] = b[i] + s * c[i] over buffers much larger than LLC. Report best-of
//! iters as `mem_bandwidth.triad` in GiB/s under `Scope::Numa`, plus an
//! all-nodes-parallel aggregate under `Scope::Node`. A degraded DIMM/channel
//! shows up as one NUMA node lagging its peers fleet-wide.

use std::hint::black_box;
use std::sync::Barrier;
use std::time::Instant;

use anyhow::Result;
use core_affinity::CoreId;

use crate::agent::EventSink;
use crate::proto::{LogLevel, MemTaskSpec, MetricRecord, Scope, TestId, TestOutcome, Unit};

const GIB: f64 = (1u64 << 30) as f64;
const SCALAR: f64 = 3.0;
/// Smallest per-thread array worth timing; below this the clock read and the
/// page-walk warm-up dominate.
const MIN_BYTES_PER_THREAD: usize = 1 << 20;

/// Single-shot triad over `bytes` per array on the current thread's NUMA
/// placement; returns GiB/s. Public for tests (plausibility bounds).
pub fn triad_gib_per_sec(bytes: usize, iters: u32) -> Result<f64> {
    let mut buffers = TriadBuffers::first_touch(bytes);
    Ok(buffers.best_of(iters, None))
}

pub fn run(sink: &EventSink, spec: &MemTaskSpec) -> Result<()> {
    let nodes = numa_topology();
    sink.log(
        LogLevel::Debug,
        format!(
            "mem triad topology: {} numa node(s), cpus {:?}",
            nodes.len(),
            nodes
                .iter()
                .map(|node| (node.id, node.cpus.len()))
                .collect::<Vec<_>>()
        ),
    );

    for node in &nodes {
        let plan = node.plan(spec.buffer_bytes_per_numa);
        let gib_per_sec = parallel_triad(&plan, spec.iters);
        if gib_per_sec.is_finite() && gib_per_sec > 0.0 {
            sink.metric(MetricRecord {
                test: TestId::MemBandwidth,
                scope: Scope::Numa { node: node.id },
                name: "triad".to_string(),
                value: gib_per_sec,
                unit: Unit::GibPerSec,
                repeat: 0,
            });
        } else {
            sink.outcome(
                TestId::MemBandwidth,
                Scope::Numa { node: node.id },
                TestOutcome::Failed {
                    reason: format!("triad produced {gib_per_sec} GiB/s"),
                },
            );
        }
    }

    // Every node hammering its own DRAM simultaneously: this is what a
    // training step actually does, and it is where a shared-fabric or
    // interleaving misconfiguration shows up.
    let all: Vec<Assignment> = nodes
        .iter()
        .flat_map(|node| node.plan(spec.buffer_bytes_per_numa))
        .collect();
    let allnode = parallel_triad(&all, spec.iters);
    if allnode.is_finite() && allnode > 0.0 {
        sink.metric(MetricRecord {
            test: TestId::MemBandwidth,
            scope: Scope::Node,
            name: "triad_allnode".to_string(),
            value: allnode,
            unit: Unit::GibPerSec,
            repeat: 0,
        });
    } else {
        sink.outcome(
            TestId::MemBandwidth,
            Scope::Node,
            TestOutcome::Failed {
                reason: format!("all-node triad produced {allnode} GiB/s"),
            },
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// The three STREAM triad arrays, allocated and written by the calling
/// thread so Linux's first-touch policy places the pages on that thread's
/// NUMA node.
struct TriadBuffers {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
}

impl TriadBuffers {
    fn first_touch(bytes: usize) -> Self {
        let len = (bytes / size_of::<f64>()).max(1);
        // Non-zero fills: an all-zero `vec!` can be served by pre-zeroed
        // pages, which would leave the memory untouched and unplaced.
        Self {
            a: vec![0.5; len],
            b: vec![1.5; len],
            c: vec![2.5; len],
        }
    }

    fn len(&self) -> usize {
        self.a.len()
    }

    /// Bytes of DRAM traffic per pass. STREAM's convention: two reads (b, c)
    /// plus one write (a) — the write-allocate read is deliberately not
    /// counted, so numbers stay comparable with published STREAM results.
    fn traffic_bytes(&self) -> f64 {
        3.0 * (self.len() * size_of::<f64>()) as f64
    }

    fn pass(&mut self) {
        for ((a, b), c) in self.a.iter_mut().zip(self.b.iter()).zip(self.c.iter()) {
            *a = c.mul_add(SCALAR, *b);
        }
    }

    /// Best of `iters` timed passes. Best-of, not mean: the goal is the
    /// machine's capability, and every perturbation is one-directional.
    fn best_of(&mut self, iters: u32, barrier: Option<&Barrier>) -> f64 {
        let traffic = self.traffic_bytes();
        let mut best = 0.0f64;
        for _ in 0..iters.max(1) {
            // Line the threads up so the timed windows overlap; otherwise a
            // straggler measures an idle memory controller.
            if let Some(barrier) = barrier {
                barrier.wait();
            }
            let start = Instant::now();
            self.pass();
            let seconds = start.elapsed().as_secs_f64();
            black_box(&self.a);
            if seconds > 0.0 {
                best = best.max(traffic / seconds / GIB);
            }
        }
        best
    }
}

/// One measuring thread: which CPU to pin to, and how much memory it owns.
#[derive(Debug, Clone, Copy)]
struct Assignment {
    cpu: usize,
    bytes: usize,
}

/// Run one thread per assignment, all timing the same windows, and sum their
/// rates. A single core cannot saturate a modern memory controller, so the
/// per-node number is only meaningful with the whole node pushing.
fn parallel_triad(plan: &[Assignment], iters: u32) -> f64 {
    if plan.is_empty() {
        return 0.0;
    }
    let barrier = Barrier::new(plan.len());
    std::thread::scope(|scope| {
        let handles: Vec<_> = plan
            .iter()
            .map(|assignment| {
                let barrier = &barrier;
                let assignment = *assignment;
                scope.spawn(move || {
                    // Pinning is what makes first-touch mean anything; an
                    // unpinned thread still measures, just less precisely.
                    core_affinity::set_for_current(CoreId { id: assignment.cpu });
                    let mut buffers = TriadBuffers::first_touch(assignment.bytes);
                    buffers.best_of(iters, Some(barrier))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap_or(0.0))
            .sum()
    })
}

// ---------------------------------------------------------------------------
// NUMA topology
// ---------------------------------------------------------------------------

struct NumaNode {
    id: u32,
    cpus: Vec<usize>,
}

impl NumaNode {
    fn plan(&self, buffer_bytes_per_numa: u64) -> Vec<Assignment> {
        let threads = self.cpus.len().max(1);
        let bytes = (buffer_bytes_per_numa as usize / threads).max(MIN_BYTES_PER_THREAD);
        self.cpus
            .iter()
            .map(|&cpu| Assignment { cpu, bytes })
            .collect()
    }
}

/// NUMA nodes with their usable CPUs, from /sys/devices/system/node.
///
/// CPUs are intersected with this process's affinity mask: inside a
/// restricted cpuset the sysfs cpulist still names CPUs we may not run on,
/// and pinning to those silently fails. A machine with no /sys NUMA view (or
/// with every node masked out) degrades to a single node 0.
fn numa_topology() -> Vec<NumaNode> {
    let allowed: Vec<usize> = core_affinity::get_core_ids()
        .unwrap_or_default()
        .into_iter()
        .map(|core| core.id)
        .collect();

    let mut nodes: Vec<NumaNode> = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(id) = name
                .strip_prefix("node")
                .filter(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
                .and_then(|rest| rest.parse::<u32>().ok())
            else {
                continue;
            };
            let Ok(cpulist) = std::fs::read_to_string(entry.path().join("cpulist")) else {
                continue;
            };
            let cpus: Vec<usize> = parse_cpulist(&cpulist)
                .into_iter()
                .filter(|cpu| allowed.is_empty() || allowed.contains(cpu))
                .collect();
            if !cpus.is_empty() {
                nodes.push(NumaNode { id, cpus });
            }
        }
    }
    nodes.sort_by_key(|node| node.id);

    if nodes.is_empty() {
        let cpus = if allowed.is_empty() { vec![0] } else { allowed };
        nodes.push(NumaNode { id: 0, cpus });
    }
    nodes
}

/// "0-7,16-23" / "3" / "" -> the CPU numbers, in ascending order.
fn parse_cpulist(raw: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for group in raw.trim().split(',').filter(|group| !group.is_empty()) {
        match group.split_once('-') {
            Some((start, end)) => {
                let (Ok(start), Ok(end)) = (start.trim().parse(), end.trim().parse::<usize>())
                else {
                    continue;
                };
                cpus.extend(start..=end);
            }
            None => {
                if let Ok(cpu) = group.trim().parse() {
                    cpus.push(cpu);
                }
            }
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    cpus
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpulists() {
        assert_eq!(parse_cpulist("0-3\n"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpulist("0-1,4-5"), vec![0, 1, 4, 5]);
        assert_eq!(parse_cpulist("7"), vec![7]);
        assert!(parse_cpulist("\n").is_empty());
        assert!(parse_cpulist("garbage").is_empty());
    }

    #[test]
    fn topology_always_has_a_node() {
        let nodes = numa_topology();
        assert!(!nodes.is_empty());
        assert!(nodes.iter().all(|node| !node.cpus.is_empty()));
    }

    #[test]
    fn plan_respects_the_byte_floor() {
        let node = NumaNode {
            id: 0,
            cpus: vec![0, 1, 2, 3],
        };
        let plan = node.plan(64 << 20);
        assert_eq!(plan.len(), 4);
        assert_eq!(plan[0].bytes, 16 << 20);
        // A tiny request still gets a measurable buffer.
        assert_eq!(node.plan(1024)[0].bytes, MIN_BYTES_PER_THREAD);
    }

    #[test]
    fn triad_traffic_counts_three_arrays() {
        let buffers = TriadBuffers::first_touch(8 * 100);
        assert_eq!(buffers.len(), 100);
        assert_eq!(buffers.traffic_bytes(), 3.0 * 800.0);
    }
}
