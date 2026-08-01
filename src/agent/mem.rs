//! Phase 1b: DRAM bandwidth, STREAM-triad style, NUMA-aware.
//!
//! For each NUMA node: pin the measuring threads to that node's cores,
//! touch-allocate the buffers there (first-touch policy), then time
//! a[i] = b[i] + s * c[i] over buffers much larger than LLC. Report best-of
//! iters as `mem_bandwidth.triad` in GiB/s under `Scope::Numa`, plus an
//! all-nodes-parallel aggregate under `Scope::Node`. A degraded DIMM/channel
//! shows up as one NUMA node lagging its peers fleet-wide.

use anyhow::Result;

use crate::agent::EventSink;
use crate::proto::MemTaskSpec;

/// Single-shot triad over `bytes` per array on the current thread's NUMA
/// placement; returns GiB/s. Public for tests (plausibility bounds).
pub fn triad_gib_per_sec(bytes: usize, iters: u32) -> Result<f64> {
    let _ = (bytes, iters);
    todo!("agent B: implement")
}

pub fn run(sink: &EventSink, spec: &MemTaskSpec) -> Result<()> {
    let _ = (sink, spec);
    todo!("agent B: implement")
}
