//! Intra-node GPU<->GPU: for every ordered pair with p2p access enabled,
//! cudaMemcpyPeer bandwidth (large transfers) and small-transfer latency.
//! Metrics under `Scope::GpuPair`: `gpu_p2p.bandwidth` GiB/s and
//! `gpu_p2p.latency` micros. Pairs without p2p access emit a Skipped
//! outcome (expected on PCIe-only boxes), not a failure.

use anyhow::Result;

use crate::agent::EventSink;

pub fn run_all_pairs(sink: &EventSink, transfer_bytes: u64) -> Result<()> {
    let _ = (sink, transfer_bytes);
    todo!("agent C: implement")
}
