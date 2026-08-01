//! GPU memory bandwidth per GPU: device-to-device copy (HBM bandwidth) plus
//! host-to-device and device-to-host over pinned host memory (PCIe reality —
//! this is the test that exposes a downtrained x16->x4 link long before
//! anyone reads lspci). Metrics under `Scope::Gpu`:
//! `gpu_mem_bandwidth.d2d` / `.h2d_pinned` / `.d2h_pinned` in GiB/s,
//! best-of-N transfers of `bandwidth_bytes`.

use anyhow::Result;

use crate::agent::EventSink;

pub fn run_on_gpu(sink: &EventSink, gpu_index: u32, bandwidth_bytes: u64) -> Result<()> {
    let _ = (sink, gpu_index, bandwidth_bytes);
    todo!("agent C: implement")
}
