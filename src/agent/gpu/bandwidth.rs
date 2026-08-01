//! GPU memory bandwidth per GPU: device-to-device copy (HBM bandwidth) plus
//! host-to-device and device-to-host over pinned host memory (PCIe reality —
//! this is the test that exposes a downtrained x16->x4 link long before
//! anyone reads lspci). Metrics under `Scope::Gpu`:
//! `gpu_mem_bandwidth.d2d` / `.h2d_pinned` / `.d2h_pinned` in GiB/s,
//! best-of-N transfers of `bandwidth_bytes`.

use std::time::Instant;

use anyhow::{Context, Result};
use cudarc::driver::CudaContext;

use crate::agent::EventSink;
use crate::proto::{MetricRecord, Scope, TestId, TestOutcome, Unit};

/// Timed transfers per direction; the best one is reported, so a stray
/// scheduler hiccup does not look like a downtrained link.
const TRANSFERS: usize = 5;
/// Floor on the transfer size: a few bytes measures launch latency, not
/// bandwidth.
const MIN_BYTES: u64 = 1 << 20;
const GIB: f64 = (1_u64 << 30) as f64;

/// A device-to-device copy both reads and writes `bytes`, so its bandwidth is
/// `2 * bytes / elapsed` (the convention NVIDIA's `bandwidthTest` uses, which
/// makes the number directly comparable to the HBM figure on the spec sheet).
/// Host transfers move each byte once.
const D2D_BYTES_PER_TRANSFER: f64 = 2.0;
const HOST_BYTES_PER_TRANSFER: f64 = 1.0;

pub fn run_on_gpu(sink: &EventSink, gpu_index: u32, bandwidth_bytes: u64) -> Result<()> {
    let scope = Scope::Gpu { index: gpu_index };
    let bytes = bandwidth_bytes.max(MIN_BYTES);
    let elements = (bytes / std::mem::size_of::<f32>() as u64).max(1) as usize;
    let bytes = (elements * std::mem::size_of::<f32>()) as f64;

    let ctx = CudaContext::new(gpu_index as usize)
        .with_context(|| format!("creating cuda context for gpu {gpu_index}"))?;
    let stream = ctx.default_stream();

    let source = stream.alloc_zeros::<f32>(elements)?;
    let mut sink_buffer = stream.alloc_zeros::<f32>(elements)?;

    // SAFETY: `alloc_pinned` hands back page-locked memory with undefined
    // contents; `as_mut_slice` is the only way to reach it and every later
    // read of the buffer (by us or by the driver's DMA engine) happens after
    // this initializing write.
    let mut pinned =
        unsafe { ctx.alloc_pinned::<f32>(elements) }.context("allocating pinned host memory")?;
    pinned
        .as_mut_slice()
        .context("mapping pinned host memory")?
        .fill(0.0);

    let d2d = best_gib_per_sec(bytes, D2D_BYTES_PER_TRANSFER, || {
        stream.memcpy_dtod(&source, &mut sink_buffer)?;
        stream.synchronize()?;
        Ok(())
    })
    .context("device-to-device copy")?;

    let h2d = best_gib_per_sec(bytes, HOST_BYTES_PER_TRANSFER, || {
        stream.memcpy_htod(&pinned, &mut sink_buffer)?;
        stream.synchronize()?;
        Ok(())
    })
    .context("pinned host-to-device copy")?;

    let d2h = best_gib_per_sec(bytes, HOST_BYTES_PER_TRANSFER, || {
        stream.memcpy_dtoh(&source, &mut pinned)?;
        stream.synchronize()?;
        Ok(())
    })
    .context("pinned device-to-host copy")?;

    for (name, value) in [("d2d", d2d), ("h2d_pinned", h2d), ("d2h_pinned", d2h)] {
        sink.metric(MetricRecord {
            test: TestId::GpuMemBandwidth,
            scope: scope.clone(),
            name: name.to_string(),
            value,
            unit: Unit::GibPerSec,
        });
    }
    sink.outcome(TestId::GpuMemBandwidth, scope, TestOutcome::Passed);
    Ok(())
}

/// One warmup transfer (first-touch page mapping, context ramp) then the best
/// of `TRANSFERS` timed ones, in GiB/s.
fn best_gib_per_sec(
    bytes: f64,
    bytes_per_transfer: f64,
    mut transfer: impl FnMut() -> Result<()>,
) -> Result<f64> {
    transfer()?;
    let mut best = 0.0_f64;
    for _ in 0..TRANSFERS {
        let start = Instant::now();
        transfer()?;
        let elapsed = start.elapsed().as_secs_f64().max(1e-12);
        let rate = gib_per_sec(bytes * bytes_per_transfer, elapsed);
        if rate > best {
            best = rate;
        }
    }
    Ok(best)
}

fn gib_per_sec(bytes: f64, elapsed_secs: f64) -> f64 {
    bytes / elapsed_secs / GIB
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gib_per_sec_converts_binary_gigabytes() {
        // One GiB in one second.
        assert!((gib_per_sec(GIB, 1.0) - 1.0).abs() < 1e-12);
        // 256 MiB in 10 ms => 25 GiB/s.
        assert!((gib_per_sec((256 << 20) as f64, 0.01) - 25.0).abs() < 1e-9);
    }

    #[test]
    fn d2d_counts_the_read_and_the_write() {
        // The same wall time over the same buffer reports twice the rate for a
        // device-to-device copy, which touches each byte twice.
        let host = gib_per_sec(GIB * HOST_BYTES_PER_TRANSFER, 1.0);
        let device = gib_per_sec(GIB * D2D_BYTES_PER_TRANSFER, 1.0);
        assert!((device - 2.0 * host).abs() < 1e-12);
    }

    #[test]
    fn best_of_n_takes_the_fastest_transfer() {
        let mut call = 0_usize;
        let best = best_gib_per_sec(GIB, 1.0, || {
            call += 1;
            // The warmup plus one slow transfer; the rest are immediate.
            if call <= 2 {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Ok(())
        })
        .expect("transfers succeed");
        // The fast transfers take microseconds, so the best rate must be far
        // above the 1 GiB / 20 ms = 50 GiB/s the slow one would report.
        assert!(best > 100.0, "best {best} looks like the slow transfer");
        assert_eq!(call, TRANSFERS + 1, "one warmup plus {TRANSFERS} timed");
    }

    #[test]
    fn transfer_errors_propagate() {
        let result = best_gib_per_sec(GIB, 1.0, || anyhow::bail!("copy failed"));
        assert!(result.is_err());
    }
}
