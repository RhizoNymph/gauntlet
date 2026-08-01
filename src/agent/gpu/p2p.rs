//! Intra-node GPU<->GPU: for every ordered pair with p2p access enabled,
//! cudaMemcpyPeer bandwidth (large transfers) and small-transfer latency.
//! Metrics under `Scope::GpuPair`: `gpu_p2p.bandwidth` GiB/s and
//! `gpu_p2p.latency` micros. Pairs without p2p access emit a Skipped
//! outcome (expected on PCIe-only boxes), not a failure.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, sys as driver_sys};

use crate::agent::EventSink;
use crate::agent::gpu::guard;
use crate::proto::{MetricRecord, Scope, TestId, TestOutcome, Unit};

/// Timed large copies per pair; the best one is reported.
const TRANSFERS: usize = 5;
/// Floor on the large-copy size.
const MIN_BYTES: u64 = 1 << 20;
/// "Small copy" for the latency probe.
const LATENCY_BYTES: usize = 8;
const LATENCY_WARMUP: usize = 32;
/// Enough iterations that the per-copy cost is well above timer resolution.
const LATENCY_ITERS: usize = 1000;
const GIB: f64 = (1_u64 << 30) as f64;

#[derive(Debug, Clone, Copy)]
struct PairMeasurement {
    bandwidth_gib_per_sec: f64,
    latency_micros: f64,
}

pub fn run_all_pairs(sink: &EventSink, transfer_bytes: u64) -> Result<()> {
    let device_count = CudaContext::device_count()
        .context("counting cuda devices")?
        .max(0) as u32;
    if device_count < 2 {
        sink.outcome(
            TestId::GpuP2p,
            Scope::Node,
            TestOutcome::Skipped {
                reason: format!("p2p needs at least two GPUs, this node has {device_count}"),
            },
        );
        return Ok(());
    }

    let contexts = (0..device_count)
        .map(|index| {
            CudaContext::new(index as usize)
                .with_context(|| format!("creating cuda context for gpu {index}"))
        })
        .collect::<Result<Vec<_>>>()?;

    for a in 0..device_count {
        for b in 0..device_count {
            if a == b {
                continue;
            }
            let scope = Scope::GpuPair { a, b };
            let source = &contexts[a as usize];
            let target = &contexts[b as usize];
            // A single bad pair (a dead NVLink, a context that refuses to
            // enable peer access) is a per-pair finding, never the end of the
            // sweep.
            match guard("p2p pair", || run_pair(source, target, transfer_bytes)) {
                Ok(Some(measurement)) => {
                    sink.metric(MetricRecord {
                        test: TestId::GpuP2p,
                        scope: scope.clone(),
                        name: "bandwidth".to_string(),
                        value: measurement.bandwidth_gib_per_sec,
                        unit: Unit::GibPerSec,
                    });
                    sink.metric(MetricRecord {
                        test: TestId::GpuP2p,
                        scope: scope.clone(),
                        name: "latency".to_string(),
                        value: measurement.latency_micros,
                        unit: Unit::Micros,
                    });
                    sink.outcome(TestId::GpuP2p, scope, TestOutcome::Passed);
                }
                // No peer capability is the normal state on PCIe-only boxes.
                Ok(None) => sink.outcome(
                    TestId::GpuP2p,
                    scope,
                    TestOutcome::Skipped {
                        reason: format!("gpu {a} cannot access gpu {b} directly"),
                    },
                ),
                Err(reason) => sink.outcome(TestId::GpuP2p, scope, TestOutcome::Failed { reason }),
            }
        }
    }
    Ok(())
}

/// Measure one ordered pair (source -> target). `Ok(None)` means the pair has
/// no peer capability, which is a Skipped outcome rather than a failure.
fn run_pair(
    source: &Arc<CudaContext>,
    target: &Arc<CudaContext>,
    transfer_bytes: u64,
) -> Result<Option<PairMeasurement>> {
    // Peer capability is a property of the link, so both directions have to be
    // usable before the copy engine will take the direct path.
    if !can_access_peer(target, source)? || !can_access_peer(source, target)? {
        return Ok(None);
    }
    enable_peer_access(target, source)?;
    enable_peer_access(source, target)?;

    let source_stream = source.default_stream();
    let target_stream = target.default_stream();

    let bytes = transfer_bytes.max(MIN_BYTES) as usize;
    let src = source_stream.alloc_zeros::<u8>(bytes)?;
    let mut dst = target_stream.alloc_zeros::<u8>(bytes)?;

    // Driven from the target's stream: cudarc routes a cross-context
    // `memcpy_dtod` through `cuMemcpyPeerAsync`.
    target_stream.memcpy_dtod(&src, &mut dst)?;
    target_stream.synchronize()?;

    let mut bandwidth_gib_per_sec = 0.0_f64;
    for _ in 0..TRANSFERS {
        let start = Instant::now();
        target_stream.memcpy_dtod(&src, &mut dst)?;
        target_stream.synchronize()?;
        let elapsed = start.elapsed().as_secs_f64().max(1e-12);
        let rate = bytes as f64 / elapsed / GIB;
        if rate > bandwidth_gib_per_sec {
            bandwidth_gib_per_sec = rate;
        }
    }
    drop(src);
    drop(dst);

    let small_src = source_stream.alloc_zeros::<u8>(LATENCY_BYTES)?;
    let mut small_dst = target_stream.alloc_zeros::<u8>(LATENCY_BYTES)?;
    for _ in 0..LATENCY_WARMUP {
        target_stream.memcpy_dtod(&small_src, &mut small_dst)?;
    }
    target_stream.synchronize()?;

    // Queue the copies back to back and synchronize once: dividing the total
    // by the count gives per-copy cost without folding a host-side
    // synchronization into every sample (same shape as NVIDIA's
    // p2pBandwidthLatencyTest).
    let start = Instant::now();
    for _ in 0..LATENCY_ITERS {
        target_stream.memcpy_dtod(&small_src, &mut small_dst)?;
    }
    target_stream.synchronize()?;
    let latency_micros = per_iteration_micros(start.elapsed().as_secs_f64(), LATENCY_ITERS);

    Ok(Some(PairMeasurement {
        bandwidth_gib_per_sec,
        latency_micros,
    }))
}

/// Can `accessor`'s device read `peer`'s memory directly?
fn can_access_peer(accessor: &Arc<CudaContext>, peer: &Arc<CudaContext>) -> Result<bool> {
    let mut capable: std::ffi::c_int = 0;
    // SAFETY: `capable` is a live, correctly typed out-parameter and both
    // device handles come from live `CudaContext`s. cudarc exposes no safe
    // wrapper for this driver call.
    unsafe {
        driver_sys::cuDeviceCanAccessPeer(&raw mut capable, accessor.cu_device(), peer.cu_device())
    }
    .result()
    .context("cuDeviceCanAccessPeer")?;
    Ok(capable != 0)
}

/// Enable `accessor` -> `peer` mappings. Already-enabled is success: contexts
/// are primary and shared, so a previous pair may have set this up.
fn enable_peer_access(accessor: &Arc<CudaContext>, peer: &Arc<CudaContext>) -> Result<()> {
    accessor
        .bind_to_thread()
        .context("binding accessor context")?;
    // SAFETY: `peer.cu_ctx()` is a live context handle and the accessing
    // context is current on this thread, which is what cuCtxEnablePeerAccess
    // requires. cudarc exposes no safe wrapper for this driver call.
    let status = unsafe { driver_sys::cuCtxEnablePeerAccess(peer.cu_ctx(), 0) };
    if status == driver_sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
        return Ok(());
    }
    status.result().context("cuCtxEnablePeerAccess")?;
    Ok(())
}

fn per_iteration_micros(elapsed_secs: f64, iters: usize) -> f64 {
    if iters == 0 {
        return 0.0;
    }
    elapsed_secs * 1e6 / iters as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_iteration_micros_divides_the_batch() {
        // 10 ms across 1000 copies is 10 us each.
        assert!((per_iteration_micros(0.01, 1000) - 10.0).abs() < 1e-9);
        assert_eq!(per_iteration_micros(1.0, 0), 0.0);
        // A batch of LATENCY_ITERS is long enough that the per-copy figure is
        // well clear of Instant's resolution.
        assert!(per_iteration_micros(1e-9 * LATENCY_ITERS as f64, LATENCY_ITERS) > 0.0);
    }
}
