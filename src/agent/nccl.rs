//! Phase 3b: NCCL collective sweeps (`gauntlet agent nccl`).
//!
//! Rendezvous: the orchestrator sends rank 0 a `GenerateId` directive; the
//! agent prints the `NcclUniqueId` JSON on stdout. The orchestrator then
//! sends every rank a `Participate` directive carrying that id. No
//! filesystem or TCP-store dependency.
//!
//! Sweep: for each size in `sizes`, `iters_per_size` all-reduce (and
//! all-gather) iterations after a warmup; emit per-size elapsed micros as
//! `nccl_all_reduce.elapsed_us` metrics with the size recorded in a
//! companion `msg_bytes` metric under the same scope, plus computed bus
//! bandwidth `bus_gib_per_sec` (the alpha-beta fit itself happens
//! orchestrator-side in `analysis::fit`). Rank 0 emits the events; other
//! ranks emit only Fatal on error.
//!
//! One process per node, one GPU per rank for v1 (world_size == node count;
//! intra-node NVLink is covered by the p2p test). `socket_ifname`, when
//! set, is exported as NCCL_SOCKET_IFNAME before init.

use std::io::BufRead;

use anyhow::{Context, Result};

use crate::proto::NcclDirective;

const GIB: f64 = (1_u64 << 30) as f64;

/// Read an `NcclDirective` JSON document from stdin and execute it.
pub fn run_from_stdin() -> Result<()> {
    let mut line = String::new();
    std::io::stdin()
        .lock()
        .read_line(&mut line)
        .context("reading NcclDirective from stdin")?;
    let directive: NcclDirective =
        serde_json::from_str(line.trim()).context("parsing NcclDirective")?;
    execute(&directive)
}

#[cfg(feature = "gpu")]
fn execute(directive: &NcclDirective) -> Result<()> {
    // libcuda/libnccl are dlopened, and cudarc panics rather than erroring
    // when the library or a symbol is missing. The gpu phase's guard turns
    // that back into an ordinary error, so a node without the NCCL stack
    // fails this directive instead of aborting the process.
    use crate::agent::gpu::guard;
    match directive {
        NcclDirective::GenerateId => {
            let id = guard("nccl generate id", imp::generate_id)
                .map_err(|reason| anyhow::anyhow!("{reason}"))?;
            // The one and only stdout line this mode produces.
            println!("{}", serde_json::to_string(&id)?);
            Ok(())
        }
        NcclDirective::Participate { .. } => {
            guard("nccl participate", || imp::participate(directive))
                .map_err(|reason| anyhow::anyhow!("{reason}"))
        }
    }
}

#[cfg(not(feature = "gpu"))]
fn execute(_directive: &NcclDirective) -> Result<()> {
    anyhow::bail!("built without gpu feature")
}

// ---------------------------------------------------------------------------
// Bus bandwidth
// ---------------------------------------------------------------------------

/// Traffic factor `(n-1)/n`: the fraction of the message every rank has to
/// actually move over the interconnect. A world of one moves nothing.
fn collective_factor(world_size: u32) -> f64 {
    if world_size <= 1 {
        return 0.0;
    }
    (f64::from(world_size) - 1.0) / f64::from(world_size)
}

fn bus_gib_per_sec(message_bytes: f64, elapsed_secs: f64, factor: f64) -> f64 {
    if elapsed_secs <= 0.0 {
        return 0.0;
    }
    message_bytes / elapsed_secs * factor / GIB
}

/// All-reduce bus bandwidth.
///
/// `algBW = message_bytes / elapsed`, and a ring all-reduce is a
/// reduce-scatter followed by an all-gather, so each rank pushes
/// `2*(n-1)/n` of the message across the wire:
///
/// ```text
/// busBW = (message_bytes / elapsed) * 2 * (n - 1) / n
/// ```
///
/// (same definition as nccl-tests' `AllReduceGetBusBw`, which is what
/// operators compare against the link's peak.)
pub fn all_reduce_bus_gib_per_sec(message_bytes: f64, elapsed_secs: f64, world_size: u32) -> f64 {
    bus_gib_per_sec(
        message_bytes,
        elapsed_secs,
        2.0 * collective_factor(world_size),
    )
}

/// All-gather bus bandwidth. Every rank already owns its own shard and
/// receives the other `n-1`:
///
/// ```text
/// busBW = (message_bytes / elapsed) * (n - 1) / n
/// ```
///
/// where `message_bytes` is the size of the *gathered* result.
pub fn all_gather_bus_gib_per_sec(message_bytes: f64, elapsed_secs: f64, world_size: u32) -> f64 {
    bus_gib_per_sec(message_bytes, elapsed_secs, collective_factor(world_size))
}

#[cfg(feature = "gpu")]
pub mod imp {
    use std::ffi::c_char;
    use std::sync::Arc;
    use std::time::Instant;

    use anyhow::{Context, Result, anyhow, bail};
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD as BASE64;
    use cudarc::driver::{CudaContext, CudaStream};
    use cudarc::nccl::result::NcclError;
    use cudarc::nccl::{Comm, Id, ReduceOp};

    use crate::agent::EventSink;
    use crate::proto::{
        AgentEvent, MetricRecord, NcclDirective, NcclUniqueId, PROTO_VERSION, Scope, TestId, Unit,
    };

    /// NCCL's opaque rendezvous token is exactly 128 bytes.
    const UNIQUE_ID_BYTES: usize = 128;
    /// Untimed iterations before the sweep, so channel setup and algorithm
    /// selection do not land in the first measured size.
    const WARMUP_ITERS: usize = 5;
    const F32_BYTES: usize = std::mem::size_of::<f32>();

    /// `NcclError` implements neither `Display` nor `std::error::Error`, so it
    /// cannot ride `?` into anyhow; the raw `ncclResult_t` is the useful part.
    fn nccl_error(what: &str, error: NcclError) -> anyhow::Error {
        anyhow!("{what}: {:?}", error.0)
    }

    pub fn generate_id() -> Result<NcclUniqueId> {
        let id = Id::new().map_err(|error| nccl_error("ncclGetUniqueId", error))?;
        Ok(NcclUniqueId {
            unique_id_b64: encode_id(&id),
        })
    }

    /// The raw 128 bytes of an `ncclUniqueId`, base64'd so it survives a JSON
    /// round trip through the orchestrator.
    pub fn encode_id(id: &Id) -> String {
        let bytes: Vec<u8> = id.internal().iter().map(|byte| *byte as u8).collect();
        BASE64.encode(bytes)
    }

    pub fn decode_id(encoded: &str) -> Result<Id> {
        let bytes = BASE64
            .decode(encoded)
            .context("decoding nccl unique id from base64")?;
        if bytes.len() != UNIQUE_ID_BYTES {
            bail!(
                "nccl unique id decoded to {} bytes, expected {UNIQUE_ID_BYTES}",
                bytes.len()
            );
        }
        let mut internal: [c_char; UNIQUE_ID_BYTES] = [0; UNIQUE_ID_BYTES];
        for (slot, byte) in internal.iter_mut().zip(bytes) {
            *slot = byte as c_char;
        }
        Ok(Id::uninit(internal))
    }

    pub fn participate(directive: &NcclDirective) -> Result<()> {
        let NcclDirective::Participate {
            unique_id_b64,
            rank,
            world_size,
            sizes,
            iters_per_size,
            socket_ifname,
        } = directive
        else {
            bail!("participate requires a Participate directive");
        };
        if *world_size == 0 {
            bail!("world_size must be at least 1");
        }
        if rank >= world_size {
            bail!("rank {rank} is outside a world of {world_size}");
        }

        if let Some(ifname) = socket_ifname {
            // NCCL reads this once, at communicator init.
            //
            // SAFETY: nothing has been spawned yet — no communicator, no NCCL
            // helper threads, no threads of our own — so no other thread can
            // be reading the environment concurrently with this write.
            unsafe { std::env::set_var("NCCL_SOCKET_IFNAME", ifname) };
        }

        // v1: one process per node, one rank, driving device 0. Intra-node
        // GPU<->GPU is covered by the p2p test, so nothing is lost by not
        // fanning out across the local GPUs here.
        let ctx = CudaContext::new(0).context("creating cuda context for nccl rank")?;
        let stream = ctx.default_stream();
        let id = decode_id(unique_id_b64)?;
        let comm = Comm::from_rank(
            Arc::clone(&stream),
            *rank as usize,
            *world_size as usize,
            id,
        )
        .map_err(|error| nccl_error("ncclCommInitRank", error))?;

        let max_elements = sizes
            .iter()
            .map(|size| *size as usize / F32_BYTES)
            .max()
            .unwrap_or(0)
            .max(1);
        let send = stream.alloc_zeros::<f32>(max_elements)?;
        let mut recv = stream.alloc_zeros::<f32>(max_elements)?;

        // Only rank 0 speaks the event protocol; every other rank stays silent
        // and reports failures through its exit status.
        let sink = (*rank == 0).then(EventSink::stdout);
        if let Some(sink) = &sink {
            sink.emit(&AgentEvent::Hello {
                proto_version: PROTO_VERSION,
                hostname: crate::agent::hostname()?,
            });
        }

        for _ in 0..WARMUP_ITERS {
            comm.all_reduce(&send, &mut recv, &ReduceOp::Sum)
                .map_err(|error| nccl_error("warmup all_reduce", error))?;
        }
        stream.synchronize()?;

        let iters = (*iters_per_size).max(1);
        let world = *world_size as usize;

        for &size in sizes {
            let elements = (size as usize / F32_BYTES).clamp(1, max_elements);

            let elapsed = timed(&stream, iters, || {
                comm.all_reduce(
                    &send.slice(0..elements),
                    &mut recv.slice_mut(0..elements),
                    &ReduceOp::Sum,
                )
                .map_err(|error| nccl_error("all_reduce", error))?;
                Ok(())
            })?;
            if let Some(sink) = &sink {
                emit_collective(
                    sink,
                    TestId::NcclAllReduce,
                    (elements * F32_BYTES) as f64,
                    elapsed / f64::from(iters),
                    *world_size,
                );
            }

            // All-gather: every rank contributes one shard and receives the
            // whole thing, so the *gathered* result is the message size the
            // bus-bandwidth formula wants. Sizes too small to split across the
            // world have no meaningful all-gather and are skipped.
            let shard = elements / world;
            if shard == 0 {
                continue;
            }
            let gathered = shard * world;
            let elapsed = timed(&stream, iters, || {
                comm.all_gather(&send.slice(0..shard), &mut recv.slice_mut(0..gathered))
                    .map_err(|error| nccl_error("all_gather", error))?;
                Ok(())
            })?;
            if let Some(sink) = &sink {
                emit_collective(
                    sink,
                    TestId::NcclAllGather,
                    (gathered * F32_BYTES) as f64,
                    elapsed / f64::from(iters),
                    *world_size,
                );
            }
        }

        Ok(())
    }

    /// Total wall seconds for `iters` collectives. The stream is synchronized
    /// on both sides of the timed region, so the interval covers exactly the
    /// device work — collectives are enqueued asynchronously and would
    /// otherwise be timed at launch cost.
    fn timed(
        stream: &Arc<CudaStream>,
        iters: u32,
        mut op: impl FnMut() -> Result<()>,
    ) -> Result<f64> {
        stream.synchronize()?;
        let start = Instant::now();
        for _ in 0..iters {
            op()?;
        }
        stream.synchronize()?;
        Ok(start.elapsed().as_secs_f64())
    }

    fn emit_collective(
        sink: &EventSink,
        test: TestId,
        message_bytes: f64,
        per_iter_secs: f64,
        world_size: u32,
    ) {
        let bus = match test {
            TestId::NcclAllGather => {
                super::all_gather_bus_gib_per_sec(message_bytes, per_iter_secs, world_size)
            }
            _ => super::all_reduce_bus_gib_per_sec(message_bytes, per_iter_secs, world_size),
        };
        for (name, value, unit) in [
            ("elapsed_us", per_iter_secs * 1e6, Unit::Micros),
            ("msg_bytes", message_bytes, Unit::Bytes),
            ("bus_gib_per_sec", bus, Unit::GibPerSec),
        ] {
            sink.metric(MetricRecord {
                test,
                scope: Scope::Node,
                name: name.to_string(),
                value,
                unit,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_reduce_bus_bandwidth_matches_the_ring_factor() {
        // 1 GiB moved in 1 s across 2 ranks: factor 2*(1/2) = 1.
        assert!((all_reduce_bus_gib_per_sec(GIB, 1.0, 2) - 1.0).abs() < 1e-12);
        // Across 4 ranks: factor 2*(3/4) = 1.5.
        assert!((all_reduce_bus_gib_per_sec(GIB, 1.0, 4) - 1.5).abs() < 1e-12);
        // Across 8 ranks: factor 2*(7/8) = 1.75.
        assert!((all_reduce_bus_gib_per_sec(GIB, 1.0, 8) - 1.75).abs() < 1e-12);
        // The factor approaches 2 but never reaches it.
        assert!(all_reduce_bus_gib_per_sec(GIB, 1.0, 1024) < 2.0);
    }

    #[test]
    fn all_gather_bus_bandwidth_is_half_the_all_reduce_factor() {
        for world in [2_u32, 4, 8, 256] {
            let reduce = all_reduce_bus_gib_per_sec(GIB, 0.5, world);
            let gather = all_gather_bus_gib_per_sec(GIB, 0.5, world);
            assert!((reduce - 2.0 * gather).abs() < 1e-12, "world {world}");
        }
        // 512 MiB gathered in 250 ms across 4 ranks: algBW 2 GiB/s, x 3/4.
        let gather = all_gather_bus_gib_per_sec((512 << 20) as f64, 0.25, 4);
        assert!((gather - 1.5).abs() < 1e-12, "{gather}");
    }

    #[test]
    fn a_world_of_one_moves_nothing_and_zero_time_is_not_infinite() {
        assert_eq!(all_reduce_bus_gib_per_sec(GIB, 1.0, 1), 0.0);
        assert_eq!(all_gather_bus_gib_per_sec(GIB, 1.0, 1), 0.0);
        assert_eq!(all_reduce_bus_gib_per_sec(GIB, 0.0, 8), 0.0);
        assert_eq!(all_gather_bus_gib_per_sec(GIB, -1.0, 8), 0.0);
    }

    #[test]
    fn bus_bandwidth_scales_with_message_size_and_time() {
        let base = all_reduce_bus_gib_per_sec(GIB, 1.0, 8);
        assert!((all_reduce_bus_gib_per_sec(2.0 * GIB, 1.0, 8) - 2.0 * base).abs() < 1e-12);
        assert!((all_reduce_bus_gib_per_sec(GIB, 0.5, 8) - 2.0 * base).abs() < 1e-12);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn nccl_unique_id_survives_a_base64_round_trip() {
        use std::ffi::c_char;

        // A 128-byte token with every byte distinct mod 256, including the
        // high half that would break a naive signed/unsigned conversion.
        let mut internal: [c_char; 128] = [0; 128];
        for (index, slot) in internal.iter_mut().enumerate() {
            *slot = (index as u8).wrapping_mul(2).wrapping_add(129) as c_char;
        }
        let id = cudarc::nccl::Id::uninit(internal);

        let encoded = super::imp::encode_id(&id);
        assert!(!encoded.is_empty());
        let decoded = super::imp::decode_id(&encoded).expect("round trip");
        assert_eq!(decoded.internal(), &internal);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn a_malformed_unique_id_is_rejected() {
        assert!(super::imp::decode_id("not base64!!").is_err());
        // Correct base64, wrong length.
        assert!(super::imp::decode_id("YWJj").is_err());
    }
}
