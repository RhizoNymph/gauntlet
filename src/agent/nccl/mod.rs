//! Phase 3b: NCCL collective sweeps (`gauntlet agent nccl`).
//!
//! World: one NCCL rank per GPU. Each NCCL-capable host owns a contiguous
//! block of global ranks ordered by local GPU index
//! (`proto::RankAssignment`); one `agent nccl` process per host drives its
//! whole block from a single thread (`local`): grouped `ncclCommInitRank`
//! for every local device inside `ncclGroupStart`/`ncclGroupEnd`, then
//! every collective issued once per local rank inside a group per
//! operation. `socket_ifname`, when set, is exported as
//! NCCL_SOCKET_IFNAME before init.
//!
//! Rendezvous: the host holding global rank 0 runs the `Lead` directive,
//! mints the `NcclUniqueId` in-process and announces it as an `NcclId`
//! event; the orchestrator relays it to every other host's `Participate`
//! directive. No filesystem or TCP-store dependency.
//!
//! Sweep (`sweep`): for each size in `sizes`, `iters_per_size` all-reduce
//! (and all-gather) iterations after a warmup; the lead host emits
//! per-size elapsed micros (timed on global rank 0) as
//! `nccl_all_reduce.elapsed_us` metrics with the size recorded in a
//! companion `msg_bytes` metric under the same scope, plus computed bus
//! bandwidth `bus_gib_per_sec` (the alpha-beta fit itself happens
//! orchestrator-side in `analysis::fit`). Other hosts emit only Fatal on
//! error — and one `NcclBarrierTimings` per local rank when the
//! barrier-skew probe rides along.
//!
//! Fleet overlap (`fleet_overlap`): the directive's `NcclWorkload::Overlap`
//! runs the combined-load protocol instead of the sweep — an isolated fleet
//! all-reduce baseline, then the same all-reduce while every local GPU
//! hammers GEMMs (`gpu::worker`), with window boundaries agreed through a
//! MIN-reduced control word (`agent::window`) so every rank leaves each
//! window in the same iteration without cross-host clock comparison. Every
//! rank (= GPU) emits its own `OverlapFleetReport`.

use std::io::BufRead;

use anyhow::{Context, Result};

use crate::proto::NcclDirective;

// Pure; its only consumer is gpu-gated, but the tests run everywhere.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
mod completion;
#[cfg(feature = "gpu")]
mod fleet_overlap;
#[cfg(feature = "gpu")]
mod local;
#[cfg(feature = "gpu")]
mod sweep;

const GIB: f64 = (1_u64 << 30) as f64;
/// Collective payloads are f32 elements throughout.
// Consumers (overlap phase, fleet overlap protocol) are gpu-gated; keep
// the definitions unconditional so the pure tests cover them everywhere.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) const F32_BYTES: usize = std::mem::size_of::<f32>();

/// All-reduce payload element count for a requested message size; never
/// zero, so a degenerate spec still exercises the collective. Shared by
/// the intra-node overlap phase and the fleet overlap protocol.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn message_elements(msg_bytes: u64) -> usize {
    (msg_bytes as usize / F32_BYTES).max(1)
}

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
        NcclDirective::Lead { .. } => guard("nccl lead", || imp::lead(directive))
            .map_err(|reason| anyhow::anyhow!("{reason}")),
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
    use anyhow::{Result, bail};
    use cudarc::nccl::Id;

    use super::fleet_overlap::overlap_fleet;
    use super::local::{LocalRanks, nccl_error};
    use super::sweep::run_sweep;
    use crate::agent::EventSink;
    use crate::proto::{AgentEvent, NcclDirective, NcclWorkload, PROTO_VERSION, RankAssignment};

    pub use super::local::{decode_id, encode_id};

    /// The host holding global rank 0: mint the id in *this* process
    /// (ncclGetUniqueId opens the bootstrap listen socket here, so the
    /// process must stay alive through communicator init), announce it,
    /// then drive every local rank through the workload.
    pub fn lead(directive: &NcclDirective) -> Result<()> {
        let NcclDirective::Lead {
            assignment,
            socket_ifname,
            workload,
        } = directive
        else {
            bail!("lead requires a Lead directive");
        };
        if !assignment.block().holds_lead() {
            bail!(
                "the Lead directive must hold global rank 0, got a block starting at {}",
                assignment.block().base()
            );
        }
        set_socket_ifname(socket_ifname);

        let sink = EventSink::stdout();
        sink.emit(&AgentEvent::Hello {
            proto_version: PROTO_VERSION,
            hostname: crate::agent::hostname()?,
        });
        let id = Id::new().map_err(|error| nccl_error("ncclGetUniqueId", error))?;
        sink.emit(&AgentEvent::NcclId {
            unique_id_b64: encode_id(&id),
        });
        run_block(&sink, id, *assignment, workload)
    }

    /// Every other host: run the workload silently on every local rank,
    /// but report per-rank barrier timings and fleet-overlap results —
    /// per-rank local measurement is the whole point of those benchmarks.
    pub fn participate(directive: &NcclDirective) -> Result<()> {
        let NcclDirective::Participate {
            unique_id_b64,
            assignment,
            socket_ifname,
            workload,
        } = directive
        else {
            bail!("participate requires a Participate directive");
        };
        if assignment.block().holds_lead() {
            bail!("the block holding global rank 0 must run the Lead directive");
        }
        set_socket_ifname(socket_ifname);
        let id = decode_id(unique_id_b64)?;
        let sink = EventSink::stdout();
        sink.emit(&AgentEvent::Hello {
            proto_version: PROTO_VERSION,
            hostname: crate::agent::hostname()?,
        });
        run_block(&sink, id, *assignment, workload)
    }

    fn set_socket_ifname(socket_ifname: &Option<String>) {
        if let Some(ifname) = socket_ifname {
            // NCCL reads this once, at communicator init.
            //
            // SAFETY: nothing has been spawned yet — no communicator, no NCCL
            // helper threads, no threads of our own — so no other thread can
            // be reading the environment concurrently with this write.
            unsafe { std::env::set_var("NCCL_SOCKET_IFNAME", ifname) };
        }
    }

    /// Grouped init of every local rank, then the workload. An init
    /// failure fails the whole block (the node's participation).
    fn run_block(
        sink: &EventSink,
        id: Id,
        assignment: RankAssignment,
        workload: &NcclWorkload,
    ) -> Result<()> {
        let ranks = LocalRanks::init(assignment, id)?;
        match workload {
            NcclWorkload::Sweep {
                sizes,
                iters_per_size,
                barrier,
            } => run_sweep(sink, &ranks, sizes, *iters_per_size, *barrier),
            NcclWorkload::Overlap(spec) => overlap_fleet(sink, &ranks, spec),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_elements_are_f32_sized_and_never_zero() {
        assert_eq!(message_elements(64 << 20), (64 << 20) / 4);
        assert_eq!(message_elements(4), 1);
        assert_eq!(message_elements(0), 1);
        assert_eq!(message_elements(3), 1);
    }

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
