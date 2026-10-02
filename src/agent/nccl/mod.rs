//! Phase 3b: NCCL collective sweeps (`gauntlet agent nccl`).
//!
//! World: each NCCL-capable host owns a contiguous block of global ranks
//! on a contiguous run of local GPUs (`proto::RankAssignment`; rank
//! `base + i` on GPU `first_gpu + i`). The rank-per-GPU world gives a host
//! every GPU from 0; the NIC-forcing shapes give it one rank on one chosen
//! GPU (the orchestrator decides — the agent only follows the block). One
//! `agent nccl` process per host drives its whole block from a single
//! thread (`local`): grouped `ncclCommInitRank` for every local device
//! inside `ncclGroupStart`/`ncclGroupEnd`, then every collective issued
//! once per local rank inside a group per operation.
//!
//! NCCL env (NCCL_SOCKET_IFNAME and any `[nccl] env` knobs) is never set
//! here: the orchestrator puts it on the remote `env ... gauntlet agent`
//! command line, so it is in the process environment before `main` builds
//! the multi-threaded tokio runtime. `std::env::set_var` after that point
//! would be unsound.
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
//! orchestrator-side in `analysis::fit`), and after the last size one
//! fleet-level headline per collective (`headline`). The workload's
//! `SweepSeries` picks the test ids (`nccl_all_*` for rank-per-GPU,
//! `nccl_inter_all_*` for the NIC-forcing shapes) and the headline name.
//! Other hosts emit only Fatal on error — and one `NcclBarrierTimings`
//! per local rank when the barrier-skew probe rides along
//! (`NcclWorkload::Barrier` runs that probe alone).
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
use thiserror::Error;

use crate::agent::EventSink;
use crate::proto::{AgentEvent, NcclDirective, PROTO_VERSION};

// Pure; its only consumer is gpu-gated, but the tests run everywhere.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
mod completion;
#[cfg(feature = "gpu")]
mod fleet_overlap;
// Pure; its only consumer is gpu-gated, but the tests run everywhere.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
mod headline;
#[cfg(feature = "gpu")]
mod local;
#[cfg(feature = "gpu")]
mod sweep;
#[cfg(feature = "gpu")]
mod watchdog;

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
///
/// `Hello` goes out first, before anything can fail — the directive parse,
/// its validation, the rendezvous id decode, the device checks — and every
/// failure after it is reported as a typed `Fatal` naming the real reason.
/// (A failure before `Hello` would reach the orchestrator only as "first
/// event was not hello".)
pub fn run_from_stdin() -> Result<()> {
    let sink = EventSink::stdout();
    sink.emit(&AgentEvent::Hello {
        proto_version: PROTO_VERSION,
        hostname: crate::agent::hostname().unwrap_or_else(|_| "(unknown)".to_string()),
    });
    let outcome = read_directive().and_then(|directive| execute(&sink, &directive));
    if let Err(error) = &outcome {
        sink.emit(&AgentEvent::Fatal {
            message: format!("{error:#}"),
        });
    }
    outcome
}

fn read_directive() -> Result<NcclDirective> {
    let mut line = String::new();
    std::io::stdin()
        .lock()
        .read_line(&mut line)
        .context("reading NcclDirective from stdin")?;
    serde_json::from_str(line.trim()).context("parsing NcclDirective")
}

#[cfg(feature = "gpu")]
fn execute(sink: &EventSink, directive: &NcclDirective) -> Result<()> {
    // libcuda/libnccl are dlopened, and cudarc panics rather than erroring
    // when the library or a symbol is missing. The gpu phase's guard turns
    // that back into an ordinary error, so a node without the NCCL stack
    // fails this directive instead of aborting the process.
    use crate::agent::gpu::guard;
    guard("agent nccl", || imp::run(sink, directive)).map_err(|reason| anyhow::anyhow!(reason))
}

#[cfg(not(feature = "gpu"))]
fn execute(_sink: &EventSink, _directive: &NcclDirective) -> Result<()> {
    anyhow::bail!("built without gpu feature")
}

/// A host's rank block cannot run on the GPUs CUDA actually exposes.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum LocalDeviceError {
    #[error(
        "rank block needs {needed} local GPUs but CUDA sees {visible} \
         (GPU off the bus, MIG enabled, or CUDA_VISIBLE_DEVICES set?)"
    )]
    TooFewDevices { needed: u32, visible: i32 },
}

/// Validate a rank block against CUDA's visible device count, before any
/// NCCL call: a host that cannot open every rank of its block must fail
/// fast (with the reason) rather than join grouped init and strand the
/// world.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn check_local_devices(needed: u32, visible: i32) -> Result<(), LocalDeviceError> {
    if i64::from(needed) > i64::from(visible) {
        return Err(LocalDeviceError::TooFewDevices { needed, visible });
    }
    Ok(())
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

    use super::fleet_overlap::{OverlapBuffers, overlap_fleet};
    use super::local::{PreparedRanks, nccl_error};
    use super::sweep::{SweepBuffers, SweepRun, run_barrier, run_sweep};
    use super::watchdog::{CascadeAbort, exit_cascade};
    use crate::agent::EventSink;
    use crate::proto::{AgentEvent, NcclDirective, NcclWorkload};

    pub use super::local::{decode_id, encode_id};

    /// Where this host's rendezvous id comes from.
    enum Rendezvous {
        /// Holds global rank 0: mint the id in *this* process
        /// (ncclGetUniqueId opens the bootstrap listen socket here, so the
        /// process must stay alive through communicator init) and announce
        /// it.
        Mint,
        /// Joins the lead's world with the relayed id.
        Join(Id),
    }

    /// Validate the directive, then run it. A [`CascadeAbort`] — this host
    /// stopped because the fleet did — leaves through a cascade exit
    /// instead of an ordinary error, so the orchestrator attributes it to
    /// the host that actually failed.
    pub fn run(sink: &EventSink, directive: &NcclDirective) -> Result<()> {
        let outcome = run_directive(sink, directive);
        if let Err(error) = &outcome
            && let Some(CascadeAbort(message)) = error.downcast_ref::<CascadeAbort>()
        {
            exit_cascade(sink, message);
        }
        outcome
    }

    fn run_directive(sink: &EventSink, directive: &NcclDirective) -> Result<()> {
        let (assignment, workload, rendezvous) = match directive {
            NcclDirective::Lead {
                assignment,
                workload,
            } => {
                if !assignment.block().holds_lead() {
                    bail!(
                        "the Lead directive must hold global rank 0, got a block starting at {}",
                        assignment.block().base()
                    );
                }
                (assignment, workload, Rendezvous::Mint)
            }
            NcclDirective::Participate {
                unique_id_b64,
                assignment,
                workload,
            } => {
                if assignment.block().holds_lead() {
                    bail!("the block holding global rank 0 must run the Lead directive");
                }
                let id = decode_id(unique_id_b64)?;
                (assignment, workload, Rendezvous::Join(id))
            }
        };

        // Stage 1, no NCCL: device check, contexts, binds, buffers. The
        // lead finishes it before minting the id, so a lead that cannot
        // run never recruits the followers.
        let prepared = PreparedRanks::new(*assignment)?;
        let buffers = match workload {
            NcclWorkload::Sweep { sizes, .. } => {
                Buffers::Sweep(SweepBuffers::alloc(&prepared, sizes)?)
            }
            NcclWorkload::Overlap(spec) => {
                Buffers::Overlap(OverlapBuffers::alloc(&prepared, spec)?)
            }
            NcclWorkload::Barrier(spec) => {
                Buffers::Sweep(SweepBuffers::alloc_for_barrier(&prepared, *spec)?)
            }
        };
        let id = match rendezvous {
            Rendezvous::Join(id) => id,
            Rendezvous::Mint => {
                let id = Id::new().map_err(|error| nccl_error("ncclGetUniqueId", error))?;
                sink.emit(&AgentEvent::NcclId {
                    unique_id_b64: encode_id(&id),
                });
                id
            }
        };
        // Stage 2: grouped communicator init.
        let ranks = prepared.connect(id)?;
        match (workload, buffers) {
            (
                NcclWorkload::Sweep {
                    sizes,
                    iters_per_size,
                    barrier,
                    series,
                },
                Buffers::Sweep(buffers),
            ) => run_sweep(
                sink,
                &ranks,
                buffers,
                SweepRun {
                    sizes,
                    iters_per_size: *iters_per_size,
                    barrier: *barrier,
                    series: *series,
                },
            ),
            (NcclWorkload::Overlap(spec), Buffers::Overlap(buffers)) => {
                overlap_fleet(sink, &ranks, buffers, spec)
            }
            (NcclWorkload::Barrier(spec), Buffers::Sweep(buffers)) => {
                run_barrier(sink, &ranks, buffers, *spec)
            }
            _ => bail!("workload buffers do not match the workload"),
        }
    }

    /// The workload's buffers, allocated in stage 1.
    enum Buffers {
        Sweep(SweepBuffers),
        Overlap(OverlapBuffers),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_larger_than_the_visible_devices_fails_fast() {
        assert_eq!(check_local_devices(8, 8), Ok(()));
        assert_eq!(check_local_devices(4, 8), Ok(()));
        assert_eq!(
            check_local_devices(8, 7),
            Err(LocalDeviceError::TooFewDevices {
                needed: 8,
                visible: 7
            })
        );
        // A negative count from the driver is never "enough".
        assert!(check_local_devices(1, -1).is_err());
        let message = check_local_devices(8, 0)
            .expect_err("none visible")
            .to_string();
        assert!(
            message.contains("needs 8") && message.contains("sees 0"),
            "{message}"
        );
    }

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
