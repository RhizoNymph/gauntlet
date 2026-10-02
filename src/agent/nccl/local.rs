//! One process, many ranks: the host's whole rank block of the fleet NCCL
//! world, driven from a single thread (gpu feature only).
//!
//! Two stages, so that nothing fallible except `ncclCommInitRank` itself
//! runs inside the init group:
//!
//! 1. [`PreparedRanks::new`] — no NCCL calls. Checks the block against
//!    CUDA's visible device count (`check_local_devices`), creates a
//!    context per local GPU, proves each one binds to this thread, and
//!    creates the per-rank completion events. Callers allocate their
//!    collective buffers on [`PreparedRanks::streams`] before connecting.
//! 2. [`PreparedRanks::connect`] — NCCL's documented "multiple GPUs from
//!    one thread" pattern: every local device's `ncclCommInitRank` (cudarc
//!    `Comm::from_rank`, with that device's context bound so NCCL picks the
//!    right device) inside one `ncclGroupStart`/`ncclGroupEnd`, so the
//!    blocking per-rank inits complete together at group end.
//!
//! Any failure fails the whole node's participation — the world was sized
//! for every local GPU, so a partial block cannot join. Every collective is
//! then issued once per local rank inside a group per operation.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream, DriverError, sys};
use cudarc::nccl::result::{NcclError, NcclStatus};
use cudarc::nccl::{Comm, Id, group_end, group_start};

use super::check_local_devices;
use super::completion::{Readiness, first_ready_times};
use crate::proto::RankAssignment;

/// NCCL's opaque rendezvous token is exactly 128 bytes.
const UNIQUE_ID_BYTES: usize = 128;

/// `NcclError` implements neither `Display` nor `std::error::Error`, so it
/// cannot ride `?` into anyhow; the raw `ncclResult_t` is the useful part.
pub(super) fn nccl_error(what: &str, error: NcclError) -> anyhow::Error {
    anyhow!("{what}: {:?}", error.0)
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
    let mut internal: [std::ffi::c_char; UNIQUE_ID_BYTES] = [0; UNIQUE_ID_BYTES];
    for (slot, byte) in internal.iter_mut().zip(bytes) {
        *slot = byte as std::ffi::c_char;
    }
    Ok(Id::uninit(internal))
}

/// One local rank before the communicator exists.
struct PreparedRank {
    global: u32,
    ctx: Arc<CudaContext>,
    done: CudaEvent,
}

/// Every rank of this host's block, validated and set up, not yet
/// connected: stage 1.
pub(super) struct PreparedRanks {
    assignment: RankAssignment,
    ranks: Vec<PreparedRank>,
}

impl PreparedRanks {
    /// Every fallible non-NCCL step, for all local ranks, before any NCCL
    /// call: device-count check, contexts, thread binding, events.
    pub(super) fn new(assignment: RankAssignment) -> Result<Self> {
        let block = assignment.block();
        let visible = CudaContext::device_count().context("counting cuda devices")?;
        // The block's highest GPU must exist: `gpus().end` devices.
        check_local_devices(block.gpus().end, visible)?;
        let mut ranks = Vec::with_capacity(block.count() as usize);
        for (global, local) in block.ranks().zip(block.gpus()) {
            let ctx = CudaContext::new(local as usize)
                .with_context(|| format!("creating cuda context for local gpu {local}"))?;
            ctx.bind_to_thread()
                .with_context(|| format!("binding local gpu {local} to the driver thread"))?;
            let done = ctx
                .new_event(None)
                .with_context(|| format!("creating completion event for rank {global}"))?;
            ranks.push(PreparedRank { global, ctx, done });
        }
        Ok(Self { assignment, ranks })
    }

    /// Each local rank's collective stream (its device's default stream),
    /// local GPU order — the stream the communicator will be bound to.
    pub(super) fn streams(&self) -> Vec<Arc<CudaStream>> {
        self.ranks
            .iter()
            .map(|rank| rank.ctx.default_stream())
            .collect()
    }

    /// Stage 2: create every communicator inside one NCCL group. The only
    /// fallible calls inside the group are the binds (already proven in
    /// stage 1) and `ncclCommInitRank`.
    pub(super) fn connect(self, id: Id) -> Result<LocalRanks> {
        let world_size = self.assignment.world_size() as usize;
        group_start().map_err(|error| nccl_error("ncclGroupStart (init)", error))?;
        let mut comms: Vec<Comm> = Vec::with_capacity(self.ranks.len());
        for rank in &self.ranks {
            let issued = rank
                .ctx
                .bind_to_thread()
                .map_err(|error| anyhow!("binding gpu {}: {error}", rank.ctx.ordinal()))
                .and_then(|()| {
                    // ncclCommInitRank picks the *current* device.
                    Comm::from_rank(
                        rank.ctx.default_stream(),
                        rank.global as usize,
                        world_size,
                        id,
                    )
                    .map_err(|error| nccl_error("ncclCommInitRank", error))
                });
            match issued {
                Ok(comm) => comms.push(comm),
                Err(error) => {
                    abandon_partial_init(comms);
                    return Err(error.context(format!("grouped init of rank {}", rank.global)));
                }
            }
        }
        if let Err(error) = group_end() {
            abandon_partial_init(comms);
            return Err(nccl_error("ncclGroupEnd (init)", error));
        }
        let ranks = self
            .ranks
            .into_iter()
            .zip(comms)
            .map(|(rank, comm)| LocalRank {
                global: rank.global,
                comm,
                done: rank.done,
            })
            .collect();
        Ok(LocalRanks {
            assignment: self.assignment,
            ranks,
        })
    }
}

/// Failure path of a grouped init. Deliberately does *not* end the group:
/// `ncclGroupEnd` would execute the already-queued inits of the earlier
/// local ranks, which then block until the whole world joins — which this
/// host no longer will. Deliberately does *not* drop the half-built
/// communicators either: cudarc's `Drop` calls
/// `comm_abort(..).expect(..)`, which can panic on a communicator whose
/// init never ran. The caller is about to fail the node and exit (the
/// orchestrator then aborts the rest of the world), so leaking both is the
/// correct cleanup; process exit reclaims them.
fn abandon_partial_init(comms: Vec<Comm>) {
    for comm in comms {
        std::mem::forget(comm);
    }
}

/// One connected local rank. The collective stream and the device context
/// are the communicator's own (single source of truth).
pub(super) struct LocalRank {
    pub global: u32,
    pub comm: Comm,
    done: CudaEvent,
}

impl LocalRank {
    pub(super) fn stream(&self) -> Arc<CudaStream> {
        self.comm.stream()
    }

    pub(super) fn ctx(&self) -> &Arc<CudaContext> {
        self.comm.context()
    }
}

/// Every rank of this host's block, connected; index `i` is block position
/// `i` (rank `base + i`, on GPU `first_gpu + i`).
pub(super) struct LocalRanks {
    assignment: RankAssignment,
    ranks: Vec<LocalRank>,
}

impl LocalRanks {
    pub(super) fn assignment(&self) -> RankAssignment {
        self.assignment
    }

    pub(super) fn world_size(&self) -> u32 {
        self.assignment.world_size()
    }

    pub(super) fn ranks(&self) -> &[LocalRank] {
        &self.ranks
    }

    pub(super) fn contexts(&self) -> Vec<Arc<CudaContext>> {
        self.ranks
            .iter()
            .map(|rank| Arc::clone(rank.ctx()))
            .collect()
    }

    /// Issue one collective on every local rank inside one NCCL group.
    /// `op(index, rank)` enqueues rank `index`'s share.
    pub(super) fn collective(
        &self,
        what: &str,
        mut op: impl FnMut(usize, &LocalRank) -> Result<NcclStatus, NcclError>,
    ) -> Result<()> {
        grouped(what, || {
            for (index, rank) in self.ranks.iter().enumerate() {
                op(index, rank).map_err(|error| nccl_error(what, error))?;
            }
            Ok(())
        })
    }

    pub(super) fn sync_all(&self) -> Result<()> {
        for rank in &self.ranks {
            rank.stream().synchronize()?;
        }
        Ok(())
    }

    /// Seconds from `since` until each local rank's queued work completed,
    /// stamped by round-robin polling (no index-order bias; see
    /// `completion`). Leaves every stream drained.
    pub(super) fn completion_secs(&self, since: Instant) -> Result<Vec<f64>> {
        for rank in &self.ranks {
            rank.done.record(&rank.stream())?;
        }
        let secs = first_ready_times(
            self.ranks.len(),
            |index| query(&self.ranks[index].done),
            || since.elapsed().as_secs_f64(),
        )?;
        Ok(secs)
    }
}

/// Non-blocking completion probe of an event.
fn query(event: &CudaEvent) -> Result<Readiness, DriverError> {
    // SAFETY: the event is a live, owned CUevent (never null).
    match unsafe { cudarc::driver::result::event::query(event.cu_event()) } {
        Ok(()) => Ok(Readiness::Ready),
        Err(DriverError(sys::cudaError_enum::CUDA_ERROR_NOT_READY)) => Ok(Readiness::Pending),
        Err(error) => Err(error),
    }
}

/// Run a collective `body` between `ncclGroupStart` and `ncclGroupEnd`.
/// The group is always closed — even when `body` fails part-way — so
/// NCCL's group depth never leaks into the next call; the body's error
/// wins over the end's. (Collectives only; grouped *init* has its own
/// failure path, `abandon_partial_init`.)
fn grouped(what: &str, body: impl FnOnce() -> Result<()>) -> Result<()> {
    group_start().map_err(|error| nccl_error(&format!("{what}: ncclGroupStart"), error))?;
    let issued = body();
    let ended = group_end().map_err(|error| nccl_error(&format!("{what}: ncclGroupEnd"), error));
    issued?;
    ended?;
    Ok(())
}
