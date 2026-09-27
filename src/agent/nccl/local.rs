//! One process, many ranks: the host's whole rank block of the fleet NCCL
//! world, driven from a single thread (gpu feature only).
//!
//! Init follows NCCL's documented "multiple GPUs from one thread" pattern:
//! every local device's communicator is created with `ncclCommInitRank`
//! (cudarc `Comm::from_rank`, with that device's context bound so NCCL
//! picks the right device) inside one `ncclGroupStart`/`ncclGroupEnd`, so
//! the blocking per-rank inits complete together at group end instead of
//! deadlocking each other. Every collective is then issued once per local
//! rank inside a group per operation. Any failure during grouped init
//! fails the whole node's participation — the world was sized for every
//! local GPU, so a partial block cannot join.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream, DriverError, sys};
use cudarc::nccl::result::{NcclError, NcclStatus};
use cudarc::nccl::{Comm, Id, group_end, group_start};

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

/// One local rank: global rank, its device's context, the default stream
/// that carries its collectives, its communicator, and a completion
/// marker for per-rank timing.
pub(super) struct LocalRank {
    pub global: u32,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub comm: Comm,
    done: CudaEvent,
}

/// Every rank of this host's block, local GPU `i` at index `i`.
pub(super) struct LocalRanks {
    assignment: RankAssignment,
    ranks: Vec<LocalRank>,
}

impl LocalRanks {
    /// Create a context per local GPU, then every communicator in one NCCL
    /// group. All-or-nothing: an error anywhere fails the node.
    pub(super) fn init(assignment: RankAssignment, id: Id) -> Result<Self> {
        let block = assignment.block();
        let world_size = assignment.world_size() as usize;
        let mut contexts = Vec::with_capacity(block.count() as usize);
        for local in 0..block.count() {
            contexts.push(
                CudaContext::new(local as usize)
                    .with_context(|| format!("creating cuda context for local gpu {local}"))?,
            );
        }

        let mut comms: Vec<Comm> = Vec::with_capacity(contexts.len());
        let issued = grouped("ncclGroupStart/ncclCommInitRank", || {
            for (global, ctx) in block.ranks().zip(&contexts) {
                // ncclCommInitRank picks the *current* device.
                ctx.bind_to_thread()
                    .map_err(|error| anyhow!("binding gpu {}: {error}", ctx.ordinal()))?;
                let comm = Comm::from_rank(ctx.default_stream(), global as usize, world_size, id)
                    .map_err(|error| nccl_error("ncclCommInitRank", error))?;
                comms.push(comm);
            }
            Ok(())
        });
        issued.context("grouped communicator init")?;

        let ranks = block
            .ranks()
            .zip(contexts)
            .zip(comms)
            .map(|((global, ctx), comm)| {
                let done = ctx
                    .new_event(None)
                    .with_context(|| format!("creating completion event for rank {global}"))?;
                Ok(LocalRank {
                    global,
                    stream: ctx.default_stream(),
                    ctx,
                    comm,
                    done,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { assignment, ranks })
    }

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
            .map(|rank| Arc::clone(&rank.ctx))
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
            rank.stream.synchronize()?;
        }
        Ok(())
    }

    /// Seconds from `since` until each local rank's queued work completed,
    /// stamped by round-robin polling (no index-order bias; see
    /// `completion`). Leaves every stream drained.
    pub(super) fn completion_secs(&self, since: Instant) -> Result<Vec<f64>> {
        for rank in &self.ranks {
            rank.done.record(&rank.stream)?;
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

/// Run `body` between `ncclGroupStart` and `ncclGroupEnd`. The group is
/// always closed — even when `body` fails part-way — so NCCL's group depth
/// never leaks into the next call; the body's error wins over the end's.
fn grouped(what: &str, body: impl FnOnce() -> Result<()>) -> Result<()> {
    group_start().map_err(|error| nccl_error(&format!("{what}: ncclGroupStart"), error))?;
    let issued = body();
    let ended = group_end().map_err(|error| nccl_error(&format!("{what}: ncclGroupEnd"), error));
    issued?;
    ended?;
    Ok(())
}
