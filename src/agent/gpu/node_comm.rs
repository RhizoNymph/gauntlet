//! Node-local NCCL communicator: one process, one rank per local GPU.
//!
//! Shared by the intra-node overlap phase and the intra-node sweep. One
//! `CudaContext` per GPU, each rank's collectives on its device's default
//! stream, the communicator from `Comm::from_devices` (`ncclCommInitAll`) —
//! no rendezvous, no orchestrator relay, no socket-transport data path.
//! A single thread drives every rank, so each collective round is wrapped
//! in `ncclGroupStart`/`ncclGroupEnd` (NCCL group semantics: without the
//! group, the first rank's call would block waiting for peers that the same
//! thread has not launched yet).

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use cudarc::nccl::result::NcclError;
use cudarc::nccl::{Comm, result as nccl_result};

/// `NcclError` implements neither `Display` nor `std::error::Error`; the raw
/// `ncclResult_t` is the useful part.
pub(crate) fn nccl_error(what: &str, error: NcclError) -> anyhow::Error {
    anyhow!("{what}: {:?}", error.0)
}

pub(crate) struct NodeComm {
    /// One per GPU, index = device ordinal = rank.
    pub contexts: Vec<Arc<CudaContext>>,
    /// Each rank's default stream; carries that rank's collectives.
    pub streams: Vec<Arc<CudaStream>>,
    /// One communicator per rank, same order.
    pub comms: Vec<Comm>,
}

impl NodeComm {
    /// Contexts, streams, and the node-local communicator over devices
    /// `0..device_count`.
    pub fn init(device_count: u32) -> Result<Self> {
        let mut contexts = Vec::with_capacity(device_count as usize);
        let mut streams = Vec::with_capacity(device_count as usize);
        for index in 0..device_count {
            let ctx = CudaContext::new(index as usize)
                .with_context(|| format!("creating cuda context for gpu {index}"))?;
            streams.push(ctx.default_stream());
            contexts.push(ctx);
        }
        let comms = Comm::from_devices(streams.clone())
            .map_err(|error| nccl_error("ncclCommInitAll", error))?;
        Ok(Self {
            contexts,
            streams,
            comms,
        })
    }

    /// One zeroed f32 buffer of `elements` per rank, on that rank's stream.
    pub fn alloc_per_rank(&self, elements: usize) -> Result<Vec<CudaSlice<f32>>> {
        self.streams
            .iter()
            .map(|stream| Ok(stream.alloc_zeros::<f32>(elements)?))
            .collect()
    }

    /// Launch one collective on every rank inside one NCCL group. The
    /// group is always closed, even when a launch fails, so NCCL's group
    /// state never leaks into the next call; the first error wins.
    pub fn grouped(&self, mut launch: impl FnMut(usize, &Comm) -> Result<()>) -> Result<()> {
        nccl_result::group_start().map_err(|error| nccl_error("ncclGroupStart", error))?;
        let mut first_error = None;
        for (rank, comm) in self.comms.iter().enumerate() {
            if let Err(error) = launch(rank, comm) {
                first_error = Some(error);
                break;
            }
        }
        let ended = nccl_result::group_end().map_err(|error| nccl_error("ncclGroupEnd", error));
        match first_error {
            Some(error) => Err(error),
            None => ended.map(|_| ()),
        }
    }

    /// Block until every rank's stream has drained.
    pub fn sync_all(&self) -> Result<()> {
        for stream in &self.streams {
            stream.synchronize()?;
        }
        Ok(())
    }
}
