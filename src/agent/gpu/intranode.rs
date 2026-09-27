//! Intra-node NCCL sweep, GPU half: the node-local communicator
//! (`NodeComm`, shared with the overlap phase) driving the shared sweep
//! loop (`agent::sweep::run_plan`, shared with the fleet sweep). Gating,
//! headline selection, and outcomes live in the GPU-independent
//! `agent::intranode`.

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaSlice};
use cudarc::nccl::ReduceOp;

use super::guard;
use super::node_comm::{NodeComm, nccl_error};
use crate::agent::EventSink;
use crate::agent::intranode::{self, MultiGpuWorld};
use crate::agent::sweep::{
    Collective, SweepCollectives, SweepLevel, SweepPlan, SweepStep, point_records, run_plan,
};
use crate::proto::{LogLevel, NcclSweepSpec};

/// Run the intra-node sweep on this node. Never errors: every failure
/// (driver, communicator, collective, loader panic) becomes a Failed
/// outcome, fewer than two GPUs a Skipped one.
pub fn run(sink: &EventSink, spec: &NcclSweepSpec) -> Result<()> {
    intranode::run_gated(
        sink,
        || guard("cuda driver init", || Ok(CudaContext::device_count()?)).map(|n| n.max(0) as u32),
        |world| {
            sink.log(
                LogLevel::Info,
                format!(
                    "intra-node nccl sweep: {} GPUs, {} sizes",
                    world.get(),
                    spec.sizes.len()
                ),
            );
            guard("intra-node nccl sweep", || execute(sink, spec, world))
        },
    );
    Ok(())
}

fn execute(sink: &EventSink, spec: &NcclSweepSpec, world: MultiGpuWorld) -> Result<()> {
    let node = NodeComm::init(world.get())?;
    let plan = SweepPlan::new(&spec.sizes, world.non_zero());
    let mut collectives = NodeCollectives {
        sends: node.alloc_per_rank(plan.max_elements())?,
        recvs: node.alloc_per_rank(plan.max_elements())?,
        node: &node,
    };
    let mut points = Vec::with_capacity(plan.steps().len());
    run_plan(&mut collectives, &plan, spec.iters_per_size, |point| {
        for record in point_records(SweepLevel::IntraNode, point, world.get()) {
            sink.metric(record);
        }
        points.push(*point);
    })?;
    intranode::emit_summary(sink, &points, world);
    Ok(())
}

/// Every local rank, driven from this thread inside one NCCL group per
/// collective.
struct NodeCollectives<'a> {
    node: &'a NodeComm,
    sends: Vec<CudaSlice<f32>>,
    recvs: Vec<CudaSlice<f32>>,
}

impl SweepCollectives for NodeCollectives<'_> {
    type Error = anyhow::Error;

    fn launch(&mut self, step: &SweepStep) -> Result<()> {
        let sends = &self.sends;
        let recvs = &mut self.recvs;
        self.node.grouped(|rank, comm| {
            let send = sends[rank].slice(0..step.send_elements);
            let mut recv = recvs[rank].slice_mut(0..step.message_elements);
            match step.collective {
                Collective::AllReduce => comm
                    .all_reduce(&send, &mut recv, &ReduceOp::Sum)
                    .map_err(|error| nccl_error("intra-node all_reduce", error))?,
                Collective::AllGather => comm
                    .all_gather(&send, &mut recv)
                    .map_err(|error| nccl_error("intra-node all_gather", error))?,
            };
            Ok(())
        })
    }

    fn sync(&mut self) -> Result<()> {
        self.node.sync_all()
    }
}
