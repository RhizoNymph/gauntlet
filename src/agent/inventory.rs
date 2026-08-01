//! Phase 0: hardware/software inventory and health counters.
//!
//! Sources: /proc (cpuinfo, meminfo, sys/kernel), /sys (numa nodes, NIC mtu
//! and speed), `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`
//! (absent nvidia-smi means "no GPUs", not an error), `ibstat`-equivalent
//! sysfs under /sys/class/infiniband, `chronyc tracking` (best effort),
//! `dmesg`-sourced Xid scan via /var/log/kern.log or journalctl (best
//! effort; missing permissions degrade to empty, never to failure).

use anyhow::Result;

use crate::agent::EventSink;
use crate::proto::{AgentEvent, InventorySnapshot};

/// Collect the snapshot and emit it as an event.
pub fn run(sink: &EventSink) -> Result<()> {
    let snapshot = collect()?;
    sink.emit(&AgentEvent::Inventory { snapshot });
    Ok(())
}

/// Collect an inventory snapshot of this node. Individual probes are best
/// effort: a missing tool or permission yields `None`/empty fields, never an
/// error. Only a totally unreadable /proc fails.
pub fn collect() -> Result<InventorySnapshot> {
    todo!("agent B: implement")
}
