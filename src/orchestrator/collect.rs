//! Aggregates decoded agent events into per-host reports, then hands the
//! lot to `report::build` for fleet analysis.
//!
//! Concurrency model: per-host driver tasks send `(host_addr, AgentEvent)`
//! over an mpsc channel; the single collector task owns all mutable state
//! (no locks). Pairwise results arrive already attributed
//! (`Scope::HostPair`) by the phase-3 driver before forwarding.

use std::collections::BTreeMap;

use crate::proto::{AgentEvent, InventorySnapshot, MetricRecord, Scope, TestId, TestOutcome};

/// Everything observed about one host during a run.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HostObservations {
    pub inventory: Option<InventorySnapshot>,
    pub metrics: Vec<MetricRecord>,
    pub outcomes: Vec<(TestId, Scope, TestOutcome)>,
    /// Fatal events, transport failures, phase timeouts.
    pub errors: Vec<String>,
}

#[derive(Debug, Default)]
pub struct Collector {
    hosts: BTreeMap<String, HostObservations>,
}

impl Collector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ingest(&mut self, host_addr: &str, event: AgentEvent) {
        let _ = (host_addr, event);
        todo!("agent A: implement")
    }

    /// Record a host-level failure not carried by an event (ssh transport
    /// error, phase timeout).
    pub fn host_error(&mut self, host_addr: &str, error: String) {
        let _ = (host_addr, error);
        todo!("agent A: implement")
    }

    pub fn into_observations(self) -> BTreeMap<String, HostObservations> {
        self.hosts
    }
}
