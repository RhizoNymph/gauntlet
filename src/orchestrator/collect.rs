//! Aggregates decoded agent events into per-host reports, then hands the
//! lot to `report::build` for fleet analysis.
//!
//! Concurrency model: per-host driver tasks send `(host_addr, AgentEvent)`
//! over an mpsc channel; the single collector task owns all mutable state
//! (no locks). Pairwise results arrive already attributed
//! (`Scope::HostPair`) by the phase-3 driver before forwarding.

use std::collections::BTreeMap;

use tracing::{debug, error, info, warn};

use crate::proto::{
    AgentEvent, InventorySnapshot, LogLevel, MetricRecord, Scope, TestId, TestOutcome,
};

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
        // Every host that says anything at all gets an entry, so a host that
        // only ever failed still appears in the report.
        let host = self.hosts.entry(host_addr.to_string()).or_default();
        match event {
            AgentEvent::Hello {
                proto_version,
                hostname,
            } => {
                debug!(host = host_addr, proto_version, hostname, "agent hello");
            }
            AgentEvent::PhaseStart { phase } => {
                debug!(host = host_addr, ?phase, "phase start");
            }
            AgentEvent::PhaseEnd { phase } => {
                debug!(host = host_addr, ?phase, "phase end");
            }
            AgentEvent::Inventory { snapshot } => {
                host.inventory = Some(snapshot);
            }
            AgentEvent::Metric { record } => {
                host.metrics.push(record);
            }
            AgentEvent::Outcome {
                test,
                scope,
                outcome,
            } => {
                host.outcomes.push((test, scope, outcome));
            }
            AgentEvent::Log { level, message } => match level {
                LogLevel::Error => error!(host = host_addr, message, "agent"),
                LogLevel::Warn => warn!(host = host_addr, message, "agent"),
                LogLevel::Info => info!(host = host_addr, message, "agent"),
                LogLevel::Debug => debug!(host = host_addr, message, "agent"),
            },
            AgentEvent::Fatal { message } => {
                error!(host = host_addr, message, "agent fatal");
                host.errors.push(message);
            }
        }
    }

    /// Record a host-level failure not carried by an event (ssh transport
    /// error, phase timeout).
    pub fn host_error(&mut self, host_addr: &str, error: String) {
        error!(host = host_addr, error, "host failed");
        self.hosts
            .entry(host_addr.to_string())
            .or_default()
            .errors
            .push(error);
    }

    pub fn into_observations(self) -> BTreeMap<String, HostObservations> {
        self.hosts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::Unit;

    fn metric(name: &str) -> MetricRecord {
        MetricRecord {
            test: TestId::CpuGflops,
            scope: Scope::Node,
            name: name.to_string(),
            value: 1.0,
            unit: Unit::Gflops,
        }
    }

    fn snapshot() -> InventorySnapshot {
        InventorySnapshot {
            hostname: "node-a".into(),
            kernel: "6.1.0".into(),
            cpu_model: "test".into(),
            logical_cores: 8,
            numa_nodes: 1,
            mem_total_bytes: 1 << 34,
            cpu_governor: Some("performance".into()),
            clock_offset_ms: Some(0.2),
            nvidia_driver: None,
            cuda_version: None,
            gpus: Vec::new(),
            nics: Vec::new(),
            ib_ports: Vec::new(),
            xid_errors: Vec::new(),
        }
    }

    #[test]
    fn events_route_to_their_buckets() {
        let mut collector = Collector::new();
        collector.ingest(
            "a",
            AgentEvent::Inventory {
                snapshot: snapshot(),
            },
        );
        collector.ingest(
            "a",
            AgentEvent::Metric {
                record: metric("gflops"),
            },
        );
        collector.ingest(
            "a",
            AgentEvent::Outcome {
                test: TestId::CpuCorrectness,
                scope: Scope::Core { id: 3 },
                outcome: TestOutcome::Passed,
            },
        );
        collector.ingest(
            "a",
            AgentEvent::Fatal {
                message: "gpu 0 fell off the bus".into(),
            },
        );

        let observations = collector.into_observations();
        let host = observations.get("a").expect("host recorded");
        assert_eq!(
            host.inventory.as_ref().map(|inv| inv.hostname.as_str()),
            Some("node-a")
        );
        assert_eq!(host.metrics.len(), 1);
        assert_eq!(host.outcomes.len(), 1);
        assert_eq!(host.errors, vec!["gpu 0 fell off the bus".to_string()]);
    }

    #[test]
    fn framing_and_log_events_do_not_accumulate_state() {
        let mut collector = Collector::new();
        collector.ingest(
            "a",
            AgentEvent::Hello {
                proto_version: crate::proto::PROTO_VERSION,
                hostname: "node-a".into(),
            },
        );
        collector.ingest(
            "a",
            AgentEvent::PhaseStart {
                phase: crate::proto::Phase::Gpu,
            },
        );
        collector.ingest(
            "a",
            AgentEvent::Log {
                level: LogLevel::Warn,
                message: "clocks throttled".into(),
            },
        );
        collector.ingest(
            "a",
            AgentEvent::PhaseEnd {
                phase: crate::proto::Phase::Gpu,
            },
        );

        let observations = collector.into_observations();
        let host = observations.get("a").expect("host recorded even so");
        assert_eq!(host, &HostObservations::default());
    }

    #[test]
    fn host_errors_accumulate_per_host() {
        let mut collector = Collector::new();
        collector.host_error("a", "connect refused".into());
        collector.host_error("a", "phase gpu timed out".into());
        collector.host_error("b", "connect refused".into());

        let observations = collector.into_observations();
        assert_eq!(observations["a"].errors.len(), 2);
        assert_eq!(observations["b"].errors.len(), 1);
    }
}
