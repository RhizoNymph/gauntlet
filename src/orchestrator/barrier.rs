//! Barrier-skew metric attribution, shared by the NCCL probe (one rank per
//! GPU: `host:gpuN` subjects) and the TCP star barrier (one rank per host:
//! node subjects).

use std::collections::BTreeSet;

use tracing::warn;

use super::ObservationSink;
use crate::analysis::skew::BarrierSkew;
use crate::proto::{MetricRecord, Scope, TestId, Unit};

/// Who a barrier rank is in the results document: its host and the scope
/// under that host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RankSubject {
    pub host: String,
    pub scope: Scope,
}

/// Turn a barrier-skew analysis into `(host, record)` pairs: per-rank
/// distribution and tally metrics against each rank's subject (`locate`),
/// fleet-level barrier-time distribution against `fleet_host` under
/// `Scope::Node` (the coordinator / the host holding global rank 0),
/// mirroring how the NCCL sweeps attribute fleet-wide numbers to rank 0.
/// A rank `locate` cannot place is logged and skipped.
///
/// Granularity: the distribution metrics (`p50_us`..`max_us`) are per
/// rank, under the rank's own subject scope (`host:gpuN` for NCCL, where
/// each GPU's stream completes at its own time). The tally metrics
/// (`slowest_frac`, `slowest_considered`) are per *host*, under
/// `Scope::Node`, emitted once: a host's ranks form one arrival group and
/// share one tally, so per-GPU copies would weight hosts by GPU count in
/// the MAD analysis and turn one late host into N identical straggler
/// rows.
///
/// Metric semantics differ by polarity — under `NcclBarrier` the per-rank
/// values are local waits (a straggler is a *low* outlier), under
/// `TcpBarrier` they are release-to-response times (a straggler is a *high*
/// outlier) — but `slowest_frac` always means "fraction of considered
/// iterations this subject's arrival group was the late arriver", which is
/// what the report's flagging rule consumes.
pub(super) fn barrier_records(
    test: TestId,
    skew: &BarrierSkew,
    locate: impl Fn(u32) -> Option<RankSubject>,
    fleet_host: Option<&str>,
) -> Vec<(String, MetricRecord)> {
    let mut records = Vec::new();
    let mut tallied_hosts: BTreeSet<String> = BTreeSet::new();
    for rank in &skew.per_rank {
        let Some(subject) = locate(rank.rank) else {
            warn!(rank = rank.rank, "barrier rank has no host");
            continue;
        };
        for (name, value) in [
            ("p50_us", rank.p50_us),
            ("p90_us", rank.p90_us),
            ("p99_us", rank.p99_us),
            ("max_us", rank.max_us),
        ] {
            records.push((
                subject.host.clone(),
                record(test, subject.scope.clone(), name, value, Unit::Micros),
            ));
        }
        // Once per host: every rank of a host carries the same tally.
        if tallied_hosts.insert(subject.host.clone()) {
            for (name, value, unit) in [
                ("slowest_frac", rank.slowest_frac, Unit::Ratio),
                (
                    "slowest_considered",
                    skew.considered_iters as f64,
                    Unit::Count,
                ),
            ] {
                records.push((
                    subject.host.clone(),
                    record(test, Scope::Node, name, value, unit),
                ));
            }
        }
    }
    if let Some(host) = fleet_host {
        for (name, value) in [
            ("fleet_span_p50_us", skew.fleet.p50_us),
            ("fleet_span_p90_us", skew.fleet.p90_us),
            ("fleet_span_p99_us", skew.fleet.p99_us),
            ("fleet_span_max_us", skew.fleet.max_us),
        ] {
            records.push((
                host.to_string(),
                MetricRecord {
                    test,
                    scope: Scope::Node,
                    name: name.to_string(),
                    value,
                    unit: Unit::Micros,
                    repeat: 0,
                },
            ));
        }
    }
    records
}

fn record(test: TestId, scope: Scope, name: &str, value: f64, unit: Unit) -> MetricRecord {
    MetricRecord {
        test,
        scope,
        name: name.to_string(),
        value,
        unit,
        repeat: 0,
    }
}

pub(super) fn emit_barrier_metrics(
    sink: &ObservationSink,
    test: TestId,
    skew: &BarrierSkew,
    locate: impl Fn(u32) -> Option<RankSubject>,
    fleet_host: Option<&str>,
) {
    for (host, record) in barrier_records(test, skew, locate, fleet_host) {
        sink.metric(&host, record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::skew::{FleetBarrier, RankSkew};

    fn skew(ranks: u32) -> BarrierSkew {
        BarrierSkew {
            iters: 2000,
            considered_iters: 1500,
            per_rank: (0..ranks)
                .map(|rank| RankSkew {
                    rank,
                    p50_us: 10.0 + f64::from(rank),
                    p90_us: 20.0,
                    p99_us: 30.0,
                    max_us: 40.0,
                    slowest_iters: 0,
                    slowest_frac: 0.0,
                })
                .collect(),
            fleet: FleetBarrier {
                p50_us: 100.0,
                p90_us: 110.0,
                p99_us: 120.0,
                max_us: 130.0,
            },
        }
    }

    #[test]
    fn per_gpu_subjects_carry_the_local_gpu_scope() {
        // Two hosts x two GPUs: rank -> (host, gpu).
        let locate = |rank: u32| {
            (rank < 4).then(|| RankSubject {
                host: if rank < 2 { "n1" } else { "n2" }.to_string(),
                scope: Scope::Gpu { index: rank % 2 },
            })
        };
        let records = barrier_records(TestId::NcclBarrier, &skew(4), locate, Some("n1"));
        let p50: Vec<(String, Scope, f64)> = records
            .iter()
            .filter(|(_, record)| record.name == "p50_us")
            .map(|(host, record)| (host.clone(), record.scope.clone(), record.value))
            .collect();
        assert_eq!(
            p50,
            [
                ("n1".to_string(), Scope::Gpu { index: 0 }, 10.0),
                ("n1".to_string(), Scope::Gpu { index: 1 }, 11.0),
                ("n2".to_string(), Scope::Gpu { index: 0 }, 12.0),
                ("n2".to_string(), Scope::Gpu { index: 1 }, 13.0),
            ]
        );
        // Fleet span stays one node-scope series on the lead host.
        let spans: Vec<&(String, MetricRecord)> = records
            .iter()
            .filter(|(_, record)| record.name.starts_with("fleet_span"))
            .collect();
        assert_eq!(spans.len(), 4);
        assert!(
            spans
                .iter()
                .all(|(host, record)| host == "n1" && record.scope == Scope::Node)
        );
    }

    #[test]
    fn tally_metrics_are_emitted_once_per_host_at_node_scope() {
        let locate = |rank: u32| {
            (rank < 4).then(|| RankSubject {
                host: if rank < 2 { "n1" } else { "n2" }.to_string(),
                scope: Scope::Gpu { index: rank % 2 },
            })
        };
        let records = barrier_records(TestId::NcclBarrier, &skew(4), locate, None);
        for name in ["slowest_frac", "slowest_considered"] {
            let tally: Vec<(&str, &Scope)> = records
                .iter()
                .filter(|(_, record)| record.name == name)
                .map(|(host, record)| (host.as_str(), &record.scope))
                .collect();
            assert_eq!(
                tally,
                [("n1", &Scope::Node), ("n2", &Scope::Node)],
                "{name}"
            );
        }
        // Distribution metrics stay one per GPU.
        assert_eq!(
            records
                .iter()
                .filter(|(_, record)| record.name == "p99_us")
                .count(),
            4
        );
    }

    #[test]
    fn unplaceable_ranks_are_skipped_not_misattributed() {
        let locate = |rank: u32| {
            (rank == 0).then(|| RankSubject {
                host: "n1".to_string(),
                scope: Scope::Node,
            })
        };
        let records = barrier_records(TestId::TcpBarrier, &skew(2), locate, None);
        assert_eq!(
            records.len(),
            6,
            "rank 0's four percentiles plus its host tally"
        );
        assert!(records.iter().all(|(host, _)| host == "n1"));
    }
}
