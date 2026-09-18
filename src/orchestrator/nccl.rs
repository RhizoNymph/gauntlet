//! Fleet-wide NCCL jobs: world selection, the rendezvous-relay driver, and
//! the two workloads that ride it — the phase-3 collective sweep and the
//! overlap phase's fleet step.
//!
//! Rank 0 mints the rendezvous id in-process (the id's bootstrap listen
//! socket must live in the process that serves as rank 0) and announces it
//! as an `NcclId` event; the driver intercepts and relays it to every other
//! rank, then supervises all rank tasks to completion.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinSet;
use tracing::{info, warn};

use super::session::HostSession;
use super::{ObservationSink, emit_barrier_metrics, gpu_bearing_hosts};
use crate::analysis::skew::{self, Margin, RankSeries, SkewPolarity};
use crate::config::FleetConfig;
use crate::proto::{
    AgentEvent, BarrierSpec, GemmDtype, InventorySnapshot, MetricRecord, NcclDirective,
    NcclWorkload, OverlapFleetReport, OverlapGpuGemm, Scope, TestId, TestOutcome, Unit,
    overlap_metric,
};

/// How long the orchestrator waits for the lead rank's NcclId event.
const NCCL_ID_WAIT: Duration = Duration::from_secs(30);

/// Hosts eligible for a fleet-wide NCCL world, in fleet order: GPU-bearing
/// (probing hosts whose inventory is missing) with a loadable libnccl.
///
/// The inventory dlopen probe knows whether libnccl actually loads. A fleet
/// without the NCCL stack skips fleet NCCL work as a structural finding
/// (visible in the inventory/consistency/bootstrap output) instead of
/// manufacturing a host failure out of a loader panic. Hosts predating the
/// probe (empty map) are given the benefit of the doubt.
async fn nccl_world(
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
) -> Vec<Arc<HostSession>> {
    let gpu_hosts = gpu_bearing_hosts(sessions, inventories).await;
    if gpu_hosts.is_empty() {
        info!("no GPU-bearing hosts; skipping fleet NCCL work");
        return Vec::new();
    }
    let nccl_hosts: Vec<_> = gpu_hosts
        .iter()
        .filter(|session| {
            inventories
                .get(session.addr())
                .map(|inv| inv.gpu_libs.get("nccl").copied().unwrap_or(true))
                .unwrap_or(true)
        })
        .cloned()
        .collect();
    if nccl_hosts.is_empty() {
        warn!(
            gpu_hosts = gpu_hosts.len(),
            "libnccl is not loadable on any GPU-bearing host; skipping fleet NCCL work"
        );
        return Vec::new();
    }
    if nccl_hosts.len() < gpu_hosts.len() {
        warn!(
            with_nccl = nccl_hosts.len(),
            without_nccl = gpu_hosts.len() - nccl_hosts.len(),
            "some GPU-bearing hosts lack a loadable libnccl and are excluded from fleet NCCL work"
        );
    }
    nccl_hosts
}

/// What one fleet-wide `agent nccl` job runs, beyond the world itself. The
/// lead and participant directives differ only in rendezvous plumbing, so
/// one job builds both.
struct NcclJob {
    socket_ifname: Option<String>,
    workload: NcclWorkload,
}

impl NcclJob {
    fn lead(&self, world_size: u32) -> NcclDirective {
        NcclDirective::Lead {
            world_size,
            socket_ifname: self.socket_ifname.clone(),
            workload: self.workload.clone(),
        }
    }

    fn participate(&self, unique_id_b64: &str, rank: u32, world_size: u32) -> NcclDirective {
        NcclDirective::Participate {
            unique_id_b64: unique_id_b64.to_string(),
            rank,
            world_size,
            socket_ifname: self.socket_ifname.clone(),
            workload: self.workload.clone(),
        }
    }
}

/// How a rank-level failure is reported. The sweep marks the host failed
/// (its numbers feed calibration and their absence is a real gap); the
/// fleet overlap step only warns here — the caller records step-level
/// Failed outcomes from the returned failure map, so the results document
/// still shows what happened, without a failed-host verdict.
#[derive(Debug, Clone, Copy)]
enum NcclFailureMode {
    HostError,
    WarnOnly,
}

/// Per-rank event filter for a fleet NCCL job: returns `None` to consume an
/// event (merged into job-local state), or the event back to forward it to
/// the collector. `NcclId` relay is handled by the driver itself.
type EventIntercept = Arc<dyn Fn(AgentEvent) -> Option<AgentEvent> + Send + Sync>;

/// Drive one fleet-wide `agent nccl` job over an established world. Returns
/// the failure map: host address -> why that rank did not complete (rank
/// error, timeout, or never being started because the rendezvous id never
/// arrived). An empty map means every rank exited cleanly.
async fn drive_fleet_nccl(
    config: &FleetConfig,
    world: &[Arc<HostSession>],
    sink: &ObservationSink,
    job: &NcclJob,
    failure: NcclFailureMode,
    intercept: EventIntercept,
) -> BTreeMap<String, String> {
    let mut failures = BTreeMap::new();
    let Some(rank0) = world.first() else {
        return failures;
    };
    let world_size = world.len() as u32;
    let timeout = Duration::from_secs(config.tests.phase_timeout_secs.max(1));
    let mut tasks = JoinSet::new();

    let document = match serde_json::to_string(&job.lead(world_size)) {
        Ok(document) => document,
        Err(error) => {
            warn!(%error, "cannot serialize the NCCL lead directive");
            let reason = format!("cannot serialize the NCCL lead directive: {error}");
            for session in world {
                failures.insert(session.addr().to_string(), reason.clone());
            }
            return failures;
        }
    };
    let (id_tx, id_rx) = tokio::sync::oneshot::channel::<String>();
    let id_slot = Arc::new(std::sync::Mutex::new(Some(id_tx)));
    {
        let session = Arc::clone(rank0);
        let sink = sink.clone();
        let id_slot = Arc::clone(&id_slot);
        let intercept = Arc::clone(&intercept);
        tasks.spawn(async move {
            let addr = session.addr().to_string();
            let outcome = tokio::time::timeout(
                timeout,
                session.run_agent(&["nccl"], Some(document), |event| match event {
                    AgentEvent::NcclId { unique_id_b64 } => {
                        if let Some(tx) = id_slot.lock().expect("id slot poisoned").take() {
                            let _ = tx.send(unique_id_b64);
                        }
                    }
                    event => {
                        if let Some(event) = intercept(event) {
                            sink.event(&addr, event);
                        }
                    }
                }),
            )
            .await;
            let failure = rank_failure(0, timeout, outcome);
            (addr, failure)
        });
    }

    let unique_id_b64 = match tokio::time::timeout(NCCL_ID_WAIT, id_rx).await {
        Ok(Ok(id)) => Some(id),
        Ok(Err(_)) | Err(_) => {
            // The lead task reports its own failure; just stop recruiting.
            warn!(
                rank0 = %rank0.addr(),
                "NCCL lead produced no rendezvous id; aborting the job"
            );
            // The other ranks never start: record why, so the caller can
            // put something truthful in the results document.
            for session in world.iter().skip(1) {
                failures.insert(
                    session.addr().to_string(),
                    "nccl rank never started: the lead produced no rendezvous id".to_string(),
                );
            }
            None
        }
    };

    if let Some(unique_id_b64) = &unique_id_b64 {
        for (rank, session) in world.iter().enumerate().skip(1) {
            let directive = job.participate(unique_id_b64, rank as u32, world_size);
            let document = match serde_json::to_string(&directive) {
                Ok(document) => document,
                Err(error) => {
                    warn!(%error, rank, "cannot serialize the NCCL directive");
                    failures.insert(
                        session.addr().to_string(),
                        format!("cannot serialize the NCCL directive: {error}"),
                    );
                    continue;
                }
            };
            let session = Arc::clone(session);
            let sink = sink.clone();
            let intercept = Arc::clone(&intercept);
            tasks.spawn(async move {
                let addr = session.addr().to_string();
                let outcome = tokio::time::timeout(
                    timeout,
                    session.run_agent(&["nccl"], Some(document), |event| {
                        if let Some(event) = intercept(event) {
                            sink.event(&addr, event);
                        }
                    }),
                )
                .await;
                let failure = rank_failure(rank, timeout, outcome);
                (addr, failure)
            });
        }
    }
    while let Some(joined) = tasks.join_next().await {
        match joined {
            Ok((addr, Some(message))) => {
                match failure {
                    NcclFailureMode::HostError => sink.error(&addr, message.clone()),
                    NcclFailureMode::WarnOnly => {
                        warn!(host = %addr, message, "fleet nccl rank failed")
                    }
                }
                failures.insert(addr, message);
            }
            Ok((_, None)) => {}
            Err(error) => warn!(%error, "nccl task did not complete"),
        }
    }
    failures
}

/// Why a rank task did not complete cleanly, if it didn't.
fn rank_failure(
    rank: usize,
    timeout: Duration,
    outcome: Result<anyhow::Result<std::process::ExitStatus>, tokio::time::error::Elapsed>,
) -> Option<String> {
    match outcome {
        Ok(Ok(status)) if status.success() => None,
        Ok(Ok(status)) => Some(format!("nccl rank {rank} exited with {status}")),
        Ok(Err(error)) => Some(format!("nccl rank {rank} failed: {error:#}")),
        Err(_) => Some(format!(
            "nccl rank {rank} timed out after {}s",
            timeout.as_secs()
        )),
    }
}

/// Fleet-wide NCCL sweep: rank 0's event stream carries the measurements;
/// every rank contributes barrier timings when the barrier-skew benchmark
/// rides along.
pub(super) async fn nccl_sweep(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
    sink: &ObservationSink,
) {
    let world = nccl_world(sessions, inventories).await;
    let Some(rank0) = world.first() else {
        return;
    };
    let world_size = world.len() as u32;
    info!(world_size, rank0 = %rank0.addr(), "NCCL sweep");

    // Barrier-skew microbenchmark rides the same communicator; a world of
    // one has no skew to measure.
    let barrier_spec = (config.tests.barrier_iters > 0 && world_size >= 2).then_some(BarrierSpec {
        iters: config.tests.barrier_iters,
        bytes: config.tests.barrier_bytes,
    });
    // Every rank reports its per-iteration barrier timings; the intercept
    // merges them here so the fleet-wide skew analysis can run once all
    // ranks are in.
    let barrier_timings: Arc<std::sync::Mutex<Vec<RankSeries>>> = Arc::default();
    let intercept: EventIntercept = {
        let barrier_timings = Arc::clone(&barrier_timings);
        Arc::new(move |event| match event {
            AgentEvent::NcclBarrierTimings { rank, elapsed_us } => {
                barrier_timings
                    .lock()
                    .expect("barrier timings poisoned")
                    .push(RankSeries { rank, elapsed_us });
                None
            }
            event => Some(event),
        })
    };
    let job = NcclJob {
        socket_ifname: config.nccl.socket_ifname.clone(),
        workload: NcclWorkload::Sweep {
            sizes: config.tests.nccl_sizes.clone(),
            iters_per_size: config.tests.nccl_iters_per_size,
            barrier: barrier_spec,
        },
    };
    drive_fleet_nccl(
        config,
        &world,
        sink,
        &job,
        NcclFailureMode::HostError,
        intercept,
    )
    .await;

    if barrier_spec.is_some() {
        let series =
            std::mem::take(&mut *barrier_timings.lock().expect("barrier timings poisoned"));
        let rank_hosts: Vec<String> = world
            .iter()
            .map(|session| session.addr().to_string())
            .collect();
        match skew::analyze(&series, SkewPolarity::LateIsMin, Margin::default()) {
            Some(skew) => emit_barrier_metrics(sink, TestId::NcclBarrier, &rank_hosts, &skew),
            None => warn!(
                ranks_reporting = series.len(),
                world_size, "NCCL barrier produced no analyzable timings"
            ),
        }
    }
}

/// Fleet-wide overlap step: every local GPU on every NCCL-capable host runs
/// the sustained GEMM while one rank per node drives a cross-node
/// all-reduce over the real fabric — the straggler signal the intra-node
/// overlap structurally cannot see (GPU<->NIC PCIe contention, GPUDirect
/// degradation under compute load, hot-host network behavior). Every rank
/// reports its own results (`OverlapFleetReport`, barrier-timings pattern);
/// this driver merges them into per-host metrics.
///
/// Failure semantics: everything short of running lands in the results
/// document as explicit outcomes — the gated skip (< 2 NCCL-capable hosts)
/// records Skipped, a rank that fails, times out, or never reports records
/// Failed against that host — but never a failed-*host* verdict. Only
/// `overlap_fleet = false` leaves no trace at all.
pub(super) async fn overlap_fleet_sweep(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
    sink: &ObservationSink,
) {
    let world = nccl_world(sessions, inventories).await;
    if world.len() < 2 {
        info!(
            nccl_hosts = world.len(),
            "fewer than two NCCL-capable hosts; skipping the fleet overlap step"
        );
        let reason = format!(
            "fleet overlap needs at least 2 NCCL-capable hosts, found {}",
            world.len()
        );
        for session in &world {
            step_outcomes(
                sink,
                session.addr(),
                |r| TestOutcome::Skipped { reason: r },
                &reason,
            );
        }
        return;
    }
    let spec = config.overlap_spec();
    let world_size = world.len() as u32;
    info!(world_size, rank0 = %world[0].addr(), "fleet overlap step");

    let reports: Arc<std::sync::Mutex<Vec<OverlapFleetReport>>> = Arc::default();
    let intercept: EventIntercept = {
        let reports = Arc::clone(&reports);
        Arc::new(move |event| match event {
            AgentEvent::OverlapFleetReport { report } => {
                reports
                    .lock()
                    .expect("overlap reports poisoned")
                    .push(*report);
                None
            }
            event => Some(event),
        })
    };
    let job = NcclJob {
        socket_ifname: config.nccl.socket_ifname.clone(),
        workload: NcclWorkload::Overlap(spec),
    };
    let failures = drive_fleet_nccl(
        config,
        &world,
        sink,
        &job,
        NcclFailureMode::WarnOnly,
        intercept,
    )
    .await;

    let reports = std::mem::take(&mut *reports.lock().expect("overlap reports poisoned"));
    if reports.is_empty() {
        warn!(world_size, "fleet overlap step produced no reports");
    }
    let mut by_rank: BTreeMap<u32, OverlapFleetReport> = BTreeMap::new();
    for report in reports {
        if by_rank.insert(report.rank, report).is_some() {
            warn!("duplicate fleet overlap report for a rank; keeping the last");
        }
    }
    // Every host in the world ends up with something in the document: the
    // rank's merged report, or a Failed outcome saying why there is none.
    for (rank, session) in world.iter().enumerate() {
        let host = session.addr();
        match by_rank.remove(&(rank as u32)) {
            Some(report) => {
                let (metrics, outcomes) = fleet_overlap_records(&report, spec.gemm_dtype);
                for record in metrics {
                    sink.metric(host, record);
                }
                for (test, scope, outcome) in outcomes {
                    sink.event(
                        host,
                        AgentEvent::Outcome {
                            test,
                            scope,
                            outcome,
                        },
                    );
                }
            }
            None => {
                let reason = failures
                    .get(host)
                    .cloned()
                    .unwrap_or_else(|| "rank produced no fleet overlap report".to_string());
                step_outcomes(sink, host, |r| TestOutcome::Failed { reason: r }, &reason);
            }
        }
    }
}

/// Node-scope outcomes for the fleet overlap step against one host,
/// mirroring the intra-node phase's `node_outcomes`: used for the gated
/// skip and for ranks that failed or never reported.
fn step_outcomes(
    sink: &ObservationSink,
    host: &str,
    outcome: impl Fn(String) -> TestOutcome,
    reason: &str,
) {
    for test in [TestId::OverlapFleetGemm, TestId::OverlapFleetAllReduce] {
        sink.event(
            host,
            AgentEvent::Outcome {
                test,
                scope: Scope::Node,
                outcome: outcome(reason.to_string()),
            },
        );
    }
}

/// Metric and outcome records for one rank's fleet-overlap report,
/// mirroring the intra-node overlap phase: bus numbers under `Scope::Node`,
/// per-GPU GEMM under `Scope::Gpu`, one GPU's failure hiding nothing else.
fn fleet_overlap_records(
    report: &OverlapFleetReport,
    dtype: GemmDtype,
) -> (Vec<MetricRecord>, Vec<(TestId, Scope, TestOutcome)>) {
    let mut metrics: Vec<MetricRecord> = [
        (
            overlap_metric::MSG_BYTES,
            report.msg_bytes as f64,
            Unit::Bytes,
        ),
        (
            overlap_metric::ISOLATED_BUS,
            report.isolated_bus_gib_per_sec,
            Unit::GibPerSec,
        ),
        (
            overlap_metric::OVERLAP_BUS,
            report.overlap_bus_gib_per_sec,
            Unit::GibPerSec,
        ),
    ]
    .into_iter()
    .map(|(name, value, unit)| MetricRecord {
        test: TestId::OverlapFleetAllReduce,
        scope: Scope::Node,
        name: name.to_string(),
        value,
        unit,
        repeat: 0,
    })
    .collect();
    let mut outcomes = vec![(
        TestId::OverlapFleetAllReduce,
        Scope::Node,
        TestOutcome::Passed,
    )];

    let tag = dtype.tag();
    for gemm in &report.gemm {
        match gemm {
            OverlapGpuGemm::Ok { gpu_index, gflops } => {
                metrics.push(MetricRecord {
                    test: TestId::OverlapFleetGemm,
                    scope: Scope::Gpu { index: *gpu_index },
                    name: format!("gflops_{tag}"),
                    value: *gflops,
                    unit: Unit::Gflops,
                    repeat: 0,
                });
                outcomes.push((
                    TestId::OverlapFleetGemm,
                    Scope::Gpu { index: *gpu_index },
                    TestOutcome::Passed,
                ));
            }
            OverlapGpuGemm::Failed { gpu_index, reason } => {
                outcomes.push((
                    TestId::OverlapFleetGemm,
                    Scope::Gpu { index: *gpu_index },
                    TestOutcome::Failed {
                        reason: reason.clone(),
                    },
                ));
            }
        }
    }
    (metrics, outcomes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fleet_overlap_records_cover_bus_and_per_gpu_gemm() {
        let report = OverlapFleetReport {
            rank: 1,
            msg_bytes: 64 << 20,
            isolated_bus_gib_per_sec: 40.0,
            overlap_bus_gib_per_sec: 30.0,
            gemm: vec![
                OverlapGpuGemm::Ok {
                    gpu_index: 0,
                    gflops: 88_000.0,
                },
                OverlapGpuGemm::Failed {
                    gpu_index: 1,
                    reason: "worker panicked".into(),
                },
            ],
        };
        let (metrics, outcomes) = fleet_overlap_records(&report, GemmDtype::Bf16);

        let names: Vec<&str> = metrics.iter().map(|record| record.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "msg_bytes",
                "isolated_bus_gib_per_sec",
                "overlap_bus_gib_per_sec",
                "gflops_bf16",
            ]
        );
        for record in &metrics[..3] {
            assert_eq!(record.test, TestId::OverlapFleetAllReduce);
            assert_eq!(record.scope, Scope::Node);
        }
        assert_eq!(metrics[3].test, TestId::OverlapFleetGemm);
        assert_eq!(metrics[3].scope, Scope::Gpu { index: 0 });
        assert_eq!(metrics[3].value, 88_000.0);

        // One outcome for the collective, one per GPU; the failed GPU is a
        // Failed outcome (a finding), never a missing entry.
        assert_eq!(outcomes.len(), 3);
        assert_eq!(
            outcomes[0],
            (
                TestId::OverlapFleetAllReduce,
                Scope::Node,
                TestOutcome::Passed
            )
        );
        assert_eq!(
            outcomes[1],
            (
                TestId::OverlapFleetGemm,
                Scope::Gpu { index: 0 },
                TestOutcome::Passed
            )
        );
        assert_eq!(
            outcomes[2],
            (
                TestId::OverlapFleetGemm,
                Scope::Gpu { index: 1 },
                TestOutcome::Failed {
                    reason: "worker panicked".into()
                }
            )
        );
    }
}
