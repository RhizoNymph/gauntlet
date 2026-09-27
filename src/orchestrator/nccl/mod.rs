//! Fleet-wide NCCL jobs: world selection, the rendezvous-relay driver, and
//! the two workloads that ride it — the phase-3 collective sweep and the
//! overlap phase's fleet step.
//!
//! World: one NCCL rank per GPU (`layout::RankLayout`). Each NCCL-capable
//! host contributes a contiguous rank block ordered by local GPU index;
//! one `agent nccl` process — one ssh session, one supervised task — per
//! host drives its whole block.
//!
//! The host holding global rank 0 mints the rendezvous id in-process (the
//! id's bootstrap listen socket must live in the process that serves as
//! rank 0) and announces it as an `NcclId` event; the driver intercepts
//! and relays it to every other host, then supervises all host tasks to
//! completion.

mod layout;
mod records;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinSet;
use tracing::{info, warn};

use self::layout::RankLayout;
use self::records::{host_overlap_records, outcomes_with};
use super::barrier::{RankSubject, emit_barrier_metrics};
use super::session::HostSession;
use super::{ObservationSink, gpu_bearing_hosts};
use crate::analysis::skew::{self, Margin, RankSeries, SkewPolarity};
use crate::config::FleetConfig;
use crate::proto::{
    AgentEvent, BarrierSpec, InventorySnapshot, NcclDirective, NcclWorkload, OverlapFleetReport,
    RankAssignment, Scope, TestId, TestOutcome,
};

/// How long the orchestrator waits for the lead rank's NcclId event.
const NCCL_ID_WAIT: Duration = Duration::from_secs(30);

/// The fleet world: NCCL-capable hosts in fleet order, each with its rank
/// block.
type FleetWorld = RankLayout<Arc<HostSession>>;

/// Hosts eligible for a fleet-wide NCCL world, laid out one rank per GPU:
/// GPU-bearing (probing hosts whose inventory is missing) with a loadable
/// libnccl, each contributing its phase-0 GPU count.
///
/// The inventory dlopen probe knows whether libnccl actually loads. A fleet
/// without the NCCL stack skips fleet NCCL work as a structural finding
/// (visible in the inventory/consistency/bootstrap output) instead of
/// manufacturing a host failure out of a loader panic. Hosts predating the
/// probe (empty map) are given the benefit of the doubt.
async fn nccl_world(
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
) -> FleetWorld {
    let gpu_hosts = gpu_bearing_hosts(sessions, inventories).await;
    if gpu_hosts.is_empty() {
        info!("no GPU-bearing hosts; skipping fleet NCCL work");
        return RankLayout::empty();
    }
    let nccl_hosts: Vec<(Arc<HostSession>, u32)> = gpu_hosts
        .iter()
        .filter_map(|session| {
            let inventory = inventories.get(session.addr())?;
            let nccl_loads = inventory.gpu_libs.get("nccl").copied().unwrap_or(true);
            let gpus = u32::try_from(inventory.gpus.len()).unwrap_or(u32::MAX);
            nccl_loads.then(|| (Arc::clone(session), gpus))
        })
        .collect();
    if nccl_hosts.is_empty() {
        warn!(
            gpu_hosts = gpu_hosts.len(),
            "libnccl is not loadable on any GPU-bearing host; skipping fleet NCCL work"
        );
        return RankLayout::empty();
    }
    if nccl_hosts.len() < gpu_hosts.len() {
        warn!(
            with_nccl = nccl_hosts.len(),
            without_nccl = gpu_hosts.len() - nccl_hosts.len(),
            "some GPU-bearing hosts lack a loadable libnccl and are excluded from fleet NCCL work"
        );
    }
    match RankLayout::new(nccl_hosts) {
        Ok(layout) => layout,
        Err(error) => {
            warn!(%error, "cannot lay out the fleet NCCL world; skipping fleet NCCL work");
            RankLayout::empty()
        }
    }
}

/// What one fleet-wide `agent nccl` job runs, beyond the world itself. The
/// lead and participant directives differ only in rendezvous plumbing, so
/// one job builds both.
struct NcclJob {
    socket_ifname: Option<String>,
    workload: NcclWorkload,
}

impl NcclJob {
    fn lead(&self, assignment: RankAssignment) -> NcclDirective {
        NcclDirective::Lead {
            assignment,
            socket_ifname: self.socket_ifname.clone(),
            workload: self.workload.clone(),
        }
    }

    fn participate(&self, unique_id_b64: &str, assignment: RankAssignment) -> NcclDirective {
        NcclDirective::Participate {
            unique_id_b64: unique_id_b64.to_string(),
            assignment,
            socket_ifname: self.socket_ifname.clone(),
            workload: self.workload.clone(),
        }
    }
}

/// How a host-level failure is reported. The sweep marks the host failed
/// (its numbers feed calibration and their absence is a real gap); the
/// fleet overlap step only warns here — the caller records step-level
/// Failed outcomes from the returned failure map, so the results document
/// still shows what happened, without a failed-host verdict.
#[derive(Debug, Clone, Copy)]
enum NcclFailureMode {
    HostError,
    WarnOnly,
}

/// Per-host event filter for a fleet NCCL job: returns `None` to consume an
/// event (merged into job-local state), or the event back to forward it to
/// the collector. `NcclId` relay is handled by the driver itself.
type EventIntercept = Arc<dyn Fn(AgentEvent) -> Option<AgentEvent> + Send + Sync>;

/// Drive one fleet-wide `agent nccl` job over an established world: one
/// process per host, each driving its rank block. Returns the failure map:
/// host address -> why that host's ranks did not complete (process error,
/// timeout, or never being started because the rendezvous id never
/// arrived). An empty map means every host exited cleanly.
async fn drive_fleet_nccl(
    config: &FleetConfig,
    world: &FleetWorld,
    sink: &ObservationSink,
    job: &NcclJob,
    failure: NcclFailureMode,
    intercept: EventIntercept,
) -> BTreeMap<String, String> {
    let mut failures = BTreeMap::new();
    let Some((lead, lead_assignment)) = world.members().first() else {
        return failures;
    };
    let timeout = Duration::from_secs(config.tests.phase_timeout_secs.max(1));
    let mut tasks = JoinSet::new();

    let document = match serde_json::to_string(&job.lead(*lead_assignment)) {
        Ok(document) => document,
        Err(error) => {
            warn!(%error, "cannot serialize the NCCL lead directive");
            let reason = format!("cannot serialize the NCCL lead directive: {error}");
            for (session, _) in world.members() {
                failures.insert(session.addr().to_string(), reason.clone());
            }
            return failures;
        }
    };
    let (id_tx, id_rx) = tokio::sync::oneshot::channel::<String>();
    let id_slot = Arc::new(std::sync::Mutex::new(Some(id_tx)));
    {
        let session = Arc::clone(lead);
        let assignment = *lead_assignment;
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
            let failure = block_failure(assignment, timeout, outcome);
            (addr, failure)
        });
    }

    let unique_id_b64 = match tokio::time::timeout(NCCL_ID_WAIT, id_rx).await {
        Ok(Ok(id)) => Some(id),
        Ok(Err(_)) | Err(_) => {
            // The lead task reports its own failure; just stop recruiting.
            warn!(
                lead = %lead.addr(),
                "NCCL lead produced no rendezvous id; aborting the job"
            );
            // The other hosts never start: record why, so the caller can
            // put something truthful in the results document.
            for (session, _) in world.members().iter().skip(1) {
                failures.insert(
                    session.addr().to_string(),
                    "nccl ranks never started: the lead produced no rendezvous id".to_string(),
                );
            }
            None
        }
    };

    if let Some(unique_id_b64) = &unique_id_b64 {
        for (session, assignment) in world.members().iter().skip(1) {
            let directive = job.participate(unique_id_b64, *assignment);
            let document = match serde_json::to_string(&directive) {
                Ok(document) => document,
                Err(error) => {
                    warn!(%error, host = %session.addr(), "cannot serialize the NCCL directive");
                    failures.insert(
                        session.addr().to_string(),
                        format!("cannot serialize the NCCL directive: {error}"),
                    );
                    continue;
                }
            };
            let session = Arc::clone(session);
            let assignment = *assignment;
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
                let failure = block_failure(assignment, timeout, outcome);
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
                        warn!(host = %addr, message, "fleet nccl host failed")
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

/// Why a host's rank block did not complete cleanly, if it didn't.
fn block_failure(
    assignment: RankAssignment,
    timeout: Duration,
    outcome: Result<anyhow::Result<std::process::ExitStatus>, tokio::time::error::Elapsed>,
) -> Option<String> {
    let block = assignment.block();
    let ranks = format!("nccl ranks {}..{}", block.base(), block.end());
    match outcome {
        Ok(Ok(status)) if status.success() => None,
        Ok(Ok(status)) => Some(format!("{ranks} exited with {status}")),
        Ok(Err(error)) => Some(format!("{ranks} failed: {error:#}")),
        Err(_) => Some(format!("{ranks} timed out after {}s", timeout.as_secs())),
    }
}

/// Fleet-wide NCCL sweep: the lead host's event stream carries the
/// measurements (timed on global rank 0); every rank contributes barrier
/// timings when the barrier-skew benchmark rides along.
pub(super) async fn nccl_sweep(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
    sink: &ObservationSink,
) {
    let world = nccl_world(sessions, inventories).await;
    let Some((lead, _)) = world.members().first() else {
        return;
    };
    let world_size = world.world_size();
    info!(
        world_size,
        hosts = world.member_count(),
        lead = %lead.addr(),
        "NCCL sweep"
    );

    // Barrier-skew microbenchmark rides the same communicator. Skew needs
    // at least two independent arrivals, and the ranks of one host share
    // its launching thread's arrival, so a one-host world has none.
    let barrier_spec =
        (config.tests.barrier_iters > 0 && world.member_count() >= 2).then_some(BarrierSpec {
            iters: config.tests.barrier_iters,
            bytes: config.tests.barrier_bytes,
        });
    // Every rank reports its per-iteration barrier timings; the intercept
    // merges them here so the fleet-wide skew analysis can run once all
    // hosts are in.
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
        match skew::analyze_grouped(
            &series,
            SkewPolarity::LateIsMin,
            Margin::default(),
            |rank| world.arrival_group(rank),
        ) {
            Some(skew) => emit_barrier_metrics(
                sink,
                TestId::NcclBarrier,
                &skew,
                |rank| gpu_subject(&world, rank),
                Some(lead.addr()),
            ),
            None => warn!(
                ranks_reporting = series.len(),
                world_size, "NCCL barrier produced no analyzable timings"
            ),
        }
    }
}

/// A global rank's results-document subject: its host, `Scope::Gpu` with
/// the local GPU index.
fn gpu_subject(world: &FleetWorld, rank: u32) -> Option<RankSubject> {
    world.locate(rank).map(|location| RankSubject {
        host: location.member.addr().to_string(),
        scope: Scope::Gpu {
            index: location.gpu,
        },
    })
}

/// Fleet-wide overlap step: every local GPU on every NCCL-capable host runs
/// the sustained GEMM *and* a rank of the cross-node all-reduce over the
/// real fabric — the straggler signal the intra-node overlap structurally
/// cannot see (GPU<->NIC PCIe contention, GPUDirect degradation under
/// compute load, hot-host network behavior), now on every GPU's path
/// rather than GPU 0's alone. Every rank reports its own results
/// (`OverlapFleetReport`, barrier-timings pattern); this driver merges
/// them into per-GPU metrics.
///
/// Failure semantics: everything short of running lands in the results
/// document as explicit outcomes — the gated skip (< 2 NCCL-capable hosts)
/// records Skipped, a host whose ranks fail, time out, or never report
/// records Failed (per GPU for a silent rank inside a reporting block) —
/// but never a failed-*host* verdict. Only `overlap_fleet = false` leaves
/// no trace at all.
pub(super) async fn overlap_fleet_sweep(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
    sink: &ObservationSink,
) {
    let world = nccl_world(sessions, inventories).await;
    if world.member_count() < 2 {
        info!(
            nccl_hosts = world.member_count(),
            "fewer than two NCCL-capable hosts; skipping the fleet overlap step"
        );
        let reason = format!(
            "fleet overlap needs at least 2 NCCL-capable hosts, found {}",
            world.member_count()
        );
        for (session, _) in world.members() {
            let skipped =
                outcomes_with(Scope::Node, |r| TestOutcome::Skipped { reason: r }, &reason);
            emit_outcomes(sink, session.addr(), skipped);
        }
        return;
    }
    let spec = config.overlap_spec();
    let world_size = world.world_size();
    info!(
        world_size,
        hosts = world.member_count(),
        lead = %world.members()[0].0.addr(),
        "fleet overlap step"
    );

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
        if report.rank >= world_size {
            warn!(
                rank = report.rank,
                world_size, "fleet overlap report outside the world"
            );
            continue;
        }
        if by_rank.insert(report.rank, report).is_some() {
            warn!("duplicate fleet overlap report for a rank; keeping the last");
        }
    }
    // Every host in the world ends up with something in the document: its
    // GPUs' merged reports, or Failed outcomes saying why there are none.
    for (session, assignment) in world.members() {
        let host = session.addr();
        let (metrics, outcomes) = host_overlap_records(
            assignment.block(),
            &by_rank,
            spec.gemm_dtype,
            failures.get(host).map(String::as_str),
        );
        for record in metrics {
            sink.metric(host, record);
        }
        emit_outcomes(sink, host, outcomes);
    }
}

fn emit_outcomes(sink: &ObservationSink, host: &str, outcomes: Vec<records::OutcomeRecord>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{GemmDtype, OverlapSpec};

    fn job() -> NcclJob {
        NcclJob {
            socket_ifname: Some("bond0".into()),
            workload: NcclWorkload::Overlap(OverlapSpec {
                duration_secs: 30,
                baseline_secs: 5,
                gemm_dim: 4096,
                gemm_dtype: GemmDtype::Bf16,
                msg_bytes: 64 << 20,
            }),
        }
    }

    #[test]
    fn directives_carry_each_hosts_block_and_the_world_size() {
        let layout = RankLayout::new([("n1", 8), ("n2", 0), ("n3", 4)]).expect("layout");
        let job = job();
        let members = layout.members();
        let lead = job.lead(members[0].1);
        let participant = job.participate("id", members[1].1);

        let NcclDirective::Lead { assignment, .. } = &lead else {
            panic!("lead directive");
        };
        assert!(assignment.block().holds_lead());
        assert_eq!(assignment.block().count(), 8);
        assert_eq!(assignment.world_size(), 12);

        let NcclDirective::Participate {
            assignment,
            unique_id_b64,
            ..
        } = &participant
        else {
            panic!("participate directive");
        };
        assert_eq!(unique_id_b64, "id");
        assert_eq!(assignment.block().ranks(), 8..12);
        assert_eq!(assignment.world_size(), 12);

        // Both survive the wire.
        for directive in [lead, participant] {
            let json = serde_json::to_string(&directive).expect("serialize");
            let back: NcclDirective = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, directive);
        }
    }

    #[test]
    fn block_failures_name_the_rank_range() {
        let layout = RankLayout::new([("n1", 4), ("n2", 4)]).expect("layout");
        let assignment = layout.members()[1].1;
        let timeout = Duration::from_secs(600);
        assert_eq!(block_failure(assignment, timeout, Ok(Ok(success()))), None);
        let failed = block_failure(assignment, timeout, Ok(Err(anyhow::anyhow!("boom"))))
            .expect("a failure");
        assert!(failed.starts_with("nccl ranks 4..8 failed"), "{failed}");
    }

    fn success() -> std::process::ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        std::process::ExitStatus::from_raw(0)
    }
}
