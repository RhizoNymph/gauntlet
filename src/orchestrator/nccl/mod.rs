//! Fleet-wide NCCL jobs: world selection, the rendezvous-relay driver, and
//! the workloads that ride it — the phase-3 collective sweep (`sweep`,
//! with the barrier-skew probe) and the overlap phase's fleet step.
//!
//! World: by default one NCCL rank per GPU (`layout::RankLayout`). Each
//! NCCL-capable host contributes a contiguous rank block ordered by local
//! GPU index, sized by the GPUs *CUDA* can open there (phase-0 inventory
//! `cuda_visible_gpus`); one `agent nccl` process — one ssh session, one
//! supervised task — per host drives its whole block. The sweep alone can
//! run in a NIC-forcing world shape instead (`shape`, `[tests]
//! nccl_world`): one rank per host, or one rail at a time. The barrier
//! probe and the fleet overlap step always run rank-per-GPU.
//!
//! The host holding global rank 0 mints the rendezvous id in-process (the
//! id's bootstrap listen socket must live in the process that serves as
//! rank 0) and announces it as an `NcclId` event; the driver intercepts
//! and relays it to every other host, then supervises all host tasks to
//! completion.
//!
//! Failure handling (`attribution`): the first primary failure (an
//! explicit error or `Fatal`, a nonzero exit, a lead without an id) aborts
//! the job — every other host's agent is killed at once instead of hanging
//! in a collective until the phase timeout — and only the culprit is
//! recorded as failed; hosts aborted by it are recorded as Skipped (overlap)
//! or warned about (sweep). Any host abandoned at the phase timeout is
//! killed remotely, never just dropped.

mod attribution;
mod layout;
mod ownership;
mod rails;
mod records;
mod shape;
mod sweep;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::task::JoinSet;
use tracing::{info, warn};

pub(super) use self::shape::IntraNodeCoverage;
pub(super) use self::sweep::nccl_sweep;

use self::attribution::{
    AbortAction, AbortTracker, Attribution, FailureKind, HostFailure, attribute,
};
use self::layout::RankLayout;
use self::ownership::accept_owned;
use self::records::{host_overlap_records, outcomes_with};
use super::barrier::RankSubject;
use super::session::HostSession;
use super::{ObservationSink, gpu_bearing_hosts, kill_remote_agent};
use crate::config::FleetConfig;
use crate::proto::{
    AGENT_EXIT_CASCADE, AgentEvent, InventorySnapshot, NcclDirective, NcclWorkload,
    OverlapFleetReport, RankAssignment, RankBlock, Scope, TestOutcome,
};

/// How long the orchestrator waits for the lead rank's NcclId event.
const NCCL_ID_WAIT: Duration = Duration::from_secs(30);

/// The fleet world: NCCL-capable hosts in fleet order, each with its rank
/// block.
type FleetWorld = RankLayout<Arc<HostSession>>;

/// How many ranks a host contributes: the GPUs CUDA can open there, never
/// the nvidia-smi count. The two differ when a GPU fell off the bus, MIG is
/// on, or CUDA_VISIBLE_DEVICES is set in the ssh environment — and a block
/// sized from nvidia-smi would then make the host fail before init and
/// strand the whole world. The mismatch itself is a phase-0 finding. A
/// host whose agent could not count (no CUDA driver) contributes nothing.
fn nccl_rank_count(inventory: &InventorySnapshot) -> u32 {
    inventory.cuda_visible_gpus.unwrap_or(0)
}

/// Whether a host's libnccl is loadable per its inventory dlopen probe.
/// Hosts predating the probe (empty map) are given the benefit of the
/// doubt. Shared by the fleet world selection and the intra-node sweep.
pub(super) fn nccl_loadable(inventory: &InventorySnapshot) -> bool {
    inventory.gpu_libs.get("nccl").copied().unwrap_or(true)
}

/// Hosts eligible for a fleet-wide NCCL world, in fleet order: GPU-bearing
/// (probing hosts whose inventory is missing) with a loadable libnccl,
/// each with its CUDA-visible GPU count (zero-GPU hosts are kept here and
/// dropped by every layout).
///
/// The inventory dlopen probe knows whether libnccl actually loads. A fleet
/// without the NCCL stack skips fleet NCCL work as a structural finding
/// (visible in the inventory/consistency/bootstrap output) instead of
/// manufacturing a host failure out of a loader panic. Hosts predating the
/// probe (empty map) are given the benefit of the doubt.
async fn nccl_hosts(
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
) -> Vec<(Arc<HostSession>, u32)> {
    let gpu_hosts = gpu_bearing_hosts(sessions, inventories).await;
    if gpu_hosts.is_empty() {
        info!("no GPU-bearing hosts; skipping fleet NCCL work");
        return Vec::new();
    }
    let nccl_hosts: Vec<(Arc<HostSession>, u32)> = gpu_hosts
        .iter()
        .filter_map(|session| {
            let inventory = inventories.get(session.addr())?;
            nccl_loadable(inventory).then(|| (Arc::clone(session), nccl_rank_count(inventory)))
        })
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
    for (session, gpus) in &nccl_hosts {
        if *gpus == 0 {
            warn!(
                host = %session.addr(),
                "CUDA sees no GPU on a GPU-bearing host; excluded from fleet NCCL work"
            );
        }
    }
    nccl_hosts
}

/// The rank-per-GPU fleet world over `hosts` (`nccl_hosts`); empty when it
/// cannot be laid out.
fn rank_per_gpu_world(hosts: Vec<(Arc<HostSession>, u32)>) -> FleetWorld {
    match RankLayout::new(hosts) {
        Ok(layout) => layout,
        Err(error) => {
            warn!(%error, "cannot lay out the fleet NCCL world; skipping fleet NCCL work");
            RankLayout::empty()
        }
    }
}

/// Host -> rank block, for ownership checks of per-rank reports.
fn block_owners(world: &FleetWorld) -> BTreeMap<String, RankBlock> {
    world
        .members()
        .iter()
        .map(|(session, assignment)| (session.addr().to_string(), assignment.block()))
        .collect()
}

/// What one fleet-wide `agent nccl` job runs, beyond the world itself. The
/// lead and participant directives differ only in rendezvous plumbing, so
/// one job builds both. NCCL env is not part of the job: every session
/// sets it on the agent's spawn command line (`HostSession::connect`).
struct NcclJob {
    workload: NcclWorkload,
}

impl NcclJob {
    fn lead(&self, assignment: RankAssignment) -> NcclDirective {
        NcclDirective::Lead {
            assignment,
            workload: self.workload.clone(),
        }
    }

    fn participate(&self, unique_id_b64: &str, assignment: RankAssignment) -> NcclDirective {
        NcclDirective::Participate {
            unique_id_b64: unique_id_b64.to_string(),
            assignment,
            workload: self.workload.clone(),
        }
    }
}

/// How the driver reports each host's *attributed* failure. The sweep
/// marks the culprit host failed (its numbers feed calibration and their
/// absence is a real gap) but only warns about hosts aborted by someone
/// else's failure, so healthy hosts never read as HostFailures; the fleet
/// overlap step only warns — the caller records step-level outcomes from
/// the returned attribution map, so the results document still shows what
/// happened, without a failed-host verdict.
#[derive(Debug, Clone, Copy)]
enum NcclFailureMode {
    HostError,
    WarnOnly,
}

/// Per-host event filter for a fleet NCCL job, given the sending host:
/// returns `None` to consume an event (merged into job-local state), or
/// the event back to forward it to the collector. `NcclId` relay and
/// `Fatal` capture are handled by the driver itself.
type EventIntercept = Arc<dyn Fn(&str, AgentEvent) -> Option<AgentEvent> + Send + Sync>;

/// Drive one fleet-wide `agent nccl` job over an established world: one
/// process per host, each driving its rank block. Returns the attributed
/// failure map (`attribution::attribute`): host address -> Failed (this
/// host caused it, or nobody identifiable did) or Skipped (aborted by
/// another host's failure). An empty map means every host exited cleanly.
///
/// - The first primary failure kills every host still running
///   (`AbortTracker`), so one dead rank costs seconds, not the phase
///   timeout, and leaves no agent blocked in a collective.
/// - A host that outlives the phase timeout is killed remotely, not just
///   abandoned.
/// - A lead that mints no rendezvous id within `NCCL_ID_WAIT` is killed
///   right away (no follower was ever started, so it can only block) and
///   blamed for every follower's "never started".
async fn drive_fleet_nccl(
    config: &FleetConfig,
    world: &FleetWorld,
    sink: &ObservationSink,
    step: &str,
    job: &NcclJob,
    mode: NcclFailureMode,
    intercept: EventIntercept,
) -> BTreeMap<String, Attribution> {
    let mut failures: BTreeMap<String, HostFailure> = BTreeMap::new();
    let Some((lead, lead_assignment)) = world.members().first() else {
        return BTreeMap::new();
    };
    let sessions: BTreeMap<String, Arc<HostSession>> = world
        .members()
        .iter()
        .map(|(session, _)| (session.addr().to_string(), Arc::clone(session)))
        .collect();
    let timeout = Duration::from_secs(config.tests.phase_timeout_secs.max(1));
    let mut tasks = JoinSet::new();
    let mut running: BTreeSet<String> = BTreeSet::new();

    let document = match serde_json::to_string(&job.lead(*lead_assignment)) {
        Ok(document) => document,
        Err(error) => {
            warn!(%error, "cannot serialize the NCCL lead directive");
            let reason = format!("cannot serialize the NCCL lead directive: {error}");
            for (session, _) in world.members() {
                failures.insert(
                    session.addr().to_string(),
                    HostFailure::new(FailureKind::Primary, reason.clone()),
                );
            }
            return report_failures(sink, step, mode, &failures);
        }
    };
    let (id_tx, id_rx) = tokio::sync::oneshot::channel::<String>();
    let id_slot = Arc::new(Mutex::new(Some(id_tx)));
    {
        let session = Arc::clone(lead);
        let assignment = *lead_assignment;
        let sink = sink.clone();
        let id_slot = Arc::clone(&id_slot);
        let intercept = Arc::clone(&intercept);
        running.insert(lead.addr().to_string());
        tasks.spawn(async move {
            let addr = session.addr().to_string();
            let failure = run_host(
                &session,
                assignment,
                document,
                timeout,
                |event| match event {
                    AgentEvent::NcclId { unique_id_b64 } => {
                        if let Some(tx) = id_slot.lock().expect("id slot poisoned").take() {
                            let _ = tx.send(unique_id_b64);
                        }
                    }
                    event => {
                        if let Some(event) = intercept(&addr, event) {
                            sink.event(&addr, event);
                        }
                    }
                },
            )
            .await;
            (addr, failure)
        });
    }

    let unique_id_b64 = match tokio::time::timeout(NCCL_ID_WAIT, id_rx).await {
        Ok(Ok(id)) => Some(id),
        Ok(Err(_)) | Err(_) => {
            warn!(
                lead = %lead.addr(),
                "NCCL lead produced no rendezvous id; aborting the job"
            );
            // A lead still running without having minted an id can only
            // block (no follower will ever join): kill it now instead of
            // letting it hold the phase until the timeout.
            kill_remote_agent(lead, "nccl").await;
            // The other hosts never start: record why, so the caller can
            // put something truthful in the results document.
            for (session, _) in world.members().iter().skip(1) {
                failures.insert(
                    session.addr().to_string(),
                    HostFailure::new(
                        FailureKind::NeverStarted,
                        "nccl ranks never started: the lead produced no rendezvous id",
                    ),
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
                        HostFailure::new(
                            FailureKind::Primary,
                            format!("cannot serialize the NCCL directive: {error}"),
                        ),
                    );
                    continue;
                }
            };
            let session = Arc::clone(session);
            let assignment = *assignment;
            let sink = sink.clone();
            let intercept = Arc::clone(&intercept);
            running.insert(session.addr().to_string());
            tasks.spawn(async move {
                let addr = session.addr().to_string();
                let failure = run_host(&session, assignment, document, timeout, |event| {
                    if let Some(event) = intercept(&addr, event) {
                        sink.event(&addr, event);
                    }
                })
                .await;
                (addr, failure)
            });
        }
    }

    let mut tracker = AbortTracker::default();
    // A follower-serialization failure is a primary failure seen before any
    // host finished: the world can never complete, so abort right away.
    if failures
        .values()
        .any(|failure| failure.kind == FailureKind::Primary)
        && tracker.primary_seen() == AbortAction::AbortRest
    {
        abort_rest(&mut tracker, &running, &sessions).await;
    }
    while let Some(joined) = tasks.join_next().await {
        match joined {
            Ok((addr, failure)) => {
                running.remove(&addr);
                let (failure, action) = tracker.observe(&addr, failure);
                if let Some(failure) = failure {
                    failures.insert(addr, failure);
                }
                if action == AbortAction::AbortRest {
                    abort_rest(&mut tracker, &running, &sessions).await;
                }
            }
            Err(error) => warn!(%error, "nccl task did not complete"),
        }
    }
    if unique_id_b64.is_none() {
        blame_the_lead(&mut failures, lead.addr());
    }
    report_failures(sink, step, mode, &failures)
}

/// Early abort: kill every host still running, concurrently, and tell the
/// tracker so their fallout is classified as the driver's doing.
async fn abort_rest(
    tracker: &mut AbortTracker,
    running: &BTreeSet<String>,
    sessions: &BTreeMap<String, Arc<HostSession>>,
) {
    if running.is_empty() {
        return;
    }
    warn!(
        hosts = running.len(),
        "primary failure in a fleet NCCL job; aborting the remaining hosts"
    );
    tracker.mark_aborted(running.iter().cloned());
    let mut kills = JoinSet::new();
    for host in running {
        if let Some(session) = sessions.get(host) {
            let session = Arc::clone(session);
            kills.spawn(async move { kill_remote_agent(&session, "nccl").await });
        }
    }
    while kills.join_next().await.is_some() {}
}

/// Run one host's `agent nccl` to completion under the phase timeout,
/// killing the remote process when the timeout abandons it. A `Fatal`
/// event is captured as the host's failure reason (it is an explicit,
/// primary failure) rather than forwarded to the collector: the driver
/// decides how the failure is reported.
async fn run_host(
    session: &HostSession,
    assignment: RankAssignment,
    document: String,
    timeout: Duration,
    mut on_event: impl FnMut(AgentEvent) + Send,
) -> Option<HostFailure> {
    let fatal: Mutex<Option<String>> = Mutex::new(None);
    let outcome = tokio::time::timeout(
        timeout,
        session.run_agent(&["nccl"], Some(document), |event| match event {
            AgentEvent::Fatal { message } => {
                *fatal.lock().expect("fatal slot poisoned") = Some(message);
            }
            event => on_event(event),
        }),
    )
    .await;
    if outcome.is_err() {
        // Dropping the future only closed our end of the ssh channel; the
        // remote agent may sit blocked in a collective indefinitely.
        kill_remote_agent(session, "nccl").await;
    }
    let fatal = fatal.into_inner().expect("fatal slot poisoned");
    block_failure(assignment, timeout, outcome, fatal)
}

/// The lead failed before the id relay: whatever its own outcome looked
/// like (even a timeout, or none at all), it is the primary failure — every
/// follower's "never started" traces back to it.
fn blame_the_lead(failures: &mut BTreeMap<String, HostFailure>, lead: &str) {
    let message = failures.get(lead).map_or_else(
        || "nccl lead exited without producing a rendezvous id".to_string(),
        |failure| failure.message.clone(),
    );
    failures.insert(
        lead.to_string(),
        HostFailure::new(FailureKind::Primary, message),
    );
}

/// Attribute the failures and report them per `mode`; returns the
/// attribution map.
fn report_failures(
    sink: &ObservationSink,
    step: &str,
    mode: NcclFailureMode,
    failures: &BTreeMap<String, HostFailure>,
) -> BTreeMap<String, Attribution> {
    let attributed = attribute(step, failures);
    for (host, attribution) in &attributed {
        match (mode, attribution) {
            (NcclFailureMode::HostError, Attribution::Failed { reason }) => {
                sink.error(host, reason.clone())
            }
            (_, Attribution::Failed { reason }) => {
                warn!(host = %host, reason, "fleet nccl host failed")
            }
            (_, Attribution::Skipped { reason }) => {
                warn!(host = %host, reason, "fleet nccl host aborted by another host's failure")
            }
        }
    }
    attributed
}

/// Why a host's rank block did not complete cleanly, if it didn't, and
/// what kind of failure that is. A `Fatal` event or any error or nonzero
/// exit is primary; a cascade exit (`AGENT_EXIT_CASCADE`) or a timeout is
/// secondary material.
fn block_failure(
    assignment: RankAssignment,
    timeout: Duration,
    outcome: Result<anyhow::Result<std::process::ExitStatus>, tokio::time::error::Elapsed>,
    fatal: Option<String>,
) -> Option<HostFailure> {
    let block = assignment.block();
    let ranks = format!("nccl ranks {}..{}", block.base(), block.end());
    if let Some(message) = fatal {
        return Some(HostFailure::new(
            FailureKind::Primary,
            format!("{ranks}: {message}"),
        ));
    }
    match outcome {
        Ok(Ok(status)) if status.success() => None,
        Ok(Ok(status)) if status.code() == Some(AGENT_EXIT_CASCADE) => Some(HostFailure::new(
            FailureKind::Cascade,
            format!("{ranks} stopped because the fleet stopped ({status})"),
        )),
        Ok(Ok(status)) => Some(HostFailure::new(
            FailureKind::Primary,
            format!("{ranks} exited with {status}"),
        )),
        Ok(Err(error)) => Some(HostFailure::new(
            FailureKind::Primary,
            format!("{ranks} failed: {error:#}"),
        )),
        Err(_) => Some(HostFailure::new(
            FailureKind::TimedOut,
            format!("{ranks} timed out after {}s", timeout.as_secs()),
        )),
    }
}

/// Report every rank-ownership violation as a structured error against the
/// offending host: a host reporting ranks it does not own is a broken
/// agent, whichever step it happened in.
fn report_violations(sink: &ObservationSink, violations: Vec<ownership::OwnershipViolation>) {
    for violation in violations {
        warn!(host = violation.host(), %violation, "rank ownership violation");
        sink.error(violation.host(), violation.to_string());
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
/// records Skipped; the host that caused a failure records Failed; hosts
/// aborted by another host's failure record Skipped naming the culprit
/// (per GPU for a silent rank inside a reporting block) — but never a
/// failed-*host* verdict. Only `overlap_fleet = false` leaves no trace.
pub(super) async fn overlap_fleet_sweep(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
    sink: &ObservationSink,
) {
    let world = rank_per_gpu_world(nccl_hosts(sessions, inventories).await);
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

    let reports: Arc<Mutex<Vec<(String, OverlapFleetReport)>>> = Arc::default();
    let intercept: EventIntercept = {
        let reports = Arc::clone(&reports);
        Arc::new(move |host, event| match event {
            AgentEvent::OverlapFleetReport { report } => {
                reports
                    .lock()
                    .expect("overlap reports poisoned")
                    .push((host.to_string(), *report));
                None
            }
            event => Some(event),
        })
    };
    let job = NcclJob {
        workload: NcclWorkload::Overlap(spec),
    };
    let failures = drive_fleet_nccl(
        config,
        &world,
        sink,
        "fleet overlap",
        &job,
        NcclFailureMode::WarnOnly,
        intercept,
    )
    .await;

    let collected = std::mem::take(&mut *reports.lock().expect("overlap reports poisoned"));
    if collected.is_empty() {
        warn!(world_size, "fleet overlap step produced no reports");
    }
    let (by_rank, violations) =
        accept_owned(&block_owners(&world), collected, |report| report.rank);
    report_violations(sink, violations);
    // Every host in the world ends up with something in the document: its
    // GPUs' merged reports, or outcomes saying why there are none.
    for (session, assignment) in world.members() {
        let host = session.addr();
        let (metrics, outcomes) = host_overlap_records(
            assignment.block(),
            &by_rank,
            spec.gemm_dtype,
            failures.get(host),
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
    use std::os::unix::process::ExitStatusExt;
    use std::process::ExitStatus;

    use super::*;
    use crate::proto::{GemmDtype, OverlapSpec};

    fn job() -> NcclJob {
        NcclJob {
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

    fn assignment() -> RankAssignment {
        RankLayout::new([("n1", 4), ("n2", 4)])
            .expect("layout")
            .members()[1]
            .1
    }

    /// `code` as a remote exit status (wait-status encoding).
    fn exited(code: i32) -> ExitStatus {
        ExitStatus::from_raw(code << 8)
    }

    #[test]
    fn block_failures_classify_by_how_the_host_ended() {
        let timeout = Duration::from_secs(600);
        let classify = |outcome, fatal: Option<&str>| {
            block_failure(assignment(), timeout, outcome, fatal.map(str::to_string))
        };
        assert_eq!(classify(Ok(Ok(exited(0))), None), None);

        let failed = classify(Ok(Err(anyhow::anyhow!("boom"))), None).expect("a failure");
        assert_eq!(failed.kind, FailureKind::Primary);
        assert!(
            failed.message.starts_with("nccl ranks 4..8 failed"),
            "{failed:?}"
        );

        let nonzero = classify(Ok(Ok(exited(1))), None).expect("a failure");
        assert_eq!(nonzero.kind, FailureKind::Primary);

        let cascade = classify(Ok(Ok(exited(AGENT_EXIT_CASCADE))), None).expect("a failure");
        assert_eq!(cascade.kind, FailureKind::Cascade);

        // A Fatal event names the real reason and is primary, even when the
        // process then exited cleanly.
        let fatal = classify(Ok(Ok(exited(0))), Some("CUDA sees 7 GPUs")).expect("a failure");
        assert_eq!(fatal.kind, FailureKind::Primary);
        assert_eq!(fatal.message, "nccl ranks 4..8: CUDA sees 7 GPUs");
    }

    #[test]
    fn a_lead_that_minted_no_id_is_always_primary() {
        let mut failures = BTreeMap::from([(
            "lead".to_string(),
            HostFailure::new(
                FailureKind::TimedOut,
                "nccl ranks 0..8 timed out after 900s",
            ),
        )]);
        blame_the_lead(&mut failures, "lead");
        assert_eq!(failures["lead"].kind, FailureKind::Primary);
        assert_eq!(
            failures["lead"].message,
            "nccl ranks 0..8 timed out after 900s"
        );

        // A lead that exited cleanly without an id is blamed too.
        let mut failures = BTreeMap::new();
        blame_the_lead(&mut failures, "lead");
        assert_eq!(failures["lead"].kind, FailureKind::Primary);
    }

    fn inventory(nvidia_smi: usize, cuda: Option<u32>) -> InventorySnapshot {
        let mut inventory: InventorySnapshot = serde_json::from_value(serde_json::json!({
            "hostname": "n1", "kernel": "6.8", "cpu_model": "x", "logical_cores": 8,
            "numa_nodes": 1, "mem_total_bytes": 1, "cpu_governor": null,
            "clock_offset_ms": null, "nvidia_driver": null, "cuda_version": null,
            "gpus": [], "nics": [], "ib_ports": [], "xid_errors": []
        }))
        .expect("inventory");
        inventory.gpus = (0..nvidia_smi)
            .map(|index| {
                serde_json::from_value(serde_json::json!({
                    "index": index, "name": "H100", "uuid": "u", "vbios": "v",
                    "mem_total_bytes": 1, "ecc_volatile_errors": null,
                    "remapped_rows_pending": null, "pcie_gen_current": null,
                    "pcie_gen_max": null, "pcie_width_current": null,
                    "pcie_width_max": null, "nvlinks_active": null,
                    "persistence_mode": null
                }))
                .expect("gpu")
            })
            .collect();
        inventory.cuda_visible_gpus = cuda;
        inventory
    }

    #[test]
    fn rank_blocks_are_sized_by_what_cuda_can_open() {
        assert_eq!(nccl_rank_count(&inventory(8, Some(8))), 8);
        // A GPU fell off the bus / MIG / CUDA_VISIBLE_DEVICES: CUDA wins.
        assert_eq!(nccl_rank_count(&inventory(8, Some(7))), 7);
        // No CUDA count at all: the host contributes no ranks.
        assert_eq!(nccl_rank_count(&inventory(8, None)), 0);
    }
}
