//! Operator-side driver for `gauntlet run` and `gauntlet bootstrap`.
//!
//! Control flow of a run:
//!   load config -> open sessions (bounded concurrency, then held open) ->
//!   ensure agent deployed (hash check) -> per-phase:
//!     phases 0-2: spawn `agent run` on every host simultaneously, decode
//!       event streams into the collector;
//!     phase 3: tournament rounds of `agent peer` pairs, then the
//!       hierarchical NCCL sweeps (per-node, pairs, full fleet) with the
//!       uniqueId relayed from rank 0 by this process;
//!   -> collector -> analysis -> report to disk + terminal, exit code.
//!
//! Per-host failures (unreachable, agent Fatal, phase timeout from
//! `tests.phase_timeout_secs`) mark that host failed and the run continues;
//! the failure lands in the report instead of aborting the fleet.

pub mod bootstrap;
pub mod collect;
pub mod deploy;
pub mod session;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::de::DeserializeOwned;
use tokio::sync::{Semaphore, mpsc};
use tokio::task::JoinSet;
use tracing::{debug, info, warn};

use self::collect::{Collector, HostObservations};
use self::session::{HostSession, single_quote};
use crate::agent::net::{BandwidthReport, LatencyReport};
use crate::analysis::schedule::{sampled_rounds, tournament_rounds};
use crate::cli::RunArgs;
use crate::config::FleetConfig;
use crate::proto::{
    AgentEvent, CounterRequest, CounterSnapshot, InventorySnapshot, MetricRecord, NcclDirective,
    Phase, Scope, TestId, Unit,
};
use crate::report;

/// How long the orchestrator waits for a peer to acknowledge a graceful
/// shutdown before falling back to killing it remotely.
const PEER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
/// Slack added to the configured probe durations for a pairwise test.
const PAIR_TIMEOUT_SLACK: Duration = Duration::from_secs(60);
/// Time given to a freshly spawned `agent peer serve` to bind its port.
const PEER_BIND_DELAY: Duration = Duration::from_millis(500);
/// How often the collector publishes a partial snapshot of a run in flight.
const PARTIAL_SNAPSHOT_INTERVAL: Duration = Duration::from_secs(2);
/// Upper bound on one error-counter pass (a handful of sysfs reads and
/// bounded tool probes; nothing like a full phase).
const COUNTER_PASS_TIMEOUT: Duration = Duration::from_secs(60);

// ---------------------------------------------------------------------------
// Observation plumbing
// ---------------------------------------------------------------------------

/// Message from a per-host driver task to the single collector task. Events
/// are boxed: an `AgentEvent` carrying an inventory dwarfs an error string,
/// and every message would otherwise pay for the largest variant.
enum Observation {
    Event {
        host: String,
        event: Box<AgentEvent>,
    },
    Error {
        host: String,
        error: String,
    },
}

/// Cheap, cloneable handle onto the collector channel. Sends never block and
/// never fail the caller: if the collector is gone the run is already over.
#[derive(Clone)]
struct ObservationSink {
    tx: mpsc::UnboundedSender<Observation>,
    /// `--repeat` iteration stamped onto every metric that flows through.
    repeat: u32,
}

impl ObservationSink {
    fn with_repeat(&self, repeat: u32) -> Self {
        Self {
            tx: self.tx.clone(),
            repeat,
        }
    }

    fn event(&self, host: &str, mut event: AgentEvent) {
        if let AgentEvent::Metric { record } = &mut event {
            record.repeat = self.repeat;
        }
        if self
            .tx
            .send(Observation::Event {
                host: host.to_string(),
                event: Box::new(event),
            })
            .is_err()
        {
            debug!(host, "collector closed; dropping event");
        }
    }

    fn metric(&self, host: &str, record: MetricRecord) {
        self.event(host, AgentEvent::Metric { record });
    }

    fn error(&self, host: &str, error: impl Into<String>) {
        let error = error.into();
        if self
            .tx
            .send(Observation::Error {
                host: host.to_string(),
                error,
            })
            .is_err()
        {
            debug!(host, "collector closed; dropping error");
        }
    }
}

/// Everything the collector task needs to publish an in-flight snapshot of
/// the run to `runs/<run_id>.partial.json`. Present only when the run writes
/// to the default history directory.
struct PartialWriter {
    run_id: String,
    config: FleetConfig,
    started_epoch_secs: u64,
}

impl PartialWriter {
    /// Analyse and publish `observations` as they stand. Snapshots are a
    /// convenience for onlookers: a failure here is logged and forgotten,
    /// never propagated into the run.
    fn write(&self, observations: BTreeMap<String, HostObservations>) {
        let mut results = report::build(
            &self.config,
            observations,
            self.started_epoch_secs,
            epoch_secs(),
        );
        results.run_id = self.run_id.clone();
        let dir = Path::new(report::history::DEFAULT_DIR);
        match report::history::save_partial(&results, dir) {
            Ok(path) => {
                debug!(path = %path.display(), run_id = %self.run_id, "partial snapshot written")
            }
            Err(error) => warn!(%error, run_id = %self.run_id, "cannot write partial snapshot"),
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn run(args: RunArgs) -> Result<()> {
    let config = FleetConfig::load(&args.config)
        .with_context(|| format!("loading {}", args.config.display()))?;
    let phases = config.resolve_phases(&args.phases)?;
    bootstrap::warn_if_debug_build();
    let started_epoch_secs = epoch_secs();
    // Fixed up front so the in-flight snapshots and the final document share
    // one identity: a viewer tailing runs/ can follow a run across completion
    // without re-keying it.
    let host_addrs: Vec<String> = config.hosts().map(|host| host.addr).collect();
    let run_id = report::make_run_id(started_epoch_secs, &host_addrs);

    let (tx, mut rx) = mpsc::unbounded_channel();
    let sink = ObservationSink { tx, repeat: 0 };
    // Snapshots only make sense for the default history directory; an
    // explicit --out is a one-shot destination, not a directory a viewer
    // tails.
    let partials = args.out.is_none().then(|| PartialWriter {
        run_id: run_id.clone(),
        config: config.clone(),
        started_epoch_secs,
    });
    let collector = tokio::spawn(async move {
        let mut collector = Collector::new();
        let mut dirty = false;
        let mut ticker = tokio::time::interval(PARTIAL_SNAPSHOT_INTERVAL);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                observation = rx.recv() => match observation {
                    Some(Observation::Event { host, event }) => {
                        collector.ingest(&host, *event);
                        dirty = true;
                    }
                    Some(Observation::Error { host, error }) => {
                        collector.host_error(&host, error);
                        dirty = true;
                    }
                    None => break,
                },
                _ = ticker.tick() => {
                    if dirty && let Some(partials) = &partials {
                        partials.write(collector.snapshot());
                        dirty = false;
                    }
                }
            }
        }
        collector.into_observations()
    });

    let sessions = connect_fleet(&config, &sink).await;
    let sessions = deploy_fleet(&config, sessions, &sink).await;
    if sessions.is_empty() {
        bail!("no hosts are usable; see the errors above");
    }
    info!(
        hosts = sessions.len(),
        phases = ?phases,
        "fleet ready"
    );

    let mut inventories: BTreeMap<String, InventorySnapshot> = BTreeMap::new();
    // Error-counter baselines, taken once before the first load phase; the
    // matching delta pass runs after the last load phase of the last repeat.
    let mut counter_baselines: BTreeMap<String, CounterSnapshot> = BTreeMap::new();
    let mut counters_started = false;
    for repeat in 0..args.repeat {
        let sink = sink.with_repeat(repeat);
        if args.repeat > 1 {
            info!(repeat, of = args.repeat, "repeat start");
        }
        for phase in &phases {
            // The inventory is a census, not a measurement.
            if *phase == Phase::Inventory && repeat > 0 {
                continue;
            }
            // Snapshot error counters after any leading inventory and
            // before the first phase that puts the hardware under load.
            if *phase != Phase::Inventory && !counters_started {
                counters_started = true;
                info!("counter baseline");
                counter_baselines = counter_baseline_pass(&config, &sessions).await;
            }
            info!(?phase, "phase start");
            match phase {
                Phase::Inventory | Phase::CpuMem | Phase::Gpu => {
                    let seen = node_phase(&config, &sessions, *phase, &sink).await;
                    inventories.extend(seen);
                }
                Phase::Network => {
                    network_phase(
                        &config,
                        &sessions,
                        &mut inventories,
                        args.sample_pairs,
                        &sink,
                    )
                    .await;
                }
            }
            info!(?phase, "phase end");
        }
    }
    if !counter_baselines.is_empty() {
        info!("counter delta pass");
        counter_delta_pass(&config, &sessions, &counter_baselines, &sink).await;
    }

    drop(sink);
    let observations = collector.await.context("collector task")?;
    let finished_epoch_secs = epoch_secs();

    let mut results = report::build(
        &config,
        observations,
        started_epoch_secs,
        finished_epoch_secs,
    );
    // The id the snapshots have been published under wins, so the final
    // document lands where onlookers were already watching.
    results.debug_build = cfg!(debug_assertions);
    results.run_id = run_id.clone();
    let path = match &args.out {
        Some(path) => {
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            let json = serde_json::to_string_pretty(&results).context("serializing results")?;
            std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))?;
            path.clone()
        }
        None => {
            let dir = Path::new(report::history::DEFAULT_DIR);
            let path = report::history::save(&results, dir).context("saving run history")?;
            // The finished document supersedes the snapshots; a stale partial
            // left behind would show a viewer a run that never ends.
            if let Err(error) = report::history::remove_partial(&run_id, dir) {
                debug!(%error, run_id = %run_id, "leftover partial snapshot not removed");
            }
            path
        }
    };
    info!(path = %path.display(), run_id = %results.run_id, "results written");

    let mut stdout = std::io::stdout();
    report::render_table(&results, &mut stdout)?;
    stdout.flush().context("flushing stdout")?;

    std::process::exit(report::verdict(&results).exit_code());
}

fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Session establishment and deploy
// ---------------------------------------------------------------------------

/// Open every session with `ssh.max_concurrent` establishments in flight.
/// Established sessions are held for the whole run (the permit is only for
/// the handshake). Order follows the config so host indices are stable.
async fn connect_fleet(config: &FleetConfig, sink: &ObservationSink) -> Vec<Arc<HostSession>> {
    let permits = Arc::new(Semaphore::new(config.ssh.max_concurrent.max(1)));
    let mut tasks = JoinSet::new();
    let mut count = 0usize;
    for (index, host) in config.hosts().enumerate() {
        count += 1;
        let ssh = config.ssh.clone();
        let permits = Arc::clone(&permits);
        let sink = sink.clone();
        tasks.spawn(async move {
            let _permit = permits.acquire_owned().await.ok();
            let addr = host.addr.clone();
            match HostSession::connect(host, &ssh).await {
                Ok(session) => (index, Some(Arc::new(session))),
                Err(error) => {
                    sink.error(&addr, format!("ssh connect failed: {error:#}"));
                    (index, None)
                }
            }
        });
    }

    let mut slots: Vec<Option<Arc<HostSession>>> = vec![None; count];
    while let Some(joined) = tasks.join_next().await {
        match joined {
            Ok((index, session)) => slots[index] = session,
            Err(error) => warn!(%error, "connect task did not complete"),
        }
    }
    slots.into_iter().flatten().collect()
}

/// Hash-compare and upload the agent everywhere, dropping hosts that cannot
/// take it. Bounded by the same concurrency limit as connects: uploads are
/// the heavy part of bootstrap.
async fn deploy_fleet(
    config: &FleetConfig,
    sessions: Vec<Arc<HostSession>>,
    sink: &ObservationSink,
) -> Vec<Arc<HostSession>> {
    let permits = Arc::new(Semaphore::new(config.ssh.max_concurrent.max(1)));
    let timeout = Duration::from_secs(config.tests.phase_timeout_secs.max(1));
    let mut tasks = JoinSet::new();
    let count = sessions.len();
    for (index, session) in sessions.into_iter().enumerate() {
        let permits = Arc::clone(&permits);
        let sink = sink.clone();
        tasks.spawn(async move {
            let _permit = permits.acquire_owned().await.ok();
            let addr = session.addr().to_string();
            match tokio::time::timeout(timeout, deploy::ensure_agent(&session)).await {
                Ok(Ok(uploaded)) => {
                    debug!(host = %addr, uploaded, "agent ready");
                    (index, Some(session))
                }
                Ok(Err(error)) => {
                    sink.error(&addr, format!("agent deploy failed: {error:#}"));
                    (index, None)
                }
                Err(_) => {
                    sink.error(
                        &addr,
                        format!("agent deploy timed out after {}s", timeout.as_secs()),
                    );
                    (index, None)
                }
            }
        });
    }

    let mut slots: Vec<Option<Arc<HostSession>>> = vec![None; count];
    while let Some(joined) = tasks.join_next().await {
        match joined {
            Ok((index, session)) => slots[index] = session,
            Err(error) => warn!(%error, "deploy task did not complete"),
        }
    }
    slots.into_iter().flatten().collect()
}

// ---------------------------------------------------------------------------
// Phases 0-2: embarrassingly parallel per node
// ---------------------------------------------------------------------------

/// Run one node-local phase on every host at once. Returns the inventories
/// observed on the way through, which phase 3 needs to pick NCCL ranks.
async fn node_phase(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
    phase: Phase,
    sink: &ObservationSink,
) -> BTreeMap<String, InventorySnapshot> {
    let spec = config.task_spec(&[phase]);
    let document = match serde_json::to_string(&spec) {
        Ok(document) => document,
        Err(error) => {
            warn!(%error, ?phase, "cannot serialize task spec; skipping phase");
            return BTreeMap::new();
        }
    };
    let timeout = Duration::from_secs(config.tests.phase_timeout_secs.max(1));

    let mut tasks = JoinSet::new();
    for session in sessions {
        let session = Arc::clone(session);
        let sink = sink.clone();
        let document = document.clone();
        tasks.spawn(async move {
            let addr = session.addr().to_string();
            let mut inventory = None;
            let outcome = tokio::time::timeout(
                timeout,
                session.run_agent(&["run"], Some(document), |event| {
                    if let AgentEvent::Inventory { snapshot } = &event {
                        inventory = Some(snapshot.clone());
                    }
                    sink.event(&addr, event);
                }),
            )
            .await;
            match outcome {
                Ok(Ok(status)) if status.success() => {}
                Ok(Ok(status)) => sink.error(
                    &addr,
                    format!("agent exited with {status} during {phase:?} phase"),
                ),
                Ok(Err(error)) => sink.error(&addr, format!("{phase:?} phase failed: {error:#}")),
                Err(_) => sink.error(
                    &addr,
                    format!("{phase:?} phase timed out after {}s", timeout.as_secs()),
                ),
            }
            (addr, inventory)
        });
    }

    let mut inventories = BTreeMap::new();
    while let Some(joined) = tasks.join_next().await {
        match joined {
            Ok((addr, Some(inventory))) => {
                inventories.insert(addr, *inventory);
            }
            Ok((_, None)) => {}
            Err(error) => warn!(%error, ?phase, "host task did not complete"),
        }
    }
    inventories
}

// ---------------------------------------------------------------------------
// Error-counter passes (baseline before load, deltas after)
// ---------------------------------------------------------------------------

/// Snapshot error counters on every host. Baselines are held here, not sent
/// to the collector: they only exist to be handed back for the delta pass.
/// Failures are logged and the host simply has no baseline (and therefore
/// no deltas) — counter collection never fails a host.
async fn counter_baseline_pass(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
) -> BTreeMap<String, CounterSnapshot> {
    let mut spec = config.task_spec(&[]);
    spec.counters = Some(CounterRequest::Baseline);
    let document = match serde_json::to_string(&spec) {
        Ok(document) => document,
        Err(error) => {
            warn!(%error, "cannot serialize the counter baseline spec; skipping counters");
            return BTreeMap::new();
        }
    };

    let mut tasks = JoinSet::new();
    for session in sessions {
        let session = Arc::clone(session);
        let document = document.clone();
        tasks.spawn(async move {
            let addr = session.addr().to_string();
            let mut baseline = None;
            let outcome = tokio::time::timeout(
                COUNTER_PASS_TIMEOUT,
                session.run_agent(&["run"], Some(document), |event| {
                    if let AgentEvent::CounterBaseline { snapshot } = event {
                        baseline = Some(*snapshot);
                    }
                }),
            )
            .await;
            match outcome {
                Ok(Ok(status)) if status.success() => {}
                Ok(Ok(status)) => {
                    warn!(host = %addr, %status, "counter baseline exited abnormally")
                }
                Ok(Err(error)) => warn!(host = %addr, %error, "counter baseline failed"),
                Err(_) => warn!(host = %addr, "counter baseline timed out"),
            }
            (addr, baseline)
        });
    }

    let mut baselines = BTreeMap::new();
    while let Some(joined) = tasks.join_next().await {
        match joined {
            Ok((addr, Some(baseline))) => {
                baselines.insert(addr, baseline);
            }
            Ok((_, None)) => {}
            Err(error) => warn!(%error, "counter baseline task did not complete"),
        }
    }
    baselines
}

/// Re-snapshot on every baselined host, diff on the agent, and feed the
/// resulting `CounterDeltas` events into the collector.
async fn counter_delta_pass(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
    baselines: &BTreeMap<String, CounterSnapshot>,
    sink: &ObservationSink,
) {
    let mut tasks = JoinSet::new();
    for session in sessions {
        let Some(baseline) = baselines.get(session.addr()) else {
            warn!(host = %session.addr(), "no counter baseline; skipping delta pass");
            continue;
        };
        let mut spec = config.task_spec(&[]);
        spec.counters = Some(CounterRequest::Delta {
            baseline: baseline.clone(),
        });
        let document = match serde_json::to_string(&spec) {
            Ok(document) => document,
            Err(error) => {
                warn!(host = %session.addr(), %error, "cannot serialize the counter delta spec");
                continue;
            }
        };
        let session = Arc::clone(session);
        let sink = sink.clone();
        tasks.spawn(async move {
            let addr = session.addr().to_string();
            let outcome = tokio::time::timeout(
                COUNTER_PASS_TIMEOUT,
                session.run_agent(&["run"], Some(document), |event| {
                    if matches!(&event, AgentEvent::CounterDeltas { .. }) {
                        sink.event(&addr, event);
                    }
                }),
            )
            .await;
            match outcome {
                Ok(Ok(status)) if status.success() => {}
                Ok(Ok(status)) => {
                    warn!(host = %addr, %status, "counter delta pass exited abnormally")
                }
                Ok(Err(error)) => warn!(host = %addr, %error, "counter delta pass failed"),
                Err(_) => warn!(host = %addr, "counter delta pass timed out"),
            }
        });
    }
    while let Some(joined) = tasks.join_next().await {
        if let Err(error) = joined {
            warn!(%error, "counter delta task did not complete");
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3: pairwise TCP, then the NCCL fleet sweep
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct PairParams {
    latency_secs: u64,
    bandwidth_secs: u64,
    timeout: Duration,
}

async fn network_phase(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
    sample_pairs: Option<usize>,
    sink: &ObservationSink,
) {
    pairwise_sweep(config, sessions, sample_pairs, sink).await;
    nccl_sweep(config, sessions, inventories, sink).await;
}

/// Tournament rounds of disjoint pairs: within a round every pair runs
/// concurrently, and no host appears twice, so nothing contends.
async fn pairwise_sweep(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
    sample_pairs: Option<usize>,
    sink: &ObservationSink,
) {
    let rounds = match sample_pairs {
        Some(per_host) => sampled_rounds(sessions.len(), per_host),
        None => tournament_rounds(sessions.len()),
    };
    if rounds.is_empty() {
        info!("fewer than two hosts; skipping pairwise network tests");
        return;
    }
    let params = PairParams {
        latency_secs: config.tests.net_latency_secs,
        bandwidth_secs: config.tests.net_bandwidth_secs,
        timeout: Duration::from_secs(
            config.tests.net_latency_secs + config.tests.net_bandwidth_secs,
        ) + PAIR_TIMEOUT_SLACK,
    };

    for (index, round) in rounds.iter().enumerate() {
        debug!(round = index, pairs = round.len(), "pairwise round");
        let mut tasks = JoinSet::new();
        for (slot, (a, b)) in round.iter().enumerate() {
            let (Some(server), Some(client)) = (sessions.get(*a), sessions.get(*b)) else {
                warn!(a, b, "scheduler produced an out-of-range pair");
                continue;
            };
            let Some(port) = port_for_slot(config.tests.net_port_base, slot) else {
                warn!(slot, "no free port for this pair slot");
                continue;
            };
            let server = Arc::clone(server);
            let client = Arc::clone(client);
            let sink = sink.clone();
            tasks.spawn(async move { run_pair(params, server, client, port, sink).await });
        }
        while let Some(joined) = tasks.join_next().await {
            if let Err(error) = joined {
                warn!(%error, "pair task did not complete");
            }
        }
    }
}

fn port_for_slot(base: u16, slot: usize) -> Option<u16> {
    u16::try_from(slot)
        .ok()
        .and_then(|slot| base.checked_add(slot))
}

/// One scheduled pair: serve on `server`, probe from `client`, attribute the
/// results to both, then always tear the listener down.
async fn run_pair(
    params: PairParams,
    server: Arc<HostSession>,
    client: Arc<HostSession>,
    port: u16,
    sink: ObservationSink,
) {
    let server_addr = server.addr().to_string();
    let client_addr = client.addr().to_string();
    let port_text = port.to_string();

    let child = match server
        .spawn_agent(&["peer", "serve", "--port", &port_text])
        .await
    {
        Ok(child) => child,
        Err(error) => {
            sink.error(
                &client_addr,
                format!("peer serve on {server_addr}:{port} failed to start: {error:#}"),
            );
            return;
        }
    };
    tokio::time::sleep(PEER_BIND_DELAY).await;

    // Benchmark traffic goes over the data plane when one is configured;
    // ssh (and therefore `server_addr`) may ride a management NIC.
    let endpoint = server
        .host
        .data_addr
        .clone()
        .unwrap_or_else(|| peer_endpoint(&server_addr).to_string());
    let target = format!("{endpoint}:{port}");
    let outcome = tokio::time::timeout(params.timeout, probe_pair(&client, &target, params)).await;

    stop_peer(&server, &target, port, child).await;

    match outcome {
        Ok(Ok((latency, bandwidth))) => {
            for record in pair_metrics(&latency, &bandwidth, &server_addr) {
                sink.metric(&client_addr, record);
            }
            for record in pair_metrics(&latency, &bandwidth, &client_addr) {
                sink.metric(&server_addr, record);
            }
            debug!(
                client = %client_addr,
                server = %server_addr,
                rtt_p50_us = latency.rtt_p50_us,
                gib_per_sec = bandwidth.gib_per_sec,
                "pair complete"
            );
        }
        Ok(Err(error)) => sink.error(
            &client_addr,
            format!("pair {client_addr} -> {server_addr}:{port} failed: {error:#}"),
        ),
        Err(_) => sink.error(
            &client_addr,
            format!(
                "pair {client_addr} -> {server_addr}:{port} timed out after {}s",
                params.timeout.as_secs()
            ),
        ),
    }
}

async fn probe_pair(
    client: &HostSession,
    target: &str,
    params: PairParams,
) -> Result<(LatencyReport, BandwidthReport)> {
    let latency_secs = params.latency_secs.to_string();
    let latency: LatencyReport = client
        .run_agent_capture(
            &["peer", "latency", target, "--duration-secs", &latency_secs],
            None,
        )
        .await
        .and_then(|output| agent_json(output, "peer latency"))?;

    let bandwidth_secs = params.bandwidth_secs.to_string();
    let bandwidth: BandwidthReport = client
        .run_agent_capture(
            &[
                "peer",
                "bandwidth",
                target,
                "--duration-secs",
                &bandwidth_secs,
            ],
            None,
        )
        .await
        .and_then(|output| agent_json(output, "peer bandwidth"))?;

    Ok((latency, bandwidth))
}

/// Graceful shutdown frame first; a remote `pkill` as the backstop for
/// fleets where the operator cannot reach the peer port directly.
async fn stop_peer(
    server: &HostSession,
    target: &str,
    port: u16,
    child: openssh::Child<Arc<openssh::Session>>,
) {
    match tokio::time::timeout(
        PEER_SHUTDOWN_TIMEOUT,
        crate::agent::net::shutdown_peer(target),
    )
    .await
    {
        Ok(Ok(())) => debug!(target, "peer shut down"),
        Ok(Err(error)) => debug!(target, %error, "graceful peer shutdown failed"),
        Err(_) => debug!(target, "graceful peer shutdown timed out"),
    }
    // The bracket keeps the pattern from matching the pkill invocation itself.
    let pattern = format!("[g]auntlet-agent peer serve --port {port}");
    if let Err(error) = server
        .exec_capture(&format!("pkill -f {}", single_quote(&pattern)))
        .await
    {
        debug!(host = %server.addr(), %error, "peer cleanup command failed");
    }
    if tokio::time::timeout(PEER_SHUTDOWN_TIMEOUT, child.wait())
        .await
        .is_err()
    {
        debug!(host = %server.addr(), port, "peer did not exit promptly");
    }
}

/// Pairwise results are attributed to both ends, each recorded against the
/// other host as `Scope::HostPair`.
fn pair_metrics(
    latency: &LatencyReport,
    bandwidth: &BandwidthReport,
    peer: &str,
) -> Vec<MetricRecord> {
    let scope = Scope::HostPair {
        peer: peer.to_string(),
    };
    vec![
        MetricRecord {
            test: TestId::NetLatency,
            scope: scope.clone(),
            name: "rtt_p50".into(),
            value: latency.rtt_p50_us,
            unit: Unit::Micros,
            repeat: 0,
        },
        MetricRecord {
            test: TestId::NetLatency,
            scope: scope.clone(),
            name: "rtt_p99".into(),
            value: latency.rtt_p99_us,
            unit: Unit::Micros,
            repeat: 0,
        },
        MetricRecord {
            test: TestId::NetLatency,
            scope: scope.clone(),
            name: "rtt_max".into(),
            value: latency.rtt_max_us,
            unit: Unit::Micros,
            repeat: 0,
        },
        MetricRecord {
            test: TestId::NetBandwidth,
            scope,
            name: "gib_per_sec".into(),
            value: bandwidth.gib_per_sec,
            unit: Unit::GibPerSec,
            repeat: 0,
        },
    ]
}

/// The address peers use to reach a host: the ssh destination minus the
/// login user and any ssh port.
fn peer_endpoint(ssh_addr: &str) -> &str {
    let host = ssh_addr
        .rsplit_once('@')
        .map(|(_, host)| host)
        .unwrap_or(ssh_addr);
    if let Some((name, port)) = host.rsplit_once(':') {
        // A second colon means a bare IPv6 literal, which has no ssh port.
        if !name.contains(':') && !port.is_empty() && port.chars().all(|c| c.is_ascii_digit()) {
            return name;
        }
    }
    host
}

/// Fleet-wide NCCL sweep: rank 0 mints the unique id, this process relays it
/// to every rank, and rank 0's event stream carries the measurements.
async fn nccl_sweep(
    config: &FleetConfig,
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
    sink: &ObservationSink,
) {
    let gpu_hosts = gpu_bearing_hosts(sessions, inventories).await;
    if gpu_hosts.is_empty() {
        info!("no GPU-bearing hosts; skipping the NCCL sweep");
        return;
    }
    // The inventory dlopen probe knows whether libnccl actually loads. A
    // fleet without the NCCL stack skips the sweep as a structural finding
    // (visible in the inventory/consistency/bootstrap output) instead of
    // manufacturing a host failure out of a loader panic. Hosts predating
    // the probe (empty map) are given the benefit of the doubt.
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
            "libnccl is not loadable on any GPU-bearing host; skipping the NCCL sweep"
        );
        return;
    }
    if nccl_hosts.len() < gpu_hosts.len() {
        warn!(
            with_nccl = nccl_hosts.len(),
            without_nccl = gpu_hosts.len() - nccl_hosts.len(),
            "some GPU-bearing hosts lack a loadable libnccl and are excluded from the sweep"
        );
    }
    let gpu_hosts = nccl_hosts;
    let rank0 = &gpu_hosts[0];
    let world_size = gpu_hosts.len() as u32;

    info!(world_size, rank0 = %rank0.addr(), "NCCL sweep");

    let timeout = Duration::from_secs(config.tests.phase_timeout_secs.max(1));
    let mut tasks = JoinSet::new();

    // Rank 0 leads: it mints the rendezvous id in-process (the id's bootstrap
    // listen socket must live in the process that serves as rank 0) and
    // announces it as an NcclId event, which is intercepted here and relayed
    // to the other ranks.
    let lead = NcclDirective::Lead {
        world_size,
        sizes: config.tests.nccl_sizes.clone(),
        iters_per_size: config.tests.nccl_iters_per_size,
        socket_ifname: config.nccl.socket_ifname.clone(),
    };
    let document = match serde_json::to_string(&lead) {
        Ok(document) => document,
        Err(error) => {
            warn!(%error, "cannot serialize the NCCL lead directive");
            return;
        }
    };
    let (id_tx, id_rx) = tokio::sync::oneshot::channel::<String>();
    let id_slot = Arc::new(std::sync::Mutex::new(Some(id_tx)));
    {
        let session = Arc::clone(rank0);
        let sink = sink.clone();
        let id_slot = Arc::clone(&id_slot);
        tasks.spawn(async move {
            let addr = session.addr().to_string();
            let outcome = tokio::time::timeout(
                timeout,
                session.run_agent(&["nccl"], Some(document), |event| {
                    if let AgentEvent::NcclId { unique_id_b64 } = &event {
                        if let Some(tx) = id_slot.lock().expect("id slot poisoned").take() {
                            let _ = tx.send(unique_id_b64.clone());
                        }
                    } else {
                        sink.event(&addr, event);
                    }
                }),
            )
            .await;
            report_rank_outcome(&sink, &addr, 0, timeout, outcome);
        });
    }

    let unique_id_b64 = match tokio::time::timeout(NCCL_ID_WAIT, id_rx).await {
        Ok(Ok(id)) => id,
        Ok(Err(_)) | Err(_) => {
            // The lead task reports its own failure; just stop recruiting.
            warn!(
                rank0 = %rank0.addr(),
                "NCCL lead produced no rendezvous id; aborting the sweep"
            );
            while let Some(joined) = tasks.join_next().await {
                if let Err(error) = joined {
                    warn!(%error, "nccl task did not complete");
                }
            }
            return;
        }
    };

    for (rank, session) in gpu_hosts.iter().enumerate().skip(1) {
        let directive = NcclDirective::Participate {
            unique_id_b64: unique_id_b64.clone(),
            rank: rank as u32,
            world_size,
            sizes: config.tests.nccl_sizes.clone(),
            iters_per_size: config.tests.nccl_iters_per_size,
            socket_ifname: config.nccl.socket_ifname.clone(),
        };
        let document = match serde_json::to_string(&directive) {
            Ok(document) => document,
            Err(error) => {
                warn!(%error, rank, "cannot serialize the NCCL directive");
                continue;
            }
        };
        let session = Arc::clone(session);
        let sink = sink.clone();
        tasks.spawn(async move {
            let addr = session.addr().to_string();
            let outcome = tokio::time::timeout(
                timeout,
                session.run_agent(&["nccl"], Some(document), |event| sink.event(&addr, event)),
            )
            .await;
            report_rank_outcome(&sink, &addr, rank, timeout, outcome);
        });
    }
    while let Some(joined) = tasks.join_next().await {
        if let Err(error) = joined {
            warn!(%error, "nccl task did not complete");
        }
    }
}

/// Hosts with at least one GPU, in fleet order. Inventories collected during
/// phase 0 are reused; hosts without one (e.g. `--phases network`) are probed.
/// How long the orchestrator waits for the lead rank's NcclId event.
const NCCL_ID_WAIT: Duration = Duration::from_secs(30);

fn report_rank_outcome(
    sink: &ObservationSink,
    addr: &str,
    rank: usize,
    timeout: Duration,
    outcome: Result<anyhow::Result<std::process::ExitStatus>, tokio::time::error::Elapsed>,
) {
    match outcome {
        Ok(Ok(status)) if status.success() => {}
        Ok(Ok(status)) => sink.error(addr, format!("nccl rank {rank} exited with {status}")),
        Ok(Err(error)) => sink.error(addr, format!("nccl rank {rank} failed: {error:#}")),
        Err(_) => sink.error(
            addr,
            format!("nccl rank {rank} timed out after {}s", timeout.as_secs()),
        ),
    }
}

async fn gpu_bearing_hosts(
    sessions: &[Arc<HostSession>],
    inventories: &mut BTreeMap<String, InventorySnapshot>,
) -> Vec<Arc<HostSession>> {
    let missing: Vec<(usize, Arc<HostSession>)> = sessions
        .iter()
        .enumerate()
        .filter(|(_, session)| !inventories.contains_key(session.addr()))
        .map(|(index, session)| (index, Arc::clone(session)))
        .collect();

    if !missing.is_empty() {
        debug!(hosts = missing.len(), "probing hosts for GPU presence");
        let mut tasks = JoinSet::new();
        for (_, session) in missing {
            tasks.spawn(async move {
                let addr = session.addr().to_string();
                let probed = session
                    .run_agent_capture(&["probe"], None)
                    .await
                    .and_then(|output| agent_json::<InventorySnapshot>(output, "probe"));
                (addr, probed)
            });
        }
        while let Some(joined) = tasks.join_next().await {
            match joined {
                Ok((addr, Ok(inventory))) => {
                    inventories.insert(addr, inventory);
                }
                Ok((addr, Err(error))) => debug!(host = %addr, %error, "probe failed"),
                Err(error) => warn!(%error, "probe task did not complete"),
            }
        }
    }

    sessions
        .iter()
        .filter(|session| {
            inventories
                .get(session.addr())
                .is_some_and(|inventory| !inventory.gpus.is_empty())
        })
        .map(Arc::clone)
        .collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse the JSON document an agent sub-command printed on stdout. The whole
/// stream is tried first (pretty-printed documents span lines); failing that,
/// the last well-formed line wins, so a stray banner cannot break parsing.
fn agent_json<T: DeserializeOwned>(output: session::RemoteOutput, what: &str) -> Result<T> {
    if !output.success() {
        bail!("{what} failed: {}", output.detail());
    }
    parse_json_document(&output.stdout).with_context(|| format!("parsing {what} output"))
}

fn parse_json_document<T: DeserializeOwned>(text: &str) -> Result<T> {
    if let Ok(value) = serde_json::from_str(text.trim()) {
        return Ok(value);
    }
    for line in text.lines().rev() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str(line) {
            return Ok(value);
        }
    }
    bail!("no JSON document on stdout")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_endpoints_drop_the_login_user_and_ssh_port() {
        assert_eq!(peer_endpoint("node1"), "node1");
        assert_eq!(peer_endpoint("root@node1"), "node1");
        assert_eq!(peer_endpoint("root@node1:2222"), "node1");
        assert_eq!(peer_endpoint("10.0.0.4:2222"), "10.0.0.4");
        // Bare IPv6 literals keep every colon they came with.
        assert_eq!(peer_endpoint("fd00::1"), "fd00::1");
        assert_eq!(peer_endpoint("user@fd00::1"), "fd00::1");
    }

    #[test]
    fn pair_slots_get_consecutive_ports_and_stop_at_the_ceiling() {
        assert_eq!(port_for_slot(29500, 0), Some(29500));
        assert_eq!(port_for_slot(29500, 7), Some(29507));
        assert_eq!(port_for_slot(65535, 1), None);
        assert_eq!(port_for_slot(29500, usize::MAX), None);
    }

    #[test]
    fn pair_metrics_cover_the_distribution_and_bandwidth() {
        let latency = LatencyReport {
            samples: 100,
            rtt_p50_us: 12.0,
            rtt_p99_us: 40.0,
            rtt_max_us: 90.0,
        };
        let bandwidth = BandwidthReport {
            bytes_sent: 1 << 30,
            gib_per_sec: 11.5,
        };
        let records = pair_metrics(&latency, &bandwidth, "node2");
        let names: Vec<&str> = records.iter().map(|record| record.name.as_str()).collect();
        assert_eq!(names, ["rtt_p50", "rtt_p99", "rtt_max", "gib_per_sec"]);
        for record in &records {
            assert_eq!(
                record.scope,
                Scope::HostPair {
                    peer: "node2".to_string()
                }
            );
        }
        assert_eq!(records[0].unit, Unit::Micros);
        assert_eq!(records[3].unit, Unit::GibPerSec);
        assert_eq!(records[3].test, TestId::NetBandwidth);
    }

    #[test]
    fn json_documents_survive_pretty_printing_and_leading_noise() {
        #[derive(serde::Deserialize)]
        struct Doc {
            unique_id_b64: String,
        }
        let pretty = "{\n  \"unique_id_b64\": \"abc\"\n}\n";
        let parsed: Doc = parse_json_document(pretty).expect("pretty json");
        assert_eq!(parsed.unique_id_b64, "abc");

        let noisy = "warming up\n{\"unique_id_b64\":\"xyz\"}\n";
        let parsed: Doc = parse_json_document(noisy).expect("last line json");
        assert_eq!(parsed.unique_id_b64, "xyz");

        assert!(parse_json_document::<Doc>("nothing here").is_err());
    }
}
