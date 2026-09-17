//! Barrier-skew fallback for CPU-only fleets: a TCP star barrier
//! (`gauntlet agent barrier`).
//!
//! One host serves as the coordinator; every fleet member (including the
//! coordinator's own host, as a separate `join` process) connects and
//! announces its orchestrator-assigned rank. Per iteration the coordinator
//! releases all ranks at once and measures, on its single clock, each
//! rank's release-to-response time. A rank that is consistently the last
//! to respond is the straggler — the same signal the NCCL barrier
//! extracts, but observable without GPUs.
//!
//! What the number means: release-to-response = server->client latency +
//! client scheduling delay + client->server latency. Network-path variance
//! and host OS jitter are deliberately *both* in the measurement — that is
//! what a training-step barrier experiences. A rank with a systematically
//! longer path shows a higher baseline; the per-iteration tally with a
//! margin discriminates a consistent late arriver from baseline offsets
//! only when the gap is nontrivial, and the per-rank distribution shape
//! (p50 vs p99) separates "far away" from "jittery".
//!
//! Wire format: client hello `[0xB7][rank: u32 be]`, then per iteration
//! one release byte from the server answered by one ack byte from the
//! client. TCP_NODELAY everywhere; single-byte frames, so the measurement
//! is response time, not serialization.
//!
//! The serving side prints a single `TcpBarrierReport` JSON line on
//! stdout (the orchestrator parses it); `join` prints nothing.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, watch};

use crate::analysis::skew::{self, BarrierSkew, Margin, SkewPolarity};
use crate::cli::{BarrierArgs, BarrierCommand};

/// First byte of a client hello.
const MAGIC_JOIN: u8 = 0xB7;
/// Server -> client "go" byte for one iteration.
const RELEASE: u8 = 0x52;
/// Client -> server response byte.
const ACK: u8 = 0x41;
/// Window for the full fleet to connect and announce ranks.
const ACCEPT_TIMEOUT: Duration = Duration::from_secs(60);
/// A rank that has not responded to a release within this long is wedged;
/// fail the run instead of hanging the phase.
const ITER_TIMEOUT: Duration = Duration::from_secs(30);
/// Clients may start before the server has bound; retry connects within
/// this window.
const JOIN_CONNECT_WINDOW: Duration = Duration::from_secs(15);
const JOIN_CONNECT_RETRY: Duration = Duration::from_millis(250);

/// The coordinator's result document: the full skew analysis, computed
/// server-side because the coordinator is the only process that sees every
/// rank on one clock (per-iteration raw vectors would be megabytes at
/// fleet scale; the analysis is the compact part).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TcpBarrierReport {
    pub world: u32,
    pub iters: u32,
    pub skew: BarrierSkew,
}

/// CLI entry: dispatch serve / join.
pub async fn barrier(args: BarrierArgs) -> Result<()> {
    match args.command {
        BarrierCommand::Serve { port, world, iters } => {
            let listener = TcpListener::bind(("0.0.0.0", port))
                .await
                .with_context(|| format!("binding barrier listener on 0.0.0.0:{port}"))?;
            let report = run_server(listener, world, iters).await?;
            println!("{}", serde_json::to_string(&report)?);
            Ok(())
        }
        BarrierCommand::Join {
            target,
            rank,
            iters,
        } => join(&target, rank, iters).await,
    }
}

/// Accept `world` ranks, run `iters` release/ack rounds, and analyze.
/// Callable in-process with an ephemeral listener (tests).
pub async fn run_server(listener: TcpListener, world: u32, iters: u32) -> Result<TcpBarrierReport> {
    if world < 2 {
        bail!("a barrier over {world} rank(s) has no skew to measure");
    }
    if iters == 0 {
        bail!("barrier iterations must be at least 1");
    }

    let clients = accept_fleet(&listener, world).await?;

    // One task per rank owns its socket. A watch channel releases every
    // task at (nearly) the same instant; each task measures its own
    // write-release-to-read-ack elapsed locally, so sequential send bias
    // never enters the numbers, and results return over an mpsc channel.
    let (release_tx, _) = watch::channel(0u32);
    let (result_tx, mut result_rx) = mpsc::channel::<(u32, Result<f64, String>)>(world as usize);
    for (rank, mut stream) in clients {
        let mut release_rx = release_tx.subscribe();
        let result_tx = result_tx.clone();
        tokio::spawn(async move {
            let mut ack = [0u8; 1];
            for _ in 0..iters {
                if release_rx.changed().await.is_err() {
                    // Coordinator gave up; nothing left to report.
                    return;
                }
                let start = Instant::now();
                let round = async {
                    stream.write_all(&[RELEASE]).await.context("release")?;
                    let read = tokio::time::timeout(ITER_TIMEOUT, stream.read_exact(&mut ack))
                        .await
                        .context("ack timed out")?;
                    read.context("reading ack")?;
                    if ack[0] != ACK {
                        bail!("rank answered 0x{:02x}, not an ack", ack[0]);
                    }
                    Ok::<f64, anyhow::Error>(start.elapsed().as_secs_f64() * 1e6)
                }
                .await;
                let failed = round.is_err();
                let message = round.map_err(|error| format!("{error:#}"));
                if result_tx.send((rank, message)).await.is_err() || failed {
                    return;
                }
            }
        });
    }
    drop(result_tx);

    let mut per_rank: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    for iteration in 1..=iters {
        release_tx
            .send(iteration)
            .ok()
            .context("all barrier rank tasks exited early")?;
        for _ in 0..world {
            let waited = tokio::time::timeout(ITER_TIMEOUT, result_rx.recv())
                .await
                .with_context(|| format!("iteration {iteration} stalled"))?;
            let Some((rank, outcome)) = waited else {
                bail!("a rank task exited before iteration {iteration} completed");
            };
            let elapsed_us = outcome.map_err(|message| {
                anyhow::anyhow!("rank {rank} failed at iteration {iteration}: {message}")
            })?;
            per_rank.entry(rank).or_default().push(elapsed_us);
        }
    }

    let series: Vec<skew::RankSeries> = per_rank
        .into_iter()
        .map(|(rank, elapsed_us)| skew::RankSeries { rank, elapsed_us })
        .collect();
    let skew = skew::analyze(&series, SkewPolarity::LateIsMax, Margin::default())
        .context("barrier produced no analyzable iterations")?;
    Ok(TcpBarrierReport { world, iters, skew })
}

/// Collect `world` distinct ranks within the accept window.
async fn accept_fleet(listener: &TcpListener, world: u32) -> Result<BTreeMap<u32, TcpStream>> {
    let deadline = Instant::now() + ACCEPT_TIMEOUT;
    let mut clients: BTreeMap<u32, TcpStream> = BTreeMap::new();
    while clients.len() < world as usize {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .context("timed out waiting for the fleet to join the barrier")?;
        let (mut stream, remote) = tokio::time::timeout(remaining, listener.accept())
            .await
            .context("timed out waiting for the fleet to join the barrier")?
            .context("accepting barrier client")?;
        stream.set_nodelay(true).context("TCP_NODELAY")?;
        let mut hello = [0u8; 5];
        if let Err(error) = tokio::time::timeout(ITER_TIMEOUT, stream.read_exact(&mut hello)).await
        {
            tracing::warn!(%remote, %error, "barrier client sent no hello");
            continue;
        }
        if hello[0] != MAGIC_JOIN {
            bail!(
                "client {remote} sent magic 0x{:02x}, not a barrier join",
                hello[0]
            );
        }
        let rank = u32::from_be_bytes([hello[1], hello[2], hello[3], hello[4]]);
        if rank >= world {
            bail!("client {remote} announced rank {rank} outside a world of {world}");
        }
        if clients.insert(rank, stream).is_some() {
            bail!("two clients announced rank {rank}");
        }
    }
    Ok(clients)
}

/// Client side: announce `rank`, then answer `iters` releases.
pub async fn join(target: &str, rank: u32, iters: u32) -> Result<()> {
    let mut stream = connect_with_retry(target).await?;
    stream.set_nodelay(true).context("TCP_NODELAY")?;

    let mut hello = [0u8; 5];
    hello[0] = MAGIC_JOIN;
    hello[1..5].copy_from_slice(&rank.to_be_bytes());
    stream.write_all(&hello).await.context("sending hello")?;

    let mut release = [0u8; 1];
    for iteration in 0..iters {
        stream
            .read_exact(&mut release)
            .await
            .with_context(|| format!("awaiting release {iteration}"))?;
        if release[0] != RELEASE {
            bail!("coordinator sent 0x{:02x}, not a release", release[0]);
        }
        stream
            .write_all(&[ACK])
            .await
            .with_context(|| format!("acking release {iteration}"))?;
    }
    Ok(())
}

/// The orchestrator starts serve and join concurrently; retry connects for
/// a bounded window so the join side tolerates the serve side binding late.
async fn connect_with_retry(target: &str) -> Result<TcpStream> {
    let deadline = Instant::now() + JOIN_CONNECT_WINDOW;
    loop {
        match TcpStream::connect(target).await {
            Ok(stream) => return Ok(stream),
            Err(error) => {
                if Instant::now() >= deadline {
                    return Err(error)
                        .with_context(|| format!("connecting to barrier coordinator {target}"));
                }
                tokio::time::sleep(JOIN_CONNECT_RETRY).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn local_listener() -> (TcpListener, String) {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind ephemeral");
        let target = listener.local_addr().expect("local addr").to_string();
        (listener, target)
    }

    #[tokio::test]
    async fn a_three_rank_barrier_produces_a_full_report() {
        let (listener, target) = local_listener().await;
        let world = 3u32;
        let iters = 50u32;
        let server = tokio::spawn(run_server(listener, world, iters));
        let mut joins = Vec::new();
        for rank in 0..world {
            let target = target.clone();
            joins.push(tokio::spawn(
                async move { join(&target, rank, iters).await },
            ));
        }
        for handle in joins {
            handle.await.expect("join task").expect("join ok");
        }
        let report = server.await.expect("server task").expect("report");
        assert_eq!(report.world, world);
        assert_eq!(report.iters, iters);
        assert_eq!(report.skew.iters, u64::from(iters));
        assert_eq!(report.skew.per_rank.len(), world as usize);
        for (index, rank) in report.skew.per_rank.iter().enumerate() {
            assert_eq!(rank.rank, index as u32);
            assert!(rank.p50_us > 0.0);
            assert!(rank.p50_us <= rank.p90_us);
            assert!(rank.p90_us <= rank.p99_us);
            assert!(rank.p99_us <= rank.max_us);
        }
        let tallied: u64 = report
            .skew
            .per_rank
            .iter()
            .map(|rank| rank.slowest_iters)
            .sum();
        assert_eq!(tallied, report.skew.considered_iters);
    }

    #[tokio::test]
    async fn duplicate_ranks_are_rejected() {
        let (listener, target) = local_listener().await;
        let server = tokio::spawn(run_server(listener, 2, 10));
        let a = tokio::spawn({
            let target = target.clone();
            async move { join(&target, 0, 10).await }
        });
        let b = tokio::spawn(async move { join(&target, 0, 10).await });
        let error = server
            .await
            .expect("server task")
            .expect_err("duplicate rank must fail the serve side");
        assert!(error.to_string().contains("rank 0"), "{error:#}");
        // The joining clients fail too (connection dropped); outcome shape
        // does not matter, only that they terminate.
        let _ = a.await;
        let _ = b.await;
    }

    #[tokio::test]
    async fn out_of_world_ranks_are_rejected() {
        let (listener, target) = local_listener().await;
        let server = tokio::spawn(run_server(listener, 2, 10));
        let client = tokio::spawn(async move { join(&target, 7, 10).await });
        let error = server
            .await
            .expect("server task")
            .expect_err("rank outside the world must fail");
        assert!(error.to_string().contains("rank 7"), "{error:#}");
        let _ = client.await;
    }

    #[tokio::test]
    async fn worlds_of_one_are_refused() {
        let (listener, _) = local_listener().await;
        let error = run_server(listener, 1, 10)
            .await
            .expect_err("no skew in a world of one");
        assert!(error.to_string().contains("no skew"), "{error:#}");
    }

    #[test]
    fn reports_round_trip_through_json() {
        let report = TcpBarrierReport {
            world: 2,
            iters: 3,
            skew: BarrierSkew {
                iters: 3,
                considered_iters: 2,
                per_rank: vec![
                    skew::RankSkew {
                        rank: 0,
                        p50_us: 10.0,
                        p90_us: 12.0,
                        p99_us: 13.0,
                        max_us: 14.0,
                        slowest_iters: 0,
                        slowest_frac: 0.0,
                    },
                    skew::RankSkew {
                        rank: 1,
                        p50_us: 200.0,
                        p90_us: 220.0,
                        p99_us: 230.0,
                        max_us: 240.0,
                        slowest_iters: 2,
                        slowest_frac: 1.0,
                    },
                ],
                fleet: skew::FleetBarrier {
                    p50_us: 200.0,
                    p90_us: 220.0,
                    p99_us: 230.0,
                    max_us: 240.0,
                },
            },
        };
        let json = serde_json::to_string(&report).expect("serialize");
        let back: TcpBarrierReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, report);
    }
}
