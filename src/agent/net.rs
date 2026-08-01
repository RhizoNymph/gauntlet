//! Phase 3a: two-sided TCP tests between agents (`gauntlet agent peer`).
//!
//! The orchestrator starts a `Serve` peer on one host of each scheduled
//! pair, then a `Latency` and a `Bandwidth` client on the other, and
//! attributes both hosts' events to the pair (`Scope::HostPair`).
//!
//! Latency: ping-pong of 8-byte payloads with TCP_NODELAY; report the RTT
//! distribution — p50/p99/max in microseconds — never just the mean, because
//! collectives ride on the tail. Bandwidth: sustained bulk stream, GiB/s.
//!
//! Wire format between peers: 1 magic byte selecting mode (latency /
//! bandwidth / shutdown) then mode-specific framing. The serving peer
//! handles clients sequentially and exits on shutdown.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::cli::{PeerArgs, PeerCommand};

/// Mode selector, sent by the client as the very first byte.
const MAGIC_LATENCY: u8 = 0x01;
const MAGIC_BANDWIDTH: u8 = 0x02;
const MAGIC_SHUTDOWN: u8 = 0xFF;

/// Ping-pong payload. Eight bytes carry a sequence number and stay well
/// inside one segment, so the measurement is round-trip time, not
/// serialization time.
const LATENCY_PAYLOAD_BYTES: usize = 8;
/// Bulk transfer unit. Big enough to keep the send window full without
/// making the duration check coarse.
const BANDWIDTH_CHUNK_BYTES: usize = 4 << 20;
/// Connecting to a wedged or firewalled host must fail, not hang: the
/// orchestrator schedules rounds and cannot wait on one pair forever.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const GIB: f64 = (1u64 << 30) as f64;

/// Result of a latency probe, printed as JSON by the client peer and also
/// emitted as metrics by the orchestrator attribution layer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LatencyReport {
    pub samples: u64,
    pub rtt_p50_us: f64,
    pub rtt_p99_us: f64,
    pub rtt_max_us: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BandwidthReport {
    pub bytes_sent: u64,
    pub gib_per_sec: f64,
}

/// CLI entry: dispatch serve / latency / bandwidth.
///
/// Client modes print their report as a single JSON line on stdout, which is
/// what the orchestrator parses. `serve` prints nothing.
pub async fn peer(args: PeerArgs) -> Result<()> {
    match args.command {
        PeerCommand::Serve { port } => serve(port).await,
        PeerCommand::Latency {
            target,
            duration_secs,
        } => {
            let report = measure_latency(&target, duration_secs).await?;
            println!("{}", serde_json::to_string(&report)?);
            Ok(())
        }
        PeerCommand::Bandwidth {
            target,
            duration_secs,
        } => {
            let report = measure_bandwidth(&target, duration_secs).await?;
            println!("{}", serde_json::to_string(&report)?);
            Ok(())
        }
    }
}

/// Listen on `port` until a shutdown frame arrives.
///
/// Clients are served strictly one at a time: a concurrent probe would
/// contend for the same NIC and corrupt the measurement it is trying to
/// take. A client that dies mid-test is logged and skipped; only the
/// shutdown frame ends the loop.
pub async fn serve(port: u16) -> Result<()> {
    let listener = TcpListener::bind(("0.0.0.0", port))
        .await
        .with_context(|| format!("binding peer listener on 0.0.0.0:{port}"))?;
    loop {
        let (mut socket, remote) = listener.accept().await.context("accepting peer")?;
        let mut magic = [0u8; 1];
        if let Err(error) = socket.read_exact(&mut magic).await {
            tracing::warn!(%remote, %error, "peer closed before selecting a mode");
            continue;
        }
        match magic[0] {
            MAGIC_SHUTDOWN => {
                tracing::debug!(%remote, "shutdown frame received");
                return Ok(());
            }
            MAGIC_LATENCY => {
                if let Err(error) = serve_latency(&mut socket).await {
                    tracing::warn!(%remote, %error, "latency session ended early");
                }
            }
            MAGIC_BANDWIDTH => {
                if let Err(error) = serve_bandwidth(&mut socket).await {
                    tracing::warn!(%remote, %error, "bandwidth session ended early");
                }
            }
            other => tracing::warn!(%remote, magic = other, "unknown peer mode"),
        }
    }
}

/// Echo fixed-size payloads until the client hangs up.
async fn serve_latency(socket: &mut TcpStream) -> Result<()> {
    socket.set_nodelay(true).context("TCP_NODELAY")?;
    let mut payload = [0u8; LATENCY_PAYLOAD_BYTES];
    loop {
        match socket.read_exact(&mut payload).await {
            Ok(_) => socket.write_all(&payload).await.context("echo")?,
            // A clean close is how the client says "done".
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(error).context("reading ping"),
        }
    }
}

/// Drain the client's stream, then report the byte count back in a trailing
/// 8-byte big-endian frame. Counting on the receiving side is the point:
/// bytes the client handed to the kernel but that never arrived must not be
/// credited as throughput.
async fn serve_bandwidth(socket: &mut TcpStream) -> Result<()> {
    let mut buffer = vec![0u8; BANDWIDTH_CHUNK_BYTES];
    let mut received: u64 = 0;
    loop {
        let read = socket
            .read(&mut buffer)
            .await
            .context("reading bulk data")?;
        if read == 0 {
            break;
        }
        received += read as u64;
    }
    socket
        .write_all(&received.to_be_bytes())
        .await
        .context("sending byte count")?;
    socket.flush().await.context("flushing byte count")?;
    Ok(())
}

/// Client side, callable in-process (tests run serve+measure over
/// localhost).
pub async fn measure_latency(target: &str, duration_secs: u64) -> Result<LatencyReport> {
    let mut socket = connect(target).await?;
    socket.write_all(&[MAGIC_LATENCY]).await.context("mode")?;

    let budget = Duration::from_secs(duration_secs);
    let start = Instant::now();
    let mut payload = [0u8; LATENCY_PAYLOAD_BYTES];
    let mut echoed = [0u8; LATENCY_PAYLOAD_BYTES];
    let mut samples_us: Vec<f64> = Vec::new();
    let mut sequence: u64 = 0;

    loop {
        payload.copy_from_slice(&sequence.to_be_bytes());
        let sent = Instant::now();
        socket.write_all(&payload).await.context("sending ping")?;
        socket
            .read_exact(&mut echoed)
            .await
            .context("awaiting echo")?;
        let rtt = sent.elapsed();
        if echoed != payload {
            bail!("peer echoed a corrupted payload at sequence {sequence}");
        }
        samples_us.push(rtt.as_secs_f64() * 1e6);
        sequence += 1;
        if start.elapsed() >= budget {
            break;
        }
    }

    // Half-close so the server's echo loop sees EOF and moves on to the next
    // client instead of blocking on a read.
    socket.shutdown().await.context("half-closing")?;

    samples_us.sort_by(f64::total_cmp);
    Ok(LatencyReport {
        samples: samples_us.len() as u64,
        rtt_p50_us: percentile(&samples_us, 0.50),
        rtt_p99_us: percentile(&samples_us, 0.99),
        rtt_max_us: samples_us.last().copied().unwrap_or(0.0),
    })
}

pub async fn measure_bandwidth(target: &str, duration_secs: u64) -> Result<BandwidthReport> {
    let mut socket = connect(target).await?;
    socket.write_all(&[MAGIC_BANDWIDTH]).await.context("mode")?;

    let chunk = vec![0xa5u8; BANDWIDTH_CHUNK_BYTES];
    let budget = Duration::from_secs(duration_secs);
    let start = Instant::now();
    loop {
        socket.write_all(&chunk).await.context("streaming")?;
        if start.elapsed() >= budget {
            break;
        }
    }
    socket.shutdown().await.context("half-closing")?;

    let mut trailer = [0u8; 8];
    socket
        .read_exact(&mut trailer)
        .await
        .context("reading the peer's byte count")?;
    // The window runs until the receiver has acknowledged everything: bytes
    // still in flight at half-close are part of the transfer, not free.
    let seconds = start.elapsed().as_secs_f64();
    let received = u64::from_be_bytes(trailer);

    if seconds <= 0.0 || !seconds.is_finite() {
        bail!("timer returned {seconds}s for the transfer window");
    }
    Ok(BandwidthReport {
        bytes_sent: received,
        gib_per_sec: received as f64 / seconds / GIB,
    })
}

/// Ask a serving peer to exit.
pub async fn shutdown_peer(target: &str) -> Result<()> {
    let mut socket = connect(target).await?;
    socket
        .write_all(&[MAGIC_SHUTDOWN])
        .await
        .context("sending shutdown")?;
    socket.flush().await.context("flushing shutdown")?;
    socket.shutdown().await.context("closing")?;
    Ok(())
}

async fn connect(target: &str) -> Result<TcpStream> {
    let socket = tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(target))
        .await
        .with_context(|| format!("connecting to peer {target} timed out"))?
        .with_context(|| format!("connecting to peer {target}"))?;
    // Every mode wants latency-faithful framing; Nagle would merge pings.
    socket.set_nodelay(true).context("TCP_NODELAY")?;
    Ok(socket)
}

/// Nearest-rank percentile over an ascending slice. Monotone in `fraction`,
/// so p50 <= p99 <= max holds by construction.
fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let last = sorted.len() - 1;
    let index = (fraction * last as f64).ceil() as usize;
    sorted[index.min(last)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_are_ordered() {
        let samples: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let p50 = percentile(&samples, 0.50);
        let p99 = percentile(&samples, 0.99);
        let max = *samples.last().expect("non-empty");
        assert!(p50 <= p99 && p99 <= max, "{p50} {p99} {max}");
        assert_eq!(max, 100.0);
    }

    #[test]
    fn percentiles_of_degenerate_inputs() {
        assert_eq!(percentile(&[], 0.5), 0.0);
        assert_eq!(percentile(&[7.0], 0.99), 7.0);
    }
}
