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

use anyhow::Result;

use serde::{Deserialize, Serialize};

use crate::cli::PeerArgs;

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
pub async fn peer(args: PeerArgs) -> Result<()> {
    let _ = args;
    todo!("agent B: implement")
}

/// Listen on `port` until a shutdown frame arrives.
pub async fn serve(port: u16) -> Result<()> {
    let _ = port;
    todo!("agent B: implement")
}

/// Client side, callable in-process (tests run serve+measure over
/// localhost).
pub async fn measure_latency(target: &str, duration_secs: u64) -> Result<LatencyReport> {
    let _ = (target, duration_secs);
    todo!("agent B: implement")
}

pub async fn measure_bandwidth(target: &str, duration_secs: u64) -> Result<BandwidthReport> {
    let _ = (target, duration_secs);
    todo!("agent B: implement")
}

/// Ask a serving peer to exit.
pub async fn shutdown_peer(target: &str) -> Result<()> {
    let _ = target;
    todo!("agent B: implement")
}
