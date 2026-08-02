use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "gauntlet",
    about = "Cluster pre-flight benchmark and health check"
)]
pub struct Cli {
    /// Enable debug logging (stderr).
    #[arg(short, long, global = true)]
    pub verbose: bool,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the benchmark suite across the fleet.
    Run(RunArgs),
    /// Prepare every node: connectivity, agent deploy, capability probe, optional tuning.
    Bootstrap(BootstrapArgs),
    /// Re-render a saved results JSON as a table.
    Report(ReportArgs),
    /// Node-side mode; invoked over ssh by the orchestrator, not by hand.
    Agent(AgentArgs),
}

#[derive(Debug, Args)]
pub struct RunArgs {
    /// Fleet config (TOML).
    #[arg(long, short, default_value = "gauntlet.toml")]
    pub config: PathBuf,
    /// Write results JSON here (default: runs/<run-id>.json).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Restrict to a subset of phases (inventory, cpu_mem, gpu, network).
    #[arg(long, value_delimiter = ',')]
    pub phases: Vec<String>,
    /// Sampled network mode: test only this many pairs per host instead of full mesh.
    #[arg(long)]
    pub sample_pairs: Option<usize>,
}

#[derive(Debug, Args)]
pub struct BootstrapArgs {
    /// Fleet config (TOML).
    #[arg(long, short, default_value = "gauntlet.toml")]
    pub config: PathBuf,
    /// Apply node tuning (GPU persistence mode, performance governor). Needs sudo on nodes.
    #[arg(long)]
    pub tune: bool,
    /// Emit the readiness report as JSON on stdout instead of a table.
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct ReportArgs {
    /// Path to a results JSON produced by `gauntlet run`.
    pub input: PathBuf,
    /// Emit JSON to stdout instead of a table.
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct AgentArgs {
    #[command(subcommand)]
    pub command: AgentCommand,
}

#[derive(Debug, Subcommand)]
pub enum AgentCommand {
    /// Execute test phases; reads an AgentTaskSpec JSON from stdin,
    /// emits AgentEvent JSON-lines on stdout.
    Run(AgentRunArgs),
    /// Print an InventorySnapshot JSON (used by bootstrap for capability detection).
    Probe,
    /// Two-sided network test peer.
    Peer(PeerArgs),
    /// NCCL participant; reads an NcclDirective JSON from stdin.
    Nccl,
}

#[derive(Debug, Args)]
pub struct AgentRunArgs {
    /// Override phases from the task spec (testing convenience).
    #[arg(long, value_delimiter = ',')]
    pub phases: Vec<String>,
}

#[derive(Debug, Args)]
pub struct PeerArgs {
    #[command(subcommand)]
    pub command: PeerCommand,
}

#[derive(Debug, Subcommand)]
pub enum PeerCommand {
    /// Listen for latency/bandwidth probes from other agents.
    Serve {
        #[arg(long)]
        port: u16,
    },
    /// Measure RTT distribution to a serving peer.
    Latency {
        target: String,
        #[arg(long, default_value_t = 3)]
        duration_secs: u64,
    },
    /// Measure sustained TCP throughput to a serving peer.
    Bandwidth {
        target: String,
        #[arg(long, default_value_t = 5)]
        duration_secs: u64,
    },
}
