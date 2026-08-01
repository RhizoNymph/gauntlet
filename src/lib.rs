pub mod agent;
pub mod analysis;
pub mod cli;
pub mod config;
pub mod orchestrator;
pub mod proto;
pub mod report;

use tracing_subscriber::EnvFilter;

/// Initialize structured logging. Log output goes to stderr so agent-mode
/// stdout stays reserved for the JSON-lines event protocol.
pub fn init_tracing(verbose: bool) {
    let default = if verbose { "debug" } else { "info" };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}
