use anyhow::Result;
use clap::Parser;
use gauntlet::cli::{AgentCommand, Cli, Command};

fn main() -> Result<()> {
    let cli = Cli::parse();
    gauntlet::init_tracing(cli.verbose);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    match cli.command {
        Command::Run(args) => runtime.block_on(gauntlet::orchestrator::run(args)),
        Command::Bootstrap(args) => runtime.block_on(gauntlet::orchestrator::bootstrap::run(args)),
        Command::Report(args) => gauntlet::report::render_saved(args),
        Command::Agent(args) => match args.command {
            AgentCommand::Run(run) => runtime.block_on(gauntlet::agent::run(run)),
            AgentCommand::Probe => gauntlet::agent::probe(),
            AgentCommand::Peer(peer) => runtime.block_on(gauntlet::agent::net::peer(peer)),
            AgentCommand::Nccl => gauntlet::agent::nccl::run_from_stdin(),
        },
    }
}
