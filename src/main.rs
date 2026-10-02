use std::process::ExitCode;

use anyhow::Result;
use clap::Parser;
use gauntlet::cli::{AgentCommand, Cli, Command, parse_failure_exit_code};
use gauntlet::report::EXIT_ERROR;

fn main() -> ExitCode {
    // clap's own exit would use 2 for a usage error, which is the
    // host-failures verdict; map it onto EXIT_ERROR instead.
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            // Help/version go to stdout, usage errors to stderr.
            let _ = error.print();
            return ExitCode::from(parse_failure_exit_code(&error));
        }
    };
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // Not a verdict: gauntlet itself could not do its job (bad
            // config, no usable host, I/O). A distinct code keeps this from
            // reading as exit 1 ("outliers only").
            eprintln!("Error: {error:?}");
            ExitCode::from(EXIT_ERROR)
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    gauntlet::init_tracing(cli.verbose);
    if matches!(cli.command, Command::Agent(_)) {
        // Before the runtime exists (its workers start at build time): move
        // the protocol off fd 1 so library stdout cannot corrupt it.
        gauntlet::agent::channel::isolate_stdout()?;
    }
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
            AgentCommand::Barrier(barrier) => {
                runtime.block_on(gauntlet::agent::barrier::barrier(barrier))
            }
            AgentCommand::StdoutIsolationCheck => gauntlet::agent::channel::isolation_check(),
        },
    }
}
