//! CLI surface: help text generated from the phase table, so it cannot go
//! stale when a phase is added.

use clap::CommandFactory;
use gauntlet::cli::Cli;
use gauntlet::proto::Phase;

fn run_help() -> String {
    let mut command = Cli::command();
    let run = command
        .find_subcommand_mut("run")
        .expect("run subcommand exists");
    run.render_long_help().to_string()
}

fn phases_arg_help() -> String {
    let command = Cli::command();
    let run = command
        .find_subcommand("run")
        .expect("run subcommand exists");
    let arg = run
        .get_arguments()
        .find(|arg| arg.get_id() == "phases")
        .expect("--phases exists");
    arg.get_help().expect("--phases has help").to_string()
}

#[test]
fn every_phase_appears_in_the_phases_help() {
    let help = phases_arg_help();
    for phase in Phase::ALL {
        assert!(
            help.contains(phase.name()),
            "--phases help omits {:?}: {help}",
            phase
        );
    }
    // And in the rendered `gauntlet run --help` page.
    let page = run_help();
    for phase in Phase::ALL {
        assert!(page.contains(phase.name()), "{page}");
    }
}

#[test]
fn phase_names_round_trip_through_parse_and_serde() {
    for phase in Phase::ALL {
        assert_eq!(Phase::parse(phase.name()), Some(phase));
        let json = serde_json::to_string(&phase).expect("serialize");
        assert_eq!(json, format!("\"{}\"", phase.name()));
    }
}

#[test]
fn every_parse_alias_resolves_to_a_listed_phase() {
    for (alias, phase) in Phase::PARSE_TABLE {
        assert_eq!(Phase::parse(alias), Some(*phase));
        assert!(Phase::ALL.contains(phase));
    }
    // Every phase is reachable by its canonical name.
    for phase in Phase::ALL {
        assert!(
            Phase::PARSE_TABLE
                .iter()
                .any(|(alias, target)| *alias == phase.name() && *target == phase)
        );
    }
    assert_eq!(Phase::parse("warp_drive"), None);
}

#[test]
fn phases_help_lists_aliases() {
    let help = phases_arg_help();
    assert!(help.contains("cpu"), "{help}");
    assert!(help.contains("net"), "{help}");
}
