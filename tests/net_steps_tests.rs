//! Network-phase step selection: `[tests] net_steps` / `--net-steps`.

use clap::CommandFactory;
use gauntlet::cli::Cli;
use gauntlet::config::{ConfigError, FleetConfig};
use gauntlet::net_steps::{DISABLED_REASON, NetStep, disabled_outcomes};
use gauntlet::proto::{TestId, TestOutcome};

fn parse(text: &str) -> Result<FleetConfig, ConfigError> {
    FleetConfig::from_toml_str(text, std::path::Path::new("test.toml"))
}

fn steps(config: &FleetConfig, cli: &[&str]) -> Result<Vec<NetStep>, ConfigError> {
    let cli: Vec<String> = cli.iter().map(|s| s.to_string()).collect();
    config
        .resolve_net_steps(&cli)
        .map(|steps| steps.iter().collect())
}

#[test]
fn every_step_runs_by_default() {
    let config = parse(r#"hosts = ["a", "b"]"#).expect("config");
    assert_eq!(config.tests.net_steps, NetStep::ALL.to_vec());
    assert_eq!(steps(&config, &[]).expect("resolve"), NetStep::ALL.to_vec());
}

#[test]
fn config_selects_a_subset_in_run_order() {
    let config = parse(
        r#"
        hosts = ["a", "b"]
        [tests]
        net_steps = ["nccl", "intranode"]
        "#,
    )
    .expect("config");
    assert_eq!(
        steps(&config, &[]).expect("resolve"),
        vec![NetStep::Intranode, NetStep::Nccl]
    );
}

#[test]
fn cli_overrides_config() {
    let config = parse(
        r#"
        hosts = ["a", "b"]
        [tests]
        net_steps = ["pairwise"]
        "#,
    )
    .expect("config");
    assert_eq!(
        steps(&config, &["nccl"]).expect("resolve"),
        vec![NetStep::Nccl]
    );
    // Aliases and duplicates.
    assert_eq!(
        steps(&config, &["tcp", "barrier", "pairwise"]).expect("resolve"),
        vec![NetStep::Pairwise, NetStep::Barrier]
    );
}

#[test]
fn unknown_and_empty_selections_are_rejected() {
    let config = parse(r#"hosts = ["a"]"#).expect("config");
    assert!(matches!(
        steps(&config, &["warp"]),
        Err(ConfigError::UnknownNetStep { .. })
    ));
    // An empty config selection loads (bootstrap and non-network runs never
    // look at it) but cannot be resolved for a network run.
    let empty = parse(
        r#"
        hosts = ["a"]
        [tests]
        net_steps = []
        "#,
    )
    .expect("empty net_steps loads");
    assert!(matches!(steps(&empty, &[]), Err(ConfigError::NoNetSteps)));
    assert!(
        parse(
            r#"
            hosts = ["a"]
            [tests]
            net_steps = ["warp"]
            "#
        )
        .is_err()
    );
}

#[test]
fn nccl_intranode_false_still_drops_the_intranode_step() {
    let config = parse(
        r#"
        hosts = ["a", "b"]
        [tests]
        nccl_intranode = false
        "#,
    )
    .expect("config");
    let resolved = steps(&config, &[]).expect("resolve");
    assert!(!resolved.contains(&NetStep::Intranode));
    assert!(resolved.contains(&NetStep::Nccl));
    // Even when asked for explicitly on the command line.
    assert_eq!(
        steps(&config, &["intranode", "nccl"]).expect("resolve"),
        vec![NetStep::Nccl]
    );
    // A selection that leaves nothing is an error, not a silent no-op.
    assert!(matches!(
        steps(&config, &["intranode"]),
        Err(ConfigError::NoNetSteps)
    ));
}

#[test]
fn nccl_only_run_skips_tcp_and_barriers_with_the_config_reason() {
    let config = parse(r#"hosts = ["a", "b"]"#).expect("config");
    let resolved = config
        .resolve_net_steps(&["nccl".to_string()])
        .expect("resolve");
    let outcomes = disabled_outcomes(&resolved);
    let tests: Vec<TestId> = outcomes.iter().map(|(test, _)| *test).collect();
    assert_eq!(
        tests,
        vec![
            TestId::NcclIntraAllReduce,
            TestId::NcclIntraAllGather,
            TestId::NetLatency,
            TestId::NetBandwidth,
            TestId::NcclBarrier,
            TestId::TcpBarrier,
        ]
    );
    for (_, outcome) in &outcomes {
        assert_eq!(
            outcome,
            &TestOutcome::Skipped {
                reason: DISABLED_REASON.to_string()
            }
        );
    }
    assert_eq!(DISABLED_REASON, "disabled by config");
}

#[test]
fn disabling_nccl_also_skips_the_nccl_barrier_that_rides_it() {
    let config = parse(r#"hosts = ["a", "b"]"#).expect("config");
    let resolved = config
        .resolve_net_steps(&["pairwise".into(), "barrier".into()])
        .expect("resolve");
    assert!(!resolved.nccl_barrier());
    assert!(resolved.contains(NetStep::Barrier));
    let tests: Vec<TestId> = disabled_outcomes(&resolved)
        .into_iter()
        .map(|(test, _)| test)
        .collect();
    assert!(tests.contains(&TestId::NcclAllReduce));
    assert!(tests.contains(&TestId::NcclAllGather));
    assert!(tests.contains(&TestId::NcclBarrier));
    assert!(!tests.contains(&TestId::TcpBarrier));
    assert!(!tests.contains(&TestId::NetLatency));
}

#[test]
fn all_steps_enabled_records_nothing() {
    let config = parse(r#"hosts = ["a", "b"]"#).expect("config");
    let resolved = config.resolve_net_steps(&[]).expect("resolve");
    assert!(resolved.nccl_barrier());
    assert!(disabled_outcomes(&resolved).is_empty());
}

#[test]
fn step_names_round_trip_and_appear_in_help() {
    for step in NetStep::ALL {
        assert_eq!(NetStep::parse(step.name()), Some(step));
        let json = serde_json::to_string(&step).expect("serialize");
        assert_eq!(json, format!("\"{}\"", step.name()));
    }
    let command = Cli::command();
    let run = command.find_subcommand("run").expect("run");
    let help = run
        .get_arguments()
        .find(|arg| arg.get_id() == "net_steps")
        .expect("--net-steps exists")
        .get_help()
        .expect("help")
        .to_string();
    for step in NetStep::ALL {
        assert!(help.contains(step.name()), "{help}");
    }
}

#[test]
fn cli_parses_comma_separated_net_steps() {
    use clap::Parser;
    use gauntlet::cli::Command;
    let cli = Cli::try_parse_from([
        "gauntlet",
        "run",
        "--phases",
        "network",
        "--net-steps",
        "intranode,nccl",
    ])
    .expect("parse");
    let Command::Run(args) = cli.command else {
        panic!("expected run");
    };
    assert_eq!(args.net_steps, vec!["intranode", "nccl"]);
}

#[test]
fn config_accepts_every_cli_spelling() {
    // The example config advertises `tcp`; config and CLI share one table.
    for (spelling, step) in NetStep::PARSE_TABLE {
        let config = parse(&format!(
            "hosts = [\"a\", \"b\"]\n[tests]\nnet_steps = [\"{spelling}\"]\n"
        ))
        .unwrap_or_else(|error| panic!("{spelling}: {error}"));
        assert_eq!(config.tests.net_steps, vec![*step], "{spelling}");
    }
    let config = parse(
        r#"
        hosts = ["a", "b"]
        [tests]
        net_steps = ["tcp", "nccl"]
        "#,
    )
    .expect("tcp alias in config");
    assert_eq!(
        steps(&config, &[]).expect("resolve"),
        vec![NetStep::Pairwise, NetStep::Nccl]
    );
}

#[test]
fn config_rejects_unknown_step_names_with_the_accepted_list() {
    let error = parse(
        r#"
        hosts = ["a"]
        [tests]
        net_steps = ["warp"]
        "#,
    )
    .expect_err("unknown step");
    let text = error.to_string();
    assert!(text.contains("warp"), "{text}");
    assert!(text.contains("pairwise (tcp)"), "{text}");
}

/// Config whose own selection resolves to nothing: `nccl_intranode = false`
/// removes the only listed step.
const SELF_EMPTYING: &str = r#"
hosts = ["a", "b"]
[tests]
nccl_intranode = false
net_steps = ["intranode"]
"#;

#[test]
fn a_self_emptying_config_still_loads() {
    // Rejecting it at load would also break `bootstrap`, which never runs
    // the network phase.
    parse(SELF_EMPTYING).expect("loads");
}

#[test]
fn cli_net_steps_override_is_applied_before_validation() {
    let config = parse(SELF_EMPTYING).expect("loads");
    let plan = config
        .run_plan(&["network".into()], &["nccl".into()])
        .expect("--net-steps nccl rescues the empty config selection");
    let steps: Vec<NetStep> = plan.net_steps().expect("network selected").iter().collect();
    assert_eq!(steps, vec![NetStep::Nccl]);
    // Without the override the network run is rejected.
    assert!(matches!(
        config.run_plan(&["network".into()], &[]),
        Err(ConfigError::NoNetSteps)
    ));
}

#[test]
fn net_steps_are_not_validated_when_the_network_phase_is_not_selected() {
    let config = parse(SELF_EMPTYING).expect("loads");
    let plan = config.run_plan(&["gpu".into()], &[]).expect("gpu-only run");
    assert_eq!(plan.phases(), &[gauntlet::proto::Phase::Gpu]);
    assert!(plan.net_steps().is_none());
    // Typos in --net-steps are still reported.
    assert!(matches!(
        config.run_plan(&["gpu".into()], &["warp".into()]),
        Err(ConfigError::UnknownNetStep { .. })
    ));
}

#[test]
fn the_default_plan_carries_net_steps_because_network_is_a_default_phase() {
    let config = parse(r#"hosts = ["a", "b"]"#).expect("config");
    let plan = config.run_plan(&[], &[]).expect("plan");
    assert!(plan.phases().contains(&gauntlet::proto::Phase::Network));
    assert_eq!(
        plan.net_steps()
            .expect("network")
            .iter()
            .collect::<Vec<_>>(),
        NetStep::ALL.to_vec()
    );
}
