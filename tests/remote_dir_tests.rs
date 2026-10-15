//! `ssh.remote_dir`: node-local default, explicit values, validation.

use gauntlet::config::FleetConfig;
use gauntlet::remote_dir::{DEFAULT_REMOTE_DIR, RemoteDir};

fn parse(text: &str) -> Result<FleetConfig, String> {
    let config: FleetConfig = toml::from_str(text).map_err(|e| e.to_string())?;
    config.validate().map_err(|e| e.to_string())?;
    Ok(config)
}

#[test]
fn default_remote_dir_is_node_local_per_user() {
    let config = parse(r#"hosts = ["a"]"#).expect("config");
    assert_eq!(config.ssh.remote_dir.as_str(), "/tmp/gauntlet-$USER");
    assert_eq!(DEFAULT_REMOTE_DIR, "/tmp/gauntlet-$USER");
    assert!(!config.ssh.remote_dir.as_str().starts_with('~'));
}

#[test]
fn an_explicit_home_remote_dir_still_works() {
    let config = parse(
        r#"
        hosts = ["a"]
        [ssh]
        remote_dir = "~/.gauntlet"
        "#,
    )
    .expect("config");
    assert_eq!(
        config.ssh.remote_dir,
        RemoteDir::parse("~/.gauntlet").expect("valid")
    );
}

#[test]
fn unsupported_variables_are_config_errors() {
    let error = parse(
        r#"
        hosts = ["a"]
        [ssh]
        remote_dir = "/scratch/$SLURM_JOB_ID"
        "#,
    )
    .expect_err("rejected");
    assert!(error.contains("$USER"), "{error}");
}
