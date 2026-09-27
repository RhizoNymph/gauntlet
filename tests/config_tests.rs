use std::path::Path;

use gauntlet::config::{ConfigError, FleetConfig};
use gauntlet::nccl_env::{NcclEnvError, NcclEnvValueError};
use gauntlet::proto::Phase;

fn parse(text: &str) -> Result<FleetConfig, String> {
    let config: FleetConfig = toml::from_str(text).map_err(|e| e.to_string())?;
    config.validate().map_err(|e| e.to_string())?;
    Ok(config)
}

#[test]
fn example_config_in_repo_is_valid() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("gauntlet.example.toml");
    let config = FleetConfig::load(&path).expect("example config must stay valid");
    assert_eq!(config.hosts().count(), 3);
}

#[test]
fn minimal_config_parses_with_defaults() {
    let config = parse(r#"hosts = ["10.0.0.1", "10.0.0.2"]"#).expect("minimal config");
    assert_eq!(config.tests.phases, Phase::ALL.to_vec());
    assert_eq!(config.ssh.max_concurrent, 32);
    assert_eq!(config.thresholds.mad_k, 4.0);
    assert!(config.tests.nccl_sizes.first() == Some(&1024));
    assert!(config.tests.nccl_sizes.last() == Some(&(1024 * 4u64.pow(10))));
}

#[test]
fn bare_and_table_hosts_normalize() {
    let config = parse(
        r#"
        hosts = [
            "10.0.0.1",
            { addr = "10.0.0.2", labels = { rack = "r1" } },
        ]
        "#,
    )
    .expect("mixed host forms");
    let hosts: Vec<_> = config.hosts().collect();
    assert_eq!(hosts[0].addr, "10.0.0.1");
    assert!(hosts[0].labels.is_empty());
    assert_eq!(hosts[1].addr, "10.0.0.2");
    assert_eq!(hosts[1].labels.get("rack").map(String::as_str), Some("r1"));
}

#[test]
fn empty_hosts_rejected() {
    let error = parse("hosts = []").expect_err("no hosts must fail");
    assert!(error.contains("no hosts"), "got: {error}");
}

#[test]
fn duplicate_hosts_rejected() {
    let error = parse(r#"hosts = ["10.0.0.1", "10.0.0.1"]"#).expect_err("duplicate must fail");
    assert!(error.contains("10.0.0.1"), "got: {error}");
}

#[test]
fn unknown_fields_rejected() {
    let error = parse(
        r#"
        hosts = ["10.0.0.1"]
        [tests]
        gemm_secz = 30
        "#,
    )
    .expect_err("typo'd key must fail");
    assert!(error.contains("gemm_secz"), "got: {error}");
}

#[test]
fn non_positive_mad_k_rejected() {
    for bad in ["0.0", "-1.0", "nan"] {
        let text = format!(
            r#"
            hosts = ["10.0.0.1"]
            [thresholds]
            mad_k = {bad}
            "#
        );
        assert!(parse(&text).is_err(), "mad_k = {bad} must be rejected");
    }
}

#[test]
fn resolve_phases_defaults_and_aliases() {
    let config = parse(r#"hosts = ["10.0.0.1"]"#).expect("config");
    assert_eq!(
        config.resolve_phases(&[]).expect("default"),
        Phase::ALL.to_vec()
    );
    let picked = config
        .resolve_phases(&["cpu".into(), "net".into()])
        .expect("aliases resolve");
    assert_eq!(picked, vec![Phase::CpuMem, Phase::Network]);
    let error = config
        .resolve_phases(&["warp_drive".into()])
        .expect_err("unknown phase");
    assert!(matches!(error, ConfigError::UnknownPhase { .. }));
}

#[test]
fn data_addr_parses_and_defaults_off() {
    let config = parse(
        r#"
        hosts = [
            "10.0.0.1",
            { addr = "10.0.0.2", data_addr = "10.1.1.2" },
        ]
        "#,
    )
    .expect("data_addr host form");
    let hosts: Vec<_> = config.hosts().collect();
    assert_eq!(hosts[0].data_addr, None);
    assert_eq!(hosts[1].data_addr.as_deref(), Some("10.1.1.2"));
}

#[test]
fn overlap_defaults_and_task_spec_mapping() {
    use gauntlet::proto::GemmDtype;
    let config = parse(r#"hosts = ["10.0.0.1"]"#).expect("config");
    assert_eq!(config.tests.overlap_secs, 30);
    assert_eq!(config.tests.overlap_baseline_secs, 5);
    assert_eq!(config.tests.overlap_msg_mib, 64);

    let spec = config.task_spec(&[Phase::Overlap]);
    assert_eq!(spec.overlap.duration_secs, 30);
    assert_eq!(spec.overlap.baseline_secs, 5);
    assert_eq!(spec.overlap.msg_bytes, 64 * 1024 * 1024);
    assert_eq!(spec.overlap.gemm_dim, config.tests.gemm_dim);
    // The compute leg uses the first configured GEMM dtype.
    assert_eq!(
        Some(&spec.overlap.gemm_dtype),
        config.tests.gemm_dtypes.first()
    );

    let custom = parse(
        r#"
        hosts = ["10.0.0.1"]
        [tests]
        overlap_secs = 12
        overlap_baseline_secs = 3
        overlap_msg_mib = 16
        gemm_dtypes = ["bf16", "f16"]
        "#,
    )
    .expect("custom overlap config");
    let spec = custom.task_spec(&[Phase::Overlap]);
    assert_eq!(spec.overlap.duration_secs, 12);
    assert_eq!(spec.overlap.baseline_secs, 3);
    assert_eq!(spec.overlap.msg_bytes, 16 * 1024 * 1024);
    assert_eq!(spec.overlap.gemm_dtype, GemmDtype::Bf16);
}

#[test]
fn fleet_overlap_toggle_defaults_on() {
    let config = parse(r#"hosts = ["10.0.0.1"]"#).expect("config");
    assert!(config.tests.overlap_fleet);
    // Both steps are handed the same spec value — one mapper, no drift.
    assert_eq!(
        config.task_spec(&[Phase::Overlap]).overlap,
        config.overlap_spec()
    );

    let off = parse(
        r#"
        hosts = ["10.0.0.1"]
        [tests]
        overlap_fleet = false
        "#,
    )
    .expect("overlap_fleet override");
    assert!(!off.tests.overlap_fleet);
}

#[test]
fn task_spec_converts_units() {
    let config = parse(
        r#"
        hosts = ["10.0.0.1"]
        [tests]
        mem_buffer_mib_per_numa = 2
        disk_file_mib = 3
        gpu_bandwidth_mib = 5
        "#,
    )
    .expect("config");
    let spec = config.task_spec(&[Phase::CpuMem]);
    assert_eq!(spec.phases, vec![Phase::CpuMem]);
    assert_eq!(spec.mem.buffer_bytes_per_numa, 2 * 1024 * 1024);
    assert_eq!(spec.disk.file_bytes, 3 * 1024 * 1024);
    assert_eq!(spec.gpu.bandwidth_bytes, 5 * 1024 * 1024);
}

#[test]
fn hot_sdc_knobs_default_on_and_flow_into_the_spec() {
    let config = parse(r#"hosts = ["10.0.0.1"]"#).expect("config");
    let spec = config.task_spec(&[Phase::CpuMem, Phase::Gpu]);
    // The hot screens default to enabled: they only matter on fleets that
    // never touch the config knobs.
    assert!(spec.cpu.sdc_hot_secs > 0);
    assert!(spec.gpu.sdc_check_secs > 0);

    let tuned = parse(
        r#"
        hosts = ["10.0.0.1"]
        [tests]
        cpu_sdc_hot_secs = 7
        gemm_sdc_check_secs = 0
        "#,
    )
    .expect("config");
    let spec = tuned.task_spec(&[Phase::CpuMem, Phase::Gpu]);
    assert_eq!(spec.cpu.sdc_hot_secs, 7);
    assert_eq!(spec.gpu.sdc_check_secs, 0);
}

// ---------------------------------------------------------------------------
// [nccl] env passthrough
// ---------------------------------------------------------------------------

fn load_str(text: &str) -> Result<FleetConfig, ConfigError> {
    FleetConfig::from_toml_str(text, Path::new("inline.toml"))
}

#[test]
fn socket_ifname_only_config_still_loads() {
    // The shape of every pre-passthrough config (deny_unknown_fields must
    // keep accepting it).
    let text = r#"
        hosts = ["10.0.0.1", "10.0.0.2"]
        [nccl]
        socket_ifname = "bond0"
    "#;
    let config = load_str(text).expect("socket_ifname-only config loads");
    assert_eq!(config.nccl.socket_ifname(), Some("bond0"));
    let env = config.nccl_env();
    assert_eq!(
        env.iter().collect::<Vec<_>>(),
        vec![("NCCL_SOCKET_IFNAME", "bond0")]
    );
    // The serde-only path (no typed pre-check) agrees.
    let parsed = parse(text).expect("parses");
    assert_eq!(parsed.nccl_env(), env);
}

#[test]
fn unknown_nccl_fields_are_still_rejected() {
    let error = parse(
        r#"
        hosts = ["10.0.0.1"]
        [nccl]
        socket_ifnam = "bond0"
        "#,
    )
    .expect_err("typo'd [nccl] key must fail");
    assert!(error.contains("socket_ifnam"), "got: {error}");
}

#[test]
fn no_nccl_section_means_an_empty_env() {
    let config = load_str(r#"hosts = ["10.0.0.1"]"#).expect("config");
    assert!(config.nccl_env().is_empty());
    assert_eq!(config.nccl.socket_ifname(), None);
}

#[test]
fn nccl_env_map_parses_and_resolves_with_socket_ifname() {
    let config = load_str(
        r#"
        hosts = ["10.0.0.1"]
        [nccl]
        socket_ifname = "bond0"
        env = { NCCL_IB_HCA = "mlx5_0,mlx5_1", NCCL_DEBUG = "WARN", NCCL_IB_GID_INDEX = "3" }
        "#,
    )
    .expect("env config loads");
    let env = config.nccl_env();
    assert_eq!(env.get("NCCL_IB_HCA"), Some("mlx5_0,mlx5_1"));
    assert_eq!(env.get("NCCL_DEBUG"), Some("WARN"));
    assert_eq!(env.get("NCCL_IB_GID_INDEX"), Some("3"));
    assert_eq!(env.get("NCCL_SOCKET_IFNAME"), Some("bond0"));
    assert_eq!(env.len(), 4);
}

#[test]
fn nccl_env_accepts_table_syntax() {
    let config = load_str(
        r#"
        hosts = ["10.0.0.1"]
        [nccl]
        socket_ifname = "bond0"
        [nccl.env]
        NCCL_P2P_LEVEL = "NVL"
        "#,
    )
    .expect("config");
    let env = config.nccl_env();
    assert_eq!(env.get("NCCL_P2P_LEVEL"), Some("NVL"));
    assert_eq!(env.get("NCCL_SOCKET_IFNAME"), Some("bond0"));
    // `resolved_env` is the same map the accessor hands out.
    assert_eq!(config.nccl.resolved_env(), &env);
}

#[test]
fn non_nccl_env_keys_are_typed_config_errors() {
    for key in ["LD_PRELOAD", "CUDA_VISIBLE_DEVICES", "nccl_debug", "NCCL_"] {
        let text = format!(
            r#"
            hosts = ["10.0.0.1"]
            [nccl.env]
            "{key}" = "x"
            "#
        );
        let error = load_str(&text).expect_err("key must be rejected");
        assert!(
            matches!(
                &error,
                ConfigError::Nccl {
                    source: NcclEnvError::InvalidKey { key: bad }
                } if bad == key
            ),
            "{key}: got {error:?}"
        );
        // The serde-only path rejects it too.
        let message = parse(&text).expect_err("serde path rejects");
        assert!(message.contains(key), "{key}: got {message}");
    }
}

#[test]
fn socket_ifname_conflict_is_a_typed_config_error() {
    let text = r#"
        hosts = ["10.0.0.1"]
        [nccl]
        socket_ifname = "bond0"
        env = { NCCL_SOCKET_IFNAME = "bond0" }
    "#;
    let error = load_str(text).expect_err("conflict must fail");
    assert!(
        matches!(
            error,
            ConfigError::Nccl {
                source: NcclEnvError::SocketIfnameConflict
            }
        ),
        "got {error:?}"
    );
    assert!(parse(text).is_err());

    // Through the map alone it is just another knob.
    let config = load_str(
        r#"
        hosts = ["10.0.0.1"]
        [nccl]
        env = { NCCL_SOCKET_IFNAME = "bond0" }
        "#,
    )
    .expect("map-only ifname");
    assert_eq!(config.nccl_env().get("NCCL_SOCKET_IFNAME"), Some("bond0"));
    assert_eq!(config.nccl.socket_ifname(), None);
}

#[test]
fn empty_and_nul_values_are_typed_config_errors() {
    for (value, reason) in [
        (r#""""#, NcclEnvValueError::Empty),
        (r#""WA\u0000RN""#, NcclEnvValueError::ContainsNul),
    ] {
        let text = format!(
            r#"
            hosts = ["10.0.0.1"]
            [nccl.env]
            NCCL_DEBUG = {value}
            "#
        );
        let error = load_str(&text).expect_err("bad value must fail");
        assert!(
            matches!(
                &error,
                ConfigError::Nccl {
                    source: NcclEnvError::InvalidValue { key, reason: got }
                } if key == "NCCL_DEBUG" && *got == reason
            ),
            "{value}: got {error:?}"
        );
    }
    let error = load_str(
        r#"
        hosts = ["10.0.0.1"]
        [nccl]
        socket_ifname = ""
        "#,
    )
    .expect_err("empty socket_ifname must fail");
    assert!(
        matches!(
            &error,
            ConfigError::Nccl {
                source: NcclEnvError::InvalidValue {
                    key,
                    reason: NcclEnvValueError::Empty
                }
            } if key == "NCCL_SOCKET_IFNAME"
        ),
        "got {error:?}"
    );
}

#[test]
fn syntax_and_shape_errors_stay_parse_errors() {
    let error = load_str("hosts = [").expect_err("broken TOML");
    assert!(matches!(error, ConfigError::Parse { .. }), "got {error:?}");
    // A non-string env value is a shape error, not an NCCL policy error.
    let error = load_str(
        r#"
        hosts = ["10.0.0.1"]
        [nccl.env]
        NCCL_IB_GID_INDEX = 3
        "#,
    )
    .expect_err("integer value");
    assert!(matches!(error, ConfigError::Parse { .. }), "got {error:?}");
}
