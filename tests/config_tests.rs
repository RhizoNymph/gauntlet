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

fn nccl_error(text: &str) -> NcclEnvError {
    match load_str(text) {
        Err(ConfigError::Nccl { source }) => source,
        other => panic!("expected ConfigError::Nccl, got {other:?}"),
    }
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
    assert_eq!(config.nccl.socket_ifname.as_deref(), Some("bond0"));
    let env = config.nccl_env().expect("validated");
    assert_eq!(
        env.iter().collect::<Vec<_>>(),
        vec![("NCCL_SOCKET_IFNAME", "bond0")]
    );
    // The serde + validate path agrees.
    let parsed = parse(text).expect("parses");
    assert_eq!(parsed.nccl_env().expect("validated"), env);
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
    assert!(config.nccl_env().expect("validated").is_empty());
    assert_eq!(config.nccl.socket_ifname, None);
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
    let env = config.nccl_env().expect("validated");
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
    let env = config.nccl_env().expect("validated");
    assert_eq!(env.get("NCCL_P2P_LEVEL"), Some("NVL"));
    assert_eq!(env.get("NCCL_SOCKET_IFNAME"), Some("bond0"));
}

#[test]
fn integer_and_boolean_values_are_stringified() {
    let config = load_str(
        r#"
        hosts = ["10.0.0.1"]
        [nccl]
        env = { NCCL_IB_GID_INDEX = 3, NCCL_IB_DISABLE = true, NCCL_CROSS_NIC = false, NCCL_NET_GDR_LEVEL = "PHB" }
        "#,
    )
    .expect("scalar values load");
    let env = config.nccl_env().expect("validated");
    assert_eq!(env.get("NCCL_IB_GID_INDEX"), Some("3"));
    assert_eq!(env.get("NCCL_IB_DISABLE"), Some("1"));
    assert_eq!(env.get("NCCL_CROSS_NIC"), Some("0"));
    assert_eq!(env.get("NCCL_NET_GDR_LEVEL"), Some("PHB"));
}

#[test]
fn float_array_and_table_values_are_typed_config_errors() {
    for (value, found) in [
        ("1.5", "float"),
        ("[0, 1]", "array"),
        ("{ port = 1 }", "table"),
    ] {
        let text = format!(
            r#"
            hosts = ["10.0.0.1"]
            [nccl.env]
            NCCL_IB_GID_INDEX = {value}
            "#
        );
        assert_eq!(
            nccl_error(&text),
            NcclEnvError::UnsupportedValueType {
                key: "NCCL_IB_GID_INDEX".into(),
                found
            },
            "{value}"
        );
    }
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
        assert_eq!(
            nccl_error(&text),
            NcclEnvError::InvalidKey { key: key.into() },
            "{key}"
        );
        // The serde + validate path rejects it too.
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
    assert_eq!(nccl_error(text), NcclEnvError::SocketIfnameConflict);
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
    assert_eq!(
        config
            .nccl_env()
            .expect("validated")
            .get("NCCL_SOCKET_IFNAME"),
        Some("bond0")
    );
    assert_eq!(config.nccl.socket_ifname, None);
}

#[test]
fn empty_and_control_character_values_are_typed_config_errors() {
    for (value, reason) in [
        (r#""""#, NcclEnvValueError::Empty),
        (r#""WA\u0000RN""#, NcclEnvValueError::ControlCharacter),
        (r#""WA\nRN""#, NcclEnvValueError::ControlCharacter),
        (r#""WA\tRN""#, NcclEnvValueError::ControlCharacter),
        (r#""WA\u007fRN""#, NcclEnvValueError::ControlCharacter),
    ] {
        let text = format!(
            r#"
            hosts = ["10.0.0.1"]
            [nccl.env]
            NCCL_DEBUG = {value}
            "#
        );
        assert_eq!(
            nccl_error(&text),
            NcclEnvError::InvalidValue {
                key: "NCCL_DEBUG".into(),
                reason
            },
            "{value}"
        );
    }
    let error = nccl_error(
        r#"
        hosts = ["10.0.0.1"]
        [nccl]
        socket_ifname = ""
        "#,
    );
    assert_eq!(
        error,
        NcclEnvError::InvalidValue {
            key: "NCCL_SOCKET_IFNAME".into(),
            reason: NcclEnvValueError::Empty
        }
    );
}

#[test]
fn syntax_errors_stay_parse_errors() {
    let error = load_str("hosts = [").expect_err("broken TOML");
    assert!(matches!(error, ConfigError::Parse { .. }), "got {error:?}");
}

#[test]
fn an_unvalidated_config_still_never_yields_an_invalid_env() {
    // Skipping validate() does not bypass the policy: the accessor resolves
    // (and validates) on first use.
    let config: FleetConfig = toml::from_str(
        r#"
        hosts = ["10.0.0.1"]
        [nccl.env]
        LD_PRELOAD = "/tmp/x.so"
        "#,
    )
    .expect("shape is fine");
    assert!(matches!(
        config.nccl_env(),
        Err(ConfigError::Nccl {
            source: NcclEnvError::InvalidKey { .. }
        })
    ));
}

#[test]
fn intranode_sweep_defaults_on_and_reuses_the_fleet_sweep_knobs() {
    let config = parse(r#"hosts = ["10.0.0.1"]"#).expect("config");
    assert!(config.tests.nccl_intranode);
    let spec = config.intranode_sweep_spec().expect("enabled by default");
    // Same sizes and iteration count as the fleet sweep: one set of knobs
    // for every level of the hierarchy.
    assert_eq!(spec.sizes, config.tests.nccl_sizes);
    assert_eq!(spec.iters_per_size, config.tests.nccl_iters_per_size);
    // The network-phase task spec carries it; one mapper, no drift.
    assert_eq!(
        config.task_spec(&[Phase::Network]).nccl_intranode,
        Some(spec)
    );

    let custom = parse(
        r#"
        hosts = ["10.0.0.1"]
        [tests]
        nccl_sizes = [4096, 1048576]
        nccl_iters_per_size = 7
        "#,
    )
    .expect("custom sweep config");
    let spec = custom.intranode_sweep_spec().expect("enabled");
    assert_eq!(spec.sizes, vec![4096, 1_048_576]);
    assert_eq!(spec.iters_per_size, 7);
}

#[test]
fn intranode_sweep_toggle_disables_the_spec() {
    let off = parse(
        r#"
        hosts = ["10.0.0.1"]
        [tests]
        nccl_intranode = false
        "#,
    )
    .expect("nccl_intranode override");
    assert!(!off.tests.nccl_intranode);
    assert_eq!(off.intranode_sweep_spec(), None);
    assert_eq!(off.task_spec(&[Phase::Network]).nccl_intranode, None);
}

#[test]
fn the_example_config_enables_the_intranode_sweep() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("gauntlet.example.toml");
    let text = std::fs::read_to_string(&path).expect("example config");
    // The key must be spelled out in the example, not just defaulted.
    assert!(text.contains("nccl_intranode = true"), "{text}");
    let config = FleetConfig::load(&path).expect("example config must stay valid");
    assert!(config.tests.nccl_intranode);
}

#[test]
fn the_nccl_world_shape_defaults_to_rank_per_gpu() {
    use gauntlet::config::NcclWorldShape;
    let config = parse(r#"hosts = ["10.0.0.1"]"#).expect("config");
    assert_eq!(config.tests.nccl_world, NcclWorldShape::RankPerGpu);
    for (wire, shape) in [
        ("rank_per_gpu", NcclWorldShape::RankPerGpu),
        ("rank_per_node", NcclWorldShape::RankPerNode),
        ("per_rail", NcclWorldShape::PerRail),
    ] {
        let config = parse(&format!(
            "hosts = [\"10.0.0.1\"]\n[tests]\nnccl_world = \"{wire}\"\n"
        ))
        .expect("world shape");
        assert_eq!(config.tests.nccl_world, shape, "{wire}");
    }
}

#[test]
fn an_unknown_nccl_world_shape_is_rejected() {
    let error = parse(
        r#"
        hosts = ["10.0.0.1"]
        [tests]
        nccl_world = "ranks_per_node"
        "#,
    )
    .expect_err("unknown shape");
    assert!(error.contains("ranks_per_node"), "{error}");
}

#[test]
fn the_example_config_spells_out_the_nccl_world_shape() {
    use gauntlet::config::NcclWorldShape;
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("gauntlet.example.toml");
    let text = std::fs::read_to_string(&path).expect("example config");
    assert!(text.contains("nccl_world = \"rank_per_gpu\""), "{text}");
    let config = FleetConfig::load(&path).expect("example config must stay valid");
    assert_eq!(config.tests.nccl_world, NcclWorldShape::RankPerGpu);
}
