//! `[nccl.levels.<level>]` overrides: parsing, validation, resolution, and
//! the level each NCCL-hosting process runs under.

use std::path::Path;

use gauntlet::config::{ConfigError, FleetConfig};
use gauntlet::nccl_env::{NcclEnvError, NcclEnvValueError};
use gauntlet::nccl_level::NcclLevel;
use gauntlet::proto::{BarrierSpec, NcclWorkload, OverlapSpec, Phase};

fn load(text: &str) -> Result<FleetConfig, ConfigError> {
    FleetConfig::from_toml_str(text, Path::new("test.toml"))
}

#[test]
fn level_overrides_layer_on_the_global_env() {
    let config = load(
        r#"
        hosts = ["10.0.0.1", "10.0.0.2"]
        [nccl]
        socket_ifname = "bond0"
        env = { NCCL_ALGO = "Tree", NCCL_IB_HCA = "mlx5_0" }
        [nccl.levels.intranode]
        env = { NCCL_ALGO = "Ring" }
        [nccl.levels.fleet]
        env = { NCCL_P2P_DISABLE = true, NCCL_SHM_DISABLE = 1 }
        "#,
    )
    .expect("valid config");
    let levels = config.nccl_levels().expect("validated");

    let intranode = levels.level(NcclLevel::Intranode);
    assert_eq!(intranode.get("NCCL_ALGO"), Some("Ring"));
    assert_eq!(intranode.get("NCCL_IB_HCA"), Some("mlx5_0"));
    assert_eq!(intranode.get("NCCL_SOCKET_IFNAME"), Some("bond0"));
    assert_eq!(intranode.get("NCCL_P2P_DISABLE"), None);

    let fleet = levels.level(NcclLevel::Fleet);
    assert_eq!(fleet.get("NCCL_ALGO"), Some("Tree"));
    assert_eq!(fleet.get("NCCL_P2P_DISABLE"), Some("1"));
    assert_eq!(fleet.get("NCCL_SHM_DISABLE"), Some("1"));

    for level in [
        NcclLevel::Barrier,
        NcclLevel::OverlapIntranode,
        NcclLevel::OverlapFleet,
    ] {
        assert_eq!(levels.level(level), levels.global(), "{level}");
    }
    // `nccl_env()` is still the global map.
    assert_eq!(config.nccl_env().expect("validated"), levels.global());
    assert_eq!(
        config.nccl_env().expect("validated").get("NCCL_ALGO"),
        Some("Tree")
    );
    // The fleet override means the barrier can no longer share its
    // communicator.
    assert!(!levels.barrier_shares_fleet_comm());
}

#[test]
fn no_levels_means_every_level_runs_the_global_env() {
    let config = load(
        r#"
        hosts = ["10.0.0.1"]
        [nccl]
        env = { NCCL_DEBUG = "WARN" }
        "#,
    )
    .expect("valid config");
    let levels = config.nccl_levels().expect("validated");
    for level in NcclLevel::ALL {
        assert_eq!(levels.level(level), levels.global(), "{level}");
    }
    assert!(levels.barrier_shares_fleet_comm());
}

#[test]
fn every_level_name_is_accepted() {
    let config = load(
        r#"
        hosts = ["10.0.0.1"]
        [nccl.levels.intranode]
        env = { NCCL_ALGO = "Ring" }
        [nccl.levels.fleet]
        env = { NCCL_ALGO = "Tree" }
        [nccl.levels.barrier]
        env = { NCCL_PROTO = "LL" }
        [nccl.levels.overlap_intranode]
        env = { NCCL_MIN_NCHANNELS = 4 }
        [nccl.levels.overlap_fleet]
        env = { NCCL_IB_QPS_PER_CONNECTION = 2 }
        "#,
    )
    .expect("valid config");
    let levels = config.nccl_levels().expect("validated");
    assert_eq!(
        levels.level(NcclLevel::Barrier).get("NCCL_PROTO"),
        Some("LL")
    );
    assert_eq!(
        levels
            .level(NcclLevel::OverlapIntranode)
            .get("NCCL_MIN_NCHANNELS"),
        Some("4")
    );
    assert_eq!(
        levels
            .level(NcclLevel::OverlapFleet)
            .get("NCCL_IB_QPS_PER_CONNECTION"),
        Some("2")
    );
    assert!(levels.global().is_empty());
}

#[test]
fn unknown_levels_and_fields_are_parse_errors() {
    for text in [
        "hosts = [\"a\"]\n[nccl.levels.overlap]\nenv = { NCCL_ALGO = \"Ring\" }\n",
        "hosts = [\"a\"]\n[nccl.levels.fleet]\nenvs = { NCCL_ALGO = \"Ring\" }\n",
        "hosts = [\"a\"]\n[nccl.levels.fleet]\nsocket_ifname = \"eth0\"\n",
    ] {
        let error = load(text).expect_err("must be rejected");
        assert!(
            matches!(error, ConfigError::Parse { .. }),
            "{text}: {error:?}"
        );
    }
}

#[test]
fn level_entries_get_the_same_validation_as_the_global_map() {
    let error = load(
        r#"
        hosts = ["10.0.0.1"]
        [nccl.levels.barrier]
        env = { LD_PRELOAD = "/tmp/x.so" }
        "#,
    )
    .expect_err("non-NCCL key");
    assert!(
        matches!(
            &error,
            ConfigError::NcclLevel {
                level: NcclLevel::Barrier,
                source: NcclEnvError::InvalidKey { key }
            } if key == "LD_PRELOAD"
        ),
        "{error:?}"
    );
    assert!(
        error.to_string().contains("[nccl.levels.barrier]"),
        "{error}"
    );

    let error = load(
        r#"
        hosts = ["10.0.0.1"]
        [nccl.levels.overlap_fleet]
        env = { NCCL_IB_HCA = "" }
        "#,
    )
    .expect_err("empty value");
    assert!(
        matches!(
            error,
            ConfigError::NcclLevel {
                level: NcclLevel::OverlapFleet,
                source: NcclEnvError::InvalidValue {
                    reason: NcclEnvValueError::Empty,
                    ..
                }
            }
        ),
        "{error:?}"
    );

    let error = load(
        r#"
        hosts = ["10.0.0.1"]
        [nccl.levels.intranode]
        env = { NCCL_ALGO = 1.5 }
        "#,
    )
    .expect_err("float");
    assert!(
        matches!(
            error,
            ConfigError::NcclLevel {
                level: NcclLevel::Intranode,
                source: NcclEnvError::UnsupportedValueType { .. }
            }
        ),
        "{error:?}"
    );
}

#[test]
fn a_level_may_override_the_typed_socket_ifname() {
    // Layering is the point: not the global both-places conflict.
    let config = load(
        r#"
        hosts = ["10.0.0.1"]
        [nccl]
        socket_ifname = "bond0"
        [nccl.levels.fleet]
        env = { NCCL_SOCKET_IFNAME = "ib0" }
        "#,
    )
    .expect("valid config");
    let levels = config.nccl_levels().expect("validated");
    assert_eq!(
        levels.level(NcclLevel::Fleet).get("NCCL_SOCKET_IFNAME"),
        Some("ib0")
    );
    assert_eq!(levels.global().get("NCCL_SOCKET_IFNAME"), Some("bond0"));
}

#[test]
fn an_unvalidated_config_never_yields_an_invalid_level_env() {
    let config: FleetConfig = toml::from_str(
        r#"
        hosts = ["10.0.0.1"]
        [nccl.levels.fleet]
        env = { CUDA_VISIBLE_DEVICES = "0" }
        "#,
    )
    .expect("shape is fine");
    assert!(matches!(
        config.nccl_levels(),
        Err(ConfigError::NcclLevel {
            level: NcclLevel::Fleet,
            ..
        })
    ));
    // The global accessor refuses too: a config is valid as a whole or not.
    assert!(config.nccl_env().is_err());
}

#[test]
fn every_nccl_call_site_maps_to_one_level() {
    // agent run: one phase per spawn, at most one level per process.
    assert_eq!(Phase::Network.nccl_level(), Some(NcclLevel::Intranode));
    assert_eq!(
        Phase::Overlap.nccl_level(),
        Some(NcclLevel::OverlapIntranode)
    );
    for phase in [Phase::Inventory, Phase::CpuMem, Phase::Gpu] {
        assert_eq!(phase.nccl_level(), None, "{phase:?}");
    }
    // agent nccl: the workload names the level.
    let barrier = BarrierSpec {
        iters: 10,
        bytes: 8,
    };
    let sweep = NcclWorkload::Sweep {
        sizes: vec![1024],
        iters_per_size: 1,
        barrier: Some(barrier),
    };
    assert_eq!(sweep.level(), NcclLevel::Fleet);
    assert_eq!(NcclWorkload::Barrier(barrier).level(), NcclLevel::Barrier);
    let overlap = NcclWorkload::Overlap(OverlapSpec {
        duration_secs: 1,
        baseline_secs: 1,
        gemm_dim: 64,
        gemm_dtype: gauntlet::proto::GemmDtype::F32,
        msg_bytes: 1024,
    });
    assert_eq!(overlap.level(), NcclLevel::OverlapFleet);
}

#[test]
fn the_barrier_workload_survives_the_wire() {
    use gauntlet::proto::{NcclDirective, RankAssignment, RankBlock};
    let assignment =
        RankAssignment::new(RankBlock::new(0, 4).expect("block"), 8).expect("assignment");
    let directive = NcclDirective::Lead {
        assignment,
        workload: NcclWorkload::Barrier(BarrierSpec {
            iters: 2000,
            bytes: 8,
        }),
    };
    let json = serde_json::to_string(&directive).expect("serialize");
    assert!(json.contains(r#""kind":"barrier""#), "{json}");
    let back: NcclDirective = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, directive);
}
