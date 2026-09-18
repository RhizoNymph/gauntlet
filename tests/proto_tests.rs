mod common;

use gauntlet::proto::{
    AgentEvent, InventorySnapshot, LogLevel, MetricRecord, PROTO_VERSION, Phase, ProtoError, Scope,
    TestId, TestOutcome, Unit, consistency_fields, decode_event, encode_event, expect_hello,
};

fn sample_inventory() -> InventorySnapshot {
    InventorySnapshot {
        hostname: "node1".into(),
        kernel: "6.8.0".into(),
        cpu_model: "EPYC 9654".into(),
        logical_cores: 192,
        numa_nodes: 2,
        mem_total_bytes: 1 << 40,
        cpu_governor: Some("performance".into()),
        clock_offset_ms: Some(0.03),
        nvidia_driver: Some("560.35.03".into()),
        cuda_version: Some("12.6".into()),
        gpus: vec![],
        nics: vec![],
        ib_ports: vec![],
        xid_errors: vec![79],
        gpu_libs: std::collections::BTreeMap::new(),
    }
}

#[test]
fn events_round_trip() {
    let events = vec![
        AgentEvent::Hello {
            proto_version: PROTO_VERSION,
            hostname: "node1".into(),
        },
        AgentEvent::PhaseStart {
            phase: Phase::CpuMem,
        },
        AgentEvent::Inventory {
            snapshot: Box::new(sample_inventory()),
        },
        AgentEvent::Metric {
            record: MetricRecord {
                test: TestId::MemBandwidth,
                scope: Scope::Numa { node: 1 },
                name: "triad".into(),
                value: 210.5,
                unit: Unit::GibPerSec,
                repeat: 0,
            },
        },
        AgentEvent::Outcome {
            test: TestId::CpuCorrectness,
            scope: Scope::Core { id: 17 },
            outcome: TestOutcome::Failed {
                reason: "checksum mismatch after 412 rounds".into(),
            },
        },
        AgentEvent::Log {
            level: LogLevel::Warn,
            message: "chrony not found".into(),
        },
        AgentEvent::PhaseEnd {
            phase: Phase::CpuMem,
        },
        AgentEvent::NcclId {
            unique_id_b64: "S1QrZi9HZldHV1VDQUs3ckNnRUE=".into(),
        },
        AgentEvent::Fatal {
            message: "boom".into(),
        },
    ];
    for event in events {
        let line = encode_event(&event);
        assert!(!line.contains('\n'), "one event per line: {line}");
        let back = decode_event(&line).expect("round trip");
        assert_eq!(back, event);
    }
}

/// Wire-format stability: this exact JSON must keep decoding. Breaking this
/// test means PROTO_VERSION must be bumped.
#[test]
fn wire_format_is_stable() {
    let line = r#"{"event":"metric","test":"gpu_gemm_perf","scope":{"kind":"gpu","index":3},"name":"gflops_bf16","value":712000.0,"unit":"gflops"}"#;
    let event = decode_event(line).expect("stable wire format");
    match event {
        AgentEvent::Metric { record } => {
            assert_eq!(record.test, TestId::GpuGemmPerf);
            assert_eq!(record.scope, Scope::Gpu { index: 3 });
            assert_eq!(record.name, "gflops_bf16");
            assert_eq!(record.unit, Unit::Gflops);
        }
        other => panic!("expected metric, got {other:?}"),
    }
}

#[test]
fn hello_validation() {
    let ok = AgentEvent::Hello {
        proto_version: PROTO_VERSION,
        hostname: "n1".into(),
    };
    assert_eq!(expect_hello(&ok).expect("valid hello"), "n1");

    let stale = AgentEvent::Hello {
        proto_version: PROTO_VERSION + 1,
        hostname: "n1".into(),
    };
    assert!(matches!(
        expect_hello(&stale),
        Err(ProtoError::VersionMismatch { .. })
    ));

    let not_hello = AgentEvent::PhaseStart {
        phase: Phase::Inventory,
    };
    assert!(matches!(
        expect_hello(&not_hello),
        Err(ProtoError::MissingHello { .. })
    ));
}

#[test]
fn overlap_phase_exists_and_runs_last() {
    assert_eq!(Phase::parse("overlap"), Some(Phase::Overlap));
    assert_eq!(Phase::ALL.len(), 5);
    // The overlap phase needs the isolated phase-2/3 baselines from the same
    // run, so it must be scheduled after them.
    assert_eq!(Phase::ALL.last(), Some(&Phase::Overlap));
}

#[test]
fn overlap_task_spec_round_trips() {
    use gauntlet::proto::{
        AgentTaskSpec, CpuTaskSpec, DiskTaskSpec, GemmDtype, GpuTaskSpec, MemTaskSpec, OverlapSpec,
    };
    let spec = AgentTaskSpec {
        phases: vec![Phase::Gpu, Phase::Overlap],
        cpu: CpuTaskSpec {
            correctness_secs_per_core: 1,
            gflops_secs: 1,
            sdc_hot_secs: 0,
        },
        mem: MemTaskSpec {
            buffer_bytes_per_numa: 1 << 20,
            iters: 2,
        },
        disk: DiskTaskSpec {
            paths: vec!["/tmp".into()],
            file_bytes: 1 << 20,
        },
        gpu: GpuTaskSpec {
            gemm_secs: 3,
            gemm_dtypes: vec![GemmDtype::Bf16],
            gemm_dim: 2048,
            bandwidth_bytes: 1 << 20,
            sdc_check_secs: 0,
        },
        overlap: OverlapSpec {
            duration_secs: 30,
            baseline_secs: 5,
            gemm_dim: 2048,
            gemm_dtype: GemmDtype::Bf16,
            msg_bytes: 64 << 20,
        },
        counters: None,
    };
    let json = serde_json::to_string(&spec).expect("serialize");
    let back: AgentTaskSpec = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, spec);
}

#[test]
fn overlap_metric_events_round_trip() {
    for (test, name, scope) in [
        (TestId::OverlapGemm, "gflops_bf16", Scope::Gpu { index: 1 }),
        (
            TestId::OverlapAllReduce,
            "overlap_bus_gib_per_sec",
            Scope::Node,
        ),
        (TestId::OverlapRetention, "all_reduce", Scope::Node),
    ] {
        let event = AgentEvent::Metric {
            record: MetricRecord {
                test,
                scope,
                name: name.into(),
                value: 0.87,
                unit: Unit::Ratio,
                repeat: 0,
            },
        };
        let back = decode_event(&encode_event(&event)).expect("round trip");
        assert_eq!(back, event);
    }
}

#[test]
fn fleet_overlap_events_round_trip() {
    use gauntlet::proto::{OverlapFleetReport, OverlapGpuGemm};
    let event = AgentEvent::OverlapFleetReport {
        report: Box::new(OverlapFleetReport {
            rank: 1,
            msg_bytes: 64 << 20,
            isolated_bus_gib_per_sec: 44.0,
            overlap_bus_gib_per_sec: 33.0,
            gemm: vec![
                OverlapGpuGemm::Ok {
                    gpu_index: 0,
                    gflops: 88_000.0,
                },
                OverlapGpuGemm::Failed {
                    gpu_index: 1,
                    reason: "overlap gemm setup: CUDA_ERROR_OUT_OF_MEMORY".into(),
                },
            ],
        }),
    };
    let back = decode_event(&encode_event(&event)).expect("round trip");
    assert_eq!(back, event);

    // Metric events under the new test ids round-trip like any other.
    for (test, name, unit) in [
        (TestId::OverlapFleetGemm, "gflops_bf16", Unit::Gflops),
        (
            TestId::OverlapFleetAllReduce,
            "overlap_bus_gib_per_sec",
            Unit::GibPerSec,
        ),
    ] {
        let event = AgentEvent::Metric {
            record: MetricRecord {
                test,
                scope: Scope::Node,
                name: name.into(),
                value: 12.5,
                unit,
                repeat: 0,
            },
        };
        let back = decode_event(&encode_event(&event)).expect("round trip");
        assert_eq!(back, event);
    }
}

#[test]
fn nccl_workloads_are_mutually_exclusive_by_construction() {
    use gauntlet::proto::{NcclDirective, NcclWorkload, OverlapSpec};
    // A sweep workload written without the optional barrier probe.
    let sweep = r#"{"directive":"lead","world_size":3,"socket_ifname":null,
        "workload":{"kind":"sweep","sizes":[1024],"iters_per_size":20}}"#;
    let directive: NcclDirective = serde_json::from_str(sweep).expect("decode sweep lead");
    let NcclDirective::Lead {
        workload: NcclWorkload::Sweep { barrier, .. },
        ..
    } = directive
    else {
        panic!("expected a Lead sweep directive");
    };
    assert_eq!(barrier, None);

    let with_overlap = NcclDirective::Participate {
        unique_id_b64: "abc".into(),
        rank: 2,
        world_size: 3,
        socket_ifname: Some("bond0".into()),
        workload: NcclWorkload::Overlap(OverlapSpec {
            duration_secs: 30,
            baseline_secs: 5,
            gemm_dim: 8192,
            gemm_dtype: gauntlet::proto::GemmDtype::Bf16,
            msg_bytes: 64 << 20,
        }),
    };
    let json = serde_json::to_string(&with_overlap).expect("serialize");
    let back: NcclDirective = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, with_overlap);
}

#[test]
fn sdc_events_round_trip() {
    let events = vec![
        AgentEvent::Outcome {
            test: TestId::CpuSdcHot,
            scope: Scope::Core { id: 5 },
            outcome: TestOutcome::Failed {
                reason: "2 mismatches over 811 hot rounds".into(),
            },
        },
        AgentEvent::Outcome {
            test: TestId::GpuGemmSdc,
            scope: Scope::Gpu { index: 1 },
            outcome: TestOutcome::Failed {
                reason: "bf16: check #3 deviated by 1.2e-1 at 1410 MHz / 84 C".into(),
            },
        },
        AgentEvent::Metric {
            record: MetricRecord {
                test: TestId::GpuGemmSdc,
                scope: Scope::Gpu { index: 0 },
                name: "mismatches_f32".into(),
                value: 0.0,
                unit: Unit::Count,
                repeat: 0,
            },
        },
    ];
    for event in events {
        let back = decode_event(&encode_event(&event)).expect("round trip");
        assert_eq!(back, event);
    }
}

#[test]
fn task_spec_sdc_fields_round_trip() {
    use gauntlet::proto::AgentTaskSpec;
    let spec: AgentTaskSpec = serde_json::from_str(
        r#"{
            "phases": ["cpu_mem", "gpu"],
            "cpu": {"correctness_secs_per_core": 5, "gflops_secs": 5, "sdc_hot_secs": 8},
            "mem": {"buffer_bytes_per_numa": 1048576, "iters": 3},
            "disk": {"paths": [], "file_bytes": 0},
            "gpu": {"gemm_secs": 30, "gemm_dtypes": ["f32"], "gemm_dim": 4096,
                    "bandwidth_bytes": 1048576, "sdc_check_secs": 5},
            "overlap": {"duration_secs": 30, "baseline_secs": 5, "gemm_dim": 4096,
                        "gemm_dtype": "f32", "msg_bytes": 1048576}
        }"#,
    )
    .expect("spec with sdc fields decodes");
    assert_eq!(spec.cpu.sdc_hot_secs, 8);
    assert_eq!(spec.gpu.sdc_check_secs, 5);

    let text = serde_json::to_string(&spec).expect("encode");
    let back: AgentTaskSpec = serde_json::from_str(&text).expect("decode");
    assert_eq!(back, spec);
}

/// v1 task specs (no sdc fields) must keep decoding: absence means disabled.
#[test]
fn task_spec_sdc_fields_default_to_disabled() {
    use gauntlet::proto::{CpuTaskSpec, GpuTaskSpec};
    let cpu: CpuTaskSpec =
        serde_json::from_str(r#"{"correctness_secs_per_core": 5, "gflops_secs": 5}"#)
            .expect("v1 cpu spec decodes");
    assert_eq!(cpu.sdc_hot_secs, 0);
    let gpu: GpuTaskSpec = serde_json::from_str(
        r#"{"gemm_secs": 30, "gemm_dtypes": [], "gemm_dim": 4096, "bandwidth_bytes": 1}"#,
    )
    .expect("v1 gpu spec decodes");
    assert_eq!(gpu.sdc_check_secs, 0);
}

#[test]
fn malformed_lines_error() {
    assert!(decode_event("not json").is_err());
    assert!(decode_event(r#"{"event":"warp"}"#).is_err());
}

#[test]
fn consistency_fields_cover_version_skew_sources() {
    let inv = sample_inventory();
    let fields = consistency_fields(&inv);
    assert_eq!(fields.get("kernel").map(String::as_str), Some("6.8.0"));
    assert_eq!(
        fields.get("nvidia_driver").map(String::as_str),
        Some("560.35.03")
    );
    assert_eq!(fields.get("cuda_version").map(String::as_str), Some("12.6"));
    assert_eq!(fields.get("gpu_count").map(String::as_str), Some("0"));
}

#[test]
fn event_sink_is_thread_safe_and_line_delimited() {
    let (sink, buf) = common::capturing_sink();
    let sink = std::sync::Arc::new(sink);
    std::thread::scope(|scope| {
        for thread_id in 0..8u32 {
            let sink = sink.clone();
            scope.spawn(move || {
                for i in 0..50u32 {
                    sink.metric(MetricRecord {
                        test: TestId::CpuGflops,
                        scope: Scope::Core {
                            id: thread_id * 100 + i,
                        },
                        name: "gflops".into(),
                        value: 42.0,
                        unit: Unit::Gflops,
                        repeat: 0,
                    });
                }
            });
        }
    });
    let events = common::decode_events(&buf);
    assert_eq!(events.len(), 400, "no torn or interleaved lines");
}

#[test]
fn consistency_marks_absent_optional_fields() {
    let mut inv = sample_inventory();
    inv.nvidia_driver = None;
    inv.cuda_version = None;
    let fields = consistency_fields(&inv);
    assert_eq!(
        fields.get("nvidia_driver").map(String::as_str),
        Some("(absent)"),
        "absence must dissent against present values"
    );
    assert_eq!(
        fields.get("cuda_version").map(String::as_str),
        Some("(absent)")
    );
}

#[test]
fn consistency_includes_gpu_libs_only_on_gpu_hosts() {
    let mut inv = sample_inventory();
    inv.gpu_libs = [("nccl".to_string(), false), ("cuda".to_string(), true)]
        .into_iter()
        .collect();
    // No GPUs: library state is noise, not skew.
    assert!(
        !consistency_fields(&inv)
            .keys()
            .any(|k| k.starts_with("lib:"))
    );

    inv.gpus.push(gauntlet::proto::GpuInventory {
        index: 0,
        name: "H100".into(),
        uuid: "u".into(),
        vbios: "v".into(),
        mem_total_bytes: 1,
        ecc_volatile_errors: None,
        remapped_rows_pending: None,
        pcie_gen_current: None,
        pcie_gen_max: None,
        pcie_width_current: None,
        pcie_width_max: None,
        nvlinks_active: None,
        persistence_mode: None,
    });
    let fields = consistency_fields(&inv);
    assert_eq!(fields.get("lib:nccl").map(String::as_str), Some("absent"));
    assert_eq!(fields.get("lib:cuda").map(String::as_str), Some("present"));
}
