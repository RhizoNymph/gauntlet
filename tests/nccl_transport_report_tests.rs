//! Transport capture in the results document: collection, the
//! socket-fallback finding and its verdict, per-level env recording, the
//! terminal rendering, and decoding of pre-v13 documents.

use std::collections::BTreeMap;
use std::path::Path;

use gauntlet::config::FleetConfig;
use gauntlet::nccl_level::NcclLevel;
use gauntlet::nccl_transport::parse::parse_log;
use gauntlet::nccl_transport::{CommSpan, NcclTransportReport, TransportCapture, UnknownTransport};
use gauntlet::orchestrator::collect::{Collector, HostObservations};
use gauntlet::proto::{AgentEvent, decode_event, encode_event};
use gauntlet::report::{self, RunResults, Verdict};

const SOCKET_LOG: &str = include_str!("../src/nccl_transport/fixtures/socket_fallback_2_23.log");
const IB_LOG: &str = include_str!("../src/nccl_transport/fixtures/ib_2_18.log");
const INTRANODE_LOG: &str = include_str!("../src/nccl_transport/fixtures/intranode_2_30.log");

fn config(extra: &str) -> FleetConfig {
    let text = format!("hosts = [\"n1\", \"n2\"]\n{extra}");
    FleetConfig::from_toml_str(&text, Path::new("test.toml")).expect("valid config")
}

fn captured(log: &str) -> TransportCapture {
    TransportCapture::Captured {
        info: parse_log(log).expect("fixture has transport lines"),
    }
}

fn report(level: NcclLevel, span: CommSpan, capture: TransportCapture) -> AgentEvent {
    AgentEvent::NcclTransport {
        report: Box::new(NcclTransportReport {
            level,
            span,
            capture,
        }),
    }
}

/// n1's fleet communicator fell back to sockets; n2's ran on IB. Both
/// intra-node communicators are single-host.
fn observations() -> BTreeMap<String, HostObservations> {
    let mut collector = Collector::new();
    for (host, fleet_log) in [("n1", SOCKET_LOG), ("n2", IB_LOG)] {
        for event in [
            report(
                NcclLevel::Intranode,
                CommSpan::SingleHost,
                captured(INTRANODE_LOG),
            ),
            report(NcclLevel::Fleet, CommSpan::MultiHost, captured(fleet_log)),
        ] {
            // Through the wire, as the orchestrator receives it.
            let event = decode_event(&encode_event(&event)).expect("round trip");
            collector.ingest(host, event);
        }
    }
    collector.into_observations()
}

fn build(config: &FleetConfig) -> RunResults {
    report::build(config, observations(), 1_700_000_000, 1_700_000_600)
}

#[test]
fn the_collector_keeps_every_transport_report() {
    let observations = observations();
    let n1 = &observations["n1"];
    assert_eq!(n1.nccl_transports.len(), 2);
    assert_eq!(n1.nccl_transports[1].level, NcclLevel::Fleet);
    assert_eq!(n1.nccl_transports[1].span, CommSpan::MultiHost);
}

#[test]
fn a_socket_fallback_is_a_straggler_finding() {
    let results = build(&config(""));
    assert_eq!(results.schema_version, report::SCHEMA_VERSION);
    let findings = &results.fleet.socket_fallbacks;
    assert_eq!(findings.keys().collect::<Vec<_>>(), vec!["n1"]);
    assert_eq!(findings["n1"][0].level, NcclLevel::Fleet);
    assert_eq!(findings["n1"][0].ifaces[0].name, "bond0");
    assert_eq!(report::verdict(&results), Verdict::Stragglers);
}

#[test]
fn sockets_requested_on_purpose_are_not_a_finding() {
    // IB disabled for the fleet level only: the requested configuration.
    let results = build(&config(
        "[nccl.levels.fleet]\nenv = { NCCL_IB_DISABLE = 1 }\n",
    ));
    assert!(results.fleet.socket_fallbacks.is_empty());
    assert_eq!(report::verdict(&results), Verdict::Clean);

    // Disabled globally: every level inherits it.
    let results = build(&config("[nccl]\nenv = { NCCL_IB_DISABLE = true }\n"));
    assert!(results.fleet.socket_fallbacks.is_empty());

    // Disabled for another level only: the fleet fallback still counts.
    let results = build(&config(
        "[nccl.levels.overlap_fleet]\nenv = { NCCL_IB_DISABLE = 1 }\n",
    ));
    assert_eq!(results.fleet.socket_fallbacks.len(), 1);
}

#[test]
fn unknown_captures_are_recorded_but_never_findings() {
    let mut observations = BTreeMap::new();
    let mut collector = Collector::new();
    collector.ingest(
        "n1",
        report(
            NcclLevel::Fleet,
            CommSpan::MultiHost,
            TransportCapture::Unknown {
                reason: UnknownTransport::DebugLevel {
                    level: Some("WARN".into()),
                },
            },
        ),
    );
    observations.extend(collector.into_observations());
    let results = report::build(&config(""), observations, 1, 2);
    assert!(results.fleet.socket_fallbacks.is_empty());
    assert_eq!(report::verdict(&results), Verdict::Clean);
    let mut table = Vec::new();
    report::render_table(&results, &mut table).expect("render");
    let table = String::from_utf8(table).expect("utf8");
    assert!(table.contains("NCCL_DEBUG=WARN"), "{table}");
}

#[test]
fn the_effective_env_of_every_level_is_recorded() {
    let results = build(&config(
        "[nccl]\nenv = { NCCL_DEBUG = \"WARN\" }\n\
         [nccl.levels.intranode]\nenv = { NCCL_ALGO = \"Ring\" }\n",
    ));
    let levels = results.nccl_level_env.as_ref().expect("recorded");
    assert_eq!(levels.len(), NcclLevel::ALL.len());
    assert_eq!(levels[&NcclLevel::Intranode]["NCCL_ALGO"], "Ring");
    assert_eq!(levels[&NcclLevel::Intranode]["NCCL_DEBUG"], "WARN");
    assert!(!levels[&NcclLevel::Fleet].contains_key("NCCL_ALGO"));
    // The run-level map stays the global one.
    let global = results.nccl_env.as_ref().expect("recorded");
    assert!(!global.contains_key("NCCL_ALGO"));
    // Capture variables are not part of the recorded env.
    assert!(!levels[&NcclLevel::Fleet].contains_key("NCCL_DEBUG_FILE"));
}

#[test]
fn the_table_shows_transports_findings_and_level_overrides() {
    let results = build(&config(
        "[nccl.levels.intranode]\nenv = { NCCL_ALGO = \"Ring\" }\n",
    ));
    let mut table = Vec::new();
    report::render_table(&results, &mut table).expect("render");
    let table = String::from_utf8(table).expect("utf8");
    assert!(
        table.contains("nccl env [intranode]: NCCL_ALGO=Ring"),
        "{table}"
    );
    assert!(!table.contains("nccl env [fleet]"), "{table}");
    assert!(table.contains("nccl transports"), "{table}");
    assert!(table.contains("Socket bond0(10.20.4.12)"), "{table}");
    assert!(table.contains("IB mlx5_0:1,mlx5_1:1"), "{table}");
    assert!(table.contains("nccl socket fallback"), "{table}");
    assert!(table.contains("2.23.4+cuda12.6"), "{table}");
}

#[test]
fn pre_v13_documents_decode_without_the_new_fields() {
    let results = build(&config(""));
    let mut json = serde_json::to_value(&results).expect("serialize");
    let object = json.as_object_mut().expect("object");
    object.remove("nccl_level_env");
    object["fleet"]
        .as_object_mut()
        .expect("fleet")
        .remove("socket_fallbacks");
    for host in object["hosts"].as_object_mut().expect("hosts").values_mut() {
        host.as_object_mut()
            .expect("host")
            .remove("nccl_transports");
    }
    let old: RunResults = serde_json::from_value(json).expect("pre-v13 shape decodes");
    assert_eq!(old.nccl_level_env, None);
    assert!(old.fleet.socket_fallbacks.is_empty());
    assert!(old.hosts.values().all(|obs| obs.nccl_transports.is_empty()));
}
