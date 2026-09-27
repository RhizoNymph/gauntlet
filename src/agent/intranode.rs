//! Intra-node NCCL sweep: the first level of the phase-3 hierarchy.
//!
//! One process drives every local GPU as one rank of a node-local
//! communicator (`ncclCommInitAll`, no rendezvous), and runs the shared
//! message-size sweep (`agent::sweep`) for all-reduce and all-gather over
//! NVLink/PCIe. Per size it emits the same `elapsed_us` / `msg_bytes` /
//! `bus_gib_per_sec` records as the fleet sweep (a per-host series, so the
//! report excludes it from MAD and fits it into `calibration.links`), then
//! one node-level headline per collective:
//!
//! - `bus_gib_per_sec_peak`: the best bus bandwidth across the sweep's
//!   sizes. One value per node, so it is fleet-comparable — the straggler
//!   signal for degraded NVLink, a downtrained PCIe switch, or a missing
//!   P2P path that pairwise p2p numbers can hide under collective traffic.
//! - `ranks`: the communicator size (local GPU count), which keys the
//!   intra-node calibration link class.
//!
//! This module is the GPU-independent half — gating, headline selection,
//! outcome/record assembly — so it compiles and tests everywhere; the
//! communicator lives in `agent::gpu::intranode`.

use std::num::NonZeroU32;

use crate::agent::EventSink;
use crate::agent::sweep::{Collective, SweepLevel, SweepPoint};
use crate::proto::{MetricRecord, Scope, TestId, TestOutcome, Unit, nccl_metric};

/// Fewest local GPUs that make an intra-node collective meaningful: a world
/// of one moves nothing.
pub const MIN_GPUS: u32 = 2;

/// A node-local communicator size that is worth sweeping (`>= MIN_GPUS`).
/// The only way to reach the sweep is through one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiGpuWorld(NonZeroU32);

impl MultiGpuWorld {
    pub fn new(gpus: u32) -> Option<Self> {
        if gpus < MIN_GPUS {
            return None;
        }
        NonZeroU32::new(gpus).map(Self)
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }

    pub fn non_zero(self) -> NonZeroU32 {
        self.0
    }
}

/// What to do on this node, decided from the driver's device count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gate {
    Run(MultiGpuWorld),
    /// Fewer than `MIN_GPUS` visible devices (or no gpu build).
    Skip {
        reason: String,
    },
    /// The driver stack could not even count devices.
    Fail {
        reason: String,
    },
}

pub fn gate(device_count: Result<u32, String>) -> Gate {
    match device_count {
        Err(reason) => Gate::Fail { reason },
        Ok(count) => match MultiGpuWorld::new(count) {
            Some(world) => Gate::Run(world),
            None => Gate::Skip {
                reason: format!(
                    "intra-node nccl sweep needs at least {MIN_GPUS} GPUs, found {count}"
                ),
            },
        },
    }
}

/// The same outcome for both intra-node tests (skip, node-level failure).
pub fn node_outcomes(
    outcome: impl Fn(String) -> TestOutcome,
    reason: &str,
) -> Vec<(TestId, TestOutcome)> {
    Collective::ALL
        .iter()
        .map(|collective| {
            (
                SweepLevel::IntraNode.test(*collective),
                outcome(reason.to_string()),
            )
        })
        .collect()
}

fn emit_outcomes(sink: &EventSink, outcomes: Vec<(TestId, TestOutcome)>) {
    for (test, outcome) in outcomes {
        sink.outcome(test, Scope::Node, outcome);
    }
}

/// Gate on `probe`'s device count, then hand the world to `execute`.
/// Skip and failure paths (including `execute` failing) always land as
/// explicit outcomes on both tests — never a silent gap.
pub fn run_gated(
    sink: &EventSink,
    probe: impl FnOnce() -> Result<u32, String>,
    execute: impl FnOnce(MultiGpuWorld) -> Result<(), String>,
) {
    match gate(probe()) {
        Gate::Run(world) => {
            if let Err(reason) = execute(world) {
                emit_outcomes(
                    sink,
                    node_outcomes(|r| TestOutcome::Failed { reason: r }, &reason),
                );
            }
        }
        Gate::Skip { reason } => emit_outcomes(
            sink,
            node_outcomes(|r| TestOutcome::Skipped { reason: r }, &reason),
        ),
        Gate::Fail { reason } => emit_outcomes(
            sink,
            node_outcomes(|r| TestOutcome::Failed { reason: r }, &reason),
        ),
    }
}

/// Headline bus bandwidth for one collective: the maximum across the
/// sweep's sizes, ignoring non-finite values. Max rather than
/// largest-size, because the largest configured size is not guaranteed to
/// be the saturating one (a sweep capped below the knee, or an all-gather
/// leg skipped at the top), and the best achieved figure is what operators
/// compare against the link's peak. `None` when nothing was measured.
pub fn peak_bus_gib_per_sec(
    points: &[SweepPoint],
    collective: Collective,
    world: MultiGpuWorld,
) -> Option<f64> {
    points
        .iter()
        .filter(|point| point.step.collective == collective)
        .map(|point| point.bus_gib_per_sec(world.get()))
        .filter(|bus| bus.is_finite())
        .reduce(f64::max)
}

/// Node-level records and outcomes once the sweep completed: per
/// collective, `bus_gib_per_sec_peak` + `ranks` and Passed, or Skipped
/// when no size of that collective could run (all-gather on sizes too
/// small to shard across the local GPUs).
pub fn summary(
    points: &[SweepPoint],
    world: MultiGpuWorld,
) -> (Vec<MetricRecord>, Vec<(TestId, TestOutcome)>) {
    let mut records = Vec::new();
    let mut outcomes = Vec::new();
    for collective in Collective::ALL {
        let test = SweepLevel::IntraNode.test(collective);
        match peak_bus_gib_per_sec(points, collective, world) {
            Some(peak) => {
                for (name, value, unit) in [
                    (nccl_metric::BUS_PEAK, peak, Unit::GibPerSec),
                    (nccl_metric::RANKS, f64::from(world.get()), Unit::Count),
                ] {
                    records.push(MetricRecord {
                        test,
                        scope: Scope::Node,
                        name: name.to_string(),
                        value,
                        unit,
                        repeat: 0,
                    });
                }
                outcomes.push((test, TestOutcome::Passed));
            }
            None => outcomes.push((
                test,
                TestOutcome::Skipped {
                    reason: format!("no configured size could run across {} GPUs", world.get()),
                },
            )),
        }
    }
    (records, outcomes)
}

/// Emit `summary` through the sink.
pub fn emit_summary(sink: &EventSink, points: &[SweepPoint], world: MultiGpuWorld) {
    let (records, outcomes) = summary(points, world);
    for record in records {
        sink.metric(record);
    }
    emit_outcomes(sink, outcomes);
}

/// Stand-in for builds without the gpu feature: both tests are Skipped.
#[cfg_attr(feature = "gpu", allow(dead_code))]
pub fn skip_without_gpu(sink: &EventSink) {
    emit_outcomes(
        sink,
        node_outcomes(
            |r| TestOutcome::Skipped { reason: r },
            "agent built without gpu feature",
        ),
    );
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::agent::sweep::SweepStep;
    use crate::proto::{AgentEvent, decode_event};

    #[derive(Clone, Default)]
    struct Buf(Arc<Mutex<Vec<u8>>>);

    impl Write for Buf {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("buf").extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn capture() -> (EventSink, Buf) {
        let buf = Buf::default();
        (EventSink::new(Box::new(buf.clone())), buf)
    }

    fn events(buf: &Buf) -> Vec<AgentEvent> {
        let text = String::from_utf8(buf.0.lock().expect("buf").clone()).expect("utf-8");
        text.lines()
            .map(|line| decode_event(line).expect("event"))
            .collect()
    }

    fn outcomes(events: &[AgentEvent]) -> Vec<(TestId, Scope, TestOutcome)> {
        events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::Outcome {
                    test,
                    scope,
                    outcome,
                } => Some((*test, scope.clone(), outcome.clone())),
                _ => None,
            })
            .collect()
    }

    fn world(n: u32) -> MultiGpuWorld {
        MultiGpuWorld::new(n).expect("multi-gpu world")
    }

    fn point(collective: Collective, message_elements: usize, secs: f64) -> SweepPoint {
        SweepPoint {
            step: SweepStep {
                collective,
                send_elements: message_elements,
                message_elements,
            },
            per_iter_secs: secs,
        }
    }

    #[test]
    fn a_multi_gpu_world_needs_two_gpus() {
        assert_eq!(MultiGpuWorld::new(0), None);
        assert_eq!(MultiGpuWorld::new(1), None);
        assert_eq!(MultiGpuWorld::new(2).map(MultiGpuWorld::get), Some(2));
        assert_eq!(MultiGpuWorld::new(8).map(MultiGpuWorld::get), Some(8));
    }

    #[test]
    fn the_gate_skips_small_nodes_and_fails_broken_drivers() {
        assert_eq!(gate(Ok(8)), Gate::Run(world(8)));
        for count in [0, 1] {
            let Gate::Skip { reason } = gate(Ok(count)) else {
                panic!("{count} GPUs must skip");
            };
            assert!(reason.contains("at least 2 GPUs"), "{reason}");
            assert!(reason.contains(&format!("found {count}")), "{reason}");
        }
        assert_eq!(
            gate(Err("cuda driver init: no libcuda".into())),
            Gate::Fail {
                reason: "cuda driver init: no libcuda".into()
            }
        );
    }

    #[test]
    fn nodes_with_one_gpu_emit_skipped_outcomes_for_both_tests() {
        let (sink, buf) = capture();
        let mut ran = false;
        run_gated(
            &sink,
            || Ok(1),
            |_| {
                ran = true;
                Ok(())
            },
        );
        assert!(!ran, "a one-GPU node must never reach the sweep");
        let outcomes = outcomes(&events(&buf));
        assert_eq!(outcomes.len(), 2);
        let tests: Vec<TestId> = outcomes.iter().map(|(test, ..)| *test).collect();
        assert_eq!(
            tests,
            [TestId::NcclIntraAllReduce, TestId::NcclIntraAllGather]
        );
        for (_, scope, outcome) in outcomes {
            assert_eq!(scope, Scope::Node);
            let TestOutcome::Skipped { reason } = outcome else {
                panic!("expected Skipped, got {outcome:?}");
            };
            assert!(reason.contains("found 1"), "{reason}");
        }
    }

    #[test]
    fn driver_and_sweep_failures_are_failed_outcomes() {
        let (sink, buf) = capture();
        run_gated(&sink, || Err("cuda driver init: boom".into()), |_| Ok(()));
        let (sink2, buf2) = capture();
        run_gated(&sink2, || Ok(4), |_| Err("ncclCommInitAll: 5".into()));
        for (buf, needle) in [(buf, "boom"), (buf2, "ncclCommInitAll")] {
            let outcomes = outcomes(&events(&buf));
            assert_eq!(outcomes.len(), 2);
            for (_, _, outcome) in outcomes {
                let TestOutcome::Failed { reason } = outcome else {
                    panic!("expected Failed, got {outcome:?}");
                };
                assert!(reason.contains(needle), "{reason}");
            }
        }
    }

    #[test]
    fn a_multi_gpu_node_runs_the_sweep_with_its_world() {
        let (sink, buf) = capture();
        let mut seen = None;
        run_gated(
            &sink,
            || Ok(8),
            |world| {
                seen = Some(world.get());
                Ok(())
            },
        );
        assert_eq!(seen, Some(8));
        // A successful execute reports its own outcomes; the gate adds none.
        assert!(events(&buf).is_empty());
    }

    #[test]
    fn the_peak_is_the_best_size_per_collective() {
        const GIB_ELEMS: usize = (1 << 30) / 4;
        let points = [
            point(Collective::AllReduce, GIB_ELEMS / 4, 1.0), // slow small size
            point(Collective::AllReduce, GIB_ELEMS, 0.5),     // the peak
            point(Collective::AllReduce, GIB_ELEMS, 1.0),     // top size, degraded
            point(Collective::AllGather, GIB_ELEMS, 0.25),
            point(Collective::AllGather, GIB_ELEMS, f64::NAN),
        ];
        let peak = peak_bus_gib_per_sec(&points, Collective::AllReduce, world(8)).expect("peak");
        // 2 GiB/s algBW x 2*(7/8).
        assert!((peak - 3.5).abs() < 1e-12, "{peak}");
        // Not the largest-size value (1.75) and not the first value.
        let gather = peak_bus_gib_per_sec(&points, Collective::AllGather, world(8)).expect("peak");
        // 4 GiB/s algBW x 7/8; the NaN point is ignored.
        assert!((gather - 3.5).abs() < 1e-12, "{gather}");
    }

    #[test]
    fn no_points_means_no_peak() {
        assert_eq!(
            peak_bus_gib_per_sec(&[], Collective::AllReduce, world(2)),
            None
        );
        let only_reduce = [point(Collective::AllReduce, 1024, 0.001)];
        assert_eq!(
            peak_bus_gib_per_sec(&only_reduce, Collective::AllGather, world(2)),
            None
        );
    }

    #[test]
    fn the_summary_carries_a_fleet_comparable_headline_per_collective() {
        let points = [
            point(Collective::AllReduce, 1 << 20, 0.001),
            point(Collective::AllGather, 1 << 20, 0.001),
        ];
        let (records, outcomes) = summary(&points, world(4));
        let shape: Vec<(TestId, &str, Unit)> = records
            .iter()
            .map(|r| (r.test, r.name.as_str(), r.unit))
            .collect();
        assert_eq!(
            shape,
            [
                (
                    TestId::NcclIntraAllReduce,
                    "bus_gib_per_sec_peak",
                    Unit::GibPerSec
                ),
                (TestId::NcclIntraAllReduce, "ranks", Unit::Count),
                (
                    TestId::NcclIntraAllGather,
                    "bus_gib_per_sec_peak",
                    Unit::GibPerSec
                ),
                (TestId::NcclIntraAllGather, "ranks", Unit::Count),
            ]
        );
        assert!(records.iter().all(|r| r.scope == Scope::Node));
        assert_eq!(records[1].value, 4.0);
        assert_eq!(
            outcomes,
            [
                (TestId::NcclIntraAllReduce, TestOutcome::Passed),
                (TestId::NcclIntraAllGather, TestOutcome::Passed),
            ]
        );
    }

    #[test]
    fn a_collective_with_no_points_is_skipped_not_passed() {
        let points = [point(Collective::AllReduce, 2, 0.001)];
        let (records, outcomes) = summary(&points, world(8));
        assert!(records.iter().all(|r| r.test == TestId::NcclIntraAllReduce));
        assert_eq!(
            outcomes[0],
            (TestId::NcclIntraAllReduce, TestOutcome::Passed)
        );
        assert!(matches!(
            &outcomes[1],
            (TestId::NcclIntraAllGather, TestOutcome::Skipped { reason }) if reason.contains("8 GPUs")
        ));
    }

    #[test]
    fn builds_without_gpus_skip_both_tests() {
        let (sink, buf) = capture();
        skip_without_gpu(&sink);
        let outcomes = outcomes(&events(&buf));
        assert_eq!(outcomes.len(), 2);
        assert!(
            outcomes
                .iter()
                .all(|(_, _, outcome)| matches!(outcome, TestOutcome::Skipped { .. }))
        );
    }
}
