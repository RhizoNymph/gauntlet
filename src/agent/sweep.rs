//! Message-size sweep shared by every level of the NCCL hierarchy.
//!
//! The fleet sweep (`agent nccl`, one rank per node) and the intra-node
//! sweep (`agent run` network phase, one rank per local GPU) run the *same*
//! per-size timing loop: a warmup at the largest size, then per message size
//! `iters` all-reduces followed by `iters` all-gathers, each bracketed by a
//! full synchronization so the interval covers completed device work only.
//! Only how a collective is launched differs (one communicator on one
//! stream vs. an NCCL group across every local rank), so that is the one
//! thing abstracted here: `SweepCollectives`. Everything else — which sizes
//! run, the all-gather sharding, the timing, the metric records — is pure
//! and unit-tested without a GPU.

use std::num::NonZeroU32;
use std::time::Instant;

use crate::agent::nccl::{F32_BYTES, all_gather_bus_gib_per_sec, all_reduce_bus_gib_per_sec};
use crate::proto::{MetricRecord, Scope, TestId, Unit, nccl_metric};

/// Untimed full-size all-reduces before the first measured size, so channel
/// setup and algorithm selection stay out of the sweep.
pub const WARMUP_ITERS: u32 = 5;

/// The two swept collectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Collective {
    AllReduce,
    AllGather,
}

impl Collective {
    pub const ALL: [Collective; 2] = [Collective::AllReduce, Collective::AllGather];

    /// Standard NCCL bus bandwidth for this collective: all-reduce factor
    /// `2(n-1)/n`, all-gather `(n-1)/n` (with `message_bytes` the gathered
    /// size) — the nccl-tests definitions, reused from `agent::nccl`.
    pub fn bus_gib_per_sec(self, message_bytes: f64, elapsed_secs: f64, world: u32) -> f64 {
        match self {
            Collective::AllReduce => all_reduce_bus_gib_per_sec(message_bytes, elapsed_secs, world),
            Collective::AllGather => all_gather_bus_gib_per_sec(message_bytes, elapsed_secs, world),
        }
    }
}

/// Which level of the hierarchy a sweep measures; decides the test ids its
/// records land under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepLevel {
    /// One rank per node across the fleet (`nccl_all_*`).
    Fleet,
    /// One rank per local GPU on one node (`nccl_intra_all_*`).
    IntraNode,
}

impl SweepLevel {
    pub fn test(self, collective: Collective) -> TestId {
        match (self, collective) {
            (SweepLevel::Fleet, Collective::AllReduce) => TestId::NcclAllReduce,
            (SweepLevel::Fleet, Collective::AllGather) => TestId::NcclAllGather,
            (SweepLevel::IntraNode, Collective::AllReduce) => TestId::NcclIntraAllReduce,
            (SweepLevel::IntraNode, Collective::AllGather) => TestId::NcclIntraAllGather,
        }
    }
}

/// One timed collective shape. For all-reduce `send_elements ==
/// message_elements`; for all-gather every rank sends one shard and
/// `message_elements = shard * world` is the gathered result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SweepStep {
    pub collective: Collective,
    /// f32 elements each rank contributes.
    pub send_elements: usize,
    /// f32 elements of the message the bus-bandwidth formula is defined on.
    pub message_elements: usize,
}

impl SweepStep {
    fn all_reduce(elements: usize) -> Self {
        Self {
            collective: Collective::AllReduce,
            send_elements: elements,
            message_elements: elements,
        }
    }

    pub fn message_bytes(&self) -> f64 {
        (self.message_elements * F32_BYTES) as f64
    }
}

/// The steps a sweep runs, in order, plus the buffer size they need. Built
/// only through `SweepPlan::new`, so every step fits the buffers
/// (`send_elements <= message_elements <= max_elements`) and `max_elements`
/// is never zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SweepPlan {
    max_elements: usize,
    steps: Vec<SweepStep>,
}

impl SweepPlan {
    /// Per size: one all-reduce step, then one all-gather step when the
    /// size can be sharded across the world (sizes too small to give every
    /// rank at least one element have no meaningful all-gather and skip
    /// that leg). Sizes are payloads of f32 elements, never zero.
    pub fn new(sizes: &[u64], world: NonZeroU32) -> Self {
        let world = world.get() as usize;
        let max_elements = sizes
            .iter()
            .map(|size| *size as usize / F32_BYTES)
            .max()
            .unwrap_or(0)
            .max(1);
        let mut steps = Vec::with_capacity(sizes.len() * 2);
        for &size in sizes {
            let elements = (size as usize / F32_BYTES).clamp(1, max_elements);
            steps.push(SweepStep::all_reduce(elements));
            let shard = elements / world;
            if shard > 0 {
                steps.push(SweepStep {
                    collective: Collective::AllGather,
                    send_elements: shard,
                    message_elements: shard * world,
                });
            }
        }
        Self {
            max_elements,
            steps,
        }
    }

    /// Elements each rank's send and receive buffer must hold.
    pub fn max_elements(&self) -> usize {
        self.max_elements
    }

    pub fn steps(&self) -> &[SweepStep] {
        &self.steps
    }
}

/// How one sweep level launches collectives. Launches are asynchronous;
/// `sync` blocks until every launched collective has completed on every
/// rank this process drives.
pub trait SweepCollectives {
    type Error;
    fn launch(&mut self, step: &SweepStep) -> Result<(), Self::Error>;
    fn sync(&mut self) -> Result<(), Self::Error>;
}

/// One measured sweep point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SweepPoint {
    pub step: SweepStep,
    /// Mean wall seconds per collective.
    pub per_iter_secs: f64,
}

impl SweepPoint {
    pub fn bus_gib_per_sec(&self, world: u32) -> f64 {
        self.step
            .collective
            .bus_gib_per_sec(self.step.message_bytes(), self.per_iter_secs, world)
    }
}

/// Run a plan: `WARMUP_ITERS` full-size all-reduces, then per step
/// `iters_per_size` (at least 1) timed launches with a sync on both sides.
/// `on_point` sees each point as soon as it is measured, so emitters can
/// stream results live. The first collective error aborts the sweep.
pub fn run_plan<C: SweepCollectives>(
    collectives: &mut C,
    plan: &SweepPlan,
    iters_per_size: u32,
    mut on_point: impl FnMut(&SweepPoint),
) -> Result<(), C::Error> {
    let warmup = SweepStep::all_reduce(plan.max_elements);
    for _ in 0..WARMUP_ITERS {
        collectives.launch(&warmup)?;
    }
    collectives.sync()?;

    let iters = iters_per_size.max(1);
    for step in plan.steps() {
        collectives.sync()?;
        let start = Instant::now();
        for _ in 0..iters {
            collectives.launch(step)?;
        }
        collectives.sync()?;
        let point = SweepPoint {
            step: *step,
            per_iter_secs: start.elapsed().as_secs_f64() / f64::from(iters),
        };
        on_point(&point);
    }
    Ok(())
}

/// The per-size records every sweep emits under `Scope::Node`, in the
/// order `elapsed_us`, `msg_bytes`, `bus_gib_per_sec`. Within one host the
/// report joins `msg_bytes` and `elapsed_us` by emission order.
pub fn point_records(level: SweepLevel, point: &SweepPoint, world: u32) -> [MetricRecord; 3] {
    let test = level.test(point.step.collective);
    let record = |name: &str, value: f64, unit: Unit| MetricRecord {
        test,
        scope: Scope::Node,
        name: name.to_string(),
        value,
        unit,
        repeat: 0,
    };
    [
        record(
            nccl_metric::ELAPSED_US,
            point.per_iter_secs * 1e6,
            Unit::Micros,
        ),
        record(
            nccl_metric::MSG_BYTES,
            point.step.message_bytes(),
            Unit::Bytes,
        ),
        record(
            nccl_metric::BUS,
            point.bus_gib_per_sec(world),
            Unit::GibPerSec,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: f64 = (1u64 << 30) as f64;

    fn world(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).expect("non-zero world")
    }

    #[test]
    fn plans_pair_every_size_with_a_sharded_all_gather() {
        let plan = SweepPlan::new(&[4096, 1 << 20], world(4));
        assert_eq!(plan.max_elements(), (1 << 20) / 4);
        assert_eq!(
            plan.steps(),
            [
                SweepStep {
                    collective: Collective::AllReduce,
                    send_elements: 1024,
                    message_elements: 1024,
                },
                SweepStep {
                    collective: Collective::AllGather,
                    send_elements: 256,
                    message_elements: 1024,
                },
                SweepStep {
                    collective: Collective::AllReduce,
                    send_elements: 262_144,
                    message_elements: 262_144,
                },
                SweepStep {
                    collective: Collective::AllGather,
                    send_elements: 65_536,
                    message_elements: 262_144,
                },
            ]
        );
    }

    #[test]
    fn all_gather_messages_round_down_to_whole_shards() {
        // 10 elements across 3 ranks: shard 3, gathered 9 (not 10).
        let plan = SweepPlan::new(&[40], world(3));
        let gather = plan.steps()[1];
        assert_eq!(gather.collective, Collective::AllGather);
        assert_eq!(gather.send_elements, 3);
        assert_eq!(gather.message_elements, 9);
        assert_eq!(gather.message_bytes(), 36.0);
    }

    #[test]
    fn sizes_too_small_to_shard_skip_the_all_gather_leg() {
        // 8 bytes = 2 elements across 8 ranks: no shard to send.
        let plan = SweepPlan::new(&[8, 4096], world(8));
        let collectives: Vec<Collective> = plan.steps().iter().map(|s| s.collective).collect();
        assert_eq!(
            collectives,
            [
                Collective::AllReduce,
                Collective::AllReduce,
                Collective::AllGather
            ]
        );
    }

    #[test]
    fn degenerate_sizes_still_exercise_one_element() {
        let plan = SweepPlan::new(&[0, 3], world(2));
        assert_eq!(plan.max_elements(), 1);
        assert!(
            plan.steps()
                .iter()
                .all(|step| step.collective == Collective::AllReduce && step.send_elements == 1)
        );
        // An empty size list still sizes a usable buffer for the warmup.
        let empty = SweepPlan::new(&[], world(2));
        assert_eq!(empty.max_elements(), 1);
        assert!(empty.steps().is_empty());
    }

    #[test]
    fn every_step_fits_the_plan_buffers() {
        let sizes: Vec<u64> = (0..=10).map(|i| 1024u64 * 4u64.pow(i)).collect();
        for n in [2, 3, 4, 7, 8, 16] {
            let plan = SweepPlan::new(&sizes, world(n));
            for step in plan.steps() {
                assert!(step.send_elements >= 1);
                assert!(step.send_elements <= step.message_elements);
                assert!(step.message_elements <= plan.max_elements());
            }
        }
    }

    #[test]
    fn bus_factors_are_the_nccl_tests_definitions() {
        for n in [2u32, 4, 8] {
            let nf = f64::from(n);
            let reduce = Collective::AllReduce.bus_gib_per_sec(GIB, 1.0, n);
            let gather = Collective::AllGather.bus_gib_per_sec(GIB, 1.0, n);
            assert!((reduce - 2.0 * (nf - 1.0) / nf).abs() < 1e-12, "n={n}");
            assert!((gather - (nf - 1.0) / nf).abs() < 1e-12, "n={n}");
        }
        // 8 GPUs: all-reduce 1.75x algBW, all-gather 0.875x.
        assert!((Collective::AllReduce.bus_gib_per_sec(GIB, 0.5, 8) - 3.5).abs() < 1e-12);
        assert!((Collective::AllGather.bus_gib_per_sec(GIB, 0.5, 8) - 1.75).abs() < 1e-12);
    }

    #[test]
    fn levels_map_to_distinct_test_ids() {
        assert_eq!(
            SweepLevel::Fleet.test(Collective::AllReduce),
            TestId::NcclAllReduce
        );
        assert_eq!(
            SweepLevel::Fleet.test(Collective::AllGather),
            TestId::NcclAllGather
        );
        assert_eq!(
            SweepLevel::IntraNode.test(Collective::AllReduce),
            TestId::NcclIntraAllReduce
        );
        assert_eq!(
            SweepLevel::IntraNode.test(Collective::AllGather),
            TestId::NcclIntraAllGather
        );
    }

    #[test]
    fn point_records_match_the_fleet_emission_shape() {
        let point = SweepPoint {
            step: SweepStep {
                collective: Collective::AllGather,
                send_elements: 1 << 25,
                message_elements: 1 << 28, // 1 GiB gathered across 8
            },
            per_iter_secs: 0.5,
        };
        let records = point_records(SweepLevel::IntraNode, &point, 8);
        let shape: Vec<(&str, Unit)> = records.iter().map(|r| (r.name.as_str(), r.unit)).collect();
        assert_eq!(
            shape,
            [
                ("elapsed_us", Unit::Micros),
                ("msg_bytes", Unit::Bytes),
                ("bus_gib_per_sec", Unit::GibPerSec),
            ]
        );
        assert!(records.iter().all(|r| r.test == TestId::NcclIntraAllGather));
        assert!(records.iter().all(|r| r.scope == Scope::Node));
        assert_eq!(records[0].value, 500_000.0);
        assert_eq!(records[1].value, GIB);
        assert!((records[2].value - 1.75).abs() < 1e-12);
    }

    /// Records every launch and sync so the driver's loop structure can be
    /// checked without a GPU.
    #[derive(Default)]
    struct Recorder {
        calls: Vec<Call>,
        fail_on_launch: Option<usize>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Call {
        Launch(Collective, usize),
        Sync,
    }

    impl SweepCollectives for Recorder {
        type Error = String;

        fn launch(&mut self, step: &SweepStep) -> Result<(), String> {
            let launches = self
                .calls
                .iter()
                .filter(|c| matches!(c, Call::Launch(..)))
                .count();
            if self.fail_on_launch == Some(launches) {
                return Err(format!("launch {launches} failed"));
            }
            self.calls
                .push(Call::Launch(step.collective, step.send_elements));
            Ok(())
        }

        fn sync(&mut self) -> Result<(), String> {
            self.calls.push(Call::Sync);
            Ok(())
        }
    }

    #[test]
    fn the_driver_warms_up_then_brackets_every_step_with_syncs() {
        let plan = SweepPlan::new(&[64, 128], world(2));
        let mut recorder = Recorder::default();
        let mut points = Vec::new();
        run_plan(&mut recorder, &plan, 3, |point| points.push(*point)).expect("sweep");

        let mut expected = vec![Call::Launch(Collective::AllReduce, 32); 5];
        expected.push(Call::Sync);
        for step in plan.steps() {
            expected.push(Call::Sync);
            expected.extend(std::iter::repeat_n(
                Call::Launch(step.collective, step.send_elements),
                3,
            ));
            expected.push(Call::Sync);
        }
        assert_eq!(recorder.calls, expected);

        // One point per step, in plan order, never a non-finite timing.
        let steps: Vec<SweepStep> = points.iter().map(|p| p.step).collect();
        assert_eq!(steps, plan.steps());
        assert!(
            points
                .iter()
                .all(|p| p.per_iter_secs.is_finite() && p.per_iter_secs >= 0.0)
        );
    }

    #[test]
    fn zero_iterations_still_time_one_collective() {
        let plan = SweepPlan::new(&[64], world(2));
        let mut recorder = Recorder::default();
        let mut count = 0;
        run_plan(&mut recorder, &plan, 0, |_| count += 1).expect("sweep");
        let timed_launches = recorder
            .calls
            .iter()
            .filter(|c| matches!(c, Call::Launch(..)))
            .count()
            - WARMUP_ITERS as usize;
        assert_eq!(timed_launches, plan.steps().len());
        assert_eq!(count, plan.steps().len());
    }

    #[test]
    fn a_collective_error_aborts_the_sweep() {
        let plan = SweepPlan::new(&[64, 128], world(2));
        // Fail on the first timed launch of the second step.
        let mut recorder = Recorder {
            fail_on_launch: Some(WARMUP_ITERS as usize + 2),
            ..Recorder::default()
        };
        let mut points = Vec::new();
        let error = run_plan(&mut recorder, &plan, 2, |point| points.push(*point))
            .expect_err("error propagates");
        assert!(error.contains("failed"), "{error}");
        assert_eq!(points.len(), 1, "only the completed step is reported");
    }
}
