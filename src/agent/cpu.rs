//! Phase 1a: per-core CPU correctness and throughput.
//!
//! Correctness: every logical core gets a pinned worker (core_affinity)
//! running deterministic workloads with known answers — an integer chain
//! (iterated FNV-style hashing) and an f64 kernel whose checksum is compared
//! against a golden value computed once on core 0. A single mismatching core
//! fails that core's scope, not the node. This is the silent-data-corruption
//! screen, so pinning is mandatory: an unpinned average hides one bad core.
//!
//! Throughput: per-core fused multiply-add loops over a small in-cache
//! buffer, reported as `cpu_gflops.gflops` per `Scope::Core`, plus one
//! all-cores-simultaneously run under `Scope::Node` (thermal/turbo reality).
//!
//! Hot SDC screen: SDC is strongly temperature- and voltage-dependent, so
//! the correctness rounds are additionally interleaved with the power-heavy
//! FMA workload on every core simultaneously — correctness exercised at max
//! package power, after the all-core throughput run has already heated the
//! package. Mismatches are counted per core (the worker keeps going, unlike
//! the isolated screen) and any nonzero count is a hard `CpuSdcHot` failure.

use std::hint::black_box;
use std::time::{Duration, Instant};

use anyhow::Result;
use core_affinity::CoreId;

use crate::agent::EventSink;
use crate::proto::{CpuTaskSpec, MetricRecord, Scope, TestId, TestOutcome, Unit};

/// Iterations per correctness round. Sized so a round takes tens of
/// microseconds: long enough to amortize the clock read, short enough that
/// `correctness_secs_per_core` is honoured with fine granularity.
const CORRECTNESS_ITERS: u64 = 20_000;
const INT_SEED: u64 = 0x5eed_1234_abcd_ef01;
const FLOAT_SEED: u64 = 0x0f10_a7c5_1234_9876;

/// f64 elements in the throughput buffer: 16 KiB, comfortably L1-resident on
/// every CPU this runs on, so the FMA loop measures the core and not DRAM.
const GFLOPS_BUFFER_LEN: usize = 2048;
/// Buffer passes between clock reads. Keeps `Instant::now()` overhead well
/// under a percent of the measured work.
const GFLOPS_PASSES_PER_CHECK: u64 = 64;
/// FMA accumulators, to keep the pipeline fed despite the loop-carried
/// dependency on each accumulator.
const GFLOPS_LANES: usize = 8;
/// Floor on any timing window so a tiny `gflops_secs` still measures
/// something rather than dividing by clock noise.
const MIN_MEASURE: Duration = Duration::from_millis(50);

/// Deterministic work unit both correctness workloads iterate. Public so
/// tests can pin down golden values.
pub fn checksum_round(seed: u64, iters: u64) -> u64 {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    let mut hash = seed ^ FNV_OFFSET;
    for i in 0..iters {
        hash ^= i.rotate_left(17);
        hash = hash.wrapping_mul(FNV_PRIME);
        // Avalanche (murmur3 finalizer constants) so a single flipped bit in
        // any intermediate propagates to the whole checksum.
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        hash ^= hash >> 33;
    }
    hash
}

/// f64 kernel checksum: sum of a fixed-seed LCG-driven FMA chain, bitcast to
/// u64. Bit-exact across runs on a correct core (no reassociation: scalar
/// ops, no -ffast-math equivalents).
pub fn float_checksum_round(seed: u64, iters: u64) -> u64 {
    // LCG constants from Knuth's MMIX; the stream drives the operands so the
    // chain is data-dependent but reproducible.
    const LCG_MUL: u64 = 6364136223846793005;
    const LCG_ADD: u64 = 1442695040888963407;
    let mut state = seed.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
    // The recurrence acc <- acc*0.5 + x*0.25 with x in [1, 2) is a
    // contraction: no overflow, no denormals, no NaN, and never zero.
    let mut acc = 1.0f64;
    for _ in 0..iters {
        state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
        // Top 24 bits -> [1, 2), exactly representable, so the operand
        // itself carries no rounding.
        let operand = 1.0 + ((state >> 40) as f64) * (1.0 / 16_777_216.0);
        acc = acc.mul_add(0.5, operand * 0.25);
    }
    acc.to_bits()
}

pub fn run(sink: &EventSink, spec: &CpuTaskSpec) -> Result<()> {
    let cores = logical_cores();
    // The golden values are computed here, on the main thread, before any
    // worker starts: a worker that disagrees is the suspect, not the oracle.
    let golden_int = checksum_round(INT_SEED, CORRECTNESS_ITERS);
    let golden_float = float_checksum_round(FLOAT_SEED, CORRECTNESS_ITERS);

    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    run_correctness(sink, spec, cores, &core_ids, golden_int, golden_float);
    run_throughput(sink, spec, cores, &core_ids);
    // Last on purpose: the all-core throughput run has just pushed the
    // package to its thermal steady state, which is where SDC shows.
    run_hot_correctness(sink, spec, cores, &core_ids, golden_int, golden_float);
    Ok(())
}

fn logical_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

// ---------------------------------------------------------------------------
// Correctness
// ---------------------------------------------------------------------------

/// Per-core verdict. `Scope::Core { id }` uses the enumeration index, so
/// every logical core gets exactly one outcome even when the affinity mask
/// exposes fewer usable CPUs than `available_parallelism` reports.
fn run_correctness(
    sink: &EventSink,
    spec: &CpuTaskSpec,
    cores: usize,
    core_ids: &[CoreId],
    golden_int: u64,
    golden_float: u64,
) {
    let budget = Duration::from_secs(spec.correctness_secs_per_core);
    let verdicts: Vec<TestOutcome> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..cores)
            .map(|index| {
                let core = core_ids.get(index).copied();
                scope.spawn(move || match core {
                    None => TestOutcome::Skipped {
                        reason: format!("core {index} not in this process's affinity mask"),
                    },
                    Some(core) if !core_affinity::set_for_current(core) => TestOutcome::Skipped {
                        reason: format!("failed to pin worker to cpu {}", core.id),
                    },
                    Some(_) => correctness_worker(budget, golden_int, golden_float),
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle.join().unwrap_or_else(|_| TestOutcome::Failed {
                    reason: "correctness worker panicked".to_string(),
                })
            })
            .collect()
    });

    for (index, verdict) in verdicts.into_iter().enumerate() {
        sink.outcome(
            TestId::CpuCorrectness,
            Scope::Core { id: index as u32 },
            verdict,
        );
    }
}

fn correctness_worker(budget: Duration, golden_int: u64, golden_float: u64) -> TestOutcome {
    let start = Instant::now();
    let mut rounds: u64 = 0;
    loop {
        let integer = checksum_round(INT_SEED, CORRECTNESS_ITERS);
        if integer != golden_int {
            return TestOutcome::Failed {
                reason: format!(
                    "integer checksum mismatch after {rounds} clean rounds: \
                     got {integer:#018x}, expected {golden_int:#018x}"
                ),
            };
        }
        let float = float_checksum_round(FLOAT_SEED, CORRECTNESS_ITERS);
        if float != golden_float {
            return TestOutcome::Failed {
                reason: format!(
                    "float checksum mismatch after {rounds} clean rounds: \
                     got {float:#018x}, expected {golden_float:#018x}"
                ),
            };
        }
        rounds += 1;
        if start.elapsed() >= budget {
            return TestOutcome::Passed;
        }
    }
}

// ---------------------------------------------------------------------------
// Hot SDC screen
// ---------------------------------------------------------------------------

/// FMA burn between correctness rounds. Sized so the duty cycle is
/// overwhelmingly the power-heavy vector workload (a correctness round is
/// ~hundreds of microseconds) while checks still land every few
/// milliseconds.
const HOT_BURN: Duration = Duration::from_millis(5);

/// One core's hot-screen tally. Unlike the isolated screen, the worker
/// records mismatches and keeps going: the count (and how it grows with
/// temperature) is diagnostic signal a first-failure abort would discard.
#[derive(Debug, Clone, PartialEq)]
pub struct HotCoreReport {
    /// Completed interleaved rounds (one int + one float check each).
    pub rounds: u64,
    pub mismatches: u64,
    /// Description of the earliest mismatch, for the failure reason.
    pub first_mismatch: Option<String>,
}

impl HotCoreReport {
    pub fn outcome(&self) -> TestOutcome {
        if self.mismatches == 0 {
            TestOutcome::Passed
        } else {
            let first = self.first_mismatch.as_deref().unwrap_or("unrecorded");
            TestOutcome::Failed {
                reason: format!(
                    "{} mismatch(es) over {} hot rounds under full-package load; first: {first}",
                    self.mismatches, self.rounds
                ),
            }
        }
    }
}

/// Correctness under load: every core simultaneously alternates between the
/// power-heavy FMA burn and the checksum workloads for `sdc_hot_secs` wall
/// seconds. Emits per-core `cpu_sdc_hot.{mismatches,rounds}` metrics and a
/// per-core outcome; mismatches are hard failures. Additional to — never a
/// replacement for — the isolated screen in `run_correctness`.
fn run_hot_correctness(
    sink: &EventSink,
    spec: &CpuTaskSpec,
    cores: usize,
    core_ids: &[CoreId],
    golden_int: u64,
    golden_float: u64,
) {
    if spec.sdc_hot_secs == 0 {
        sink.outcome(
            TestId::CpuSdcHot,
            Scope::Node,
            TestOutcome::Skipped {
                reason: "hot SDC screen disabled (sdc_hot_secs = 0)".into(),
            },
        );
        return;
    }

    let budget = Duration::from_secs(spec.sdc_hot_secs);
    let verdicts: Vec<Result<HotCoreReport, TestOutcome>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..cores)
            .map(|index| {
                let core = core_ids.get(index).copied();
                scope.spawn(move || match core {
                    None => Err(TestOutcome::Skipped {
                        reason: format!("core {index} not in this process's affinity mask"),
                    }),
                    Some(core) if !core_affinity::set_for_current(core) => {
                        Err(TestOutcome::Skipped {
                            reason: format!("failed to pin worker to cpu {}", core.id),
                        })
                    }
                    Some(_) => Ok(hot_worker(budget, golden_int, golden_float)),
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle.join().unwrap_or_else(|_| {
                    Err(TestOutcome::Failed {
                        reason: "hot SDC worker panicked".to_string(),
                    })
                })
            })
            .collect()
    });

    for (index, verdict) in verdicts.into_iter().enumerate() {
        let scope = Scope::Core { id: index as u32 };
        match verdict {
            Ok(report) => {
                for (name, value) in [
                    ("mismatches", report.mismatches as f64),
                    ("rounds", report.rounds as f64),
                ] {
                    sink.metric(MetricRecord {
                        test: TestId::CpuSdcHot,
                        scope: scope.clone(),
                        name: name.to_string(),
                        value,
                        unit: Unit::Count,
                        repeat: 0,
                    });
                }
                sink.outcome(TestId::CpuSdcHot, scope, report.outcome());
            }
            Err(outcome) => sink.outcome(TestId::CpuSdcHot, scope, outcome),
        }
    }
}

/// Alternate FMA burn and correctness rounds until the budget elapses. The
/// burn keeps this core (and, with every core running this concurrently,
/// the whole package) at maximum power draw while the checks run.
fn hot_worker(budget: Duration, golden_int: u64, golden_float: u64) -> HotCoreReport {
    let start = Instant::now();
    let mut report = HotCoreReport {
        rounds: 0,
        mismatches: 0,
        first_mismatch: None,
    };
    loop {
        // The rate is irrelevant here; the burn exists for its power draw.
        let _ = fma_gflops(HOT_BURN);
        let integer = checksum_round(INT_SEED, CORRECTNESS_ITERS);
        if integer != golden_int {
            report.mismatches += 1;
            report.first_mismatch.get_or_insert_with(|| {
                format!(
                    "integer checksum mismatch on round {}: got {integer:#018x}, \
                     expected {golden_int:#018x}",
                    report.rounds
                )
            });
        }
        let float = float_checksum_round(FLOAT_SEED, CORRECTNESS_ITERS);
        if float != golden_float {
            report.mismatches += 1;
            report.first_mismatch.get_or_insert_with(|| {
                format!(
                    "float checksum mismatch on round {}: got {float:#018x}, \
                     expected {golden_float:#018x}",
                    report.rounds
                )
            });
        }
        report.rounds += 1;
        if start.elapsed() >= budget {
            return report;
        }
    }
}

// ---------------------------------------------------------------------------
// Throughput
// ---------------------------------------------------------------------------

fn run_throughput(sink: &EventSink, spec: &CpuTaskSpec, cores: usize, core_ids: &[CoreId]) {
    // The budget is split across cores because each core is measured alone:
    // a neighbour running flat out changes the turbo bin and the answer.
    let per_core = Duration::from_secs_f64(spec.gflops_secs as f64 / cores as f64).max(MIN_MEASURE);

    for index in 0..cores {
        let core = core_ids.get(index).copied();
        let gflops = std::thread::scope(|scope| {
            scope
                .spawn(move || {
                    if let Some(core) = core {
                        // An unpinned throughput sample is still a sample;
                        // unlike correctness, it is not silently wrong.
                        core_affinity::set_for_current(core);
                    }
                    fma_gflops(per_core)
                })
                .join()
                .unwrap_or(0.0)
        });
        if gflops.is_finite() && gflops > 0.0 {
            sink.metric(MetricRecord {
                test: TestId::CpuGflops,
                scope: Scope::Core { id: index as u32 },
                name: "gflops".to_string(),
                value: gflops,
                unit: Unit::Gflops,
                repeat: 0,
            });
        } else {
            sink.outcome(
                TestId::CpuGflops,
                Scope::Core { id: index as u32 },
                TestOutcome::Failed {
                    reason: format!("throughput measurement produced {gflops}"),
                },
            );
        }
    }

    // All cores at once: what the node actually delivers under load, after
    // turbo and thermal limits bite.
    let allcore: f64 = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..cores)
            .map(|index| {
                let core = core_ids.get(index).copied();
                scope.spawn(move || {
                    if let Some(core) = core {
                        core_affinity::set_for_current(core);
                    }
                    fma_gflops(per_core)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap_or(0.0))
            .sum()
    });
    if allcore.is_finite() && allcore > 0.0 {
        sink.metric(MetricRecord {
            test: TestId::CpuGflops,
            scope: Scope::Node,
            name: "gflops_allcore".to_string(),
            value: allcore,
            unit: Unit::Gflops,
            repeat: 0,
        });
    } else {
        sink.outcome(
            TestId::CpuGflops,
            Scope::Node,
            TestOutcome::Failed {
                reason: format!("all-core throughput measurement produced {allcore}"),
            },
        );
    }
}

/// Sustained multiply-add rate of the calling thread, in GFLOP/s (a
/// multiply-add counts as two flops, the conventional roofline accounting).
///
/// Deliberately `acc * k + x` and not `mul_add`: without a compile-time FMA
/// target feature, `f64::mul_add` lowers to a libm call, which measures the
/// node's libm rather than its CPU and is ~8x slower. Written out, the two
/// operations are plain hardware instructions the backend is free to
/// vectorize — the accumulator lanes are independent, so there is no
/// reassociation and no cross-node variability.
///
/// The recurrence `acc <- acc*0.5 + buf[i]` is a contraction, so the values
/// stay bounded for any run length; `black_box` keeps the optimizer from
/// noticing the results are unused.
fn fma_gflops(budget: Duration) -> f64 {
    let buffer: Vec<f64> = (0..GFLOPS_BUFFER_LEN)
        .map(|i| 1.0 + (i as f64) * (1.0 / GFLOPS_BUFFER_LEN as f64))
        .collect();
    let buffer = black_box(&buffer[..]);
    let mut acc = [0.0f64; GFLOPS_LANES];
    let mut fmas: u64 = 0;

    let start = Instant::now();
    let elapsed = loop {
        for _ in 0..GFLOPS_PASSES_PER_CHECK {
            for chunk in buffer.as_chunks::<GFLOPS_LANES>().0 {
                for lane in 0..GFLOPS_LANES {
                    acc[lane] = acc[lane] * 0.5 + chunk[lane];
                }
            }
        }
        fmas += GFLOPS_PASSES_PER_CHECK * (buffer.len() - buffer.len() % GFLOPS_LANES) as u64;
        let elapsed = start.elapsed();
        if elapsed >= budget {
            break elapsed;
        }
    };
    black_box(&acc);

    let seconds = elapsed.as_secs_f64();
    if seconds <= 0.0 {
        return 0.0;
    }
    2.0 * fmas as f64 / seconds / 1e9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_checksum_is_seed_and_length_sensitive() {
        assert_eq!(checksum_round(1, 100), checksum_round(1, 100));
        assert_ne!(checksum_round(1, 100), checksum_round(2, 100));
        assert_ne!(checksum_round(1, 100), checksum_round(1, 101));
    }

    #[test]
    fn float_checksum_stays_finite_and_nonzero() {
        for iters in [1u64, 10, 1000, 50_000] {
            let bits = float_checksum_round(3, iters);
            let value = f64::from_bits(bits);
            assert!(value.is_finite(), "iters={iters} value={value}");
            assert!(value > 0.0, "iters={iters} value={value}");
        }
        assert_ne!(float_checksum_round(3, 1000), float_checksum_round(3, 1001));
    }

    #[test]
    fn fma_kernel_reports_positive_rate() {
        let gflops = fma_gflops(Duration::from_millis(20));
        assert!(gflops.is_finite() && gflops > 0.0, "got {gflops}");
    }
}
