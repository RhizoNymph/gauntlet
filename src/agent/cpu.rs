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

use anyhow::Result;

use crate::agent::EventSink;
use crate::proto::CpuTaskSpec;

/// Deterministic work unit both correctness workloads iterate. Public so
/// tests can pin down golden values.
pub fn checksum_round(seed: u64, iters: u64) -> u64 {
    let _ = (seed, iters);
    todo!("agent B: implement")
}

/// f64 kernel checksum: sum of a fixed-seed LCG-driven FMA chain, bitcast to
/// u64. Bit-exact across runs on a correct core (no reassociation: scalar
/// ops, no -ffast-math equivalents).
pub fn float_checksum_round(seed: u64, iters: u64) -> u64 {
    let _ = (seed, iters);
    todo!("agent B: implement")
}

pub fn run(sink: &EventSink, spec: &CpuTaskSpec) -> Result<()> {
    let _ = (sink, spec);
    todo!("agent B: implement")
}
