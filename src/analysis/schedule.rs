//! Round-robin tournament scheduling for pairwise network tests.
//!
//! Full mesh over n hosts is n(n-1)/2 pairs; scheduling them as rounds of
//! disjoint pairs makes wall time linear in n: n-1 rounds for even n, n for
//! odd n (each host sits out once).

/// Return rounds of disjoint pairs covering every unordered pair of
/// `0..n` exactly once. Every round's pairs can run simultaneously because
/// no host appears twice within a round. `n <= 1` yields no rounds.
///
/// Pairs are emitted as `(a, b)` with `a < b`.
pub fn tournament_rounds(n: usize) -> Vec<Vec<(usize, usize)>> {
    let _ = n;
    todo!("agent D: implement")
}

/// Sampled alternative for quick runs: each host is paired with
/// `pairs_per_host` distinct peers (ring-offset selection so choices spread
/// across the fleet), deduplicated, then packed into disjoint rounds.
pub fn sampled_rounds(n: usize, pairs_per_host: usize) -> Vec<Vec<(usize, usize)>> {
    let _ = (n, pairs_per_host);
    todo!("agent D: implement")
}
