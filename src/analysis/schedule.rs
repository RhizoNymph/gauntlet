//! Round-robin tournament scheduling for pairwise network tests.
//!
//! Full mesh over n hosts is n(n-1)/2 pairs; scheduling them as rounds of
//! disjoint pairs makes wall time linear in n: n-1 rounds for even n, n for
//! odd n (each host sits out once).

use std::collections::BTreeSet;

/// Return rounds of disjoint pairs covering every unordered pair of
/// `0..n` exactly once. Every round's pairs can run simultaneously because
/// no host appears twice within a round. `n <= 1` yields no rounds.
///
/// Pairs are emitted as `(a, b)` with `a < b`.
pub fn tournament_rounds(n: usize) -> Vec<Vec<(usize, usize)>> {
    if n <= 1 {
        return Vec::new();
    }
    // Circle method: seat everyone around a table, hold seat 0 fixed and
    // rotate the rest by one each round. Odd fleets get a phantom player
    // (index `n`) whose partner sits the round out.
    let seats = if n.is_multiple_of(2) { n } else { n + 1 };
    let mut order: Vec<usize> = (0..seats).collect();
    let mut rounds = Vec::with_capacity(seats - 1);

    for _ in 0..seats - 1 {
        let mut round = Vec::with_capacity(seats / 2);
        for i in 0..seats / 2 {
            let a = order[i];
            let b = order[seats - 1 - i];
            if a < n && b < n {
                round.push((a.min(b), a.max(b)));
            }
        }
        round.sort_unstable();
        rounds.push(round);
        rotate_fixing_first(&mut order);
    }
    rounds
}

/// Sampled alternative for quick runs: each host is paired with
/// `pairs_per_host` distinct peers (ring-offset selection so choices spread
/// across the fleet), deduplicated, then packed into disjoint rounds.
pub fn sampled_rounds(n: usize, pairs_per_host: usize) -> Vec<Vec<(usize, usize)>> {
    if n < 2 || pairs_per_host == 0 {
        return Vec::new();
    }
    // Offsets beyond n/2 only restate earlier ones on the ring, so a huge
    // `pairs_per_host` degrades to the full mesh instead of duplicating.
    let max_offset = n / 2;
    let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    for offset in 1..=pairs_per_host.min(max_offset) {
        for i in 0..n {
            let j = (i + offset) % n;
            if i != j {
                edges.insert((i.min(j), i.max(j)));
            }
        }
    }
    pack_into_rounds(edges)
}

/// Rotate all seats but the first one position clockwise.
fn rotate_fixing_first(order: &mut [usize]) {
    let len = order.len();
    if len < 3 {
        return;
    }
    let last = order[len - 1];
    for i in (2..len).rev() {
        order[i] = order[i - 1];
    }
    order[1] = last;
}

/// Greedy first-fit packing: each pair joins the earliest round where
/// neither endpoint is already busy. Deterministic for a sorted input.
fn pack_into_rounds(edges: BTreeSet<(usize, usize)>) -> Vec<Vec<(usize, usize)>> {
    let mut rounds: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut busy: Vec<BTreeSet<usize>> = Vec::new();
    for (a, b) in edges {
        let slot = busy
            .iter()
            .position(|hosts| !hosts.contains(&a) && !hosts.contains(&b));
        match slot {
            Some(index) => {
                busy[index].insert(a);
                busy[index].insert(b);
                rounds[index].push((a, b));
            }
            None => {
                busy.push(BTreeSet::from([a, b]));
                rounds.push(vec![(a, b)]);
            }
        }
    }
    rounds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_hosts_each_sit_out_exactly_once() {
        let rounds = tournament_rounds(3);
        assert_eq!(rounds.len(), 3);
        for round in &rounds {
            assert_eq!(round.len(), 1);
        }
        let pairs: BTreeSet<_> = rounds.iter().flatten().copied().collect();
        assert_eq!(pairs, BTreeSet::from([(0, 1), (0, 2), (1, 2)]));
    }

    #[test]
    fn schedules_are_deterministic() {
        assert_eq!(tournament_rounds(9), tournament_rounds(9));
        assert_eq!(sampled_rounds(16, 2), sampled_rounds(16, 2));
    }

    #[test]
    fn sampled_offsets_are_ring_neighbours() {
        let rounds = sampled_rounds(6, 1);
        let pairs: BTreeSet<_> = rounds.iter().flatten().copied().collect();
        assert_eq!(
            pairs,
            BTreeSet::from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)])
        );
    }

    #[test]
    fn sampled_packing_is_reasonably_tight() {
        // 6 hosts, ring neighbours only: 2 hosts busy per pair, so a perfect
        // packing is 3 rounds of 3 pairs; greedy must not be far off.
        let rounds = sampled_rounds(6, 1);
        assert!(
            rounds.len() <= 4,
            "greedy packing used {} rounds",
            rounds.len()
        );
    }

    #[test]
    fn sampled_equals_full_mesh_when_k_saturates() {
        let sampled: BTreeSet<_> = sampled_rounds(7, 99).iter().flatten().copied().collect();
        let full: BTreeSet<_> = tournament_rounds(7).iter().flatten().copied().collect();
        assert_eq!(sampled, full);
    }
}
