use std::collections::BTreeSet;

use gauntlet::analysis::schedule::{sampled_rounds, tournament_rounds};

fn all_pairs(rounds: &[Vec<(usize, usize)>]) -> Vec<(usize, usize)> {
    rounds.iter().flatten().copied().collect()
}

fn assert_rounds_disjoint(rounds: &[Vec<(usize, usize)>]) {
    for (i, round) in rounds.iter().enumerate() {
        let mut seen = BTreeSet::new();
        for &(a, b) in round {
            assert!(a < b, "pairs must be ordered (a < b), got ({a}, {b})");
            assert!(seen.insert(a), "host {a} appears twice in round {i}");
            assert!(seen.insert(b), "host {b} appears twice in round {i}");
        }
    }
}

#[test]
fn degenerate_sizes_yield_no_rounds() {
    assert!(tournament_rounds(0).is_empty());
    assert!(tournament_rounds(1).is_empty());
}

#[test]
fn two_hosts_one_round() {
    assert_eq!(tournament_rounds(2), vec![vec![(0, 1)]]);
}

#[test]
fn even_fleet_full_coverage_in_n_minus_1_rounds() {
    let n = 8;
    let rounds = tournament_rounds(n);
    assert_eq!(rounds.len(), n - 1);
    for round in &rounds {
        assert_eq!(round.len(), n / 2, "even n: every host paired each round");
    }
    assert_rounds_disjoint(&rounds);
    let pairs: BTreeSet<_> = all_pairs(&rounds).into_iter().collect();
    assert_eq!(pairs.len(), n * (n - 1) / 2, "every pair exactly once");
}

#[test]
fn odd_fleet_full_coverage_in_n_rounds() {
    let n = 7;
    let rounds = tournament_rounds(n);
    assert_eq!(rounds.len(), n);
    for round in &rounds {
        assert_eq!(round.len(), (n - 1) / 2, "odd n: one host sits out");
    }
    assert_rounds_disjoint(&rounds);
    let pairs: BTreeSet<_> = all_pairs(&rounds).into_iter().collect();
    assert_eq!(pairs.len(), n * (n - 1) / 2);
}

#[test]
fn target_scale_holds_the_invariants() {
    for n in [63, 64, 200] {
        let rounds = tournament_rounds(n);
        assert_rounds_disjoint(&rounds);
        let listed = all_pairs(&rounds);
        let unique: BTreeSet<_> = listed.iter().copied().collect();
        assert_eq!(listed.len(), unique.len(), "n={n}: pair repeated");
        assert_eq!(unique.len(), n * (n - 1) / 2, "n={n}: pair missing");
        let expected_rounds = if n % 2 == 0 { n - 1 } else { n };
        assert_eq!(rounds.len(), expected_rounds, "n={n}: linear wall time");
    }
}

#[test]
fn sampled_mode_is_a_strict_subset_with_bounded_degree() {
    let (n, k) = (32, 3);
    let rounds = sampled_rounds(n, k);
    assert_rounds_disjoint(&rounds);
    let listed = all_pairs(&rounds);
    let unique: BTreeSet<_> = listed.iter().copied().collect();
    assert_eq!(listed.len(), unique.len(), "no pair repeated");
    assert!(
        unique.len() < n * (n - 1) / 2,
        "sampling must be cheaper than full mesh"
    );
    // Ring-offset selection: every host tests against at least k peers.
    for host in 0..n {
        let degree = unique
            .iter()
            .filter(|&&(a, b)| a == host || b == host)
            .count();
        assert!(degree >= k, "host {host} degree {degree} < {k}");
    }
}

#[test]
fn sampled_mode_edge_cases() {
    assert!(sampled_rounds(5, 0).is_empty());
    assert!(sampled_rounds(0, 3).is_empty());
    assert!(sampled_rounds(1, 3).is_empty());
    // k too large degrades to (at most) full mesh without duplicates.
    let rounds = sampled_rounds(4, 100);
    let listed = all_pairs(&rounds);
    let unique: BTreeSet<_> = listed.iter().copied().collect();
    assert_eq!(listed.len(), unique.len());
    assert!(unique.len() <= 6);
}
