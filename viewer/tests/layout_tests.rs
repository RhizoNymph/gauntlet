//! Geometry tests for the graph layout and hit-testing math.

use gauntlet_view::layout::{
    RING_RADIUS, circle_positions, dist_point_segment, hit_edge, hit_node, to_px,
};

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

#[test]
fn positions_lie_on_the_ring() {
    let positions = circle_positions(6);
    assert_eq!(positions.len(), 6);
    for p in &positions {
        let r = dist(*p, (0.5, 0.5));
        assert!((r - RING_RADIUS).abs() < 1e-5, "radius {r}");
    }
    // All distinct.
    for i in 0..positions.len() {
        for j in i + 1..positions.len() {
            assert!(dist(positions[i], positions[j]) > 1e-3);
        }
    }
}

#[test]
fn first_position_is_at_the_top() {
    let positions = circle_positions(4);
    assert!(dist(positions[0], (0.5, 0.5 - RING_RADIUS)) < 1e-5);
    // Second node is a quarter turn clockwise (to the right).
    assert!(dist(positions[1], (0.5 + RING_RADIUS, 0.5)) < 1e-5);
}

#[test]
fn zero_and_one_node_layouts_are_sane() {
    assert!(circle_positions(0).is_empty());
    let single = circle_positions(1);
    assert_eq!(single.len(), 1);
    assert!(single[0].0.is_finite() && single[0].1.is_finite());
}

#[test]
fn to_px_letterboxes_into_the_short_dimension() {
    // Wide viewport: the unit square maps to a centered 100x100 region.
    assert_eq!(to_px((0.5, 0.5), 200.0, 100.0), (100.0, 50.0));
    assert_eq!(to_px((0.0, 0.0), 200.0, 100.0), (50.0, 0.0));
    // Tall viewport letterboxes vertically.
    assert_eq!(to_px((0.0, 0.0), 100.0, 200.0), (0.0, 50.0));
}

#[test]
fn point_segment_distance_handles_perpendicular_and_beyond_ends() {
    let a = (0.0, 0.0);
    let b = (1.0, 0.0);
    assert!((dist_point_segment((0.5, 1.0), a, b) - 1.0).abs() < 1e-6);
    assert!((dist_point_segment((2.0, 0.0), a, b) - 1.0).abs() < 1e-6);
    assert!((dist_point_segment((-3.0, 0.0), a, b) - 3.0).abs() < 1e-6);
    // Degenerate segment falls back to point distance.
    assert!((dist_point_segment((3.0, 4.0), a, a) - 5.0).abs() < 1e-6);
}

#[test]
fn edge_hit_testing_picks_the_closest_edge_within_tolerance() {
    let centers = vec![(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)];
    let edges = vec![(0, 1), (0, 2)];
    assert_eq!(hit_edge((50.0, 3.0), &centers, &edges, 8.0), Some(0));
    assert_eq!(hit_edge((3.0, 50.0), &centers, &edges, 8.0), Some(1));
    assert_eq!(hit_edge((50.0, 50.0), &centers, &edges, 8.0), None);
    // Out-of-range endpoint indices are skipped, not a panic.
    let broken = vec![(0, 9)];
    assert_eq!(hit_edge((50.0, 0.0), &centers, &broken, 8.0), None);
}

#[test]
fn node_hit_testing_picks_the_closest_node_within_radius() {
    let centers = vec![(0.0, 0.0), (30.0, 0.0)];
    assert_eq!(hit_node((2.0, 2.0), &centers, 16.0), Some(0));
    assert_eq!(hit_node((17.0, 0.0), &centers, 16.0), Some(1));
    assert_eq!(hit_node((100.0, 100.0), &centers, 16.0), None);
}
