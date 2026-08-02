//! Geometry for the fleet graph: unit-space circular layout, pixel
//! mapping, and hit-testing. Pure math with no gpui types so it unit tests
//! headlessly.

/// Ring radius in unit-square coordinates (center is (0.5, 0.5)).
pub const RING_RADIUS: f32 = 0.42;

/// Positions for `n` nodes on a ring, starting at the top and proceeding
/// clockwise, in unit-square coordinates.
pub fn circle_positions(n: usize) -> Vec<(f32, f32)> {
    let count = n as f32;
    (0..n)
        .map(|i| {
            let angle = -std::f32::consts::FRAC_PI_2 + std::f32::consts::TAU * (i as f32) / count;
            (
                0.5 + RING_RADIUS * angle.cos(),
                0.5 + RING_RADIUS * angle.sin(),
            )
        })
        .collect()
}

/// Map a unit-square coordinate into a `width` x `height` viewport,
/// letterboxed to the shorter dimension so the ring stays circular.
pub fn to_px(unit: (f32, f32), width: f32, height: f32) -> (f32, f32) {
    let side = width.min(height);
    let offset_x = (width - side) / 2.0;
    let offset_y = (height - side) / 2.0;
    (offset_x + unit.0 * side, offset_y + unit.1 * side)
}

/// Distance from point `p` to the segment `a`-`b`.
pub fn dist_point_segment(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (abx, aby) = (b.0 - a.0, b.1 - a.1);
    let len_squared = abx * abx + aby * aby;
    if len_squared <= f32::EPSILON {
        return (p.0 - a.0).hypot(p.1 - a.1);
    }
    let t = (((p.0 - a.0) * abx + (p.1 - a.1) * aby) / len_squared).clamp(0.0, 1.0);
    let closest = (a.0 + t * abx, a.1 + t * aby);
    (p.0 - closest.0).hypot(p.1 - closest.1)
}

/// Index of the closest node whose center is within `radius` of `p`.
pub fn hit_node(p: (f32, f32), centers: &[(f32, f32)], radius: f32) -> Option<usize> {
    centers
        .iter()
        .enumerate()
        .map(|(i, center)| (i, (p.0 - center.0).hypot(p.1 - center.1)))
        .filter(|(_, distance)| *distance <= radius)
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(i, _)| i)
}

/// Index of the closest edge (into `edges`) within `tolerance` of `p`.
/// Edges referencing out-of-range node indices are skipped.
pub fn hit_edge(
    p: (f32, f32),
    centers: &[(f32, f32)],
    edges: &[(usize, usize)],
    tolerance: f32,
) -> Option<usize> {
    edges
        .iter()
        .enumerate()
        .filter_map(|(i, (a, b))| {
            let a = centers.get(*a)?;
            let b = centers.get(*b)?;
            let distance = dist_point_segment(p, *a, *b);
            (distance <= tolerance).then_some((i, distance))
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(i, _)| i)
}
