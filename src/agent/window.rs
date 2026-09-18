//! Window consensus for the fleet overlap step: time-based measurement
//! windows agreed across NCCL ranks *through the collective itself*, with
//! no cross-host clock comparison.
//!
//! The problem: the overlap step measures two windows (an isolated
//! all-reduce baseline, then the same all-reduce under GEMM load) whose
//! boundaries must fall between the same two iterations on every rank —
//! otherwise one rank's "overlapped" iterations are another's baseline.
//! Host clocks cannot arbitrate this: phase 0's `clock_offset_ms` is
//! NTP/chrony-grade, and any wall-clock cutoff would land mid-iteration
//! differently per host anyway.
//!
//! The mechanism: every control step, all ranks all-reduce a one-element
//! control word with MIN reduction. Rank 0 (the lead) contributes
//! [`CONTROL_CLOSE`] (0.0) once its local clock says the window is over and
//! [`CONTROL_OPEN`] (1.0) before that; every other rank always contributes
//! [`CONTROL_OPEN`]. The MIN across ranks is therefore 0 exactly when the
//! lead has closed the window, and because a collective returns the same
//! value everywhere, every rank observes the closure in the same control
//! step and leaves the window between the same two payload iterations.
//! Only rank 0's clock ever matters, so clock skew cannot split the fleet.
//!
//! Everything here is pure and unit-tested without a GPU; the NCCL runner
//! (`agent::nccl`) is a thin shell around it.

/// Control word meaning "my window is still open". Any positive value
/// works; 1.0 keeps the reduce exact in f32.
pub const CONTROL_OPEN: f32 = 1.0;
/// Control word the lead contributes once its window has elapsed.
pub const CONTROL_CLOSE: f32 = 0.0;

/// Role in the consensus. Exactly one rank (NCCL rank 0) leads; its local
/// clock is the only one that can close a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowRole {
    Lead,
    Follower,
}

/// The control word this rank contributes for the current control step.
/// Followers ignore their own clocks entirely — a follower that closed
/// windows on its own clock would recreate exactly the cross-host clock
/// comparison this module exists to avoid.
pub fn contribution(role: WindowRole, deadline_reached: bool) -> f32 {
    match role {
        WindowRole::Lead if deadline_reached => CONTROL_CLOSE,
        WindowRole::Lead | WindowRole::Follower => CONTROL_OPEN,
    }
}

/// Interpret the MIN-reduced control word: the window is closed exactly
/// when the lead contributed [`CONTROL_CLOSE`]. The midpoint threshold
/// tolerates any floating-point residue a reduction could introduce.
pub fn window_closed(reduced_min: f32) -> bool {
    reduced_min < (CONTROL_OPEN + CONTROL_CLOSE) / 2.0
}

/// Payload-iteration accounting for one consensus window. Only payload
/// batches are recorded (the control reduce and its synchronizations stay
/// outside), so the per-iteration figure measures the collective, not the
/// consensus overhead.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WindowTally {
    iters: u64,
    busy_secs: f64,
}

impl WindowTally {
    /// Record one synchronized batch of `iters` payload iterations that
    /// took `secs` of wall time.
    pub fn record(&mut self, iters: u64, secs: f64) {
        self.iters += iters;
        self.busy_secs += secs;
    }

    pub fn iters(&self) -> u64 {
        self.iters
    }

    /// Mean seconds per payload iteration; never divides by zero.
    pub fn per_iter_secs(&self) -> f64 {
        self.busy_secs / self.iters.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn followers_never_close_a_window() {
        assert_eq!(contribution(WindowRole::Follower, false), CONTROL_OPEN);
        // Even a follower whose *own* clock says time is up keeps the
        // window open: only the lead's clock counts.
        assert_eq!(contribution(WindowRole::Follower, true), CONTROL_OPEN);
    }

    #[test]
    fn the_lead_closes_exactly_at_its_deadline() {
        assert_eq!(contribution(WindowRole::Lead, false), CONTROL_OPEN);
        assert_eq!(contribution(WindowRole::Lead, true), CONTROL_CLOSE);
    }

    #[test]
    fn min_reduction_closes_iff_the_lead_closed() {
        // Simulate the MIN all-reduce over a 4-rank world.
        let reduce = |words: &[f32]| words.iter().copied().fold(f32::INFINITY, f32::min);

        let open = [
            contribution(WindowRole::Lead, false),
            contribution(WindowRole::Follower, false),
            contribution(WindowRole::Follower, true),
            contribution(WindowRole::Follower, false),
        ];
        assert!(!window_closed(reduce(&open)));

        let closed = [
            contribution(WindowRole::Lead, true),
            contribution(WindowRole::Follower, false),
            contribution(WindowRole::Follower, false),
            contribution(WindowRole::Follower, false),
        ];
        assert!(window_closed(reduce(&closed)));
    }

    #[test]
    fn closure_is_robust_to_floating_point_residue() {
        assert!(window_closed(0.0));
        assert!(window_closed(1e-7));
        assert!(window_closed(-1e-7));
        assert!(!window_closed(1.0));
        assert!(!window_closed(1.0 - 1e-6));
    }

    #[test]
    fn tallies_average_over_payload_iterations_only() {
        let mut tally = WindowTally::default();
        tally.record(4, 0.4);
        tally.record(4, 0.6);
        assert_eq!(tally.iters(), 8);
        assert!((tally.per_iter_secs() - 0.125).abs() < 1e-12);
    }

    #[test]
    fn an_empty_tally_never_divides_by_zero() {
        let tally = WindowTally::default();
        assert_eq!(tally.iters(), 0);
        assert_eq!(tally.per_iter_secs(), 0.0);
    }
}
