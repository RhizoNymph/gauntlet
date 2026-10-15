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

/// Role of a global rank. The fleet world runs one rank per GPU and one
/// process drives a host's whole rank block, so the lead's *host* also
/// drives follower ranks: global rank 0 leads, and every other rank —
/// rank 0's local siblings included — follows. The MIN reduction then
/// works unchanged: the lead contributes the only close word.
pub fn role_for_rank(global_rank: u32) -> WindowRole {
    if global_rank == 0 {
        WindowRole::Lead
    } else {
        WindowRole::Follower
    }
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

/// Minimum tallied payload iterations before a close signal is honored.
///
/// On a degraded link a short window (baseline default 5s) could otherwise
/// close after a single cold batch, making the retention denominator
/// noise. Every rank applies the floor to its own tally, and every rank
/// runs exactly the same batches (windows close in the same control step),
/// so the counts — and therefore the decision — are identical fleet-wide:
/// the floor can never split the group.
pub const MIN_WINDOW_ITERS: u64 = 16;

/// Whether to actually leave the window: the lead has signalled closure
/// *and* enough payload iterations are in the tally for the window's
/// figure to mean something.
pub fn should_close(reduced_min: f32, tallied_iters: u64) -> bool {
    window_closed(reduced_min) && tallied_iters >= MIN_WINDOW_ITERS
}

/// Floor of the follower failsafe, so a tiny configured window still
/// leaves room for warm-up and the iteration floor.
pub const FAILSAFE_FLOOR_SECS: u64 = 10;

/// Follower failsafe budget for one window, in seconds: if the lead has
/// not been seen to *signal* close within twice the window's configured
/// duration (with a floor), the lead is presumed dead and the follower
/// must end the protocol with a structured error instead of hammering the
/// fabric until an external kill. Only helps while collectives still
/// complete — a rank blocked *inside* a collective can only be reaped by
/// the orchestrator's phase timeout.
pub fn failsafe_secs(budget_secs: u64) -> u64 {
    budget_secs.saturating_mul(2).max(FAILSAFE_FLOOR_SECS)
}

/// Whether a follower should abandon the protocol: only while the lead's
/// close signal has *never* been observed. Once the close word has been
/// seen the lead is provably alive and the loop is bounded by the
/// iteration floor — on a degraded fabric a window legitimately runs past
/// the failsafe while it accumulates its floor, and cutting it down there
/// would strand the other ranks in a blocking collective, the exact
/// failure the failsafe exists to prevent. (A lead that dies *after*
/// signaling close leaves ranks blocked inside a collective, which is the
/// phase timeout's job.)
pub fn failsafe_tripped(close_seen: bool, elapsed_secs: f64, failsafe_secs: u64) -> bool {
    !close_seen && elapsed_secs > failsafe_secs as f64
}

/// Slack on top of both windows' failsafe budgets in the hard deadline:
/// warmup collectives, worker setup/teardown, report emission.
pub const HARD_DEADLINE_SLACK_SECS: u64 = 120;

/// Hard wall-clock budget, in seconds from protocol start, of a whole
/// overlap step (isolated window + overlapped window). Past it the step is
/// presumed wedged — typically every rank blocked inside a collective
/// because some other rank died — and nothing the step started may keep
/// running: GEMM workers stop on their own at this deadline (independent
/// of the driver ever reaching `GemmLoad::finish`), and the fleet step's
/// watchdog terminates the process shortly after.
///
/// Each window gets its follower failsafe budget (so the in-protocol
/// failsafe always gets the first chance to end things cleanly) plus
/// [`HARD_DEADLINE_SLACK_SECS`]. Saturating, so absurd specs cannot
/// overflow into a tiny deadline.
pub fn hard_deadline_secs(baseline_secs: u64, duration_secs: u64) -> u64 {
    failsafe_secs(baseline_secs)
        .saturating_add(failsafe_secs(duration_secs))
        .saturating_add(HARD_DEADLINE_SLACK_SECS)
}

/// Grace between the hard deadline (GEMM workers stop) and the watchdog
/// terminating the process, so a protocol that is merely finishing up can
/// still exit through its normal path.
pub const WATCHDOG_GRACE_SECS: u64 = 15;

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
    fn exactly_one_lead_across_multi_rank_blocks() {
        // Three hosts x four GPUs: ranks 0..12, blocks 0..4, 4..8, 8..12.
        let world = 12u32;
        let leads: Vec<u32> = (0..world)
            .filter(|rank| role_for_rank(*rank) == WindowRole::Lead)
            .collect();
        assert_eq!(leads, [0], "only global rank 0 leads");
        // Rank 0's local siblings are followers.
        for sibling in 1..4 {
            assert_eq!(role_for_rank(sibling), WindowRole::Follower);
        }
    }

    #[test]
    fn min_consensus_with_multiple_local_ranks_per_host() {
        let reduce = |words: &[f32]| words.iter().copied().fold(f32::INFINITY, f32::min);
        let world = 12u32;
        // Every host's clock except the lead's says time is up: the window
        // must stay open — rank 0's siblings share its host clock but are
        // followers, so they cannot close it either.
        let open: Vec<f32> = (0..world)
            .map(|rank| contribution(role_for_rank(rank), rank != 0))
            .collect();
        assert!(!window_closed(reduce(&open)));
        // Only the lead's deadline closes it, regardless of every other
        // rank's clock.
        for others_expired in [false, true] {
            let closed: Vec<f32> = (0..world)
                .map(|rank| contribution(role_for_rank(rank), rank == 0 || others_expired))
                .collect();
            assert!(window_closed(reduce(&closed)));
        }
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
    fn close_signals_are_ignored_until_the_iteration_floor() {
        let closed = CONTROL_CLOSE;
        assert!(!should_close(closed, 0));
        assert!(!should_close(closed, MIN_WINDOW_ITERS - 1));
        assert!(should_close(closed, MIN_WINDOW_ITERS));
        assert!(should_close(closed, MIN_WINDOW_ITERS + 100));
        // No amount of iterations closes a window the lead holds open.
        assert!(!should_close(CONTROL_OPEN, u64::MAX));
    }

    #[test]
    fn the_failsafe_never_trips_once_the_close_signal_was_seen() {
        // Degraded-fabric scenario the iteration floor was added for: a
        // healthy lead signaled close at its 5s deadline, but slow
        // iterations hold the window open past the 10s failsafe while the
        // floor accumulates. The follower must keep going.
        assert!(!failsafe_tripped(true, 25.0, 10));
        // A lead that has never signaled close past the failsafe is
        // presumed dead.
        assert!(failsafe_tripped(false, 10.5, 10));
        // Before the failsafe elapses nothing trips either way.
        assert!(!failsafe_tripped(false, 9.5, 10));
        assert!(!failsafe_tripped(true, 9.5, 10));
    }

    #[test]
    fn the_failsafe_is_twice_the_budget_with_a_floor() {
        assert_eq!(failsafe_secs(30), 60);
        assert_eq!(failsafe_secs(5), 10);
        // Tiny and zero budgets keep a workable floor.
        assert_eq!(failsafe_secs(1), FAILSAFE_FLOOR_SECS);
        assert_eq!(failsafe_secs(0), FAILSAFE_FLOOR_SECS);
        // Absurd budgets must not overflow.
        assert_eq!(failsafe_secs(u64::MAX), u64::MAX);
    }

    #[test]
    fn the_hard_deadline_covers_both_failsafes_plus_slack() {
        // Defaults: 5s baseline, 30s overlapped window.
        assert_eq!(
            hard_deadline_secs(5, 30),
            10 + 60 + HARD_DEADLINE_SLACK_SECS
        );
        // Tiny windows keep the failsafe floors.
        assert_eq!(
            hard_deadline_secs(0, 0),
            2 * FAILSAFE_FLOOR_SECS + HARD_DEADLINE_SLACK_SECS
        );
        // Always strictly later than both in-protocol failsafes combined,
        // so the clean path gets the first chance.
        for (baseline, duration) in [(1, 1), (5, 30), (60, 600)] {
            assert!(
                hard_deadline_secs(baseline, duration)
                    > failsafe_secs(baseline) + failsafe_secs(duration)
            );
        }
        // Absurd specs saturate instead of wrapping to a tiny deadline.
        assert_eq!(hard_deadline_secs(u64::MAX, 30), u64::MAX);
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
