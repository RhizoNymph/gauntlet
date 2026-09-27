//! Per-rank completion timing for a process that drives several NCCL
//! ranks from one thread.
//!
//! Synchronizing the local streams one after another would stamp local
//! rank 0 first and rank n-1 last on *every* iteration — a systematic,
//! index-ordered bias that the fleet-relative MAD analysis would read as
//! per-GPU skew. Instead, every rank's completion marker is polled
//! round-robin and each rank is stamped the first time it is seen done, so
//! no rank is favoured by its position (the residual error is one polling
//! sweep, a handful of cheap driver queries).
//!
//! Pure over the probe and the clock so it is unit-tested without a GPU.

/// State of one rank's completion marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Readiness {
    Ready,
    Pending,
}

/// Poll `ranks` completion probes until every one is ready. Returns, per
/// rank index, the `now()` reading at which that rank was first observed
/// ready. A probe error aborts polling and is returned as-is (a failed
/// stream never becomes ready, so waiting on would spin forever).
pub(crate) fn first_ready_times<E>(
    ranks: usize,
    mut probe: impl FnMut(usize) -> Result<Readiness, E>,
    mut now: impl FnMut() -> f64,
) -> Result<Vec<f64>, E> {
    let mut stamps: Vec<Option<f64>> = vec![None; ranks];
    let mut pending = ranks;
    while pending > 0 {
        for (index, stamp) in stamps.iter_mut().enumerate() {
            if stamp.is_some() {
                continue;
            }
            if probe(index)? == Readiness::Ready {
                *stamp = Some(now());
                pending -= 1;
            }
        }
        if pending > 0 {
            std::hint::spin_loop();
        }
    }
    Ok(stamps.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A probe where rank `i` becomes ready on its `ready_after[i]`-th
    /// query, and a clock that ticks once per reading.
    fn simulate(ready_after: &[u32]) -> Vec<f64> {
        let mut queries = vec![0u32; ready_after.len()];
        let mut clock = 0.0;
        first_ready_times::<()>(
            ready_after.len(),
            |index| {
                queries[index] += 1;
                Ok(if queries[index] >= ready_after[index] {
                    Readiness::Ready
                } else {
                    Readiness::Pending
                })
            },
            || {
                clock += 1.0;
                clock
            },
        )
        .expect("no probe errors")
    }

    #[test]
    fn every_rank_gets_exactly_one_stamp_in_rank_order() {
        let stamps = simulate(&[3, 1, 2]);
        assert_eq!(stamps.len(), 3);
        // Rank 1 is seen first, then rank 2, then rank 0.
        assert!(stamps[1] < stamps[2]);
        assert!(stamps[2] < stamps[0]);
    }

    #[test]
    fn index_order_does_not_decide_the_stamp_order() {
        // The last rank finishing first gets the earliest stamp, unlike
        // sequential per-stream synchronization.
        let stamps = simulate(&[5, 5, 5, 1]);
        assert!(stamps[3] < stamps[0]);
        assert!(stamps[3] < stamps[1]);
        assert!(stamps[3] < stamps[2]);
    }

    #[test]
    fn zero_ranks_is_an_empty_result() {
        assert!(simulate(&[]).is_empty());
    }

    #[test]
    fn probe_errors_abort_instead_of_spinning() {
        let result = first_ready_times(
            2,
            |index| {
                if index == 1 {
                    Err("stream failed")
                } else {
                    Ok(Readiness::Pending)
                }
            },
            || 0.0,
        );
        assert_eq!(result, Err("stream failed"));
    }
}
