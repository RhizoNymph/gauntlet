//! Bounded-concurrency fan-out shared by bootstrap and deploy.

use std::future::Future;
use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::warn;

/// Run `op` over `items` with at most `max_concurrent` futures in flight,
/// returning results in input order. A slot is `None` only when its task
/// panicked (logged); callers turn that into their own failure value.
pub(crate) async fn fan_out<I, T, F, Fut>(
    items: Vec<I>,
    max_concurrent: usize,
    op: F,
) -> Vec<Option<T>>
where
    I: Send + 'static,
    T: Send + 'static,
    F: Fn(I) -> Fut,
    Fut: Future<Output = T> + Send + 'static,
{
    let permits = Arc::new(Semaphore::new(max_concurrent.max(1)));
    let mut tasks = JoinSet::new();
    let count = items.len();
    for (index, item) in items.into_iter().enumerate() {
        let permits = Arc::clone(&permits);
        let future = op(item);
        tasks.spawn(async move {
            let _permit = permits.acquire_owned().await.ok();
            (index, future.await)
        });
    }
    let mut slots: Vec<Option<T>> = (0..count).map(|_| None).collect();
    while let Some(joined) = tasks.join_next().await {
        match joined {
            Ok((index, value)) => slots[index] = Some(value),
            Err(error) => warn!(%error, "fan-out task did not complete"),
        }
    }
    slots
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn results_keep_input_order_and_concurrency_stays_bounded() {
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let results = fan_out((0..20u64).collect(), 3, |item| {
            let live = Arc::clone(&live);
            let peak = Arc::clone(&peak);
            async move {
                let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                // Later items finish first: order must still be input order.
                tokio::time::sleep(Duration::from_millis(20 - item)).await;
                live.fetch_sub(1, Ordering::SeqCst);
                item * 10
            }
        })
        .await;
        let values: Vec<u64> = results
            .into_iter()
            .map(|slot| slot.expect("done"))
            .collect();
        assert_eq!(values, (0..20u64).map(|i| i * 10).collect::<Vec<_>>());
        assert!(peak.load(Ordering::SeqCst) <= 3);
    }

    #[tokio::test]
    async fn a_panicking_task_leaves_an_empty_slot() {
        let results = fan_out(vec![1, 2, 3], 2, |item| async move {
            assert_ne!(item, 2, "boom");
            item
        })
        .await;
        assert_eq!(results, vec![Some(1), None, Some(3)]);
    }
}
