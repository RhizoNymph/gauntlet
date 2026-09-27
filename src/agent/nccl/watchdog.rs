//! Hard-deadline watchdog and cascade exits for the fleet NCCL agent.
//!
//! When a rank dies mid-collective, every other rank blocks inside the
//! collective forever: no error surfaces, the driver thread never returns,
//! and the orchestrator's phase timeout only drops its ssh future. The
//! agent must therefore end itself.
//!
//! Why process termination, not `ncclCommAbort`: cudarc 0.19 does expose
//! the raw `ncclCommAbort` (`nccl::result::comm_abort`), but the safe
//! `Comm` keeps its `ncclComm_t` private (its only abort is in `Drop`, on
//! the thread that owns it — the thread that is blocked). Reaching the
//! handle would mean replacing every cudarc collective with raw FFI.
//! Terminating the process is the stronger guarantee anyway: process exit
//! destroys its CUDA contexts, which kills in-flight collective and GEMM
//! kernels alike, and frees the GPUs.
//!
//! Both exits here use [`AGENT_EXIT_CASCADE`] and a `Log` event (never
//! `Fatal`, which the collector would record as a host failure): the host
//! stopped because the fleet stopped, and the orchestrator attributes it to
//! whichever host failed first.

use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use thiserror::Error;

use crate::agent::EventSink;
use crate::proto::{AGENT_EXIT_CASCADE, AgentEvent, LogLevel};

/// A rank gave up because the rest of the fleet stopped (the lead's close
/// signal never came). Distinct from a local fault: `run_block` turns it
/// into a cascade exit instead of an ordinary error.
#[derive(Debug, Error)]
#[error("{0}")]
pub(super) struct CascadeAbort(pub String);

/// Run `body` under a watchdog that terminates the process with a cascade
/// exit if `body` has not returned within `budget`. The watchdog is
/// disarmed (and joined) as soon as `body` returns or unwinds.
pub(super) fn guarded<T>(
    sink: &EventSink,
    what: &str,
    budget: Duration,
    body: impl FnOnce() -> T,
) -> T {
    let (disarm, armed) = mpsc::channel::<()>();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            // Disconnect (the sender dropped: `body` finished or unwound)
            // disarms; only a genuine timeout fires.
            if let Err(RecvTimeoutError::Timeout) = armed.recv_timeout(budget) {
                exit_cascade(
                    sink,
                    &format!(
                        "{what}: hard deadline of {}s exceeded (a peer rank likely died \
                         mid-collective); terminating so no compute load or blocked \
                         collective outlives the step",
                        budget.as_secs()
                    ),
                );
            }
        });
        let out = body();
        drop(disarm);
        out
    })
}

/// Report why, then terminate immediately with [`AGENT_EXIT_CASCADE`].
pub(super) fn exit_cascade(sink: &EventSink, message: &str) -> ! {
    sink.emit(&AgentEvent::Log {
        level: LogLevel::Warn,
        message: message.to_string(),
    });
    // SAFETY: `_exit` only ends the process. It deliberately skips atexit
    // handlers and destructors, which could block on the wedged CUDA/NCCL
    // state (a stream that never completes); the kernel reclaims the
    // process's GPU contexts. The event above was flushed by `emit`.
    unsafe { libc::_exit(AGENT_EXIT_CASCADE) }
}
