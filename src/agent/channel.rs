//! The agent's protocol channel: a private duplicate of the original stdout.
//!
//! Every agent mode speaks to the orchestrator over stdout (JSON-lines
//! events, or a single JSON document for probe/peer/barrier). Native
//! libraries loaded into the same process do not know that: NCCL with
//! NCCL_DEBUG set, CUDA, cuBLAS and friends write diagnostics to fd 1. A
//! partial library line flushed just before an event write merges with it
//! into one undecodable line — lose an `NcclId` and the rendezvous stalls,
//! lose a report and its results vanish.
//!
//! `isolate_stdout` therefore runs at agent startup, before any thread
//! exists: it duplicates fd 1 to a private close-on-exec fd (the protocol
//! channel) and then points fd 1 at stderr (`dup2(2, 1)`). From then on
//! anything any library (or a stray `println!`) writes to "stdout" lands in
//! the stderr log the orchestrator already drains, and only
//! `protocol_writer` reaches the orchestrator's decoder.

use std::fs::File;
use std::io::{self, Write};
use std::os::fd::{FromRawFd, RawFd};
use std::sync::OnceLock;

use thiserror::Error;

/// The protocol channel, set once by `isolate_stdout`.
static PROTOCOL: OnceLock<File> = OnceLock::new();

#[derive(Debug, Error)]
pub enum ChannelError {
    #[error("duplicating stdout into the protocol channel: {0}")]
    Duplicate(io::Error),
    #[error("redirecting stdout to stderr: {0}")]
    Redirect(io::Error),
    #[error("stdout is already isolated")]
    AlreadyIsolated,
}

/// Move the protocol off fd 1 (see the module docs). Call once, from
/// `main`, for agent subcommands only, before building the tokio runtime:
/// with no other threads alive, nothing can be mid-write on fd 1 while it
/// is swapped, and nothing has buffered output in `std::io::stdout` yet.
pub fn isolate_stdout() -> Result<(), ChannelError> {
    if PROTOCOL.get().is_some() {
        return Err(ChannelError::AlreadyIsolated);
    }
    // Nothing should be buffered yet; flush anyway so no early bytes
    // strand on the wrong side of the swap.
    io::stdout().flush().map_err(ChannelError::Duplicate)?;

    // SAFETY: plain fd syscalls on fds this process owns. F_DUPFD_CLOEXEC
    // returns a fresh fd (>= 3) so the channel never leaks into children.
    let protocol_fd: RawFd = unsafe { libc::fcntl(libc::STDOUT_FILENO, libc::F_DUPFD_CLOEXEC, 3) };
    if protocol_fd < 0 {
        return Err(ChannelError::Duplicate(io::Error::last_os_error()));
    }
    // SAFETY: `protocol_fd` was just returned by fcntl and is owned by
    // nobody else; the `File` takes sole ownership.
    let protocol = unsafe { File::from_raw_fd(protocol_fd) };

    // SAFETY: dup2 on process-owned fds; atomically replaces fd 1.
    if unsafe { libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) } < 0 {
        return Err(ChannelError::Redirect(io::Error::last_os_error()));
    }
    PROTOCOL
        .set(protocol)
        .map_err(|_| ChannelError::AlreadyIsolated)
}

/// Writer for protocol output: the private channel once `isolate_stdout`
/// ran, else plain stdout (in-process tests, which inject their own
/// `EventSink` writer anyway).
pub fn protocol_writer() -> Box<dyn Write + Send> {
    match PROTOCOL.get() {
        Some(file) => Box::new(file),
        None => Box::new(io::stdout()),
    }
}

/// Write one line to the protocol channel and flush it.
pub fn write_line(line: &str) -> io::Result<()> {
    let mut out = protocol_writer();
    writeln!(out, "{line}")?;
    out.flush()
}

/// Hidden `agent stdout-isolation-check` mode: interleave protocol events
/// with raw writes to fd 1 of the kind NCCL/CUDA produce — a partial line
/// with no newline right before an event, a full line, a `println!` — so
/// an end-to-end test can prove the orchestrator-side stream stays clean.
pub fn isolation_check() -> anyhow::Result<()> {
    use crate::agent::EventSink;
    use crate::proto::{AgentEvent, LogLevel, PROTO_VERSION};

    let raw_stdout = |bytes: &[u8]| {
        // SAFETY: writing an owned, live buffer to fd 1.
        unsafe { libc::write(libc::STDOUT_FILENO, bytes.as_ptr().cast(), bytes.len()) }
    };
    let sink = EventSink::stdout();
    sink.emit(&AgentEvent::Hello {
        proto_version: PROTO_VERSION,
        hostname: "isolation-check".into(),
    });
    raw_stdout(b"node:123:456 [0] NCCL INFO partial line without newline");
    sink.log(LogLevel::Info, "after partial");
    raw_stdout(b"node:123:456 [0] NCCL INFO a whole line\n");
    println!("stray println from library-ish code");
    sink.log(LogLevel::Info, "after full");
    Ok(())
}
