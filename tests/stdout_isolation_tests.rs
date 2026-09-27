//! The agent's protocol channel survives library writes to raw fd 1.
//!
//! Runs the real binary's hidden `agent stdout-isolation-check` mode, which
//! interleaves protocol events with NCCL-style raw writes to fd 1 (a
//! partial line right before an event, a full line, a `println!`), and
//! decodes stdout exactly as the orchestrator does.

use std::process::{Command, Stdio};

use gauntlet::proto::{AgentEvent, LogLevel, decode_event, expect_hello};

#[test]
fn library_stdout_never_reaches_the_event_stream() {
    let output = Command::new(env!("CARGO_BIN_EXE_gauntlet"))
        .args(["agent", "stdout-isolation-check"])
        .stdin(Stdio::null())
        .output()
        .expect("run the agent binary");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "agent failed: {stderr}");

    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let events: Vec<AgentEvent> = stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            decode_event(line).unwrap_or_else(|error| panic!("undecodable line {line:?}: {error}"))
        })
        .collect();

    assert_eq!(events.len(), 3, "exactly the emitted events: {events:?}");
    expect_hello(&events[0]).expect("hello first");
    let logs: Vec<&str> = events[1..]
        .iter()
        .map(|event| match event {
            AgentEvent::Log {
                level: LogLevel::Info,
                message,
            } => message.as_str(),
            other => panic!("unexpected event {other:?}"),
        })
        .collect();
    assert_eq!(logs, ["after partial", "after full"]);

    // Nothing was dropped: the library noise went to stderr instead.
    for noise in [
        "NCCL INFO partial line without newline",
        "NCCL INFO a whole line",
        "stray println from library-ish code",
    ] {
        assert!(
            stderr.contains(noise),
            "{noise:?} missing from stderr: {stderr}"
        );
        assert!(!stdout.contains(noise), "{noise:?} leaked into stdout");
    }
}
