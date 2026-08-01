//! Shared helpers for integration tests.

use std::io::Write;
use std::sync::{Arc, Mutex};

use gauntlet::agent::EventSink;
use gauntlet::proto::AgentEvent;

/// A writer that appends into a shared buffer, so tests can hand an
/// `EventSink` to agent code and decode what it emitted.
#[derive(Clone, Default)]
pub struct SharedBuf(pub Arc<Mutex<Vec<u8>>>);

impl Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .expect("buffer poisoned")
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// An EventSink plus a handle to read back everything emitted through it.
pub fn capturing_sink() -> (EventSink, SharedBuf) {
    let buf = SharedBuf::default();
    let sink = EventSink::new(Box::new(buf.clone()));
    (sink, buf)
}

/// Decode every JSON line in the buffer; panics on any malformed line
/// (agents must never interleave non-protocol output on stdout).
pub fn decode_events(buf: &SharedBuf) -> Vec<AgentEvent> {
    let bytes = buf.0.lock().expect("buffer poisoned").clone();
    let text = String::from_utf8(bytes).expect("event stream must be utf-8");
    text.lines()
        .map(|line| {
            gauntlet::proto::decode_event(line)
                .unwrap_or_else(|error| panic!("malformed event line {line:?}: {error}"))
        })
        .collect()
}

/// A unique scratch directory under the system temp dir. Callers clean up.
pub fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock before epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("gauntlet-test-{tag}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("creating scratch dir");
    dir
}
