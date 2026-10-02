//! Agent side of transport capture: find this process's NCCL INFO log from
//! the debug variables actually in its environment, parse it
//! (`nccl_transport::parse`, pure), and report a typed
//! `NcclTransportReport`.
//!
//! The environment is only *read* here (never set: the agent runs on a
//! multi-threaded runtime). The orchestrator put NCCL_DEBUG,
//! NCCL_DEBUG_SUBSYS and NCCL_DEBUG_FILE on the spawn command line per
//! `nccl_transport::debug::capture_additions`; reading them back rather
//! than trusting a wire copy means the agent parses exactly the file NCCL
//! was told to write.
//!
//! Call `CaptureWindow::open` before the first NCCL call and
//! `CaptureWindow::emit` once the communicator has carried traffic: NCCL
//! connects peers lazily (at the first collective, since 2.22), so the
//! channel lines exist only after the workload ran. NCCL opens its debug
//! file unbuffered, so every line is on disk by then.

use std::path::Path;
use std::time::{Duration, SystemTime};

use crate::agent::EventSink;
use crate::nccl_level::NcclLevel;
use crate::nccl_transport::debug::{
    DebugSettings, NCCL_DEBUG, NCCL_DEBUG_FILE, NCCL_DEBUG_SUBSYS, log_source,
};
use crate::nccl_transport::parse::parse_log;
use crate::nccl_transport::{CommSpan, NcclTransportReport, TransportCapture, UnknownTransport};
use crate::proto::AgentEvent;

/// Filesystem timestamp slack: a log modified this long before the window
/// opened still counts as this process's (coarse mtime granularity on some
/// filesystems).
const MTIME_SLACK: Duration = Duration::from_secs(2);

/// The time before this process's NCCL init: a debug file not modified
/// since is a previous run's leftover.
#[derive(Debug, Clone, Copy)]
pub struct CaptureWindow {
    opened: SystemTime,
}

impl CaptureWindow {
    pub fn open() -> Self {
        Self {
            opened: SystemTime::now(),
        }
    }

    /// Capture and emit the transport report for `level`.
    pub fn emit(self, sink: &EventSink, level: NcclLevel, span: CommSpan) {
        let report = NcclTransportReport {
            level,
            span,
            capture: self.capture(),
        };
        sink.emit(&AgentEvent::NcclTransport {
            report: Box::new(report),
        });
    }

    fn capture(self) -> TransportCapture {
        let level = std::env::var(NCCL_DEBUG).ok();
        let subsys = std::env::var(NCCL_DEBUG_SUBSYS).ok();
        let file = std::env::var(NCCL_DEBUG_FILE).ok();
        let settings = DebugSettings {
            level: level.as_deref(),
            subsys: subsys.as_deref(),
            file: file.as_deref(),
        };
        let hostname = crate::agent::hostname().unwrap_or_default();
        match log_source(settings, &hostname, std::process::id()) {
            Ok(path) => capture_from(&path, self.opened),
            Err(reason) => TransportCapture::Unknown { reason },
        }
    }
}

/// Read and parse the log at `path`, written no earlier than `opened`.
fn capture_from(path: &Path, opened: SystemTime) -> TransportCapture {
    let shown = path.display().to_string();
    let unreadable = |error: std::io::Error| TransportCapture::Unknown {
        reason: UnknownTransport::Unreadable {
            path: shown.clone(),
            error: error.to_string(),
        },
    };
    let modified = match std::fs::metadata(path).and_then(|meta| meta.modified()) {
        Ok(modified) => modified,
        Err(error) => return unreadable(error),
    };
    if modified + MTIME_SLACK < opened {
        return TransportCapture::Unknown {
            reason: UnknownTransport::Stale { path: shown },
        };
    }
    let text = match std::fs::read(path) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(error) => return unreadable(error),
    };
    match parse_log(&text) {
        Some(info) => TransportCapture::Captured { info },
        None => TransportCapture::Unknown {
            reason: UnknownTransport::NoTransportLines { path: shown },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nccl_transport::NetTransport;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("gauntlet-transport-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir.join("nccl.log")
    }

    #[test]
    fn a_fresh_log_is_parsed() {
        let path = scratch("fresh");
        std::fs::write(
            &path,
            include_str!("../nccl_transport/fixtures/socket_fallback_2_23.log"),
        )
        .expect("write log");
        let TransportCapture::Captured { info } = capture_from(&path, SystemTime::now()) else {
            panic!("expected a capture");
        };
        assert!(info.net.as_ref().is_some_and(NetTransport::is_socket));
    }

    #[test]
    fn a_log_from_before_the_window_is_stale() {
        let path = scratch("stale");
        std::fs::write(
            &path,
            include_str!("../nccl_transport/fixtures/ib_2_18.log"),
        )
        .expect("write log");
        let later = SystemTime::now() + Duration::from_secs(60);
        assert!(matches!(
            capture_from(&path, later),
            TransportCapture::Unknown {
                reason: UnknownTransport::Stale { .. }
            }
        ));
    }

    #[test]
    fn missing_and_empty_logs_say_why() {
        let path = scratch("missing").with_file_name("does-not-exist.log");
        assert!(matches!(
            capture_from(&path, SystemTime::now()),
            TransportCapture::Unknown {
                reason: UnknownTransport::Unreadable { .. }
            }
        ));
        let path = scratch("empty");
        std::fs::write(&path, "nothing here\n").expect("write log");
        assert!(matches!(
            capture_from(&path, SystemTime::now()),
            TransportCapture::Unknown {
                reason: UnknownTransport::NoTransportLines { .. }
            }
        ));
    }
}
