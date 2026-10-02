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
//! Usage: `CaptureWindow::open()` before the first NCCL call, then
//! `window.run(sink, level, span, || { ...every NCCL call... })`. `run`
//! emits exactly one report whatever the body returns — success, an init
//! failure (`ncclCommInitRank`, `ncclGetUniqueId`), a collective error —
//! and then hands the body's result back. A failed init is exactly when
//! the diagnosis matters (wrong NCCL_IB_HCA, GID index), and NCCL has
//! written its `NET/IB` and `Using network` lines by then.
//!
//! Freshness: `open` deletes whatever exists at the resolved log path
//! before NCCL starts. Whatever is there afterwards was written by this
//! process's NCCL, by construction — no timestamp comparison, so an NFS
//! server's clock (which sets mtime there) cannot matter. NCCL truncates
//! the file at open anyway, so deleting it loses nothing. NCCL opens it
//! unbuffered, so every line is on disk when `run` reads it.

use std::path::{Path, PathBuf};

use crate::agent::EventSink;
use crate::nccl_level::NcclLevel;
use crate::nccl_transport::debug::{
    DebugSettings, NCCL_DEBUG, NCCL_DEBUG_FILE, NCCL_DEBUG_SUBSYS, log_source,
};
use crate::nccl_transport::parse::parse_log;
use crate::nccl_transport::{CommSpan, NcclTransportReport, TransportCapture, UnknownTransport};
use crate::proto::AgentEvent;

/// Where this process's NCCL log will be (already cleared of any previous
/// file), or why there will be none.
#[derive(Debug)]
pub struct CaptureWindow {
    source: Result<PathBuf, UnknownTransport>,
}

impl CaptureWindow {
    /// Resolve the log path from the process env and clear it. Call before
    /// the first NCCL call of the process.
    pub fn open() -> Self {
        let level = std::env::var(NCCL_DEBUG).ok();
        let subsys = std::env::var(NCCL_DEBUG_SUBSYS).ok();
        let file = std::env::var(NCCL_DEBUG_FILE).ok();
        let settings = DebugSettings {
            level: level.as_deref(),
            subsys: subsys.as_deref(),
            file: file.as_deref(),
        };
        let hostname = crate::agent::hostname().unwrap_or_default();
        Self::open_with(settings, &hostname, std::process::id())
    }

    /// `open` with explicit inputs (the seam the tests use).
    pub fn open_with(settings: DebugSettings<'_>, hostname: &str, pid: u32) -> Self {
        Self {
            source: log_source(settings, hostname, pid).and_then(clear),
        }
    }

    /// Run `body` (every NCCL call of the communicator), then emit this
    /// window's report for `level` — on every outcome — and return the
    /// body's result unchanged.
    pub fn run<T, E>(
        self,
        sink: &EventSink,
        level: NcclLevel,
        span: CommSpan,
        body: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        let outcome = body();
        let report = NcclTransportReport {
            level,
            span,
            capture: self.capture(),
        };
        sink.emit(&AgentEvent::NcclTransport {
            report: Box::new(report),
        });
        outcome
    }

    fn capture(&self) -> TransportCapture {
        match &self.source {
            Ok(path) => capture_from(path),
            Err(reason) => TransportCapture::Unknown {
                reason: reason.clone(),
            },
        }
    }
}

/// Remove a previous log at `path` so what exists after NCCL ran is fresh.
fn clear(path: PathBuf) -> Result<PathBuf, UnknownTransport> {
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(path),
        Err(error) => Err(UnknownTransport::Uncleared {
            path: path.display().to_string(),
            error: error.to_string(),
        }),
    }
}

/// Read and parse the log at `path`.
fn capture_from(path: &Path) -> TransportCapture {
    let shown = path.display().to_string();
    let text = match std::fs::read(path) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(error) => {
            return TransportCapture::Unknown {
                reason: UnknownTransport::Unreadable {
                    path: shown,
                    error: error.to_string(),
                },
            };
        }
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
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::nccl_transport::NetTransport;
    use crate::proto::decode_event;

    const SOCKET_LOG: &str = include_str!("../nccl_transport/fixtures/socket_fallback_2_23.log");
    const IB_LOG: &str = include_str!("../nccl_transport/fixtures/ib_2_18.log");

    /// A writer the test can read back.
    #[derive(Clone, Default)]
    struct Shared(Arc<Mutex<Vec<u8>>>);

    impl Write for Shared {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("buffer").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn sink() -> (EventSink, Shared) {
        let shared = Shared::default();
        (EventSink::new(Box::new(shared.clone())), shared)
    }

    fn reports(shared: &Shared) -> Vec<NcclTransportReport> {
        let bytes = shared.0.lock().expect("buffer").clone();
        String::from_utf8(bytes)
            .expect("utf8")
            .lines()
            .map(|line| match decode_event(line).expect("event") {
                AgentEvent::NcclTransport { report } => *report,
                other => panic!("unexpected event {other:?}"),
            })
            .collect()
    }

    /// A per-test directory and the NCCL_DEBUG_FILE pattern inside it.
    fn scratch(name: &str) -> (PathBuf, String) {
        let dir =
            std::env::temp_dir().join(format!("gauntlet-transport-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let pattern = format!("{}/nccl.%h.log", dir.display());
        (dir.join("nccl.node0.log"), pattern)
    }

    fn window(pattern: &str) -> CaptureWindow {
        CaptureWindow::open_with(
            DebugSettings {
                level: Some("INFO"),
                subsys: None,
                file: Some(pattern),
            },
            "node0.cluster",
            1,
        )
    }

    #[test]
    fn a_failed_body_still_emits_once_and_keeps_its_error() {
        let (path, pattern) = scratch("init-failure");
        let (sink, shared) = sink();
        let result: Result<(), String> =
            window(&pattern).run(&sink, NcclLevel::Fleet, CommSpan::MultiHost, || {
                // NCCL wrote its NET lines, then ncclCommInitRank failed.
                std::fs::write(&path, SOCKET_LOG).expect("write log");
                Err("ncclCommInitRank: ncclSystemError".to_string())
            });
        assert_eq!(result, Err("ncclCommInitRank: ncclSystemError".to_string()));
        let reports = reports(&shared);
        assert_eq!(reports.len(), 1, "exactly one report");
        assert_eq!(reports[0].level, NcclLevel::Fleet);
        let TransportCapture::Captured { info } = &reports[0].capture else {
            panic!("expected a capture: {:?}", reports[0].capture);
        };
        assert!(info.net.as_ref().is_some_and(NetTransport::is_socket));
    }

    #[test]
    fn a_successful_body_emits_once_and_returns_its_value() {
        let (path, pattern) = scratch("success");
        let (sink, shared) = sink();
        let value = window(&pattern).run(&sink, NcclLevel::Intranode, CommSpan::SingleHost, || {
            std::fs::write(&path, IB_LOG).expect("write log");
            Ok::<_, String>(42)
        });
        assert_eq!(value, Ok(42));
        assert_eq!(reports(&shared).len(), 1);
    }

    #[test]
    fn a_previous_runs_log_is_cleared_before_nccl_starts() {
        let (path, pattern) = scratch("stale");
        std::fs::write(&path, IB_LOG).expect("leftover log");
        let window = window(&pattern);
        assert!(!path.exists(), "open must clear the leftover");
        // NCCL never wrote (it could not open the file): no stale parse.
        let (sink, shared) = sink();
        let _ = window.run(&sink, NcclLevel::Fleet, CommSpan::MultiHost, || {
            Ok::<_, String>(())
        });
        assert!(matches!(
            reports(&shared)[0].capture,
            TransportCapture::Unknown {
                reason: UnknownTransport::Unreadable { .. }
            }
        ));
    }

    #[test]
    fn an_uncleared_path_is_unknown_not_stale_data() {
        // A directory at the log path cannot be removed as a file.
        let (path, pattern) = scratch("uncleared");
        std::fs::create_dir_all(&path).expect("dir in the way");
        let (sink, shared) = sink();
        let _ = window(&pattern).run(&sink, NcclLevel::Barrier, CommSpan::MultiHost, || {
            Ok::<_, String>(())
        });
        assert!(matches!(
            reports(&shared)[0].capture,
            TransportCapture::Unknown {
                reason: UnknownTransport::Uncleared { .. }
            }
        ));
    }

    #[test]
    fn a_log_without_transport_lines_says_so() {
        let (path, pattern) = scratch("empty");
        let (sink, shared) = sink();
        let _ = window(&pattern).run(&sink, NcclLevel::Fleet, CommSpan::MultiHost, || {
            std::fs::write(&path, "nothing here\n").expect("write log");
            Ok::<_, String>(())
        });
        assert!(matches!(
            reports(&shared)[0].capture,
            TransportCapture::Unknown {
                reason: UnknownTransport::NoTransportLines { .. }
            }
        ));
    }

    #[test]
    fn no_debug_settings_report_why() {
        let (sink, shared) = sink();
        let window = CaptureWindow::open_with(DebugSettings::default(), "n", 1);
        let _ = window.run(&sink, NcclLevel::Fleet, CommSpan::MultiHost, || {
            Ok::<_, String>(())
        });
        assert_eq!(
            reports(&shared)[0].capture,
            TransportCapture::Unknown {
                reason: UnknownTransport::DebugLevel { level: None }
            }
        );
    }
}
