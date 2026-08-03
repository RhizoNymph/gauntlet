//! Phase 1c: sequential disk throughput on the paths training actually uses
//! (dataset reads, checkpoint writes). O_DIRECT where the filesystem allows
//! it, buffered + fsync fallback otherwise; the temp file is always removed,
//! including on error paths. Reports `disk_io.seq_write` / `disk_io.seq_read`
//! in GiB/s under `Scope::Disk { path }`.

use std::fmt;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};

use crate::agent::EventSink;
use crate::proto::{DiskTaskSpec, LogLevel, MetricRecord, Scope, TestId, TestOutcome, Unit};

const GIB: f64 = (1u64 << 30) as f64;
/// O_DIRECT demands the buffer address, the file offset and the transfer
/// length all be block aligned. 4096 covers every logical block size in
/// practice (512e drives included).
const DIRECT_ALIGN: usize = 4096;
/// Transfer size: large enough to keep a NVMe queue busy, small enough that
/// the aligned bounce buffer stays cheap.
const CHUNK_BYTES: usize = 4 << 20;

/// Which IO path a measurement actually took. O_DIRECT is unavailable on
/// tmpfs and on some network filesystems, and the distinction matters when
/// reading the numbers: buffered results include page-cache effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoMode {
    Direct,
    Buffered,
}

impl fmt::Display for IoMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            IoMode::Direct => "o_direct",
            IoMode::Buffered => "buffered",
        })
    }
}

pub struct DiskThroughput {
    pub write_gib_per_sec: f64,
    pub read_gib_per_sec: f64,
    pub write_mode: IoMode,
    pub read_mode: IoMode,
}

/// Measure sequential write-then-read of `file_bytes` in `dir`.
pub fn measure(dir: &str, file_bytes: u64) -> Result<DiskThroughput> {
    if file_bytes == 0 {
        bail!("file_bytes must be positive");
    }
    let scratch = ScratchFile::new(Path::new(dir));
    let mut buffer =
        AlignedBuffer::new(CHUNK_BYTES).context("allocating a block-aligned transfer buffer")?;
    // O_DIRECT can only move whole blocks, so a request that is not a
    // multiple of the alignment goes straight to the buffered path.
    let direct_possible = file_bytes.is_multiple_of(DIRECT_ALIGN as u64);

    let (write_seconds, write_mode) = write_best_mode(
        &scratch.path,
        file_bytes,
        buffer.as_slice(),
        direct_possible,
    )
    .with_context(|| format!("writing {file_bytes} bytes under {dir}"))?;

    let (read_seconds, read_mode) = read_best_mode(
        &scratch.path,
        file_bytes,
        buffer.as_mut_slice(),
        direct_possible,
    )
    .with_context(|| format!("reading {file_bytes} bytes back under {dir}"))?;

    Ok(DiskThroughput {
        write_gib_per_sec: rate(file_bytes, write_seconds)?,
        read_gib_per_sec: rate(file_bytes, read_seconds)?,
        write_mode,
        read_mode,
    })
}

pub fn run(sink: &EventSink, spec: &DiskTaskSpec) -> Result<()> {
    if spec.paths.is_empty() {
        sink.outcome(
            TestId::DiskIo,
            Scope::Node,
            TestOutcome::Skipped {
                reason: "no disk paths configured".to_string(),
            },
        );
        return Ok(());
    }

    for path in &spec.paths {
        let scope = Scope::Disk { path: path.clone() };
        // A bad path is that path's failure, not the agent's: the other
        // paths and the remaining phases still have to run.
        match measure(path, spec.file_bytes) {
            Ok(throughput) => {
                sink.log(
                    LogLevel::Info,
                    format!(
                        "disk_io path={path} bytes={} write_mode={} read_mode={}",
                        spec.file_bytes, throughput.write_mode, throughput.read_mode
                    ),
                );
                sink.metric(MetricRecord {
                    test: TestId::DiskIo,
                    scope: scope.clone(),
                    name: "seq_write".to_string(),
                    value: throughput.write_gib_per_sec,
                    unit: Unit::GibPerSec,
                    repeat: 0,
                });
                sink.metric(MetricRecord {
                    test: TestId::DiskIo,
                    scope: scope.clone(),
                    name: "seq_read".to_string(),
                    value: throughput.read_gib_per_sec,
                    unit: Unit::GibPerSec,
                    repeat: 0,
                });
                sink.outcome(TestId::DiskIo, scope, TestOutcome::Passed);
            }
            Err(error) => sink.outcome(
                TestId::DiskIo,
                scope,
                TestOutcome::Failed {
                    reason: format!("{error:#}"),
                },
            ),
        }
    }
    Ok(())
}

fn rate(bytes: u64, seconds: f64) -> Result<f64> {
    if !seconds.is_finite() || seconds <= 0.0 {
        bail!("timer returned {seconds}s for {bytes} bytes");
    }
    Ok(bytes as f64 / seconds / GIB)
}

/// Try O_DIRECT, fall back to buffered. Filesystems that reject O_DIRECT
/// (tmpfs, some network mounts) fail at `open` or at the first `write`, so
/// the fallback re-runs the whole pass against a freshly truncated file.
fn write_best_mode(
    path: &Path,
    file_bytes: u64,
    chunk: &[u8],
    direct_possible: bool,
) -> Result<(f64, IoMode)> {
    if direct_possible && let Ok(seconds) = write_pass(path, file_bytes, chunk, IoMode::Direct) {
        return Ok((seconds, IoMode::Direct));
    }
    Ok((
        write_pass(path, file_bytes, chunk, IoMode::Buffered)?,
        IoMode::Buffered,
    ))
}

fn read_best_mode(
    path: &Path,
    file_bytes: u64,
    chunk: &mut [u8],
    direct_possible: bool,
) -> Result<(f64, IoMode)> {
    if direct_possible && let Ok(seconds) = read_pass(path, file_bytes, chunk, IoMode::Direct) {
        return Ok((seconds, IoMode::Direct));
    }
    Ok((
        read_pass(path, file_bytes, chunk, IoMode::Buffered)?,
        IoMode::Buffered,
    ))
}

// ---------------------------------------------------------------------------
// IO passes
// ---------------------------------------------------------------------------

/// Sequentially write `file_bytes`, returning the elapsed seconds.
///
/// `sync_all` is inside the timed region on purpose: a checkpoint that only
/// reached the page cache has not been written, and excluding the flush
/// would report DRAM bandwidth on the buffered path.
fn write_pass(path: &Path, file_bytes: u64, chunk: &[u8], mode: IoMode) -> Result<f64> {
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    if mode == IoMode::Direct {
        options.custom_flags(libc::O_DIRECT);
    }

    let start = Instant::now();
    let mut file = options.open(path).context("opening scratch file")?;
    let mut remaining = file_bytes;
    while remaining > 0 {
        let take = chunk.len().min(remaining as usize);
        file.write_all(&chunk[..take]).context("write")?;
        remaining -= take as u64;
    }
    file.sync_all().context("sync_all")?;
    Ok(start.elapsed().as_secs_f64())
}

/// Sequentially read the file back, returning the elapsed seconds. The file
/// is freshly opened either way, and O_DIRECT bypasses the cache the write
/// just populated; the buffered fallback cannot, which is exactly why the
/// mode is reported alongside the number.
fn read_pass(path: &Path, expect_bytes: u64, chunk: &mut [u8], mode: IoMode) -> Result<f64> {
    let mut options = OpenOptions::new();
    options.read(true);
    if mode == IoMode::Direct {
        options.custom_flags(libc::O_DIRECT);
    }

    let start = Instant::now();
    let mut file = options
        .open(path)
        .context("opening scratch file for read")?;
    let mut total: u64 = 0;
    loop {
        let read = file.read(chunk).context("read")?;
        if read == 0 {
            break;
        }
        total += read as u64;
    }
    let seconds = start.elapsed().as_secs_f64();
    if total != expect_bytes {
        bail!("short read: {total} of {expect_bytes} bytes");
    }
    Ok(seconds)
}

// ---------------------------------------------------------------------------
// Aligned buffer + scratch file
// ---------------------------------------------------------------------------

/// A block-aligned byte buffer, carved out of an over-allocated `Vec` so no
/// `unsafe` allocator dance is needed.
struct AlignedBuffer {
    storage: Vec<u8>,
    offset: usize,
    len: usize,
}

impl AlignedBuffer {
    fn new(len: usize) -> Option<Self> {
        let mut storage = vec![0u8; len + DIRECT_ALIGN];
        let offset = storage.as_ptr().align_offset(DIRECT_ALIGN);
        if offset == usize::MAX || offset + len > storage.len() {
            return None;
        }
        // A non-trivial pattern: an all-zero file invites transparent
        // compression or sparse allocation, which would not be a disk test.
        for (i, byte) in storage[offset..offset + len].iter_mut().enumerate() {
            *byte = (i % 251) as u8;
        }
        Some(Self {
            storage,
            offset,
            len,
        })
    }

    fn as_slice(&self) -> &[u8] {
        &self.storage[self.offset..self.offset + self.len]
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.storage[self.offset..self.offset + self.len]
    }
}

/// Owns the temp file's path and deletes it on drop, so an error or a panic
/// anywhere between creation and the last read cannot leave gigabytes behind
/// on a node's dataset volume.
struct ScratchFile {
    path: PathBuf,
}

impl ScratchFile {
    fn new(dir: &Path) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_nanos())
            .unwrap_or(0);
        Self {
            path: dir.join(format!(
                "gauntlet-diskio-{}-{nanos}.tmp",
                std::process::id()
            )),
        }
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::*;

    #[test]
    fn buffer_is_block_aligned_and_patterned() {
        let buffer = AlignedBuffer::new(CHUNK_BYTES).expect("aligned buffer");
        assert_eq!(buffer.as_slice().len(), CHUNK_BYTES);
        assert_eq!(buffer.as_slice().as_ptr() as usize % DIRECT_ALIGN, 0);
        assert!(buffer.as_slice().iter().any(|byte| *byte != 0));
    }

    #[test]
    fn scratch_file_is_removed_on_drop() {
        let dir = std::env::temp_dir();
        let path = {
            let scratch = ScratchFile::new(&dir);
            File::create(&scratch.path).expect("create");
            assert!(scratch.path.exists());
            scratch.path.clone()
        };
        assert!(!path.exists());
    }

    #[test]
    fn zero_sized_request_is_rejected() {
        let dir = std::env::temp_dir();
        assert!(measure(dir.to_str().expect("utf8"), 0).is_err());
    }

    #[test]
    fn unwritable_directory_is_an_error_not_a_panic() {
        assert!(measure("/gauntlet-does-not-exist", 4096).is_err());
    }

    #[test]
    fn rate_rejects_a_stopped_clock() {
        assert!(rate(1024, 0.0).is_err());
        assert!(rate(1024, f64::NAN).is_err());
        assert!((rate(GIB as u64, 1.0).expect("rate") - 1.0).abs() < 1e-9);
    }
}
