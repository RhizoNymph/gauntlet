//! Phase 1c: sequential disk throughput on the paths training actually uses
//! (dataset reads, checkpoint writes). O_DIRECT where the filesystem allows
//! it, buffered + fsync fallback otherwise; the temp file is always removed,
//! including on error paths. Reports `disk_io.seq_write` / `disk_io.seq_read`
//! in GiB/s under `Scope::Disk { path }`.

use anyhow::Result;

use crate::agent::EventSink;
use crate::proto::DiskTaskSpec;

pub struct DiskThroughput {
    pub write_gib_per_sec: f64,
    pub read_gib_per_sec: f64,
}

/// Measure sequential write-then-read of `file_bytes` in `dir`.
pub fn measure(dir: &str, file_bytes: u64) -> Result<DiskThroughput> {
    let _ = (dir, file_bytes);
    todo!("agent B: implement")
}

pub fn run(sink: &EventSink, spec: &DiskTaskSpec) -> Result<()> {
    let _ = (sink, spec);
    todo!("agent B: implement")
}
