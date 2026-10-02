//! Phase 0: hardware/software inventory and health counters.
//!
//! Sources: /proc (cpuinfo, meminfo, sys/kernel), /sys (numa nodes, NIC mtu
//! and speed), `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`
//! (absent nvidia-smi means "no GPUs", not an error), `ibstat`-equivalent
//! sysfs under /sys/class/infiniband, `chronyc tracking` (best effort),
//! `dmesg`-sourced Xid scan via /var/log/kern.log or journalctl (best
//! effort; missing permissions degrade to empty, never to failure).

use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::agent::EventSink;
use crate::agent::gpu_occupancy::{self, AgentIdentity, PciBusId};
use crate::agent::sysfs::read_trimmed;
use crate::agent::{ib, pci};
use crate::proto::{
    AgentEvent, GpuInventory, GpuOccupancy, InventorySnapshot, NicInventory, PciLocation, Scope,
    TestId, TestOutcome, gpu_visibility_mismatch,
};

/// Upper bound on any single external probe. `collect()` runs at most a
/// handful of these, keeping the documented < 5s budget with room to spare.
pub(crate) const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const QUICK_TIMEOUT: Duration = Duration::from_secs(1);
/// Kernel logs can be huge; a bounded read keeps the Xid scan cheap.
const MAX_CAPTURE_BYTES: u64 = 8 << 20;

/// Fields queried from `nvidia-smi` in one shot. `driver_version` is folded
/// into the same query rather than costing a second process spawn.
/// New fields go at the end so existing positions stay put.
const GPU_QUERY: &str = concat!(
    "--query-gpu=index,name,uuid,vbios_version,memory.total,",
    "ecc.errors.uncorrected.volatile.total,remapped_rows.pending,",
    "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,",
    "pcie.link.width.max,persistence_mode,driver_version,",
    // Occupancy (proto v10): the bus id joins the compute-apps query onto
    // a GPU index; memory.used feeds the gpu_idle threshold.
    "pci.bus_id,memory.used"
);

/// Collect the snapshot and emit it as an event.
pub fn run(sink: &EventSink) -> Result<()> {
    let snapshot = Box::new(collect()?);
    // A GPU-visibility mismatch is a per-host finding: this host will run
    // fewer fleet NCCL ranks than it has GPUs.
    let mismatch = gpu_visibility_mismatch(&snapshot);
    sink.emit(&AgentEvent::Inventory { snapshot });
    if let Some(reason) = mismatch {
        sink.outcome(
            TestId::Inventory,
            Scope::Node,
            TestOutcome::Failed { reason },
        );
    }
    Ok(())
}

/// Collect an inventory snapshot of this node. Individual probes are best
/// effort: a missing tool or permission yields `None`/empty fields, never an
/// error. Only a totally unreadable /proc fails.
pub fn collect() -> Result<InventorySnapshot> {
    let hostname = crate::agent::hostname()?;
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").context("reading /proc/cpuinfo")?;
    let meminfo = std::fs::read_to_string("/proc/meminfo").context("reading /proc/meminfo")?;

    let (gpus, nvidia_driver) = probe_gpus();
    let cuda_version = if gpus.is_empty() {
        None
    } else {
        probe_cuda_version()
    };

    let ib = ib::probe(Path::new(ib::INFINIBAND_ROOT));

    Ok(InventorySnapshot {
        hostname,
        kernel: kernel_release(),
        cpu_model: cpu_model(&cpuinfo),
        logical_cores: logical_cores(&cpuinfo),
        numa_nodes: numa_node_count(),
        mem_total_bytes: mem_total_bytes(&meminfo),
        cpu_governor: read_trimmed("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        clock_offset_ms: probe_clock_offset_ms(),
        nvidia_driver,
        cuda_version,
        gpus,
        nics: probe_nics(),
        ib_ports: ib.ports,
        ib_devices: ib.devices,
        xid_errors: probe_xid_errors(),
        gpu_libs: probe_gpu_libs(),
        cuda_visible_gpus: probe_cuda_visible_gpus(),
    })
}

/// How many devices the CUDA driver can open — the count the fleet NCCL
/// world sizes rank blocks from. Guarded: a missing libcuda panics inside
/// cudarc, which must degrade to "unknown", never abort the probe.
#[cfg(feature = "gpu")]
fn probe_cuda_visible_gpus() -> Option<u32> {
    crate::agent::gpu::guard("cuda device count", || {
        Ok(cudarc::driver::CudaContext::device_count()?)
    })
    .ok()
    .and_then(|count| u32::try_from(count).ok())
}

#[cfg(not(feature = "gpu"))]
fn probe_cuda_visible_gpus() -> Option<u32> {
    None
}

// ---------------------------------------------------------------------------
// /proc + /sys probes
// ---------------------------------------------------------------------------

/// `6.14.0-28-generic` out of `Linux version 6.14.0-28-generic (…) …`,
/// falling back to `uname -r` and finally to a placeholder so the field is
/// never empty (fleet consistency analysis keys on it).
fn kernel_release() -> String {
    if let Some(version) = read_trimmed("/proc/version")
        && let Some(release) = version.split_whitespace().nth(2)
    {
        return release.to_string();
    }
    if let Some(release) = run_capture("uname", &["-r"], QUICK_TIMEOUT) {
        let release = release.trim();
        if !release.is_empty() {
            return release.to_string();
        }
    }
    "unknown".to_string()
}

/// x86 exposes `model name`; other architectures use different keys, so try
/// a few before falling back to the machine type.
fn cpu_model(cpuinfo: &str) -> String {
    const KEYS: [&str; 4] = ["model name", "Model Name", "cpu model", "Hardware"];
    for key in KEYS {
        if let Some(value) = cpuinfo_field(cpuinfo, key) {
            return value;
        }
    }
    if let Some(machine) = run_capture("uname", &["-m"], QUICK_TIMEOUT) {
        let machine = machine.trim();
        if !machine.is_empty() {
            return machine.to_string();
        }
    }
    "unknown".to_string()
}

fn cpuinfo_field(cpuinfo: &str, key: &str) -> Option<String> {
    cpuinfo.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name.trim() != key {
            return None;
        }
        let value = value.trim();
        (!value.is_empty()).then(|| value.to_string())
    })
}

fn logical_cores(cpuinfo: &str) -> u32 {
    let counted = cpuinfo
        .lines()
        .filter(|line| {
            line.split(':')
                .next()
                .is_some_and(|k| k.trim() == "processor")
        })
        .count();
    if counted > 0 {
        return counted as u32;
    }
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

fn numa_node_count() -> u32 {
    let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") else {
        return 1;
    };
    let count = entries
        .flatten()
        .filter(|entry| numa_node_id(&entry.file_name().to_string_lossy()).is_some())
        .count();
    count.max(1) as u32
}

/// `node7` -> 7; anything else -> None.
fn numa_node_id(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("node")?;
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

fn mem_total_bytes(meminfo: &str) -> u64 {
    meminfo
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            if key.trim() != "MemTotal" {
                return None;
            }
            let kib: u64 = value.split_whitespace().next()?.parse().ok()?;
            Some(kib * 1024)
        })
        .unwrap_or(0)
}

fn probe_nics() -> Vec<NicInventory> {
    let Ok(entries) = std::fs::read_dir("/sys/class/net") else {
        return Vec::new();
    };
    let mut nics: Vec<NicInventory> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            // Loopback tells us nothing about the fabric.
            if name == "lo" {
                return None;
            }
            let dir = entry.path();
            let mtu: u32 = read_trimmed(dir.join("mtu"))?.parse().ok()?;
            // `speed` is EINVAL on wireless/virtual links and -1 when the
            // carrier is down; both mean "unknown", not zero.
            let speed_mbps = read_trimmed(dir.join("speed"))
                .and_then(|raw| raw.parse::<i64>().ok())
                .filter(|mbps| *mbps > 0)
                .map(|mbps| mbps as u64);
            Some(NicInventory {
                name,
                mtu,
                speed_mbps,
            })
        })
        .collect();
    nics.sort_by(|a, b| a.name.cmp(&b.name));
    nics
}

// ---------------------------------------------------------------------------
// nvidia-smi
// ---------------------------------------------------------------------------

/// Returns the per-GPU inventory (occupancy included) plus the driver
/// version reported alongside it. An absent or failing `nvidia-smi` means
/// "no GPUs", not an error; a failing compute-apps query leaves every GPU's
/// process list unknown.
fn probe_gpus() -> (Vec<GpuInventory>, Option<String>) {
    let Some(gpu_query) = run_capture(
        "nvidia-smi",
        &[GPU_QUERY, "--format=csv,noheader,nounits"],
        PROBE_TIMEOUT,
    ) else {
        return (Vec::new(), None);
    };
    let compute_apps = run_capture(
        "nvidia-smi",
        &[
            gpu_occupancy::COMPUTE_APPS_QUERY,
            "--format=csv,noheader,nounits",
        ],
        PROBE_TIMEOUT,
    );
    let (mut gpus, driver) = gpus_from_nvidia_smi(
        &gpu_query,
        compute_apps.as_deref(),
        &AgentIdentity::current(),
        gpu_occupancy::exe_basename,
    );
    let pci_root = Path::new(pci::PCI_DEVICES_ROOT);
    for location in gpus.iter_mut().filter_map(|gpu| gpu.pci.as_mut()) {
        pci::resolve(pci_root, location);
    }
    (gpus, driver)
}

/// Pure core of `probe_gpus`: the `GPU_QUERY` CSV joined with the
/// compute-apps CSV (`None` = that query failed) by PCI bus id.
pub(crate) fn gpus_from_nvidia_smi(
    gpu_query: &str,
    compute_apps: Option<&str>,
    identity: &AgentIdentity,
    exe_of: impl Fn(u32) -> Option<String>,
) -> (Vec<GpuInventory>, Option<String>) {
    let (rows, driver) = parse_gpu_query(gpu_query);
    let by_bus = gpu_occupancy::processes_by_bus(
        compute_apps.map(gpu_occupancy::parse_compute_apps),
        identity,
        exe_of,
    );
    let gpus = rows
        .into_iter()
        .map(|(mut gpu, bus_id)| {
            gpu.occupancy.compute_processes = gpu_occupancy::processes_for(by_bus.as_ref(), bus_id);
            gpu
        })
        .collect();
    (gpus, driver)
}

/// Parse the `GPU_QUERY` CSV into per-GPU inventories (occupancy memory
/// filled, process list still unknown) paired with each GPU's bus id, plus
/// the driver version from the first row. Rows are sorted by index.
fn parse_gpu_query(output: &str) -> (Vec<(GpuInventory, Option<PciBusId>)>, Option<String>) {
    let mut gpus = Vec::new();
    let mut driver = None;
    for line in output.lines() {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 12 {
            continue;
        }
        let Some(index) = csv_field(fields[0]).and_then(|raw| raw.parse::<u32>().ok()) else {
            continue;
        };
        if driver.is_none() {
            driver = fields
                .get(12)
                .copied()
                .and_then(csv_field)
                .map(str::to_string);
        }
        let field_u64 = |position: usize| {
            fields
                .get(position)
                .copied()
                .and_then(csv_field)
                .and_then(|raw| raw.parse::<u64>().ok())
        };
        // `nounits` renders memory.total / memory.used in MiB.
        let memory_total_mib = field_u64(4);
        let bus_id = fields
            .get(13)
            .copied()
            .and_then(csv_field)
            .and_then(|raw| raw.parse::<PciBusId>().ok());
        let gpu = GpuInventory {
            index,
            name: csv_field(fields[1]).unwrap_or("unknown").to_string(),
            uuid: csv_field(fields[2]).unwrap_or_default().to_string(),
            vbios: csv_field(fields[3]).unwrap_or_default().to_string(),
            mem_total_bytes: memory_total_mib.map(|mib| mib * 1024 * 1024).unwrap_or(0),
            ecc_volatile_errors: csv_field(fields[5]).and_then(|raw| raw.parse::<u64>().ok()),
            remapped_rows_pending: csv_field(fields[6]).and_then(parse_yes_no),
            pcie_gen_current: csv_field(fields[7]).and_then(|raw| raw.parse::<u32>().ok()),
            pcie_gen_max: csv_field(fields[8]).and_then(|raw| raw.parse::<u32>().ok()),
            pcie_width_current: csv_field(fields[9]).and_then(|raw| raw.parse::<u32>().ok()),
            pcie_width_max: csv_field(fields[10]).and_then(|raw| raw.parse::<u32>().ok()),
            // NVLink topology needs `nvidia-smi nvlink -s`, a separate and
            // much slower query; phase 2 measures the links directly.
            nvlinks_active: None,
            persistence_mode: csv_field(fields[11]).and_then(parse_yes_no),
            occupancy: GpuOccupancy {
                memory_used_mib: field_u64(14),
                memory_total_mib,
                compute_processes: None,
            },
            // The address only; `probe_gpus` resolves NUMA node and
            // upstream bridges against sysfs.
            pci: bus_id.map(|bus| PciLocation::address_only(bus.to_string())),
        };
        gpus.push((gpu, bus_id));
    }
    gpus.sort_by_key(|(gpu, _)| gpu.index);
    (gpus, driver)
}

/// `nvidia-smi` prints "N/A", "[N/A]" or "[Not Supported]" for fields the
/// device or driver does not expose.
pub(crate) fn csv_field(raw: &str) -> Option<&str> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let normalized = trimmed.trim_start_matches('[').trim_end_matches(']');
    if normalized.eq_ignore_ascii_case("N/A")
        || normalized.eq_ignore_ascii_case("Not Supported")
        || normalized.eq_ignore_ascii_case("Unknown Error")
    {
        return None;
    }
    Some(trimmed)
}

fn parse_yes_no(raw: &str) -> Option<bool> {
    match raw.to_ascii_lowercase().as_str() {
        "yes" | "enabled" | "active" | "1" => Some(true),
        "no" | "disabled" | "0" => Some(false),
        _ => None,
    }
}

/// Prefer the toolkit's own manifest; fall back to the CUDA version the
/// driver advertises in the `nvidia-smi` banner.
fn probe_cuda_version() -> Option<String> {
    if let Ok(text) = std::fs::read_to_string("/usr/local/cuda/version.json")
        && let Ok(doc) = serde_json::from_str::<serde_json::Value>(&text)
        && let Some(version) = doc.get("cuda").and_then(|cuda| cuda.get("version"))
        && let Some(version) = version.as_str()
    {
        return Some(version.to_string());
    }
    let banner = run_capture("nvidia-smi", &[], PROBE_TIMEOUT)?;
    banner.lines().find_map(|line| {
        let (_, rest) = line.split_once("CUDA Version:")?;
        let version = rest.split_whitespace().next()?;
        (!version.is_empty()).then(|| version.to_string())
    })
}

// ---------------------------------------------------------------------------
// Clock sync + kernel log
// ---------------------------------------------------------------------------

/// chrony reports "Last offset : +0.000004567 seconds"; systemd-timesyncd's
/// `timedatectl timesync-status` reports "Offset: +1.234ms". (`timedatectl
/// show` carries no offset at all, so the timesync view is the useful
/// fallback.) Anything else degrades to None.
fn probe_clock_offset_ms() -> Option<f64> {
    if let Some(tracking) = run_capture("chronyc", &["tracking"], QUICK_TIMEOUT)
        && let Some(offset) = tracking.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            if key.trim() != "Last offset" {
                return None;
            }
            let seconds: f64 = value.split_whitespace().next()?.parse().ok()?;
            Some(seconds * 1000.0)
        })
    {
        return Some(offset);
    }
    let status = run_capture("timedatectl", &["timesync-status"], QUICK_TIMEOUT)?;
    status.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        if key.trim() != "Offset" {
            return None;
        }
        parse_duration_ms(value.trim())
    })
}

/// "+1.234ms" / "-45us" / "2.5s" -> milliseconds.
fn parse_duration_ms(raw: &str) -> Option<f64> {
    let digits_end = raw
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '+' || c == '-'))
        .unwrap_or(raw.len());
    let (number, unit) = raw.split_at(digits_end);
    let value: f64 = number.parse().ok()?;
    match unit.trim() {
        "s" => Some(value * 1000.0),
        "ms" => Some(value),
        "us" | "µs" => Some(value / 1000.0),
        "ns" => Some(value / 1_000_000.0),
        _ => None,
    }
}

/// Attempt to dlopen each GPU-stack library the benchmark phases depend on,
/// under the same process environment they will run in. Soname fallbacks
/// cover the CUDA major versions in the field.
fn probe_gpu_libs() -> std::collections::BTreeMap<String, bool> {
    // The "nccl" entry mirrors cudarc's search list, which notably does NOT
    // include libnccl.so.2 — NCCL's actual runtime soname. "nccl_runtime"
    // detects that runtime-only situation (libnccl2 installed without the
    // -dev symlink) so bootstrap can build a shim symlink for it.
    const CANDIDATES: [(&str, &[&str]); 4] = [
        ("cuda", &["libcuda.so.1", "libcuda.so"]),
        (
            "cublas",
            &[
                "libcublas.so.13",
                "libcublas.so.12",
                "libcublas.so.11",
                "libcublas.so",
            ],
        ),
        (
            "nccl",
            &[
                "libnccl.so",
                "libnccl.so.12",
                "libnccl.so.11",
                "libnccl.so.10",
                "libnccl.so.9",
                "libnccl.so.1",
            ],
        ),
        ("nccl_runtime", &["libnccl.so.2"]),
    ];
    CANDIDATES
        .into_iter()
        .map(|(name, sonames)| (name.to_string(), sonames.iter().any(|so| dlopen_works(so))))
        .collect()
}

fn dlopen_works(soname: &str) -> bool {
    // SAFETY: loading runs the library's initializers; these are the standard
    // NVIDIA runtime libraries, whose initializers are safe to run (the GPU
    // phases load exactly the same objects). The handle is deliberately
    // leaked: some driver stacks misbehave under dlclose, and the agent
    // process is short-lived anyway.
    match unsafe { libloading::Library::new(soname) } {
        Ok(library) => {
            std::mem::forget(library);
            true
        }
        Err(_) => false,
    }
}

/// Xid codes seen in the kernel ring buffer, deduplicated and sorted. Lines
/// look like `NVRM: Xid (PCI:0000:65:00): 13, pid=…`.
fn probe_xid_errors() -> Vec<u32> {
    let text = run_capture(
        "journalctl",
        &["-k", "--no-pager", "-o", "cat"],
        PROBE_TIMEOUT,
    )
    .or_else(|| std::fs::read_to_string("/var/log/kern.log").ok());
    let Some(text) = text else {
        return Vec::new();
    };
    let codes: BTreeSet<u32> = text.lines().filter_map(parse_xid_line).collect();
    codes.into_iter().collect()
}

pub(crate) fn parse_xid_line(line: &str) -> Option<u32> {
    let (_, rest) = line.split_once("NVRM: Xid")?;
    // Skip the "(PCI:0000:65:00)" device tag when present.
    let rest = match rest.split_once(')') {
        Some((_, after)) => after,
        None => rest,
    };
    let rest = rest.trim_start_matches([':', ' ']);
    let digits_end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if digits_end == 0 {
        return None;
    }
    rest[..digits_end].parse().ok()
}

// ---------------------------------------------------------------------------
// Small IO helpers
// ---------------------------------------------------------------------------

/// Run a probe command and capture its stdout, giving up after `timeout`.
///
/// The child is spawned with piped stdout drained on a helper thread (a
/// blocking `wait()` would deadlock if the child filled the pipe) and killed
/// if it outlives the deadline, so a wedged tool can never stall `collect()`.
/// A missing binary, a non-zero exit, or a timeout all yield `None`.
pub(crate) fn run_capture(program: &str, args: &[&str], timeout: Duration) -> Option<String> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let stdout = child.stdout.take()?;
    let drain = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = stdout.take(MAX_CAPTURE_BYTES).read_to_end(&mut buffer);
        buffer
    });

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(_) => {
                let _ = child.kill();
                break None;
            }
        }
    };
    // Joining is safe either way: killing the child closes the pipe, so the
    // drain thread always terminates.
    let bytes = drain.join().ok()?;
    if !status?.success() {
        return None;
    }
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kernel_and_cpu_fields() {
        let cpuinfo = "processor\t: 0\nmodel name\t: Fancy CPU 9000\n\nprocessor\t: 1\nmodel name\t: Fancy CPU 9000\n";
        assert_eq!(cpu_model(cpuinfo), "Fancy CPU 9000");
        assert_eq!(logical_cores(cpuinfo), 2);
    }

    #[test]
    fn parses_meminfo() {
        assert_eq!(
            mem_total_bytes("MemFree:  10 kB\nMemTotal:       16384 kB\n"),
            16384 * 1024
        );
        assert_eq!(mem_total_bytes("nothing here\n"), 0);
    }

    #[test]
    fn numa_dir_names() {
        assert_eq!(numa_node_id("node0"), Some(0));
        assert_eq!(numa_node_id("node12"), Some(12));
        assert_eq!(numa_node_id("node"), None);
        assert_eq!(numa_node_id("has_cpu"), None);
        assert_eq!(numa_node_id("possible"), None);
    }

    #[test]
    fn nvidia_na_fields_become_none() {
        assert_eq!(csv_field("H100"), Some("H100"));
        assert_eq!(csv_field("N/A"), None);
        assert_eq!(csv_field("[N/A]"), None);
        assert_eq!(csv_field("[Not Supported]"), None);
        assert_eq!(csv_field("  "), None);
    }

    #[test]
    fn xid_lines() {
        assert_eq!(
            parse_xid_line("NVRM: Xid (PCI:0000:65:00): 13, pid=1234, Graphics Exception"),
            Some(13)
        );
        assert_eq!(
            parse_xid_line("kernel: NVRM: Xid (0000:01:00): 79, GPU has fallen off the bus."),
            Some(79)
        );
        assert_eq!(parse_xid_line("nothing interesting"), None);
    }

    #[test]
    fn durations() {
        assert_eq!(parse_duration_ms("+1.5ms"), Some(1.5));
        assert_eq!(parse_duration_ms("-500us"), Some(-0.5));
        assert_eq!(parse_duration_ms("2s"), Some(2000.0));
        assert_eq!(parse_duration_ms("garbage"), None);
    }

    // -- GPU occupancy fixtures ------------------------------------------
    //
    // `GPU_QUERY` rows as nvidia-smi prints them under
    // `--format=csv,noheader,nounits` (a consumer RTX 3090: no ECC, no row
    // remapping), then the compute-apps rows.

    const RTX3090_BUSY: &str = "0, NVIDIA GeForce RTX 3090, GPU-5a1b, 94.02.42.00.A9, 24576, \
        [N/A], [N/A], 4, 4, 16, 16, Enabled, 570.133.07, 00000000:01:00.0, 23264\n";
    const RTX3090_IDLE: &str = "0, NVIDIA GeForce RTX 3090, GPU-5a1b, 94.02.42.00.A9, 24576, \
        [N/A], [N/A], 4, 4, 16, 16, Enabled, 570.133.07, 00000000:01:00.0, 37\n";
    /// The field report's vLLM node. nvidia-smi's process table there also
    /// showed Xorg (12 MiB) and sddm-greeter (20 MiB), both type G: the
    /// compute-apps query does not list graphics clients at all.
    const VLLM_APPS: &str = "00000000:01:00.0, 2102873, VLLM::EngineCore, 23232\n";

    fn identity(own: u32) -> AgentIdentity {
        AgentIdentity {
            lineage: [own].into_iter().collect(),
            agent_basenames: ["gauntlet-agent".to_string()].into_iter().collect(),
        }
    }

    fn parse(query: &str, apps: Option<&str>) -> Vec<GpuInventory> {
        gpus_from_nvidia_smi(query, apps, &identity(1), |_| None).0
    }

    fn outcome(gpu: &GpuInventory) -> TestOutcome {
        crate::proto::assess_gpu_idle(&gpu.occupancy, 1024).outcome()
    }

    #[test]
    fn the_vllm_node_is_busy() {
        let (gpus, driver) =
            gpus_from_nvidia_smi(RTX3090_BUSY, Some(VLLM_APPS), &identity(1), |_| {
                Some("python3.12".into())
            });
        assert_eq!(driver.as_deref(), Some("570.133.07"));
        let gpu = &gpus[0];
        assert_eq!(gpu.mem_total_bytes, 24576 << 20);
        assert_eq!(gpu.persistence_mode, Some(true));
        assert_eq!(gpu.occupancy.memory_used_mib, Some(23264));
        assert_eq!(gpu.occupancy.memory_total_mib, Some(24576));
        let processes = gpu.occupancy.compute_processes.as_ref().expect("known");
        assert_eq!(processes.len(), 1);
        assert_eq!(processes[0].owner, crate::proto::ProcessOwner::Foreign);
        let TestOutcome::Failed { reason } = outcome(gpu) else {
            panic!("expected Failed");
        };
        assert!(
            reason.contains("VLLM::EngineCore (pid 2102873, 23232 MiB)"),
            "{reason}"
        );
    }

    #[test]
    fn graphics_only_node_is_idle() {
        // Xorg + sddm-greeter hold 32 MiB between them but are not compute
        // apps: the query prints nothing.
        let gpus = parse(RTX3090_IDLE, Some(""));
        assert_eq!(gpus[0].occupancy.compute_processes, Some(vec![]));
        assert_eq!(gpus[0].occupancy.memory_used_mib, Some(37));
        assert_eq!(outcome(&gpus[0]), TestOutcome::Passed);
    }

    #[test]
    fn na_fields_and_failed_queries_are_unknown() {
        // MIG-style N/A memory and bus id.
        let na = "0, NVIDIA A100, GPU-1, 92.00, [N/A], [N/A], [N/A], 4, 4, 16, 16, \
                  Enabled, 550.54.15, [N/A], [N/A]\n";
        let gpus = parse(na, Some(VLLM_APPS));
        let occupancy = &gpus[0].occupancy;
        assert_eq!(occupancy.memory_used_mib, None);
        assert_eq!(occupancy.memory_total_mib, None);
        assert_eq!(gpus[0].mem_total_bytes, 0);
        // No bus id: nothing can be attributed, so the list is unknown.
        assert_eq!(occupancy.compute_processes, None);
        assert!(matches!(outcome(&gpus[0]), TestOutcome::Skipped { .. }));

        // Compute-apps query failed outright.
        let gpus = parse(RTX3090_IDLE, None);
        assert_eq!(gpus[0].occupancy.compute_processes, None);
        assert!(matches!(outcome(&gpus[0]), TestOutcome::Skipped { .. }));

        // A pre-v10 query shape (no bus id / memory.used columns) still
        // parses, occupancy unknown.
        let old = "0, H100, GPU-1, 96.00, 81559, 0, No, 5, 5, 16, 16, Enabled, 550.54.15\n";
        let gpus = parse(old, Some(""));
        assert_eq!(gpus[0].mem_total_bytes, 81559 << 20);
        assert_eq!(gpus[0].occupancy.memory_used_mib, None);
        assert_eq!(gpus[0].occupancy.compute_processes, None);
    }

    #[test]
    fn processes_land_on_the_right_gpu_only() {
        let query = "1, H100, GPU-b, 96.00, 81559, 0, No, 5, 5, 16, 16, Enabled, 550.54.15, \
                     00000000:41:00.0, 60000\n\
                     0, H100, GPU-a, 96.00, 81559, 0, No, 5, 5, 16, 16, Enabled, 550.54.15, \
                     00000000:1B:00.0, 4\n";
        let apps = "00000000:41:00.0, 777, python, 59990\n";
        let gpus = parse(query, Some(apps));
        assert_eq!(gpus.iter().map(|g| g.index).collect::<Vec<_>>(), [0, 1]);
        assert_eq!(gpus[0].occupancy.compute_processes, Some(vec![]));
        assert_eq!(outcome(&gpus[0]), TestOutcome::Passed);
        let busy = gpus[1].occupancy.compute_processes.as_ref().expect("known");
        assert_eq!(busy[0].pid, 777);
        assert!(matches!(outcome(&gpus[1]), TestOutcome::Failed { .. }));
    }

    #[test]
    fn a_stale_agent_is_flagged_and_our_own_is_not() {
        let apps = "00000000:01:00.0, 4242, /home/u/.gauntlet/bin/gauntlet-agent, 310\n\
                    00000000:01:00.0, 1, /home/u/.gauntlet/bin/gauntlet-agent, 300\n";
        let gpus = parse(RTX3090_IDLE, Some(apps));
        let processes = gpus[0].occupancy.compute_processes.as_ref().expect("known");
        assert_eq!(processes.len(), 1, "own pid 1 excluded: {processes:?}");
        assert_eq!(
            processes[0].owner,
            crate::proto::ProcessOwner::StaleGauntletAgent
        );
        let TestOutcome::Failed { reason } = outcome(&gpus[0]) else {
            panic!("a stale agent must fail the GPU");
        };
        assert!(reason.contains("stale gauntlet agent"), "{reason}");
        assert!(reason.contains("pid 4242"), "{reason}");
    }

    #[test]
    fn gpu_query_names_the_occupancy_fields_last() {
        assert!(GPU_QUERY.ends_with("driver_version,pci.bus_id,memory.used"));
        assert_eq!(GPU_QUERY.matches(',').count(), 14, "15 fields");
    }

    #[test]
    fn missing_binaries_are_not_errors() {
        assert_eq!(
            run_capture("gauntlet-no-such-binary", &[], QUICK_TIMEOUT),
            None
        );
    }
}
