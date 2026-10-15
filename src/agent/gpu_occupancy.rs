//! Phase 0 GPU occupancy: which compute processes hold each GPU.
//!
//! Source: `nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,
//! used_memory --format=csv,noheader,nounits`, joined onto the per-GPU
//! inventory by PCI bus id (the compute-apps query has no GPU index).
//! Only compute contexts are listed, so graphics-only clients (Xorg, display
//! managers) never appear. The reporting agent and its ancestry are dropped;
//! any other `gauntlet-agent` is kept and marked stale — a leftover from a
//! crashed run is exactly what a pre-flight check should surface.
//!
//! Everything here is a pure function of injected text or an injected
//! process view, so the parsers and the classification are unit-tested with
//! fixtures; only `AgentIdentity::current` and `exe_basename` touch /proc.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use thiserror::Error;

use crate::agent::inventory::csv_field;
use crate::proto::{GpuProcess, ProcessOwner};

/// Query for per-process GPU use; `used_memory` is MiB under `nounits`.
pub(crate) const COMPUTE_APPS_QUERY: &str =
    "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory";

/// Executable name of the deployed agent (`deploy::AGENT_RELPATH`'s file
/// name). Kept literal here: the agent side must not depend on the
/// orchestrator module.
const AGENT_BASENAME: &str = "gauntlet-agent";

/// A PCI address, normalized so that nvidia-smi's two spellings
/// (`00000000:01:00.0` from both queries on most drivers, `0000:01:00.0`
/// elsewhere) compare equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PciBusId {
    domain: u32,
    bus: u8,
    device: u8,
    function: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PciBusIdError {
    #[error("pci bus id {raw:?} is not domain:bus:device.function")]
    Shape { raw: String },
    #[error("pci bus id {raw:?} has a non-hex {part} component")]
    Component { raw: String, part: &'static str },
}

impl FromStr for PciBusId {
    type Err = PciBusIdError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let trimmed = raw.trim();
        let shape = || PciBusIdError::Shape {
            raw: raw.to_string(),
        };
        let mut colon_parts = trimmed.split(':');
        let (Some(domain), Some(bus), Some(slot), None) = (
            colon_parts.next(),
            colon_parts.next(),
            colon_parts.next(),
            colon_parts.next(),
        ) else {
            return Err(shape());
        };
        let (device, function) = slot.split_once('.').ok_or_else(shape)?;
        let component = |text: &str, part: &'static str| {
            (!text.is_empty())
                .then_some(text)
                .and_then(|text| u32::from_str_radix(text, 16).ok())
                .ok_or_else(|| PciBusIdError::Component {
                    raw: raw.to_string(),
                    part,
                })
        };
        let narrow = |value: u32, part: &'static str| {
            u8::try_from(value).map_err(|_| PciBusIdError::Component {
                raw: raw.to_string(),
                part,
            })
        };
        Ok(Self {
            domain: component(domain, "domain")?,
            bus: narrow(component(bus, "bus")?, "bus")?,
            device: narrow(component(device, "device")?, "device")?,
            function: narrow(component(function, "function")?, "function")?,
        })
    }
}

impl fmt::Display for PciBusId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

/// One row of the compute-apps query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeApp {
    pub bus_id: PciBusId,
    pub pid: u32,
    pub name: String,
    pub used_mib: Option<u64>,
}

/// Parse the compute-apps CSV. Rows that do not parse (no bus id, no pid)
/// are dropped: they cannot be attributed to a GPU anyway. The process name
/// is everything between the pid and the last field, so a name containing a
/// comma survives. Empty output (no compute processes) is an empty list.
pub fn parse_compute_apps(text: &str) -> Vec<ComputeApp> {
    text.lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 4 {
                return None;
            }
            let bus_id = csv_field(fields[0])?.parse::<PciBusId>().ok()?;
            let pid = csv_field(fields[1])?.parse::<u32>().ok()?;
            let last = fields.len() - 1;
            let name = fields[2..last].join(",").trim().to_string();
            let used_mib = csv_field(fields[last]).and_then(|raw| raw.parse::<u64>().ok());
            Some(ComputeApp {
                bus_id,
                pid,
                name: if name.is_empty() {
                    "unknown".to_string()
                } else {
                    name
                },
                used_mib,
            })
        })
        .collect()
}

/// What the agent knows about itself, for telling its own GPU contexts
/// apart from everyone else's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentIdentity {
    /// This process and its ancestors (a wrapper `env`/shell never holds a
    /// GPU context, but excluding the whole chain is free and exact).
    pub lineage: BTreeSet<u32>,
    /// Executable basenames that identify a gauntlet agent.
    pub agent_basenames: BTreeSet<String>,
}

impl AgentIdentity {
    /// The running agent's identity from /proc. Unreadable entries shrink
    /// the lineage (never below this pid) rather than failing.
    pub fn current() -> Self {
        let own = std::process::id();
        let mut lineage = BTreeSet::from([own]);
        let mut pid = own;
        // Bounded: pid namespaces are shallow, and a cycle is impossible but
        // cheap to guard against.
        for _ in 0..64 {
            let Some(parent) = std::fs::read_to_string(format!("/proc/{pid}/stat"))
                .ok()
                .as_deref()
                .and_then(parse_stat_ppid)
            else {
                break;
            };
            if parent == 0 || !lineage.insert(parent) {
                break;
            }
            pid = parent;
        }
        let mut agent_basenames = BTreeSet::from([AGENT_BASENAME.to_string()]);
        if let Some(own_exe) = exe_basename(own) {
            agent_basenames.insert(own_exe);
        }
        Self {
            lineage,
            agent_basenames,
        }
    }
}

/// Parent pid out of `/proc/<pid>/stat`: `pid (comm) state ppid ...`. The
/// comm may itself contain spaces and parentheses, so fields are counted
/// from the *last* `)`.
pub fn parse_stat_ppid(stat: &str) -> Option<u32> {
    let (_, rest) = stat.rsplit_once(')')?;
    rest.split_whitespace().nth(1)?.parse().ok()
}

/// Basename of `/proc/<pid>/exe`, without the " (deleted)" suffix a
/// replaced binary leaves behind (deploy renames a new agent over the old
/// one, so a stale agent's exe link reads exactly like that). `None` when
/// unreadable (another user's process).
pub fn exe_basename(pid: u32) -> Option<String> {
    let target = std::fs::read_link(format!("/proc/{pid}/exe")).ok()?;
    Some(basename(&target.to_string_lossy()).to_string())
}

fn basename(path: &str) -> &str {
    let path = path.trim().trim_end_matches(" (deleted)");
    path.rsplit('/').next().unwrap_or(path)
}

/// Whose process this is; `None` for the reporting agent itself (dropped).
/// A process is a gauntlet agent when either its reported name or its
/// executable has an agent basename.
pub fn classify(
    app: &ComputeApp,
    exe: Option<&str>,
    identity: &AgentIdentity,
) -> Option<ProcessOwner> {
    if identity.lineage.contains(&app.pid) {
        return None;
    }
    let named_agent = identity.agent_basenames.contains(basename(&app.name))
        || exe.is_some_and(|exe| identity.agent_basenames.contains(basename(exe)));
    Some(if named_agent {
        ProcessOwner::StaleGauntletAgent
    } else {
        ProcessOwner::Foreign
    })
}

/// Per-GPU process lists keyed by bus id, the reporting agent removed.
/// `None` in, `None` out: a failed query means "unknown", never "idle".
pub fn processes_by_bus(
    apps: Option<Vec<ComputeApp>>,
    identity: &AgentIdentity,
    exe_of: impl Fn(u32) -> Option<String>,
) -> Option<BTreeMap<PciBusId, Vec<GpuProcess>>> {
    let apps = apps?;
    let mut by_bus: BTreeMap<PciBusId, Vec<GpuProcess>> = BTreeMap::new();
    for app in apps {
        let exe = exe_of(app.pid);
        let Some(owner) = classify(&app, exe.as_deref(), identity) else {
            continue;
        };
        by_bus.entry(app.bus_id).or_default().push(GpuProcess {
            pid: app.pid,
            name: app.name,
            used_mib: app.used_mib,
            owner,
        });
    }
    for processes in by_bus.values_mut() {
        processes.sort_by_key(|process| process.pid);
    }
    Some(by_bus)
}

/// The process list for one GPU: `None` when either the query failed or
/// the GPU's own bus id is unknown (nothing can be attributed to it).
pub fn processes_for(
    by_bus: Option<&BTreeMap<PciBusId, Vec<GpuProcess>>>,
    bus_id: Option<PciBusId>,
) -> Option<Vec<GpuProcess>> {
    let by_bus = by_bus?;
    let bus_id = bus_id?;
    Some(by_bus.get(&bus_id).cloned().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bus(raw: &str) -> PciBusId {
        raw.parse().expect("bus id")
    }

    fn identity(lineage: &[u32]) -> AgentIdentity {
        AgentIdentity {
            lineage: lineage.iter().copied().collect(),
            agent_basenames: BTreeSet::from(["gauntlet-agent".to_string()]),
        }
    }

    /// The node from the field report: vLLM holding 23.2 GiB of a 24 GiB
    /// card. Xorg and sddm-greeter were on the card too, but as type G
    /// processes — the compute-apps query never lists them.
    const VLLM_BUSY: &str = "00000000:01:00.0, 2102873, VLLM::EngineCore, 23232\n";

    #[test]
    fn bus_ids_normalize_across_spellings() {
        assert_eq!(bus("00000000:01:00.0"), bus("0000:01:00.0"));
        assert_eq!(bus(" 00000000:C1:00.0 "), bus("0000:c1:00.0"));
        assert_eq!(bus("00000000:01:00.0").to_string(), "0000:01:00.0");
        assert_ne!(bus("0000:01:00.0"), bus("0000:02:00.0"));
    }

    #[test]
    fn malformed_bus_ids_are_typed_errors() {
        assert!(matches!(
            "01:00.0".parse::<PciBusId>(),
            Err(PciBusIdError::Shape { .. })
        ));
        assert!(matches!(
            "0000:zz:00.0".parse::<PciBusId>(),
            Err(PciBusIdError::Component { part: "bus", .. })
        ));
        assert!(matches!(
            "0000:100:00.0".parse::<PciBusId>(),
            Err(PciBusIdError::Component { part: "bus", .. })
        ));
        assert!("0000:01:00".parse::<PciBusId>().is_err());
        assert!("N/A".parse::<PciBusId>().is_err());
    }

    #[test]
    fn parses_the_vllm_row() {
        assert_eq!(
            parse_compute_apps(VLLM_BUSY),
            vec![ComputeApp {
                bus_id: bus("0000:01:00.0"),
                pid: 2_102_873,
                name: "VLLM::EngineCore".into(),
                used_mib: Some(23_232),
            }]
        );
    }

    #[test]
    fn empty_output_means_no_compute_processes() {
        assert!(parse_compute_apps("").is_empty());
        assert!(parse_compute_apps("\n").is_empty());
        // Some drivers print a banner instead of nothing.
        assert!(parse_compute_apps("No running processes found\n").is_empty());
    }

    #[test]
    fn na_memory_and_commas_in_names_survive() {
        let apps = parse_compute_apps(
            "00000000:01:00.0, 77, [N/A], [N/A]\n\
             00000000:01:00.0, 78, python train.py --a,b, 512\n\
             garbage line\n\
             N/A, 79, x, 1\n",
        );
        assert_eq!(apps.len(), 2, "{apps:?}");
        assert_eq!(apps[0].used_mib, None);
        assert_eq!(apps[0].name, "[N/A]");
        assert_eq!(apps[1].name, "python train.py --a,b");
        assert_eq!(apps[1].used_mib, Some(512));
    }

    #[test]
    fn stat_ppid_survives_hostile_comm() {
        assert_eq!(
            parse_stat_ppid("2102873 (VLLM::EngineCore) S 2102800 2102873 0"),
            Some(2_102_800)
        );
        assert_eq!(parse_stat_ppid("42 (a) b (c)) R 7 42 42"), Some(7));
        assert_eq!(parse_stat_ppid("garbage"), None);
    }

    #[test]
    fn own_lineage_is_dropped_and_other_agents_are_stale() {
        let own = ComputeApp {
            bus_id: bus("0000:01:00.0"),
            pid: 100,
            name: "/home/u/.gauntlet/bin/gauntlet-agent".into(),
            used_mib: Some(300),
        };
        let mut stale = own.clone();
        stale.pid = 55;
        let identity = identity(&[100, 90, 1]);
        assert_eq!(classify(&own, None, &identity), None);
        assert_eq!(
            classify(&stale, None, &identity),
            Some(ProcessOwner::StaleGauntletAgent)
        );
        // Recognized by executable even under a changed process title.
        let mut retitled = stale.clone();
        retitled.name = "worker".into();
        assert_eq!(
            classify(
                &retitled,
                Some("/home/u/.gauntlet/bin/gauntlet-agent (deleted)"),
                &identity
            ),
            Some(ProcessOwner::StaleGauntletAgent)
        );
        let vllm = &parse_compute_apps(VLLM_BUSY)[0];
        assert_eq!(
            classify(vllm, Some("python3.12"), &identity),
            Some(ProcessOwner::Foreign)
        );
    }

    #[test]
    fn processes_group_by_bus_and_unknown_stays_unknown() {
        let text = "00000000:02:00.0, 9, /opt/gauntlet-agent, 200\n\
                    00000000:02:00.0, 3, python, 4000\n\
                    00000000:02:00.0, 100, gauntlet-agent, 150\n";
        let by_bus = processes_by_bus(Some(parse_compute_apps(text)), &identity(&[100]), |_| None)
            .expect("known");
        let gpu1 = &by_bus[&bus("0000:02:00.0")];
        assert_eq!(
            gpu1.iter().map(|p| p.pid).collect::<Vec<_>>(),
            [3, 9],
            "own pid dropped, sorted by pid"
        );
        assert_eq!(gpu1[1].owner, ProcessOwner::StaleGauntletAgent);

        // A GPU the query did not mention is verified idle; an unknown bus
        // id or a failed query is unknown.
        assert_eq!(
            processes_for(Some(&by_bus), Some(bus("0000:01:00.0"))),
            Some(vec![])
        );
        assert_eq!(processes_for(Some(&by_bus), None), None);
        assert_eq!(processes_by_bus(None, &identity(&[]), |_| None), None);
        assert_eq!(processes_for(None, Some(bus("0000:01:00.0"))), None);
    }
}
