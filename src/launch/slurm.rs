//! The Slurm allocation the orchestrator runs inside (srun launch mode).
//!
//! Everything here is pure: the environment is read through an injected
//! lookup, and `scontrol show hostnames` output is parsed from a string, so
//! the policy is unit-testable on a machine without Slurm. Slurm's bracket
//! syntax (`node[01-04,07]`) is never parsed here — `scontrol` expands it
//! and this module only reads the one-hostname-per-line result.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::config::HostConfig;

/// Environment variable naming the allocation's job id.
pub const ENV_JOB_ID: &str = "SLURM_JOB_ID";
/// Environment variable carrying the allocation's compressed node list.
pub const ENV_JOB_NODELIST: &str = "SLURM_JOB_NODELIST";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SlurmError {
    #[error(
        "srun launch needs a Slurm allocation, but {ENV_JOB_ID} is not set; run gauntlet \
         inside one (`salloc -N <nodes> ...` then `gauntlet run --launch srun`, or from an \
         sbatch script), or use `--launch ssh`"
    )]
    NotInAllocation,
    #[error("{ENV_JOB_ID}={value:?} is not a Slurm job id")]
    BadJobId { value: String },
    #[error(
        "{ENV_JOB_NODELIST} is not set inside job {job_id}; cannot infer hosts from the \
         allocation (list them under `hosts` instead)"
    )]
    NoNodelist { job_id: SlurmJobId },
    #[error("`scontrol show hostnames` listed no nodes")]
    EmptyHostnames,
    #[error("`scontrol show hostnames` listed {name:?} twice")]
    DuplicateHostname { name: String },
    #[error("`scontrol show hostnames` printed an unusable line {line:?}")]
    InvalidHostname { line: String },
    #[error(
        "host(s) {hosts:?} are not part of Slurm job {job_id} (nodes: {allocation:?}); srun \
         can only launch inside the allocation — drop them from `hosts` or omit `hosts` to use \
         every allocated node"
    )]
    HostsOutsideAllocation {
        job_id: SlurmJobId,
        hosts: Vec<String>,
        allocation: Vec<String>,
    },
    #[error("`scontrol show node` printed an unusable record {line:?}")]
    InvalidNodeRecord { line: String },
}

/// A Slurm job id. Numeric by construction: array/het-job suffixes are not
/// job ids an srun step can be launched under from inside the allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SlurmJobId(u64);

impl SlurmJobId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SlurmJobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for SlurmJobId {
    type Err = SlurmError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let trimmed = value.trim();
        if trimmed.is_empty() || !trimmed.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(SlurmError::BadJobId {
                value: value.to_string(),
            });
        }
        trimmed
            .parse::<u64>()
            .map(SlurmJobId)
            .map_err(|_| SlurmError::BadJobId {
                value: value.to_string(),
            })
    }
}

/// The allocation as the environment describes it. `nodelist` stays in
/// Slurm's compressed form; `scontrol show hostnames` expands it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlurmAllocation {
    pub job_id: SlurmJobId,
    pub nodelist: Option<String>,
}

impl SlurmAllocation {
    /// Read the allocation through `lookup` (normally `std::env::var`).
    /// A missing or empty SLURM_JOB_ID means "not inside an allocation";
    /// the node list is optional here and only required when hosts are
    /// inferred (`require_nodelist`).
    pub fn from_env(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, SlurmError> {
        let job_id = match lookup(ENV_JOB_ID) {
            Some(value) if !value.trim().is_empty() => value.parse::<SlurmJobId>()?,
            _ => return Err(SlurmError::NotInAllocation),
        };
        let nodelist = lookup(ENV_JOB_NODELIST).filter(|value| !value.trim().is_empty());
        Ok(Self { job_id, nodelist })
    }

    pub fn require_nodelist(&self) -> Result<&str, SlurmError> {
        self.nodelist.as_deref().ok_or(SlurmError::NoNodelist {
            job_id: self.job_id,
        })
    }
}

/// `scontrol show hostnames <nodelist>` arguments.
pub fn scontrol_hostnames_args(nodelist: &str) -> Vec<String> {
    vec!["show".into(), "hostnames".into(), nodelist.to_string()]
}

/// Parse `scontrol show hostnames` output: one hostname per line, order
/// preserved (it is the allocation order, which becomes fleet order and
/// therefore rank order). Blank lines and surrounding whitespace are
/// ignored; a line with inner whitespace or control characters, a repeat,
/// or no hostname at all is an error rather than a guess.
pub fn parse_hostnames(stdout: &str) -> Result<Vec<String>, SlurmError> {
    let mut seen = BTreeSet::new();
    let mut hosts = Vec::new();
    for line in stdout.lines() {
        let name = line.trim();
        if name.is_empty() {
            continue;
        }
        if name.chars().any(|ch| ch.is_whitespace() || ch.is_control()) {
            return Err(SlurmError::InvalidHostname {
                line: line.to_string(),
            });
        }
        if !seen.insert(name.to_string()) {
            return Err(SlurmError::DuplicateHostname {
                name: name.to_string(),
            });
        }
        hosts.push(name.to_string());
    }
    if hosts.is_empty() {
        return Err(SlurmError::EmptyHostnames);
    }
    Ok(hosts)
}

/// The fleet for an srun launch. No configured hosts: every allocated node,
/// in allocation order (the default). Configured hosts: kept as written
/// (labels, data_addr) but every one must be an allocated node, since srun
/// cannot place a step anywhere else.
pub fn select_hosts(
    job_id: SlurmJobId,
    configured: Vec<HostConfig>,
    allocation: &[String],
) -> Result<Vec<HostConfig>, SlurmError> {
    if configured.is_empty() {
        return Ok(allocation
            .iter()
            .map(|name| HostConfig {
                addr: name.clone(),
                data_addr: None,
                labels: Default::default(),
            })
            .collect());
    }
    let allocated: BTreeSet<&str> = allocation.iter().map(String::as_str).collect();
    let outside: Vec<String> = configured
        .iter()
        .filter(|host| !allocated.contains(host.addr.as_str()))
        .map(|host| host.addr.clone())
        .collect();
    if !outside.is_empty() {
        return Err(SlurmError::HostsOutsideAllocation {
            job_id,
            hosts: outside,
            allocation: allocation.to_vec(),
        });
    }
    Ok(configured)
}

/// NodeName -> NodeAddr, from `scontrol -o show node`.
pub type NodeAddrs = BTreeMap<String, String>;

/// `scontrol --oneliner show node <nodelist>`: one record per node, the
/// whole (compressed) node list resolved in a single call.
pub fn scontrol_show_nodes_args(nodelist: &str) -> Vec<String> {
    vec![
        "--oneliner".into(),
        "show".into(),
        "node".into(),
        nodelist.to_string(),
    ]
}

/// Parse `scontrol --oneliner show node` output into NodeName -> NodeAddr.
/// Each record is one line of space-separated `Key=Value` fields starting
/// with `NodeName=` (`slurm_sprint_node_table`); NodeAddr comes before
/// any free-text field (OS, Reason, Comment), and neither names nor
/// addresses contain spaces, so the first ` NodeAddr=` token is the
/// node's. A record without NodeAddr is omitted (the node is then reached
/// by its name); blank lines are ignored; any other line is an error.
pub fn parse_node_addrs(stdout: &str) -> Result<NodeAddrs, SlurmError> {
    let mut addrs = NodeAddrs::new();
    for line in stdout.lines() {
        let record = line.trim();
        if record.is_empty() {
            continue;
        }
        let invalid = || SlurmError::InvalidNodeRecord {
            line: line.to_string(),
        };
        let name = record
            .strip_prefix("NodeName=")
            .and_then(|rest| rest.split_whitespace().next())
            .filter(|name| !name.is_empty())
            .ok_or_else(invalid)?;
        let addr = record
            .split_whitespace()
            .find_map(|field| field.strip_prefix("NodeAddr="))
            .filter(|addr| !addr.is_empty() && *addr != "(null)");
        if let Some(addr) = addr {
            addrs.insert(name.to_string(), addr.to_string());
        }
    }
    Ok(addrs)
}

/// Point peer traffic at each node's NodeAddr. The host's `addr` stays the
/// Slurm NodeName (what `srun --nodelist` targets); `data_addr` — which
/// the pairwise, TCP-barrier and NCCL bootstrap traffic uses — becomes the
/// NodeAddr when the node has one that differs from its name. A configured
/// `data_addr` always wins.
pub fn apply_node_addrs(hosts: Vec<HostConfig>, addrs: &NodeAddrs) -> Vec<HostConfig> {
    hosts
        .into_iter()
        .map(|mut host| {
            if host.data_addr.is_none()
                && let Some(addr) = addrs.get(&host.addr)
                && *addr != host.addr
            {
                host.data_addr = Some(addr.clone());
            }
            host
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: BTreeMap<String, String> = pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        move |key| map.get(key).cloned()
    }

    #[test]
    fn no_job_id_means_not_in_an_allocation() {
        assert_eq!(
            SlurmAllocation::from_env(env(&[])),
            Err(SlurmError::NotInAllocation)
        );
        assert_eq!(
            SlurmAllocation::from_env(env(&[(ENV_JOB_ID, "  ")])),
            Err(SlurmError::NotInAllocation)
        );
        // The message tells the operator what to do.
        let message = SlurmError::NotInAllocation.to_string();
        assert!(message.contains("salloc"), "{message}");
        assert!(message.contains("--launch ssh"), "{message}");
    }

    #[test]
    fn job_ids_are_numeric() {
        let allocation = SlurmAllocation::from_env(env(&[
            (ENV_JOB_ID, "123456"),
            (ENV_JOB_NODELIST, "gpu[01-02]"),
        ]))
        .expect("allocation");
        assert_eq!(allocation.job_id, SlurmJobId::new(123456));
        assert_eq!(allocation.require_nodelist(), Ok("gpu[01-02]"));
        for bad in ["12a", "-3", "123_4", "1.2", "99999999999999999999999"] {
            assert!(
                matches!(
                    SlurmAllocation::from_env(env(&[(ENV_JOB_ID, bad)])),
                    Err(SlurmError::BadJobId { .. })
                ),
                "{bad}"
            );
        }
    }

    #[test]
    fn a_missing_nodelist_is_only_an_error_when_required() {
        let allocation = SlurmAllocation::from_env(env(&[(ENV_JOB_ID, "7")])).expect("allocation");
        assert_eq!(allocation.nodelist, None);
        assert_eq!(
            allocation.require_nodelist(),
            Err(SlurmError::NoNodelist {
                job_id: SlurmJobId::new(7)
            })
        );
    }

    #[test]
    fn hostnames_parse_one_per_line_in_order() {
        let stdout = "gpu-a07\ngpu-a01\n  gpu-a02  \n\ngpu-b10\n";
        assert_eq!(
            parse_hostnames(stdout).expect("hosts"),
            ["gpu-a07", "gpu-a01", "gpu-a02", "gpu-b10"]
        );
        assert_eq!(
            parse_hostnames("node0\r\nnode1\r\n").expect("crlf"),
            ["node0", "node1"]
        );
    }

    #[test]
    fn hostname_output_errors_are_typed() {
        assert_eq!(parse_hostnames(""), Err(SlurmError::EmptyHostnames));
        assert_eq!(parse_hostnames("\n \n"), Err(SlurmError::EmptyHostnames));
        assert_eq!(
            parse_hostnames("a\nb\na\n"),
            Err(SlurmError::DuplicateHostname { name: "a".into() })
        );
        assert!(matches!(
            parse_hostnames("node0 node1\n"),
            Err(SlurmError::InvalidHostname { .. })
        ));
        assert!(matches!(
            parse_hostnames("no\u{1b}de\n"),
            Err(SlurmError::InvalidHostname { .. })
        ));
    }

    #[test]
    fn scontrol_gets_the_compressed_list_verbatim() {
        assert_eq!(
            scontrol_hostnames_args("gpu[01-04,07],cpu1"),
            ["show", "hostnames", "gpu[01-04,07],cpu1"]
        );
    }

    fn host(addr: &str) -> HostConfig {
        HostConfig {
            addr: addr.into(),
            data_addr: None,
            labels: Default::default(),
        }
    }

    #[test]
    fn omitted_hosts_become_the_allocation() {
        let allocation = vec!["n1".to_string(), "n0".to_string()];
        let hosts = select_hosts(SlurmJobId::new(1), Vec::new(), &allocation).expect("hosts");
        let addrs: Vec<&str> = hosts.iter().map(|host| host.addr.as_str()).collect();
        assert_eq!(addrs, ["n1", "n0"]);
    }

    #[test]
    fn configured_hosts_must_be_allocated() {
        let allocation = vec!["n0".to_string(), "n1".to_string(), "n2".to_string()];
        let mut labelled = host("n2");
        labelled.data_addr = Some("10.1.1.2".into());
        let hosts = select_hosts(
            SlurmJobId::new(9),
            vec![labelled.clone(), host("n0")],
            &allocation,
        )
        .expect("subset");
        assert_eq!(hosts, vec![labelled, host("n0")]);

        let error = select_hosts(
            SlurmJobId::new(9),
            vec![host("n0"), host("elsewhere")],
            &allocation,
        )
        .expect_err("outside");
        assert_eq!(
            error,
            SlurmError::HostsOutsideAllocation {
                job_id: SlurmJobId::new(9),
                hosts: vec!["elsewhere".into()],
                allocation,
            }
        );
    }

    /// Shaped like `slurm_sprint_node_table` with `--oneliner`: free-text
    /// fields (OS, Reason) contain spaces and come after NodeAddr.
    const SHOW_NODES: &str = "\
NodeName=gpu-a01 Arch=x86_64 CoresPerSocket=32 CPUAlloc=64 CPUEfctv=128 CPUTot=128 CPULoad=3.10 AvailableFeatures=h100,ib ActiveFeatures=h100,ib Gres=gpu:h100:8 NodeAddr=10.1.1.11 NodeHostName=gpu-a01 Version=24.05.1 OS=Linux 5.15.0-91-generic #101-Ubuntu SMP RealMemory=2000000 State=MIXED Reason=fake NodeAddr=9.9.9.9 [root@2026-01-01]
NodeName=gpu-a02 Arch=x86_64 CoresPerSocket=32 Gres=gpu:h100:8 NodeAddr=gpu-a02 NodeHostName=gpu-a02 OS=Linux 5.15.0 #1 SMP State=IDLE

NodeName=cpu-07 Arch=aarch64 CoresPerSocket=64 Gres=(null) NodeHostName=cpu-07 State=IDLE
";

    #[test]
    fn node_addrs_parse_from_oneliner_records() {
        let addrs = parse_node_addrs(SHOW_NODES).expect("parse");
        assert_eq!(addrs.get("gpu-a01").map(String::as_str), Some("10.1.1.11"));
        // NodeAddr == NodeName is kept as reported.
        assert_eq!(addrs.get("gpu-a02").map(String::as_str), Some("gpu-a02"));
        // No NodeAddr field: not resolved.
        assert!(!addrs.contains_key("cpu-07"));
        assert_eq!(addrs.len(), 2);
        assert_eq!(parse_node_addrs(""), Ok(NodeAddrs::new()));
        assert!(matches!(
            parse_node_addrs("Node gpu-a01 not found\n"),
            Err(SlurmError::InvalidNodeRecord { .. })
        ));
        assert_eq!(
            scontrol_show_nodes_args("gpu-a[01-02],cpu-07"),
            ["--oneliner", "show", "node", "gpu-a[01-02],cpu-07"]
        );
    }

    #[test]
    fn node_addrs_become_data_addrs_unless_configured_or_identical() {
        let addrs = parse_node_addrs(SHOW_NODES).expect("parse");
        let mut configured = host("gpu-a01");
        configured.data_addr = Some("10.9.9.9".into());
        let hosts = apply_node_addrs(
            vec![
                host("gpu-a01"),
                host("gpu-a02"),
                host("cpu-07"),
                configured.clone(),
            ],
            &addrs,
        );
        let data: Vec<Option<&str>> = hosts.iter().map(|h| h.data_addr.as_deref()).collect();
        assert_eq!(data, [Some("10.1.1.11"), None, None, Some("10.9.9.9")]);
        // The step target is untouched.
        assert_eq!(hosts[0].addr, "gpu-a01");
    }
}
