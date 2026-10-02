//! Per-level NCCL environments.
//!
//! Every NCCL communicator gauntlet creates belongs to exactly one
//! `NcclLevel`, named after its call site:
//!
//! | level               | process            | communicator                                  |
//! |---------------------|--------------------|-----------------------------------------------|
//! | `intranode`         | `agent run` (network phase) | intra-node sweep, `ncclCommInitAll`  |
//! | `fleet`             | `agent nccl` (sweep workload) | rank-per-GPU fleet sweep           |
//! | `barrier`           | `agent nccl`       | NCCL barrier-skew probe                       |
//! | `overlap_intranode` | `agent run` (overlap phase) | node-local overlap, `ncclCommInitAll` |
//! | `overlap_fleet`     | `agent nccl` (overlap workload) | fleet overlap step               |
//!
//! `[nccl.levels.<level>] env = {...}` layers an override on top of the
//! global `[nccl]` env: the level's effective env is the global map with
//! every override key replacing (or adding to) it. Overrides go through the
//! same key/value validation as the global map.
//!
//! Each level has its own agent process, so its env can be set on that
//! process's spawn command line (never via `set_var`): the orchestrator
//! sends `agent run` exactly one phase per spawn, and the fleet levels are
//! separate `agent nccl` invocations. The one co-hosting case is the
//! barrier probe, which by default rides the fleet sweep's communicator;
//! when the two levels' effective envs differ, the orchestrator runs the
//! probe in its own world (`NcclWorkload::Barrier`) under the barrier env
//! instead (`NcclLevelEnvs::barrier_shares_fleet_comm`).

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::nccl_env::NcclEnv;

/// Which NCCL call site a communicator (and its agent process) belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NcclLevel {
    /// Intra-node sweep: every local GPU, one process (`agent run`, network
    /// phase).
    Intranode,
    /// Rank-per-GPU fleet sweep (`agent nccl`, sweep workload).
    Fleet,
    /// NCCL barrier-skew probe (rides the fleet communicator unless its env
    /// differs from the fleet env).
    Barrier,
    /// Node-local overlap step (`agent run`, overlap phase).
    OverlapIntranode,
    /// Fleet overlap step (`agent nccl`, overlap workload).
    OverlapFleet,
}

impl NcclLevel {
    pub const ALL: [NcclLevel; 5] = [
        NcclLevel::Intranode,
        NcclLevel::Fleet,
        NcclLevel::Barrier,
        NcclLevel::OverlapIntranode,
        NcclLevel::OverlapFleet,
    ];

    /// The config / results-document name (`[nccl.levels.<name>]`).
    pub fn as_str(self) -> &'static str {
        match self {
            NcclLevel::Intranode => "intranode",
            NcclLevel::Fleet => "fleet",
            NcclLevel::Barrier => "barrier",
            NcclLevel::OverlapIntranode => "overlap_intranode",
            NcclLevel::OverlapFleet => "overlap_fleet",
        }
    }
}

impl fmt::Display for NcclLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The resolved NCCL env of the run: the global map plus the effective
/// (global <- override) map of every level. Complete by construction —
/// every level has an env — so a lookup can never miss.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NcclLevelEnvs {
    global: NcclEnv,
    intranode: NcclEnv,
    fleet: NcclEnv,
    barrier: NcclEnv,
    overlap_intranode: NcclEnv,
    overlap_fleet: NcclEnv,
}

impl NcclLevelEnvs {
    /// Layer each level's validated override onto `global`. Levels without
    /// an override get the global env verbatim.
    pub fn resolve(global: NcclEnv, overrides: &BTreeMap<NcclLevel, NcclEnv>) -> Self {
        let effective = |level: NcclLevel| match overrides.get(&level) {
            Some(over) => global.overlay(over),
            None => global.clone(),
        };
        Self {
            intranode: effective(NcclLevel::Intranode),
            fleet: effective(NcclLevel::Fleet),
            barrier: effective(NcclLevel::Barrier),
            overlap_intranode: effective(NcclLevel::OverlapIntranode),
            overlap_fleet: effective(NcclLevel::OverlapFleet),
            global,
        }
    }

    /// Every level runs under `global` (no overrides).
    pub fn uniform(global: NcclEnv) -> Self {
        Self::resolve(global, &BTreeMap::new())
    }

    /// The `[nccl]` env without level overrides: what non-NCCL agent spawns
    /// (inventory, cpu, gpu, peer, probe, TCP barrier) are started with.
    pub fn global(&self) -> &NcclEnv {
        &self.global
    }

    /// The effective env of `level`.
    pub fn level(&self, level: NcclLevel) -> &NcclEnv {
        match level {
            NcclLevel::Intranode => &self.intranode,
            NcclLevel::Fleet => &self.fleet,
            NcclLevel::Barrier => &self.barrier,
            NcclLevel::OverlapIntranode => &self.overlap_intranode,
            NcclLevel::OverlapFleet => &self.overlap_fleet,
        }
    }

    /// Whether the barrier probe can ride the fleet sweep's communicator:
    /// only when both levels' effective envs are identical. Otherwise the
    /// probe needs its own process (and world) to run under its own env.
    pub fn barrier_shares_fleet_comm(&self) -> bool {
        self.barrier == self.fleet
    }

    /// Effective env per level as plain string maps, for the results
    /// document.
    pub fn to_record(&self) -> BTreeMap<NcclLevel, BTreeMap<String, String>> {
        NcclLevel::ALL
            .into_iter()
            .map(|level| (level, self.level(level).to_string_map()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(entries: &[(&str, &str)]) -> NcclEnv {
        let raw = entries
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        NcclEnv::from_map(&raw).expect("valid env")
    }

    #[test]
    fn levels_without_overrides_inherit_the_global_env() {
        let global = env(&[("NCCL_DEBUG", "WARN"), ("NCCL_SOCKET_IFNAME", "bond0")]);
        let envs = NcclLevelEnvs::uniform(global.clone());
        assert_eq!(envs.global(), &global);
        for level in NcclLevel::ALL {
            assert_eq!(envs.level(level), &global, "{level}");
        }
        assert!(envs.barrier_shares_fleet_comm());
    }

    #[test]
    fn an_override_replaces_and_adds_keys_for_its_level_only() {
        let global = env(&[("NCCL_ALGO", "Tree"), ("NCCL_IB_HCA", "mlx5_0")]);
        let overrides = BTreeMap::from([
            (
                NcclLevel::Intranode,
                env(&[("NCCL_ALGO", "Ring"), ("NCCL_P2P_LEVEL", "NVL")]),
            ),
            (
                NcclLevel::Fleet,
                env(&[("NCCL_P2P_DISABLE", "1"), ("NCCL_SHM_DISABLE", "1")]),
            ),
        ]);
        let envs = NcclLevelEnvs::resolve(global.clone(), &overrides);
        assert_eq!(
            envs.level(NcclLevel::Intranode),
            &env(&[
                ("NCCL_ALGO", "Ring"),
                ("NCCL_IB_HCA", "mlx5_0"),
                ("NCCL_P2P_LEVEL", "NVL"),
            ])
        );
        assert_eq!(
            envs.level(NcclLevel::Fleet),
            &env(&[
                ("NCCL_ALGO", "Tree"),
                ("NCCL_IB_HCA", "mlx5_0"),
                ("NCCL_P2P_DISABLE", "1"),
                ("NCCL_SHM_DISABLE", "1"),
            ])
        );
        for level in [
            NcclLevel::Barrier,
            NcclLevel::OverlapIntranode,
            NcclLevel::OverlapFleet,
        ] {
            assert_eq!(envs.level(level), &global, "{level}");
        }
        assert_eq!(envs.global(), &global, "the global map is untouched");
        // Fleet got an override the barrier did not: they no longer share.
        assert!(!envs.barrier_shares_fleet_comm());
    }

    #[test]
    fn identical_barrier_and_fleet_overrides_still_share_the_communicator() {
        let over = env(&[("NCCL_PROTO", "Simple")]);
        let overrides =
            BTreeMap::from([(NcclLevel::Fleet, over.clone()), (NcclLevel::Barrier, over)]);
        let envs = NcclLevelEnvs::resolve(NcclEnv::default(), &overrides);
        assert!(envs.barrier_shares_fleet_comm());
    }

    #[test]
    fn an_override_equal_to_the_global_value_changes_nothing() {
        let global = env(&[("NCCL_ALGO", "Ring")]);
        let overrides = BTreeMap::from([(NcclLevel::Barrier, env(&[("NCCL_ALGO", "Ring")]))]);
        let envs = NcclLevelEnvs::resolve(global, &overrides);
        assert!(envs.barrier_shares_fleet_comm());
    }

    #[test]
    fn the_record_lists_every_level() {
        let global = env(&[("NCCL_DEBUG", "WARN")]);
        let overrides = BTreeMap::from([(NcclLevel::OverlapFleet, env(&[("NCCL_DEBUG", "INFO")]))]);
        let record = NcclLevelEnvs::resolve(global, &overrides).to_record();
        assert_eq!(record.len(), NcclLevel::ALL.len());
        assert_eq!(record[&NcclLevel::OverlapFleet]["NCCL_DEBUG"], "INFO");
        assert_eq!(record[&NcclLevel::Fleet]["NCCL_DEBUG"], "WARN");
    }

    #[test]
    fn level_names_round_trip_through_serde_and_json_map_keys() {
        for level in NcclLevel::ALL {
            let json = serde_json::to_string(&level).expect("serialize");
            assert_eq!(json, format!("\"{}\"", level.as_str()));
            let back: NcclLevel = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, level);
        }
        let map = BTreeMap::from([(NcclLevel::OverlapIntranode, 1u32)]);
        let json = serde_json::to_string(&map).expect("enum keys serialize");
        assert_eq!(json, r#"{"overlap_intranode":1}"#);
        let back: BTreeMap<NcclLevel, u32> = serde_json::from_str(&json).expect("enum keys");
        assert_eq!(back, map);
    }
}
