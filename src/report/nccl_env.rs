//! The run's effective NCCL environment in the results document:
//! rendering, and run-to-run drift for baseline comparisons.
//!
//! The env is recorded once per run (`RunResults.nccl_env`), not per host:
//! the orchestrator resolves one map from one config and sends that same
//! map to every host and every NCCL-creating invocation, so per-host copies
//! could never disagree. What *can* change is the tuning between runs,
//! which is exactly what makes two runs' NCCL numbers incomparable — hence
//! the drift view.
//!
//! Per-level effective envs (`RunResults.nccl_level_env`, schema v13) get
//! the same treatment: the table lists every level whose env differs from
//! the global one, and the drift view lists per-level changes that the
//! global drift does not already explain.

use std::collections::BTreeMap;

use crate::nccl_level::NcclLevel;

/// Effective env per level, as recorded in `RunResults.nccl_level_env`.
pub type LevelEnvMap = BTreeMap<NcclLevel, BTreeMap<String, String>>;

/// One key's difference between a baseline run's NCCL env and the current
/// run's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NcclEnvChange {
    Added {
        key: String,
        value: String,
    },
    Removed {
        key: String,
        value: String,
    },
    Changed {
        key: String,
        from: String,
        to: String,
    },
}

impl NcclEnvChange {
    pub fn key(&self) -> &str {
        match self {
            NcclEnvChange::Added { key, .. }
            | NcclEnvChange::Removed { key, .. }
            | NcclEnvChange::Changed { key, .. } => key,
        }
    }

    /// One-line human form: "+K=v", "-K=v", "K: a -> b".
    pub fn describe(&self) -> String {
        match self {
            NcclEnvChange::Added { key, value } => format!("+{key}={value}"),
            NcclEnvChange::Removed { key, value } => format!("-{key}={value}"),
            NcclEnvChange::Changed { key, from, to } => format!("{key}: {from} -> {to}"),
        }
    }
}

/// Every key that differs between `baseline` and `current`, in key order.
/// `Some(empty)` means the two runs used identical NCCL tuning. `None` when
/// either run did not record its env (pre-v10 documents): comparing against
/// an unknown would invent drift, so there is nothing to say.
pub fn nccl_env_drift(
    baseline: Option<&BTreeMap<String, String>>,
    current: Option<&BTreeMap<String, String>>,
) -> Option<Vec<NcclEnvChange>> {
    Some(env_drift(baseline?, current?))
}

/// Per-level drift: for every level whose effective env changed between
/// the runs, the changes not already listed in `global_drift` (a changed
/// global key shows up once, globally, not once per level). Levels without
/// remaining changes are omitted, so `Some(empty)` means no level-specific
/// drift. `None` when either run did not record per-level envs (pre-v13).
pub fn nccl_level_env_drift(
    baseline: Option<&LevelEnvMap>,
    current: Option<&LevelEnvMap>,
    global_drift: &[NcclEnvChange],
) -> Option<BTreeMap<NcclLevel, Vec<NcclEnvChange>>> {
    let (baseline, current) = (baseline?, current?);
    let empty = BTreeMap::new();
    let mut levels: Vec<NcclLevel> = baseline.keys().chain(current.keys()).copied().collect();
    levels.sort();
    levels.dedup();
    Some(
        levels
            .into_iter()
            .filter_map(|level| {
                let changes: Vec<NcclEnvChange> = env_drift(
                    baseline.get(&level).unwrap_or(&empty),
                    current.get(&level).unwrap_or(&empty),
                )
                .into_iter()
                .filter(|change| !global_drift.contains(change))
                .collect();
                (!changes.is_empty()).then_some((level, changes))
            })
            .collect(),
    )
}

/// Levels whose effective env differs from the global env, with that
/// effective env rendered by `format_nccl_env`, in level order. Empty when
/// no level has an override (or nothing was recorded).
pub fn format_level_overrides(
    global: Option<&BTreeMap<String, String>>,
    levels: Option<&LevelEnvMap>,
) -> Vec<(NcclLevel, String)> {
    let (Some(global), Some(levels)) = (global, levels) else {
        return Vec::new();
    };
    levels
        .iter()
        .filter(|(_, env)| *env != global)
        .map(|(level, env)| (*level, format_nccl_env(Some(env))))
        .collect()
}

fn env_drift(
    baseline: &BTreeMap<String, String>,
    current: &BTreeMap<String, String>,
) -> Vec<NcclEnvChange> {
    let mut keys: Vec<&String> = baseline.keys().chain(current.keys()).collect();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .filter_map(|key| match (baseline.get(key), current.get(key)) {
            (None, Some(value)) => Some(NcclEnvChange::Added {
                key: key.clone(),
                value: value.clone(),
            }),
            (Some(value), None) => Some(NcclEnvChange::Removed {
                key: key.clone(),
                value: value.clone(),
            }),
            (Some(from), Some(to)) if from != to => Some(NcclEnvChange::Changed {
                key: key.clone(),
                from: from.clone(),
                to: to.clone(),
            }),
            _ => None,
        })
        .collect()
}

/// "K=v K=v" in key order, "(none)" for an untuned run, or
/// "(not recorded)" for a document that predates the field.
pub fn format_nccl_env(env: Option<&BTreeMap<String, String>>) -> String {
    let Some(env) = env else {
        return "(not recorded)".to_string();
    };
    if env.is_empty() {
        return "(none)".to_string();
    }
    env.iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    #[test]
    fn identical_envs_have_no_drift() {
        let tuned = env(&[("NCCL_DEBUG", "WARN"), ("NCCL_SOCKET_IFNAME", "bond0")]);
        assert_eq!(nccl_env_drift(Some(&tuned), Some(&tuned)), Some(vec![]));
        let untuned = BTreeMap::new();
        assert_eq!(nccl_env_drift(Some(&untuned), Some(&untuned)), Some(vec![]));
    }

    #[test]
    fn drift_reports_added_removed_and_changed_keys_in_order() {
        let baseline = env(&[
            ("NCCL_ALGO", "Ring"),
            ("NCCL_IB_HCA", "mlx5_0"),
            ("NCCL_SOCKET_IFNAME", "bond0"),
        ]);
        let current = env(&[
            ("NCCL_DEBUG", "INFO"),
            ("NCCL_IB_HCA", "mlx5_0,mlx5_1"),
            ("NCCL_SOCKET_IFNAME", "bond0"),
        ]);
        let drift = nccl_env_drift(Some(&baseline), Some(&current)).expect("both recorded");
        assert_eq!(
            drift,
            vec![
                NcclEnvChange::Removed {
                    key: "NCCL_ALGO".into(),
                    value: "Ring".into()
                },
                NcclEnvChange::Added {
                    key: "NCCL_DEBUG".into(),
                    value: "INFO".into()
                },
                NcclEnvChange::Changed {
                    key: "NCCL_IB_HCA".into(),
                    from: "mlx5_0".into(),
                    to: "mlx5_0,mlx5_1".into()
                },
            ]
        );
        let described: Vec<String> = drift.iter().map(NcclEnvChange::describe).collect();
        assert_eq!(
            described,
            [
                "-NCCL_ALGO=Ring",
                "+NCCL_DEBUG=INFO",
                "NCCL_IB_HCA: mlx5_0 -> mlx5_0,mlx5_1"
            ]
        );
        assert_eq!(drift[2].key(), "NCCL_IB_HCA");
    }

    #[test]
    fn unrecorded_envs_report_no_drift_at_all() {
        // A pre-v10 baseline says nothing about its tuning: no invented
        // "+NCCL_SOCKET_IFNAME" against it, in either direction.
        let tuned = env(&[("NCCL_SOCKET_IFNAME", "bond0")]);
        assert_eq!(nccl_env_drift(None, Some(&tuned)), None);
        assert_eq!(nccl_env_drift(Some(&tuned), None), None);
        assert_eq!(nccl_env_drift(None, None), None);
    }

    #[test]
    fn an_untuned_baseline_is_real_drift() {
        // Some(empty) is a recorded, untuned run: tuning added since is drift.
        let drift = nccl_env_drift(
            Some(&BTreeMap::new()),
            Some(&env(&[("NCCL_SOCKET_IFNAME", "bond0")])),
        );
        assert_eq!(
            drift,
            Some(vec![NcclEnvChange::Added {
                key: "NCCL_SOCKET_IFNAME".into(),
                value: "bond0".into()
            }])
        );
    }

    fn levels(entries: &[(NcclLevel, &[(&str, &str)])]) -> LevelEnvMap {
        entries
            .iter()
            .map(|(level, pairs)| (*level, env(pairs)))
            .collect()
    }

    #[test]
    fn level_drift_omits_unchanged_levels_and_global_changes() {
        let baseline = levels(&[
            (
                NcclLevel::Intranode,
                &[("NCCL_ALGO", "Ring"), ("NCCL_DEBUG", "WARN")],
            ),
            (NcclLevel::Fleet, &[("NCCL_DEBUG", "WARN")]),
            (NcclLevel::Barrier, &[("NCCL_DEBUG", "WARN")]),
        ]);
        let current = levels(&[
            (
                NcclLevel::Intranode,
                &[("NCCL_ALGO", "Tree"), ("NCCL_DEBUG", "INFO")],
            ),
            (
                NcclLevel::Fleet,
                &[("NCCL_DEBUG", "INFO"), ("NCCL_P2P_DISABLE", "1")],
            ),
            (NcclLevel::Barrier, &[("NCCL_DEBUG", "INFO")]),
        ]);
        // NCCL_DEBUG changed globally: reported there, not per level.
        let global = nccl_env_drift(
            Some(&env(&[("NCCL_DEBUG", "WARN")])),
            Some(&env(&[("NCCL_DEBUG", "INFO")])),
        )
        .expect("recorded");
        let drift =
            nccl_level_env_drift(Some(&baseline), Some(&current), &global).expect("recorded");
        assert_eq!(
            drift,
            BTreeMap::from([
                (
                    NcclLevel::Intranode,
                    vec![NcclEnvChange::Changed {
                        key: "NCCL_ALGO".into(),
                        from: "Ring".into(),
                        to: "Tree".into()
                    }]
                ),
                (
                    NcclLevel::Fleet,
                    vec![NcclEnvChange::Added {
                        key: "NCCL_P2P_DISABLE".into(),
                        value: "1".into()
                    }]
                ),
            ])
        );
        assert_eq!(
            nccl_level_env_drift(Some(&baseline), Some(&baseline), &[]),
            Some(BTreeMap::new())
        );
    }

    #[test]
    fn level_drift_needs_both_runs_recorded() {
        let recorded = levels(&[(NcclLevel::Fleet, &[("NCCL_ALGO", "Ring")])]);
        assert_eq!(nccl_level_env_drift(None, Some(&recorded), &[]), None);
        assert_eq!(nccl_level_env_drift(Some(&recorded), None, &[]), None);
    }

    #[test]
    fn only_levels_that_differ_from_the_global_env_are_listed() {
        let global = env(&[("NCCL_DEBUG", "WARN")]);
        let recorded = levels(&[
            (
                NcclLevel::Intranode,
                &[("NCCL_ALGO", "Ring"), ("NCCL_DEBUG", "WARN")],
            ),
            (NcclLevel::Fleet, &[("NCCL_DEBUG", "WARN")]),
        ]);
        assert_eq!(
            format_level_overrides(Some(&global), Some(&recorded)),
            vec![(
                NcclLevel::Intranode,
                "NCCL_ALGO=Ring NCCL_DEBUG=WARN".to_string()
            )]
        );
        assert!(format_level_overrides(None, Some(&recorded)).is_empty());
        assert!(format_level_overrides(Some(&global), None).is_empty());
    }

    #[test]
    fn formatting_is_key_ordered_and_marks_untuned_runs() {
        assert_eq!(format_nccl_env(None), "(not recorded)");
        assert_eq!(format_nccl_env(Some(&BTreeMap::new())), "(none)");
        assert_eq!(
            format_nccl_env(Some(&env(&[
                ("NCCL_SOCKET_IFNAME", "bond0"),
                ("NCCL_DEBUG", "WARN")
            ]))),
            "NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=bond0"
        );
    }
}
