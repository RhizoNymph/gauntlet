//! The run's effective NCCL environment in the results document:
//! rendering, and run-to-run drift for baseline comparisons.
//!
//! The env is recorded once per run (`RunResults.nccl_env`), not per host:
//! the orchestrator resolves one map from one config and sends that same
//! map to every host and every NCCL-creating invocation, so per-host copies
//! could never disagree. What *can* change is the tuning between runs,
//! which is exactly what makes two runs' NCCL numbers incomparable — hence
//! the drift view.

use std::collections::BTreeMap;

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
/// either run did not record its env (pre-v8 documents): comparing against
/// an unknown would invent drift, so there is nothing to say.
pub fn nccl_env_drift(
    baseline: Option<&BTreeMap<String, String>>,
    current: Option<&BTreeMap<String, String>>,
) -> Option<Vec<NcclEnvChange>> {
    Some(env_drift(baseline?, current?))
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
        // A pre-v8 baseline says nothing about its tuning: no invented
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
