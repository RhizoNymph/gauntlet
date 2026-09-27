//! Validated NCCL environment passthrough.
//!
//! `[nccl] env` in the fleet config lets operators hand NCCL the same tuning
//! the training job runs with (NCCL_IB_HCA, NCCL_NET_GDR_LEVEL, NCCL_ALGO,
//! NCCL_DEBUG, ...). NCCL reads its knobs through `getenv` at communicator
//! init, so the orchestrator places the resolved map on the remote
//! `env K=V ... gauntlet agent ...` command line of every agent spawn: the
//! variables are in the process environment before the agent starts any
//! thread. (The agent never calls `set_var`; it runs on a multi-threaded
//! tokio runtime where that would be unsound.)
//!
//! Only keys matching `^NCCL_[A-Z0-9_]+$` are representable. The map is a
//! tuning channel for NCCL, not a general way to reshape the agent's
//! process: LD_PRELOAD, LD_LIBRARY_PATH, CUDA_VISIBLE_DEVICES and friends
//! would change what every phase measures (and what code runs) in ways the
//! results document cannot show, so they are rejected rather than passed
//! through. Values must be non-empty (an empty value is almost always a
//! templating mistake, and NCCL treats "set but empty" inconsistently
//! across knobs) and NUL-free (an environment string cannot carry an
//! interior NUL). The key charset also means a key is a literal shell word;
//! values are quoted by the command-line builder.
//!
//! Every constructor validates, so an unvalidated map cannot reach a
//! command line.

use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::fmt;

use thiserror::Error;

/// The one NCCL knob that also has a first-class config field
/// (`[nccl] socket_ifname`).
pub const SOCKET_IFNAME: &str = "NCCL_SOCKET_IFNAME";

const KEY_PREFIX: &str = "NCCL_";

/// Why a value was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum NcclEnvValueError {
    #[error("value is empty")]
    Empty,
    #[error("value contains a NUL byte")]
    ContainsNul,
}

/// Why an NCCL env entry (or the config section producing it) was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NcclEnvError {
    #[error("NCCL env key {key:?} is not allowed: keys must match ^NCCL_[A-Z0-9_]+$")]
    InvalidKey { key: String },
    #[error("NCCL env value for {key} is invalid: {reason}")]
    InvalidValue {
        key: String,
        reason: NcclEnvValueError,
    },
    #[error(
        "NCCL_SOCKET_IFNAME is set by both [nccl] socket_ifname and [nccl] env; set it in one place"
    )]
    SocketIfnameConflict,
}

/// An environment variable name NCCL owns: `^NCCL_[A-Z0-9_]+$`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NcclEnvKey(String);

impl NcclEnvKey {
    pub fn parse(key: &str) -> Result<Self, NcclEnvError> {
        let valid = key.strip_prefix(KEY_PREFIX).is_some_and(|rest| {
            !rest.is_empty()
                && rest
                    .bytes()
                    .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
        });
        if valid {
            Ok(Self(key.to_string()))
        } else {
            Err(NcclEnvError::InvalidKey {
                key: key.to_string(),
            })
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// Sound: the derived `Ord` compares the inner `String`, which orders exactly
// like its `str`.
impl Borrow<str> for NcclEnvKey {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NcclEnvKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A non-empty, NUL-free environment value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NcclEnvValue(String);

impl NcclEnvValue {
    pub fn parse(value: &str) -> Result<Self, NcclEnvValueError> {
        if value.is_empty() {
            Err(NcclEnvValueError::Empty)
        } else if value.contains('\0') {
            Err(NcclEnvValueError::ContainsNul)
        } else {
            Ok(Self(value.to_string()))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A validated NCCL environment: what every agent process is started with.
/// Iteration order is the key order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NcclEnv(BTreeMap<NcclEnvKey, NcclEnvValue>);

impl NcclEnv {
    /// Validate a raw name -> value map.
    pub fn from_map(raw: &BTreeMap<String, String>) -> Result<Self, NcclEnvError> {
        raw.iter()
            .map(|(key, value)| validate_entry(key, value))
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map(Self)
    }

    /// The single resolved map for the wire: the `[nccl] env` entries plus
    /// `socket_ifname` folded in as NCCL_SOCKET_IFNAME. Setting
    /// NCCL_SOCKET_IFNAME in both places is a conflict, never a precedence
    /// rule.
    pub fn resolve(
        socket_ifname: Option<&str>,
        env: &BTreeMap<String, String>,
    ) -> Result<Self, NcclEnvError> {
        let mut resolved = Self::from_map(env)?;
        if let Some(ifname) = socket_ifname {
            if resolved.get(SOCKET_IFNAME).is_some() {
                return Err(NcclEnvError::SocketIfnameConflict);
            }
            let (key, value) = validate_entry(SOCKET_IFNAME, ifname)?;
            resolved.0.insert(key, value);
        }
        Ok(resolved)
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(NcclEnvValue::as_str)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Entries in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
    }

    /// Plain string map, for the results document.
    pub fn to_string_map(&self) -> BTreeMap<String, String> {
        self.iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }
}

fn validate_entry(key: &str, value: &str) -> Result<(NcclEnvKey, NcclEnvValue), NcclEnvError> {
    let key = NcclEnvKey::parse(key)?;
    let value = NcclEnvValue::parse(value).map_err(|reason| NcclEnvError::InvalidValue {
        key: key.to_string(),
        reason,
    })?;
    Ok((key, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
        entries
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    #[test]
    fn nccl_keys_are_accepted() {
        for key in [
            "NCCL_IB_HCA",
            "NCCL_IB_GID_INDEX",
            "NCCL_NET_GDR_LEVEL",
            "NCCL_P2P_LEVEL",
            "NCCL_ALGO",
            "NCCL_PROTO",
            "NCCL_DEBUG",
            "NCCL_DEBUG_SUBSYS",
            "NCCL_CROSS_NIC",
            "NCCL_SOCKET_IFNAME",
            "NCCL_1",
            "NCCL__",
        ] {
            let parsed = NcclEnvKey::parse(key).unwrap_or_else(|error| panic!("{key}: {error}"));
            assert_eq!(parsed.as_str(), key);
        }
    }

    #[test]
    fn non_nccl_keys_are_rejected() {
        for key in [
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "PATH",
            "",
            "NCCL",
            "NCCL_",
            "nccl_debug",
            "NCCL_debug",
            "NCCL_IB-HCA",
            "NCCL_IB HCA",
            " NCCL_DEBUG",
            "NCCL_DEBUG=INFO",
            "XNCCL_DEBUG",
            "NCCL_DÉBUG",
            "NCCL_DEBUG\0",
        ] {
            assert_eq!(
                NcclEnvKey::parse(key),
                Err(NcclEnvError::InvalidKey {
                    key: key.to_string()
                }),
                "{key:?} must be rejected"
            );
        }
    }

    #[test]
    fn values_must_be_non_empty_and_nul_free() {
        assert_eq!(NcclEnvValue::parse(""), Err(NcclEnvValueError::Empty));
        assert_eq!(
            NcclEnvValue::parse("mlx5_0\0mlx5_1"),
            Err(NcclEnvValueError::ContainsNul)
        );
        // Anything else is NCCL's business: commas, carets, spaces, '='.
        for value in ["mlx5_0,mlx5_1", "^docker0,lo", "=mlx5_0:1", "INFO", "a b"] {
            assert_eq!(
                NcclEnvValue::parse(value).map(|v| v.as_str().to_string()),
                Ok(value.to_string())
            );
        }
    }

    #[test]
    fn from_map_reports_the_offending_entry() {
        let error = NcclEnv::from_map(&map(&[("NCCL_DEBUG", "WARN"), ("LD_PRELOAD", "x.so")]))
            .expect_err("LD_PRELOAD must be rejected");
        assert_eq!(
            error,
            NcclEnvError::InvalidKey {
                key: "LD_PRELOAD".into()
            }
        );

        let error = NcclEnv::from_map(&map(&[("NCCL_IB_HCA", "")])).expect_err("empty value");
        assert_eq!(
            error,
            NcclEnvError::InvalidValue {
                key: "NCCL_IB_HCA".into(),
                reason: NcclEnvValueError::Empty,
            }
        );
    }

    #[test]
    fn socket_ifname_folds_into_the_resolved_map() {
        let resolved =
            NcclEnv::resolve(Some("bond0"), &map(&[("NCCL_DEBUG", "WARN")])).expect("resolves");
        assert_eq!(resolved.get(SOCKET_IFNAME), Some("bond0"));
        assert_eq!(resolved.get("NCCL_DEBUG"), Some("WARN"));
        assert_eq!(resolved.len(), 2);

        // socket_ifname alone: exactly the pre-passthrough behavior.
        let only = NcclEnv::resolve(Some("bond0"), &BTreeMap::new()).expect("resolves");
        assert_eq!(
            only.iter().collect::<Vec<_>>(),
            vec![(SOCKET_IFNAME, "bond0")]
        );

        // NCCL_SOCKET_IFNAME through the map alone is fine too.
        let via_map = NcclEnv::resolve(None, &map(&[(SOCKET_IFNAME, "bond0")])).expect("resolves");
        assert_eq!(via_map, only);

        assert!(
            NcclEnv::resolve(None, &BTreeMap::new())
                .expect("empty")
                .is_empty()
        );
    }

    #[test]
    fn socket_ifname_in_both_places_conflicts() {
        // Even when the two agree: one source of truth, no precedence rules.
        for env_value in ["bond0", "eth0"] {
            assert_eq!(
                NcclEnv::resolve(Some("bond0"), &map(&[(SOCKET_IFNAME, env_value)])),
                Err(NcclEnvError::SocketIfnameConflict)
            );
        }
    }

    #[test]
    fn socket_ifname_is_validated_as_a_value() {
        assert_eq!(
            NcclEnv::resolve(Some(""), &BTreeMap::new()),
            Err(NcclEnvError::InvalidValue {
                key: SOCKET_IFNAME.into(),
                reason: NcclEnvValueError::Empty,
            })
        );
        assert_eq!(
            NcclEnv::resolve(Some("bo\0nd0"), &BTreeMap::new()),
            Err(NcclEnvError::InvalidValue {
                key: SOCKET_IFNAME.into(),
                reason: NcclEnvValueError::ContainsNul,
            })
        );
    }

    #[test]
    fn to_string_map_mirrors_the_entries() {
        let env = NcclEnv::resolve(Some("bond0"), &map(&[("NCCL_ALGO", "Ring")])).expect("ok");
        assert_eq!(
            env.to_string_map(),
            map(&[("NCCL_ALGO", "Ring"), (SOCKET_IFNAME, "bond0")])
        );
    }
}
