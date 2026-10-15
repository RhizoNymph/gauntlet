//! What binary produced a document: crate version plus the git revision
//! captured at build time by `build.rs` (`GAUNTLET_GIT_REVISION`).

use std::fmt;

use serde::{Deserialize, Serialize};

/// Raw revision string from `build.rs`: `<sha>`, `<sha>-dirty` or
/// `unknown`.
const RAW_REVISION: &str = env!("GAUNTLET_GIT_REVISION");

/// `gauntlet --version` output: `<crate version> (<revision>)`.
pub const VERSION_LINE: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("GAUNTLET_GIT_REVISION"),
    ")"
);

/// The git revision a binary was built from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum GitRevision {
    /// Built from a checkout of this repository at `sha`; `dirty` when
    /// tracked files differed from that commit.
    Known { sha: String, dirty: bool },
    /// No git metadata at build time (source tarball, no git binary).
    Unknown,
}

impl GitRevision {
    /// Parse the `build.rs` encoding. Anything that is not a hex sha
    /// (optionally `-dirty`) is `Unknown`, so a malformed value can never
    /// pose as a revision.
    pub fn parse(raw: &str) -> GitRevision {
        let (sha, dirty) = match raw.strip_suffix("-dirty") {
            Some(sha) => (sha, true),
            None => (raw, false),
        };
        if sha.len() >= 7 && sha.chars().all(|c| c.is_ascii_hexdigit()) {
            GitRevision::Known {
                sha: sha.to_ascii_lowercase(),
                dirty,
            }
        } else {
            GitRevision::Unknown
        }
    }
}

impl fmt::Display for GitRevision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GitRevision::Known { sha, dirty: false } => f.write_str(sha),
            GitRevision::Known { sha, dirty: true } => write!(f, "{sha}-dirty"),
            GitRevision::Unknown => f.write_str("unknown"),
        }
    }
}

/// Crate version plus git revision; recorded run-level in `RunResults`
/// and in the bootstrap report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildInfo {
    /// `CARGO_PKG_VERSION`.
    pub version: String,
    pub git: GitRevision,
}

impl BuildInfo {
    /// This binary's build info.
    pub fn current() -> BuildInfo {
        BuildInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            git: GitRevision::parse(RAW_REVISION),
        }
    }
}

impl fmt::Display for BuildInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.version, self.git)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_clean_dirty_and_unknown() {
        assert_eq!(
            GitRevision::parse("0123456789ab"),
            GitRevision::Known {
                sha: "0123456789ab".into(),
                dirty: false
            }
        );
        assert_eq!(
            GitRevision::parse("0123456789ab-dirty"),
            GitRevision::Known {
                sha: "0123456789ab".into(),
                dirty: true
            }
        );
        assert_eq!(GitRevision::parse("unknown"), GitRevision::Unknown);
        assert_eq!(GitRevision::parse(""), GitRevision::Unknown);
        assert_eq!(GitRevision::parse("-dirty"), GitRevision::Unknown);
        assert_eq!(GitRevision::parse("abc"), GitRevision::Unknown);
        assert_eq!(GitRevision::parse("not-a-sha!"), GitRevision::Unknown);
    }

    #[test]
    fn display_round_trips_through_parse() {
        for raw in ["0123456789ab", "0123456789ab-dirty", "unknown"] {
            assert_eq!(GitRevision::parse(raw).to_string(), raw);
        }
    }

    #[test]
    fn version_line_matches_the_current_build_info() {
        let current = BuildInfo::current();
        assert_eq!(current.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(VERSION_LINE, current.to_string());
    }

    #[test]
    fn serializes_as_a_tagged_object() {
        let info = BuildInfo {
            version: "0.1.0".into(),
            git: GitRevision::Known {
                sha: "abcdef0".into(),
                dirty: true,
            },
        };
        let json = serde_json::to_value(&info).expect("serialize");
        assert_eq!(
            json,
            serde_json::json!({
                "version": "0.1.0",
                "git": { "state": "known", "sha": "abcdef0", "dirty": true }
            })
        );
        let unknown = serde_json::to_value(GitRevision::Unknown).expect("serialize");
        assert_eq!(unknown, serde_json::json!({ "state": "unknown" }));
        let back: BuildInfo = serde_json::from_value(json).expect("parse");
        assert_eq!(back, info);
    }
}
