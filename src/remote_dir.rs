//! `ssh.remote_dir`: the per-node scratch directory, as a validated
//! template.
//!
//! Two expansions happen on the node, never on the operator's machine: a
//! leading `~` (the remote `$HOME`) and `$USER` / `${USER}` (the remote
//! login name, via `id -un`). Every other character is literal, and any
//! other `$` is rejected up front, so a typo cannot silently become a
//! literal `$FOO` directory or an expansion nobody asked for.
//!
//! The default is node-local (`/tmp/gauntlet-$USER`): a home directory is
//! often NFS on clusters, and a shared `remote_dir` makes every node upload
//! the agent onto the same file at once.

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Default scratch directory: node-local, per user.
pub const DEFAULT_REMOTE_DIR: &str = "/tmp/gauntlet-$USER";

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RemoteDirError {
    #[error("remote_dir is empty")]
    Empty,
    #[error("remote_dir contains a control character")]
    ControlCharacter,
    /// A `$` that does not start `$USER` / `${USER}`.
    #[error(
        "remote_dir {raw:?}: unsupported `$` at byte {at}; only a leading `~` and \
         `$USER` / `${{USER}}` are expanded"
    )]
    UnsupportedVariable { raw: String, at: usize },
}

/// One piece of a parsed template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteDirPart<'a> {
    /// Literal text; the first part may start with `~` / `~/` (home).
    Literal(&'a str),
    /// The remote login name.
    User,
}

/// A validated `remote_dir` template. Construct with `parse` (or serde).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct RemoteDir(String);

const USER_TOKENS: [&str; 2] = ["${USER}", "$USER"];

impl RemoteDir {
    pub fn parse(raw: &str) -> Result<RemoteDir, RemoteDirError> {
        if raw.is_empty() {
            return Err(RemoteDirError::Empty);
        }
        if raw.chars().any(char::is_control) {
            return Err(RemoteDirError::ControlCharacter);
        }
        // Walk the template once; `split` rejects stray `$`.
        split(raw)?;
        Ok(RemoteDir(raw.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Literal runs and `$USER` references, in order. Never empty.
    pub fn parts(&self) -> Vec<RemoteDirPart<'_>> {
        // Validated at construction.
        split(&self.0).unwrap_or_default()
    }
}

/// Split at `$USER` / `${USER}`; `$USER` must not run into an identifier
/// character (`$USERNAME` is a different variable).
fn split(raw: &str) -> Result<Vec<RemoteDirPart<'_>>, RemoteDirError> {
    let mut parts = Vec::new();
    let mut literal_start = 0;
    let mut index = 0;
    while let Some(offset) = raw[index..].find('$') {
        let at = index + offset;
        let rest = &raw[at..];
        let token = USER_TOKENS
            .iter()
            .find(|token| rest.starts_with(**token))
            .filter(|token| {
                **token == "${USER}"
                    || !rest[token.len()..]
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
            })
            .ok_or_else(|| RemoteDirError::UnsupportedVariable {
                raw: raw.to_string(),
                at,
            })?;
        if at > literal_start {
            parts.push(RemoteDirPart::Literal(&raw[literal_start..at]));
        }
        parts.push(RemoteDirPart::User);
        index = at + token.len();
        literal_start = index;
    }
    if literal_start < raw.len() {
        parts.push(RemoteDirPart::Literal(&raw[literal_start..]));
    }
    Ok(parts)
}

impl Default for RemoteDir {
    fn default() -> Self {
        RemoteDir(DEFAULT_REMOTE_DIR.to_string())
    }
}

impl TryFrom<String> for RemoteDir {
    type Error = RemoteDirError;

    fn try_from(raw: String) -> Result<Self, Self::Error> {
        RemoteDir::parse(&raw)
    }
}

impl From<RemoteDir> for String {
    fn from(dir: RemoteDir) -> String {
        dir.0
    }
}

impl fmt::Display for RemoteDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_node_local_and_per_user() {
        let dir = RemoteDir::default();
        assert_eq!(dir.as_str(), "/tmp/gauntlet-$USER");
        assert_eq!(
            dir.parts(),
            vec![
                RemoteDirPart::Literal("/tmp/gauntlet-"),
                RemoteDirPart::User
            ]
        );
        assert_eq!(RemoteDir::parse(DEFAULT_REMOTE_DIR), Ok(dir));
    }

    #[test]
    fn user_references_split_out() {
        let dir = RemoteDir::parse("/scratch/${USER}/g/$USER").expect("valid");
        assert_eq!(
            dir.parts(),
            vec![
                RemoteDirPart::Literal("/scratch/"),
                RemoteDirPart::User,
                RemoteDirPart::Literal("/g/"),
                RemoteDirPart::User,
            ]
        );
        let dir = RemoteDir::parse("$USER-x").expect("valid");
        assert_eq!(
            dir.parts(),
            vec![RemoteDirPart::User, RemoteDirPart::Literal("-x")]
        );
    }

    #[test]
    fn explicit_home_paths_stay_valid_and_literal() {
        let dir = RemoteDir::parse("~/.gauntlet").expect("valid");
        assert_eq!(dir.parts(), vec![RemoteDirPart::Literal("~/.gauntlet")]);
        let dir = RemoteDir::parse("/opt/g x").expect("valid");
        assert_eq!(dir.parts(), vec![RemoteDirPart::Literal("/opt/g x")]);
    }

    #[test]
    fn other_dollars_and_bad_values_are_rejected() {
        for bad in [
            "/tmp/$HOME",
            "/tmp/$USERNAME",
            "/tmp/$",
            "$(id)",
            "/x/${USER",
        ] {
            assert!(
                matches!(
                    RemoteDir::parse(bad),
                    Err(RemoteDirError::UnsupportedVariable { .. })
                ),
                "{bad}"
            );
        }
        assert_eq!(RemoteDir::parse(""), Err(RemoteDirError::Empty));
        assert_eq!(
            RemoteDir::parse("/tmp/a\nb"),
            Err(RemoteDirError::ControlCharacter)
        );
    }

    #[test]
    fn serde_round_trips_and_validates() {
        #[derive(Debug, Deserialize, Serialize)]
        struct Doc {
            dir: RemoteDir,
        }
        let doc: Doc = toml::from_str(r#"dir = "/tmp/g-$USER""#).expect("valid");
        assert_eq!(doc.dir.as_str(), "/tmp/g-$USER");
        assert_eq!(
            toml::to_string(&doc).expect("serialize").trim(),
            r#"dir = "/tmp/g-$USER""#
        );
        assert!(toml::from_str::<Doc>(r#"dir = "/tmp/$HOME""#).is_err());
    }
}
