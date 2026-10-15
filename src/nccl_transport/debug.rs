//! How a parsable NCCL INFO log is obtained without overriding the user.
//!
//! Three variables decide what NCCL logs and where: NCCL_DEBUG (level),
//! NCCL_DEBUG_SUBSYS (subsystem mask) and NCCL_DEBUG_FILE (destination;
//! `%h` expands to the short hostname, `%p` to the pid). Precedence:
//!
//! 1. A variable set by the level's config env is never overridden.
//! 2. NCCL_DEBUG unset -> gauntlet adds `NCCL_DEBUG=INFO`, and, when
//!    NCCL_DEBUG_SUBSYS is unset too, `NCCL_DEBUG_SUBSYS=INIT,NET` (only
//!    what the parser reads, so the log stays small). A user's own
//!    NCCL_DEBUG=INFO keeps NCCL's default mask (which includes INIT).
//! 3. NCCL_DEBUG_FILE unset and the effective level is INFO or TRACE ->
//!    gauntlet adds `NCCL_DEBUG_FILE=<remote_dir>/nccl-logs/<level>.%h.log`
//!    (one file per host and level, overwritten by the next run). A user
//!    who set NCCL_DEBUG=INFO without a file therefore finds the log in
//!    that file instead of the agent's stderr.
//! 4. The config's NCCL_DEBUG below INFO (VERSION, WARN, ...) or a subsys
//!    mask without INIT -> nothing is added and the transport is recorded
//!    as unknown with the reason.
//!
//! The agent decides from the variables actually in its environment
//! (`log_source`), so the orchestrator's additions and the agent's reading
//! cannot disagree.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::UnknownTransport;
use crate::nccl_env::{NcclEnv, NcclEnvError};
use crate::nccl_level::NcclLevel;

pub const NCCL_DEBUG: &str = "NCCL_DEBUG";
pub const NCCL_DEBUG_SUBSYS: &str = "NCCL_DEBUG_SUBSYS";
pub const NCCL_DEBUG_FILE: &str = "NCCL_DEBUG_FILE";

/// The level gauntlet adds when the config sets none.
pub const CAPTURE_LEVEL: &str = "INFO";
/// The subsystem mask gauntlet adds alongside `CAPTURE_LEVEL`: INIT carries
/// every line the parser reads; NET adds the network-plugin detail.
pub const CAPTURE_SUBSYS: &str = "INIT,NET";
/// Directory under `remote_dir` holding gauntlet-managed NCCL logs.
pub const LOG_DIR: &str = "nccl-logs";

/// The three NCCL debug variables as some environment holds them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DebugSettings<'a> {
    pub level: Option<&'a str>,
    pub subsys: Option<&'a str>,
    pub file: Option<&'a str>,
}

impl<'a> DebugSettings<'a> {
    /// The settings a level's config env makes.
    pub fn from_env(env: &'a NcclEnv) -> Self {
        Self {
            level: env.get(NCCL_DEBUG),
            subsys: env.get(NCCL_DEBUG_SUBSYS),
            file: env.get(NCCL_DEBUG_FILE),
        }
    }
}

/// Whether an NCCL_DEBUG value makes NCCL write INFO lines (NCCL compares
/// case-insensitively; TRACE includes INFO).
pub fn level_logs_info(level: &str) -> bool {
    let level = level.trim();
    level.eq_ignore_ascii_case("INFO") || level.eq_ignore_ascii_case("TRACE")
}

/// Whether an NCCL_DEBUG_SUBSYS value lets INIT lines through. Unset is
/// NCCL's default mask, which includes INIT. A leading `^` inverts the
/// list; `ALL` matches every subsystem; names compare case-insensitively.
pub fn subsys_admits_init(subsys: Option<&str>) -> bool {
    let Some(subsys) = subsys.map(str::trim) else {
        return true;
    };
    let (inverted, list) = match subsys.strip_prefix('^') {
        Some(rest) => (true, rest),
        None => (false, subsys),
    };
    let names_init = list
        .split(',')
        .map(str::trim)
        .any(|name| name.eq_ignore_ascii_case("INIT") || name.eq_ignore_ascii_case("ALL"));
    names_init != inverted
}

/// `<remote_dir>/nccl-logs/<level>.%h.log`: NCCL expands `%h`, so hosts
/// sharing a remote_dir (NFS home) never write the same file.
pub fn managed_log_path(remote_dir: &str, level: NcclLevel) -> String {
    format!("{remote_dir}/{LOG_DIR}/{level}.%h.log")
}

/// The variables gauntlet adds to a level's spawn env so NCCL writes a
/// parsable log, in key order. Never names a variable `user` already sets.
/// Adds nothing at all when the config's own settings rule capture out
/// (a level below INFO, or a subsystem mask without INIT): capture is then
/// unknown anyway, and gauntlet must not switch on logging nobody asked
/// for.
pub fn capture_additions(user: DebugSettings<'_>, log_path: &str) -> Vec<(&'static str, String)> {
    let mut added = Vec::new();
    let effective_level = user.level.unwrap_or(CAPTURE_LEVEL);
    if !level_logs_info(effective_level) || !subsys_admits_init(user.subsys) {
        return added;
    }
    if user.level.is_none() {
        added.push((NCCL_DEBUG, CAPTURE_LEVEL.to_string()));
        if user.subsys.is_none() {
            added.push((NCCL_DEBUG_SUBSYS, CAPTURE_SUBSYS.to_string()));
        }
    }
    if user.file.is_none() {
        added.push((NCCL_DEBUG_FILE, log_path.to_string()));
    }
    added.sort_by_key(|(key, _)| *key);
    added
}

/// `capture_additions` as a validated `NcclEnv`, ready to overlay on the
/// level's env (its keys never collide with the env's own). Fails only
/// when the log path is not a valid env value (a control character in
/// remote_dir).
pub fn capture_env(user: DebugSettings<'_>, log_path: &str) -> Result<NcclEnv, NcclEnvError> {
    let raw: BTreeMap<String, String> = capture_additions(user, log_path)
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
    NcclEnv::from_map(&raw)
}

/// Where NCCL wrote this process's INFO log, from the variables in the
/// process env, or why it wrote none. `hostname` is the full kernel
/// hostname (NCCL's `%h` uses the part before the first dot).
pub fn log_source(
    settings: DebugSettings<'_>,
    hostname: &str,
    pid: u32,
) -> Result<PathBuf, UnknownTransport> {
    match settings.level {
        Some(level) if level_logs_info(level) => {}
        level => {
            return Err(UnknownTransport::DebugLevel {
                level: level.map(str::to_string),
            });
        }
    }
    if !subsys_admits_init(settings.subsys) {
        return Err(UnknownTransport::SubsysExcludesInit {
            subsys: settings.subsys.unwrap_or_default().to_string(),
        });
    }
    let file = settings.file.ok_or(UnknownTransport::NoDebugFile)?;
    Ok(PathBuf::from(expand_debug_file(
        file,
        short_hostname(hostname),
        pid,
    )))
}

/// NCCL's NCCL_DEBUG_FILE expansion: `%h` -> hostname, `%p` -> pid; any
/// other `%x` (and a trailing `%`) stays literal.
pub fn expand_debug_file(pattern: &str, hostname: &str, pid: u32) -> String {
    let mut out = String::with_capacity(pattern.len() + hostname.len());
    let mut chars = pattern.chars();
    while let Some(ch) = chars.next() {
        if ch != '%' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('h') => out.push_str(hostname),
            Some('p') => out.push_str(&pid.to_string()),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

/// The hostname up to the first dot, as NCCL's `getHostName(.., '.')`.
pub fn short_hostname(hostname: &str) -> &str {
    hostname.split('.').next().unwrap_or(hostname)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PATH: &str = "/g/nccl-logs/fleet.%h.log";

    fn settings<'a>(
        level: Option<&'a str>,
        subsys: Option<&'a str>,
        file: Option<&'a str>,
    ) -> DebugSettings<'a> {
        DebugSettings {
            level,
            subsys,
            file,
        }
    }

    #[test]
    fn an_untouched_config_gets_all_three_variables() {
        assert_eq!(
            capture_additions(DebugSettings::default(), PATH),
            vec![
                (NCCL_DEBUG, "INFO".to_string()),
                (NCCL_DEBUG_FILE, PATH.to_string()),
                (NCCL_DEBUG_SUBSYS, "INIT,NET".to_string()),
            ]
        );
    }

    #[test]
    fn user_values_are_never_overridden() {
        // The user's INFO keeps NCCL's default subsys; only the file is added.
        assert_eq!(
            capture_additions(settings(Some("INFO"), None, None), PATH),
            vec![(NCCL_DEBUG_FILE, PATH.to_string())]
        );
        // The user's subsys is kept; gauntlet supplies the level and file.
        assert_eq!(
            capture_additions(settings(None, Some("INIT,GRAPH"), None), PATH),
            vec![
                (NCCL_DEBUG, "INFO".to_string()),
                (NCCL_DEBUG_FILE, PATH.to_string())
            ]
        );
        // The user's own file: nothing about the destination changes.
        assert_eq!(
            capture_additions(settings(Some("INFO"), None, Some("/x/%h.%p")), PATH),
            vec![]
        );
        // WARN: gauntlet adds nothing (no file either: that would move the
        // user's WARN output off stderr), and capture becomes unknown.
        assert_eq!(
            capture_additions(settings(Some("WARN"), None, None), PATH),
            vec![]
        );
        assert_eq!(
            capture_additions(settings(Some("trace"), None, None), PATH),
            vec![(NCCL_DEBUG_FILE, PATH.to_string())]
        );
    }

    #[test]
    fn a_subsys_without_init_adds_no_logging_at_all() {
        // Rule 4: capture is impossible, so no NCCL_DEBUG=INFO nobody asked
        // for and no file.
        for subsys in ["NET,GRAPH", "^INIT", "COLL"] {
            assert_eq!(
                capture_additions(settings(None, Some(subsys), None), PATH),
                vec![],
                "{subsys}"
            );
            assert_eq!(
                log_source(settings(None, Some(subsys), None), "n", 1),
                Err(UnknownTransport::DebugLevel { level: None }),
                "{subsys}"
            );
        }
        assert_eq!(
            capture_additions(settings(Some("INFO"), Some("GRAPH"), None), PATH),
            vec![]
        );
    }

    #[test]
    fn capture_env_is_a_validated_env() {
        let env = capture_env(DebugSettings::default(), PATH).expect("valid");
        assert_eq!(env.get(NCCL_DEBUG), Some("INFO"));
        assert_eq!(env.get(NCCL_DEBUG_FILE), Some(PATH));
        assert_eq!(env.len(), 3);
        assert!(capture_env(DebugSettings::default(), "/bad\npath").is_err());
        assert!(
            capture_env(settings(Some("WARN"), None, None), PATH)
                .expect("valid")
                .is_empty()
        );
    }

    #[test]
    fn additions_from_a_real_level_env() {
        let raw = [("NCCL_DEBUG".to_string(), "WARN".to_string())]
            .into_iter()
            .collect();
        let env = NcclEnv::from_map(&raw).expect("env");
        assert_eq!(
            capture_additions(DebugSettings::from_env(&env), PATH),
            vec![]
        );
        let empty = NcclEnv::default();
        assert_eq!(
            capture_additions(DebugSettings::from_env(&empty), PATH).len(),
            3
        );
    }

    #[test]
    fn subsys_masks() {
        assert!(subsys_admits_init(None));
        assert!(subsys_admits_init(Some("INIT,NET")));
        assert!(subsys_admits_init(Some("init")));
        assert!(subsys_admits_init(Some("ALL")));
        assert!(subsys_admits_init(Some("NET, INIT")));
        assert!(!subsys_admits_init(Some("NET,GRAPH")));
        assert!(subsys_admits_init(Some("^NET")));
        assert!(!subsys_admits_init(Some("^INIT")));
        assert!(!subsys_admits_init(Some("^ALL")));
    }

    #[test]
    fn info_levels() {
        for level in ["INFO", "info", " Info ", "TRACE"] {
            assert!(level_logs_info(level), "{level}");
        }
        for level in ["WARN", "VERSION", "ABORT", "NONE", ""] {
            assert!(!level_logs_info(level), "{level}");
        }
    }

    #[test]
    fn the_agent_finds_the_file_nccl_wrote() {
        let path = log_source(
            settings(Some("INFO"), Some("INIT,NET"), Some(PATH)),
            "node0.cluster.local",
            4242,
        );
        assert_eq!(path, Ok(PathBuf::from("/g/nccl-logs/fleet.node0.log")));
        let path = log_source(settings(Some("INFO"), None, Some("/l/%h-%p.txt")), "n1", 7);
        assert_eq!(path, Ok(PathBuf::from("/l/n1-7.txt")));
    }

    #[test]
    fn the_agent_explains_a_missing_log() {
        assert_eq!(
            log_source(settings(None, None, Some(PATH)), "n", 1),
            Err(UnknownTransport::DebugLevel { level: None })
        );
        assert_eq!(
            log_source(settings(Some("WARN"), None, None), "n", 1),
            Err(UnknownTransport::DebugLevel {
                level: Some("WARN".into())
            })
        );
        assert_eq!(
            log_source(settings(Some("INFO"), Some("GRAPH"), Some(PATH)), "n", 1),
            Err(UnknownTransport::SubsysExcludesInit {
                subsys: "GRAPH".into()
            })
        );
        assert_eq!(
            log_source(settings(Some("INFO"), None, None), "n", 1),
            Err(UnknownTransport::NoDebugFile)
        );
    }

    #[test]
    fn expansion_matches_nccls() {
        assert_eq!(
            expand_debug_file("/a/%h.%p.log", "node0", 12),
            "/a/node0.12.log"
        );
        assert_eq!(expand_debug_file("/a/100%%", "n", 1), "/a/100%%");
        assert_eq!(expand_debug_file("/a/%x%", "n", 1), "/a/%x%");
        assert_eq!(expand_debug_file("plain", "n", 1), "plain");
        assert_eq!(short_hostname("node0.cluster.local"), "node0");
        assert_eq!(short_hostname("node0"), "node0");
    }

    #[test]
    fn managed_paths_are_per_level_and_host_expanded() {
        assert_eq!(
            managed_log_path("/home/u/.gauntlet", NcclLevel::OverlapFleet),
            "/home/u/.gauntlet/nccl-logs/overlap_fleet.%h.log"
        );
    }
}
