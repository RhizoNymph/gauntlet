//! Shared best-effort readers for /proc and /sys attribute files.

use std::path::Path;

/// The file's contents, trimmed. `None` when the file is missing,
/// unreadable, not UTF-8, or empty after trimming.
///
/// Empty is deliberately "no value", not `Some("")`: sysfs uses an empty
/// read for an unset attribute (an unpopulated GID's `ndevs` entry, a
/// driver that has nothing to report), and no caller can do anything with
/// an empty string that it would not also do with an absent one.
pub fn read_trimmed(path: impl AsRef<Path>) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_missing_are_none() {
        let dir = std::env::temp_dir().join(format!(
            "gauntlet-sysfs-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("empty"), "\n").expect("write");
        std::fs::write(dir.join("value"), "  4: ACTIVE\n").expect("write");
        assert_eq!(read_trimmed(dir.join("empty")), None);
        assert_eq!(read_trimmed(dir.join("missing")), None);
        assert_eq!(
            read_trimmed(dir.join("value")).as_deref(),
            Some("4: ACTIVE")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
