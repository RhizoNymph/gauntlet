//! Remote filesystem layout for a session: resolving `ssh.remote_dir` on
//! the node and staging uploads so a destination file is only ever
//! replaced by an atomic rename of a complete file. Pure command builders,
//! tested against a real `sh`.

use super::{shell_path, single_quote};
use crate::remote_dir::{RemoteDir, RemoteDirPart};

/// Upload file mode, applied before the rename so the destination is
/// never visible with the wrong bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FileMode {
    Executable,
    Regular,
}

impl FileMode {
    fn octal(self) -> &'static str {
        match self {
            FileMode::Executable => "755",
            FileMode::Regular => "644",
        }
    }
}

/// Sibling temp name for one upload: `<dest>.tmp.<16 hex>`. Same directory
/// as the destination, so the final `mv` is a same-filesystem rename(2).
pub(super) fn staging_path(dest: &str, nonce: u64) -> String {
    format!("{dest}.tmp.{nonce:016x}")
}

/// A fresh nonce per upload: randomly keyed std hasher over the clock, the
/// pid and a process-wide counter. Collisions only matter between
/// concurrent uploaders of one path (different processes or hosts), which
/// differ in key, pid or clock.
pub(super) fn upload_nonce() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos())
            .unwrap_or_default(),
    );
    hasher.write_u32(std::process::id());
    hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    hasher.finish()
}

/// Set the mode on the staged file, then atomically rename it over `dest`.
pub(super) fn install_command(staging: &str, dest: &str, mode: FileMode) -> String {
    format!(
        "chmod {mode} {staging} && mv -f {staging} {dest}",
        mode = mode.octal(),
        staging = single_quote(staging),
        dest = single_quote(dest),
    )
}

/// `remote_dir` as one POSIX-sh word: literal runs quoted by `shell_path`
/// (the first may carry a leading `~`) or `single_quote`, each `$USER`
/// replaced by `"$(id -un)"` — the remote login name, whether or not the
/// login environment exports USER.
pub(super) fn remote_dir_word(remote_dir: &RemoteDir) -> String {
    remote_dir
        .parts()
        .into_iter()
        .enumerate()
        .map(|(index, part)| match part {
            RemoteDirPart::Literal(text) if index == 0 => shell_path(text),
            RemoteDirPart::Literal(text) => single_quote(text),
            RemoteDirPart::User => "\"$(id -un)\"".to_string(),
        })
        .collect()
}

/// Create the scratch directory (mode 700 when this creates it), refuse it
/// when it is not the login user's, and print its *physical* path.
///
/// - A world-writable parent like /tmp lets anyone pre-create
///   `/tmp/gauntlet-<you>` (or a symlink by that name) and plant a binary,
///   so the directory must be owned by the login user (`[ -O . ]` after
///   `cd`), and when the last component is a symlink, the link itself must
///   be the user's too (`find -user` does not follow it).
/// - The path printed — and later used for every upload and exec — is
///   `pwd -P`, the directory that was checked, not a logical path whose
///   symlinks could be repointed afterwards.
pub(super) fn remote_dir_script(remote_dir: &RemoteDir) -> String {
    let word = remote_dir_word(remote_dir);
    format!(
        "d={word}; \
         case \"$d\" in /) ;; */) d=\"${{d%/}}\" ;; esac; \
         mkdir -p -m 700 \"$d\" || exit 1; \
         if [ -L \"$d\" ] && [ -z \"$(find \"$d\" -maxdepth 0 -user \"$(id -u)\" -print)\" ]; then \
         echo \"remote_dir $d is a symlink not owned by $(id -un)\" >&2; exit 1; fi; \
         cd \"$d\" || exit 1; \
         [ -O . ] || {{ echo \"remote_dir $(pwd -P) is not owned by $(id -un)\" >&2; exit 1; }}; \
         pwd -P"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gauntlet-session-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn sh(script: &str, home: &std::path::Path) -> std::process::Output {
        std::process::Command::new("sh")
            .arg("-c")
            .arg(script)
            .env("HOME", home)
            .output()
            .expect("run sh")
    }

    fn login_name() -> String {
        let output = std::process::Command::new("id")
            .arg("-un")
            .output()
            .expect("id -un");
        String::from_utf8(output.stdout)
            .expect("utf8")
            .trim()
            .to_string()
    }

    #[test]
    fn user_expands_on_the_remote_shell_and_nothing_else_does() {
        let dir = RemoteDir::parse("/tmp/gauntlet-$USER/a b/'q'").expect("valid");
        let word = remote_dir_word(&dir);
        assert_eq!(word, r#"'/tmp/gauntlet-'"$(id -un)"'/a b/'\''q'\'''"#);
        let home = scratch("word");
        let output = sh(&format!("printf '%s' {word}"), &home);
        assert!(output.status.success());
        assert_eq!(
            String::from_utf8(output.stdout).expect("utf8"),
            format!("/tmp/gauntlet-{}/a b/'q'", login_name())
        );
        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn remote_dir_script_creates_a_private_per_user_dir() {
        use std::os::unix::fs::PermissionsExt;
        let base = scratch("script");
        let template = format!("{}/g-$USER", base.display());
        let dir = RemoteDir::parse(&template).expect("valid");
        let output = sh(&remote_dir_script(&dir), &base);
        assert!(output.status.success(), "{output:?}");
        let resolved = String::from_utf8(output.stdout).expect("utf8");
        let expected = base.join(format!("g-{}", login_name()));
        assert_eq!(
            std::path::Path::new(resolved.trim())
                .canonicalize()
                .expect("exists"),
            expected.canonicalize().expect("exists")
        );
        let mode = std::fs::metadata(&expected)
            .expect("dir")
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o700, "{mode:o}");
        // Idempotent: a second resolve of the existing, owned dir succeeds.
        assert!(sh(&remote_dir_script(&dir), &base).status.success());
        std::fs::remove_dir_all(&base).expect("cleanup");
    }

    #[test]
    fn an_explicit_home_remote_dir_still_resolves_under_home() {
        let home = scratch("home");
        let dir = RemoteDir::parse("~/.gauntlet").expect("valid");
        let output = sh(&remote_dir_script(&dir), &home);
        assert!(output.status.success(), "{output:?}");
        let resolved = String::from_utf8(output.stdout).expect("utf8");
        assert_eq!(
            std::path::Path::new(resolved.trim())
                .canonicalize()
                .expect("exists"),
            home.join(".gauntlet").canonicalize().expect("exists")
        );
        std::fs::remove_dir_all(&home).expect("cleanup");
    }

    #[test]
    fn the_resolved_path_is_physical_through_an_owned_symlink() {
        let base = scratch("physical");
        let real = base.join("real");
        std::fs::create_dir_all(&real).expect("real dir");
        let link = base.join("link");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");
        let physical = real.canonicalize().expect("canonical");
        for spelled in [link.display().to_string(), format!("{}/", link.display())] {
            let dir = RemoteDir::parse(&spelled).expect("valid");
            let output = sh(&remote_dir_script(&dir), &base);
            assert!(output.status.success(), "{spelled}: {output:?}");
            // The exact string, not a canonicalized comparison: the path
            // stored and used later must already be the checked one.
            assert_eq!(
                String::from_utf8(output.stdout).expect("utf8").trim(),
                physical.to_str().expect("utf8"),
                "{spelled}"
            );
        }
        std::fs::remove_dir_all(&base).expect("cleanup");
    }

    #[test]
    fn a_symlink_owned_by_someone_else_is_refused() {
        // `/bin` is a root-owned symlink on merged-/usr systems; it needs a
        // non-root test user and such a system to exercise this branch.
        let is_symlink = std::fs::symlink_metadata("/bin")
            .map(|meta| meta.file_type().is_symlink())
            .unwrap_or(false);
        if login_name() == "root" || !is_symlink {
            return;
        }
        let dir = RemoteDir::parse("/bin").expect("valid");
        let output = sh(&remote_dir_script(&dir), &std::env::temp_dir());
        assert!(!output.status.success(), "{output:?}");
        assert!(output.stdout.is_empty(), "no path may be reported");
        let stderr = String::from_utf8(output.stderr).expect("utf8");
        assert!(stderr.contains("is a symlink not owned by"), "{stderr}");
    }

    #[test]
    fn staging_names_are_unique_siblings_of_the_destination() {
        let dest = "/tmp/gauntlet-u/bin/gauntlet-agent";
        let a = staging_path(dest, upload_nonce());
        let b = staging_path(dest, upload_nonce());
        assert_ne!(a, b);
        for staging in [&a, &b] {
            let (dir, name) = staging.rsplit_once('/').expect("path");
            assert_eq!(dir, "/tmp/gauntlet-u/bin");
            assert!(name.starts_with("gauntlet-agent.tmp."), "{name}");
            assert_eq!(name.len(), "gauntlet-agent.tmp.".len() + 16, "{name}");
        }
        assert_eq!(staging_path("/x/y", 0xab), "/x/y.tmp.00000000000000ab");
    }

    #[test]
    fn install_renames_a_complete_file_into_place() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("install");
        let dest = dir.join("bin dir/gauntlet-agent");
        std::fs::create_dir_all(dest.parent().expect("parent")).expect("mkdir");
        let dest = dest.to_str().expect("utf8").to_string();
        std::fs::write(&dest, b"old binary").expect("old");

        // Two uploaders racing on one shared path each stage their own file;
        // whichever renames last wins, and the destination is always one
        // complete file, never a mix.
        let first = staging_path(&dest, upload_nonce());
        let second = staging_path(&dest, upload_nonce());
        std::fs::write(&first, b"new binary").expect("stage 1");
        std::fs::write(&second, b"new binary").expect("stage 2");
        // A reader holding the old file keeps its contents across the rename.
        let held = std::fs::File::open(&dest).expect("hold old");

        for staging in [&first, &second] {
            let output = sh(&install_command(staging, &dest, FileMode::Executable), &dir);
            assert!(output.status.success(), "{output:?}");
            assert!(!std::path::Path::new(staging).exists(), "{staging}");
            assert_eq!(std::fs::read(&dest).expect("dest"), b"new binary");
            let mode = std::fs::metadata(&dest).expect("dest").permissions().mode();
            assert_eq!(mode & 0o777, 0o755, "{mode:o}");
        }
        let mut old = String::new();
        std::io::Read::read_to_string(&mut &held, &mut old).expect("read held");
        assert_eq!(old, "old binary");

        let regular = staging_path(&dest, upload_nonce());
        std::fs::write(&regular, b"data").expect("stage");
        assert!(
            sh(&install_command(&regular, &dest, FileMode::Regular), &dir)
                .status
                .success()
        );
        let mode = std::fs::metadata(&dest).expect("dest").permissions().mode();
        assert_eq!(mode & 0o777, 0o644, "{mode:o}");
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    #[test]
    fn a_failed_install_leaves_the_destination_untouched() {
        let dir = scratch("failed");
        let dest = dir
            .join("gauntlet-agent")
            .to_str()
            .expect("utf8")
            .to_string();
        std::fs::write(&dest, b"old binary").expect("old");
        // The staged file never arrived (upload killed before close).
        let staging = staging_path(&dest, upload_nonce());
        let output = sh(
            &install_command(&staging, &dest, FileMode::Executable),
            &dir,
        );
        assert!(!output.status.success());
        assert_eq!(std::fs::read(&dest).expect("dest"), b"old binary");
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}
