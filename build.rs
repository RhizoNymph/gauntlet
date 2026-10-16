//! Capture the git revision this binary is built from as
//! `GAUNTLET_GIT_REVISION`: `<short sha>`, `<short sha>-dirty`, or
//! `unknown` (no git, not a checkout of this crate — e.g. a source
//! tarball, possibly unpacked inside some other repository).
//!
//! Nothing else is embedded — no build time, hostname or path — so the
//! release binary is a pure function of the commit and the pinned toolchain
//! (see docs/features/release_ci.md, "Reproducible builds").

use std::path::{Path, PathBuf};
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    // `--no-optional-locks`: `status` would otherwise refresh the index
    // and take index.lock, racing any git command the developer runs while
    // cargo builds.
    let output = Command::new("git")
        .arg("--no-optional-locks")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|text| text.trim().to_string())
}

fn revision(manifest_dir: &Path) -> Option<String> {
    // Only trust a repository whose top level *is* this crate: a tarball
    // unpacked inside another checkout must not report that checkout's sha.
    let toplevel = PathBuf::from(git(manifest_dir, &["rev-parse", "--show-toplevel"])?);
    let same = match (toplevel.canonicalize(), manifest_dir.canonicalize()) {
        (Ok(top), Ok(manifest)) => top == manifest,
        _ => false,
    };
    if !same {
        return None;
    }
    // The full sha cut to exactly 12 characters, not `--short=12`: git
    // lengthens an abbreviation that is ambiguous among the objects of the
    // local clone, so a shallow CI checkout and a full clone could embed
    // different strings and the release binary would not rebuild
    // bit-for-bit.
    let full = git(manifest_dir, &["rev-parse", "HEAD"])?;
    let sha = full.get(..12)?.to_string();
    // Tracked changes only: untracked scratch files do not change the build.
    let dirty = git(
        manifest_dir,
        &["status", "--porcelain", "--untracked-files=no"],
    )
    .map(|status| !status.is_empty())?;
    Some(if dirty { format!("{sha}-dirty") } else { sha })
}

fn main() {
    let manifest_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"),
    );
    let revision = revision(&manifest_dir).unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=GAUNTLET_GIT_REVISION={revision}");

    // Re-run when the commit or the compiled tree changes. `src` is scanned
    // recursively, so an edit flips the dirty flag on the next build. The
    // git index is deliberately not watched: `git status` rewrites it, and
    // every such touch would otherwise rebuild the crate.
    for path in ["build.rs", "Cargo.toml", "Cargo.lock", "src", "viewer/src"] {
        println!("cargo:rerun-if-changed={path}");
    }
    if let Some(git_dir) = git(&manifest_dir, &["rev-parse", "--absolute-git-dir"]) {
        let git_dir = PathBuf::from(git_dir);
        println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
        // Branch tips live under the common dir (shared by worktrees).
        if let Some(common) = git(&manifest_dir, &["rev-parse", "--git-common-dir"]) {
            let common = manifest_dir.join(common);
            for name in ["refs/heads", "packed-refs"] {
                println!("cargo:rerun-if-changed={}", common.join(name).display());
            }
        }
    }
}
