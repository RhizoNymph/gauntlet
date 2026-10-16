#!/usr/bin/env bash
# Build the release `gauntlet` CLI exactly as release CI does, so the result
# is bit-for-bit identical to the published binary for the same commit.
#
#   CLI_TARGET=x86_64-unknown-linux-gnu GLIBC_FLOOR=2.31 scripts/build-release-cli.sh
#
# Output: ${CARGO_TARGET_DIR:-target}/$CLI_TARGET/release/gauntlet
#
# Requires the toolchain pinned in rust-toolchain.toml (rustup selects it),
# plus the zig and cargo-zigbuild versions pinned in
# .github/workflows/release.yml. Run it from a clean git checkout of the
# tag: build.rs embeds the commit sha, `-dirty` for a modified tree, or
# `unknown` without git metadata, and each of those changes the binary.
# See docs/features/release_ci.md, "Reproducible builds".
set -euo pipefail

: "${CLI_TARGET:?set CLI_TARGET, e.g. x86_64-unknown-linux-gnu}"
: "${GLIBC_FLOOR:?set GLIBC_FLOOR, e.g. 2.31}"

# Rewrite every absolute source path rustc would embed (panic locations,
# file!() in dependencies) to a location-independent one: workspace files
# become relative, registry crates become `<registry>/<crate>-<ver>/...`,
# the sysroot becomes `/rustc/<commit-hash>`. Without this the binary embeds
# $CARGO_HOME, so two builders with different home directories disagree.
# Set here rather than in Cargo.toml so it needs no `cargo-features` entry,
# which would make the published crate unbuildable on stable.
export CARGO_UNSTABLE_TRIM_PATHS=true
export CARGO_PROFILE_RELEASE_TRIM_PATHS=all

cargo zigbuild --release --locked -p gauntlet-bench \
  --target "$CLI_TARGET.$GLIBC_FLOOR"
