#!/usr/bin/env bash
# Package a CLI built by scripts/build-release-cli.sh into the three release
# artifacts, byte-for-byte as release CI does:
#
#   <out>/gauntlet-<version>-<target>          the raw binary
#   <out>/gauntlet-<version>-<target>.tar.gz   the binary in a tarball
#   <out>/gauntlet_<version>-1_amd64.deb        the glibc-floor .deb
#
#   CLI_TARGET=x86_64-unknown-linux-gnu scripts/package-release-cli.sh 0.1.0 dist
#
# Every timestamp comes from SOURCE_DATE_EPOCH, which is always the commit
# time of HEAD, so a rebuild of the same tag reproduces the same archives.
# Requires the cargo-deb version pinned in .github/workflows/release.yml.
set -euo pipefail

: "${CLI_TARGET:?set CLI_TARGET, e.g. x86_64-unknown-linux-gnu}"
version=${1:?usage: package-release-cli.sh <version> <out dir>}
out=${2:?usage: package-release-cli.sh <version> <out dir>}

SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)
export SOURCE_DATE_EPOCH

bin_dir="${CARGO_TARGET_DIR:-target}/$CLI_TARGET/release"
name="gauntlet-$version-$CLI_TARGET"
mkdir -p "$out"

cp "$bin_dir/gauntlet" "$out/$name"

# Fixed order, owner and mtime; `gzip -n` omits the name and time gzip
# would otherwise record in its header.
tar --sort=name --format=gnu --owner=0 --group=0 --numeric-owner \
  --mtime="@$SOURCE_DATE_EPOCH" -C "$bin_dir" -cf - gauntlet \
  | gzip -9n >"$out/$name.tar.gz"

# - The binary is packaged as is (--no-build); see Cargo.toml for why the
#   glibc-floor variant declares libc6 explicitly.
# - --no-strip ships the exact binary published above instead of a copy
#   rewritten by the host's binutils `strip`, whose output depends on its
#   version. The release profile already leaves no debug info to strip.
# - cargo-deb takes archive mtimes from SOURCE_DATE_EPOCH.
cargo deb -p gauntlet-bench --no-build --no-strip --target "$CLI_TARGET" \
  --variant glibc-floor --output "$out/"
