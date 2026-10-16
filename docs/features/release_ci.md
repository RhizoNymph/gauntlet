# Release CI

## Scope

Turning a `vX.Y.Z` tag into published artifacts:

- Two Debian packages, `gauntlet` (CLI) and `gauntlet-view` (GUI viewer).
- Publication of those packages into the existing reprepro-managed APT
  repository on the `apt-repo` branch of `RhizoNymph/sysdui`, served over
  GitHub Pages at `https://rhizonymph.github.io/sysdui`.
- A GitHub release carrying both `.deb` files, both `x86_64` tarballs, the
  raw CLI binary and a `SHA256SUMS` file covering all of them.
- Bit-for-bit reproducibility of the CLI artifacts (raw binary, tarball,
  `.deb`), enforced by an independent rebuild before anything is published.
- Publication of both crates to crates.io.

## Non-scope

- Per-commit / pull-request CI. The release workflow runs the test suite as a
  gate, but there is no separate push-triggered check workflow.
- Architectures other than `amd64`. The APT repository's
  `conf/distributions` declares `Architectures: amd64`, and the fleet is
  x86_64; cross-building would require adding an architecture there first.
- Creating or rotating the GPG signing key, or provisioning the APT
  repository. Both already exist and are owned by `sysdui`.
- Version bumping. The tag is the source of truth and the manifests must
  already agree with it; the workflow verifies but never edits versions.
- Reproducibility of the viewer (`gauntlet-view` binary, tarball, `.deb`).
  It is a native build at the runner's glibc that compiles C (vendored
  freetype) with the runner's system compiler; its archives get
  deterministic timestamps but nothing checks or guarantees its bytes.
- Signing the release or `SHA256SUMS`. The checksums are protected only by
  GitHub's release storage, and by anyone being able to rebuild them.

## Control flow

```
git tag vX.Y.Z && git push --tags
        |
        v
   [verify]  resolve tag -> version, check both manifests match the tag,
        |    rustup toolchain install (the rust-toolchain.toml pin),
        |    cargo test --workspace --locked
        v
   [build]   scripts/build-release-cli.sh
        |      (cargo zigbuild -p gauntlet-bench --target ...gnu.2.31,
        |       -Ztrim-paths, profile.release.trim-paths = "all")
        |    check newest GLIBC_* symbol <= 2.31
        |    SOURCE_DATE_EPOCH = commit time of the tag
        |    scripts/package-release-cli.sh -> raw binary, tarball,
        |      cargo deb --no-build --no-strip --variant glibc-floor
        |    cargo deb -p gauntlet-view (native) + viewer tarball
        |    check the .deb declares libc6 (>= GLIBC_FLOOR)
        |    sha256sum dist/* > dist/SHA256SUMS
        |    -> uploads the `packages` artifact
        v
   [reproduce]  fresh runner, checkout at rebuild/src/gauntlet, separate
        |       CARGO_HOME and CARGO_TARGET_DIR, no cache:
        |       both scripts again, every rebuilt CLI artifact's sha256
        |       must appear verbatim in dist/SHA256SUMS
        |
        +--------------------+--------------------+
        v                    v                    v
  [publish-apt]      [github-release]      [publish-crates]
  reprepro into      gh release with       gauntlet-bench, then
  sysdui@apt-repo    dist/*                gauntlet-view
```

`verify` is a hard gate: a tag whose version disagrees with the manifests
fails before anything is pushed anywhere. `reproduce` is the second gate: a
CLI build that does not reproduce publishes nothing. The three terminal jobs
run in parallel because they write to disjoint destinations.

## Files

| File | Role |
| --- | --- |
| `.github/workflows/release.yml` | The whole pipeline. Triggered by `push` on `v*` tags, plus `workflow_dispatch` with a `tag` input for re-running a release. Pins zig, cargo-zigbuild and cargo-deb versions and `GLIBC_FLOOR`. |
| `scripts/build-release-cli.sh` | The one definition of the release CLI build: `cargo zigbuild --release --locked -p gauntlet-bench --target $CLI_TARGET.$GLIBC_FLOOR` with path trimming enabled. Requires `CLI_TARGET` and `GLIBC_FLOOR`. Used by `build`, `reproduce` and humans. |
| `scripts/package-release-cli.sh` | `<version> <out dir>`: copies the raw binary, writes the deterministic tarball and the glibc-floor `.deb`, with `SOURCE_DATE_EPOCH` set to HEAD's commit time. Requires `CLI_TARGET`. |
| `rust-toolchain.toml` | Pins the compiler (`nightly-2026-08-20`) for every job and every rebuild; the workflow installs it with a bare `rustup toolchain install`. |
| `build.rs`, `src/build_info.rs` | Embed the 12-character commit sha (plus `-dirty` / `unknown`) in the binary; the only build-environment input that reaches its bytes. |
| `Cargo.toml` | Package `gauntlet-bench`; `[lib] name = "gauntlet"` and `[[bin]] name = "gauntlet"`; `[package.metadata.deb]` for the CLI package. |
| `viewer/Cargo.toml` | Package `gauntlet-view`; depends on `gauntlet-bench` under the alias `gauntlet`; `[package.metadata.deb]` for the viewer package. |
| `viewer/build.rs` | Pre-existing link shim for the unversioned `libxkbcommon-x11.so` symlink; a no-op when `libxkbcommon-x11-dev` is installed, as it is in CI. |
| `LICENSE-MIT`, `LICENSE-APACHE` | Dual license, duplicated into `viewer/` because crates.io and cargo-deb resolve `license-file` relative to the package directory. |
| `README.md`, `viewer/README.md` | crates.io front pages; also installed to `/usr/share/doc/<pkg>/`. |

## Naming

`gauntlet` was already registered on crates.io by an unrelated crate, so the
CLI publishes as **`gauntlet-bench`**. This is confined to the package name:

- `[lib] name = "gauntlet"` keeps every `use gauntlet::...` in the workspace
  working, including the 19 such imports in `viewer/`.
- `[[bin]] name = "gauntlet"` keeps the installed binary and all deployment
  paths unchanged — the orchestrator uploads and executes `gauntlet` on each
  node, and the agent protocol is unaffected.
- `[package.metadata.deb] name = "gauntlet"` keeps the Debian package name
  short; Debian and crates.io namespaces are independent.

`viewer/Cargo.toml` therefore declares
`gauntlet = { package = "gauntlet-bench", path = "..", version = "0.1.0" }`.

## Build dependencies

No CUDA toolchain is needed. `cudarc` is configured with `dynamic-loading`,
so `libcuda`/`libcublas`/`libnccl` are dlopened at runtime and the `gpu`
feature builds on a runner with no GPU and no driver.

For the viewer, the only build-time system dependency is xkbcommon: the
`xkbcommon` crate has no build script and links `libxkbcommon` and
`libxkbcommon-x11` via `#[link]`. gpui's other platform libraries — Wayland
(`wayland-backend` with the `dlopen` feature), Vulkan (via `blade-graphics`),
and fontconfig (`source-fontconfig-dlopen`) — are all dlopened, and
`freetype-sys` vendors and compiles freetype when pkg-config finds none. CI
installs `libxkbcommon-dev libxkbcommon-x11-dev pkg-config`.

Because dlopened libraries are invisible to `dpkg-shlibdeps`, the viewer's
`depends` names `libvulkan1`, `libfontconfig1` and `libwayland-client0`
explicitly alongside `$auto`.

## Required secrets

Set on the `RhizoNymph/gauntlet` repository:

| Secret | Purpose |
| --- | --- |
| `APT_REPO_TOKEN` | PAT with `contents: write` on `RhizoNymph/sysdui`. The default `GITHUB_TOKEN` is scoped to this repository only and cannot push to the APT repo. |
| `GPG_PRIVATE_KEY` | Base64-encoded private key for `4B1381326B1258D94B7559846BD38CA281A72F27`, the key named by `SignWith` in the APT repo's `conf/distributions`. Must be passphrase-less: reprepro signs non-interactively. |
| `CARGO_REGISTRY_TOKEN` | crates.io publish token. |

## Invariants and constraints

- **The tag is the version.** `vX.Y.Z` must equal the `version` of both
  `gauntlet-bench` and `gauntlet-view`, which are released in lockstep.
- **Publication order is fixed.** `gauntlet-view` depends on
  `gauntlet-bench`, so the latter must be on crates.io and visible in the
  index first. The workflow polls the crates.io API before continuing.
- **The APT repository is serialized.** `concurrency: group: release` with
  `cancel-in-progress: false` prevents two runs rewriting reprepro's `db/`
  in the `sysdui` branch at once, which would corrupt it.
- **Every publishing step is idempotent.** Re-running a release skips crates
  already on crates.io, removes a package from the APT suite before
  re-including it (reprepro rejects a re-add of the same version), and skips
  the git push when the branch is unchanged. This makes `workflow_dispatch`
  safe for resuming a partially failed release.
- **Only `amd64` is produced.** Adding an architecture requires editing
  `Architectures:` in the APT repo's `conf/distributions` first.
- **The CLI's glibc floor is 2.31, independent of the runner.** The
  `gauntlet` binary is what the orchestrator copies onto every node, so its
  glibc requirement decides which nodes can run an agent. It is therefore
  cross-linked with `cargo zigbuild --target x86_64-unknown-linux-gnu.2.31`
  rather than inheriting the runner's glibc (`ubuntu-latest` is 24.04 →
  2.39). That makes Ubuntu 20.04, Debian 11 and RHEL 9 nodes supported.
  - **Floor check.** A step reads the newest `GLIBC_*` symbol version from
    `objdump -T` and fails the release if it exceeds `GLIBC_FLOOR`. A
    dependency that starts calling a newer libc function then breaks the
    build instead of silently breaking old nodes.
  - **The .deb.** The CLI package wraps the zig-built binary as is
    (`cargo deb --no-build --target x86_64-unknown-linux-gnu --variant
    glibc-floor`). The `glibc-floor` variant in Cargo.toml declares
    `libc6 (>= 2.31)` explicitly instead of `$auto`. The binary links libdl
    and libpthread, which the runner's libc6 symbols file maps to 2.34 (the
    release that folded them into libc), so `dpkg-shlibdeps` would demand
    2.34 for a binary that runs on 2.31.
  - **Depends check.** A workflow step fails the release unless the built
    package's Depends contains `libc6 (>= $GLIBC_FLOOR)`, so the
    hand-written version cannot drift from the floor.
  - **Validation.** The zig-built binary and the installed .deb were run in
    an `ubuntu:20.04` container (glibc 2.31): `apt install` of the package
    succeeds and `gauntlet agent probe` produces a full inventory. A native
    24.04 build fails there with `GLIBC_2.32 not found`.
  - **The viewer.** `gauntlet-view` is still a native build at the runner's
    glibc. It is a desktop app that links gpui's xkbcommon and never runs on
    nodes.
  - **Raising or lowering the floor.** Change `GLIBC_FLOOR` in the workflow
    `env`. zig ships the glibc stubs, so no older runner image is needed.
- **A locally built binary keeps the builder's floor.** The orchestrator's
  ssh deploy uploads whatever binary the operator is running. A
  `cargo build` on a newer distro produces an agent that older nodes
  cannot load. Install the release artifact, or build with
  `CLI_TARGET=x86_64-unknown-linux-gnu GLIBC_FLOOR=2.31
  scripts/build-release-cli.sh` when deploying to older nodes.
- **Pinned build tools.** The compiler is pinned by `rust-toolchain.toml`
  and installed in every job by a bare `rustup toolchain install`, never a
  floating channel. `ZIG_VERSION` (installed from the `ziglang` PyPI wheel
  into `$RUNNER_TEMP`, since the runner's system Python is externally
  managed), `CARGO_ZIGBUILD_VERSION` and `CARGO_DEB_VERSION` are pinned in
  the workflow `env`. Each of them shapes the CLI artifacts' bytes, so
  changing any one changes the published checksums (expected; the
  `reproduce` job rebuilds with the same pins).
- **The CLI build is defined once.** `scripts/build-release-cli.sh` and
  `scripts/package-release-cli.sh` are what `build`, `reproduce` and a
  human rebuild all run; do not inline a variant of either command in the
  workflow.

## Reproducible builds

### What is guaranteed

For a given tag, these release assets are bit-for-bit reproducible, and
the `reproduce` job refuses to publish if they are not:

| Asset | Reproducible | Inputs that determine it |
| --- | --- | --- |
| `gauntlet-<ver>-x86_64-unknown-linux-gnu` (raw CLI) | yes | commit, rust toolchain, zig, cargo-zigbuild, `GLIBC_FLOOR` |
| `gauntlet-<ver>-x86_64-unknown-linux-gnu.tar.gz` | yes | the raw CLI + commit time; GNU tar and gzip |
| `gauntlet_<ver>-1_amd64.deb` | yes | the raw CLI + commit time + cargo-deb version |
| `gauntlet-view` tarball and `.deb` | no | native build; see Non-scope |
| `SHA256SUMS` | only its CLI lines | lists every asset, including the viewer's |

Verified locally (nightly-2026-08-20, zig 0.16.0, cargo-zigbuild 0.23.4,
cargo-deb 3.7.0): two clones of the same commit — one full, one
`--depth 1`, at different absolute paths, with different
`CARGO_TARGET_DIR`s and different `CARGO_HOME`s — produced an identical
raw binary, tarball and `.deb`.

### Non-determinism found and removed

- **`$CARGO_HOME` paths.** Panic locations and `file!()` in dependencies
  embed the absolute path of the registry source, e.g.
  `/home/<user>/.cargo/registry/src/index.crates.io-…/tokio-1.53.1/src/…`
  (233 such strings in the 0.1.0 CLI). Two builds on one machine agree, but any
  builder with a different home directory (CI's is `/home/runner`) does
  not; changing only `CARGO_HOME` changed the binary's sha256. Fixed with
  cargo's `trim-paths` (`-Ztrim-paths`, `profile.release.trim-paths =
  "all"`), set as environment variables by `build-release-cli.sh`.
  Registry sources then appear as `/cargo/registry/<hash>/<crate>-<ver>/…`,
  workspace sources as relative paths, and the sysroot as
  `/rustc/<commit>/…`. `trim-paths` was chosen over
  `--remap-path-prefix` in `RUSTFLAGS` because it covers every path kind
  without enumerating the builder's directories, and it leaves `RUSTFLAGS`
  free. It is not in `Cargo.toml` because that needs `cargo-features =
  ["trim-paths"]`, which would make the published crate unbuildable on
  stable.
- **The git revision.** `build.rs` used `git rev-parse --short=12`, which
  git lengthens when 12 characters are ambiguous among the local clone's
  objects, so a shallow CI checkout and a full clone could embed different
  strings. It now takes the full sha and cuts it to exactly 12 characters.
- **Archive timestamps and ownership.** `tar -czf` recorded the binary's
  build-time mtime and the runner's uid/gid. Tarballs are now written with
  `--sort=name --owner=0 --group=0 --numeric-owner
  --mtime=@$SOURCE_DATE_EPOCH`, piped through `gzip -9n` so the gzip
  header carries no name or timestamp. cargo-deb takes its archive mtimes
  from `SOURCE_DATE_EPOCH`; without it the `.deb`'s entries carry a
  build-time mtime and its sha256 differs. `SOURCE_DATE_EPOCH` is the tagged commit's
  committer time.
- **Host `strip`.** cargo-deb strips the binary with the host's binutils by
  default, so the packaged binary depended on the runner image. The CLI
  `.deb` is built with `--no-strip`; the release profile already strips
  debug info (the difference was 176 bytes), and the packaged binary is now
  the published raw binary byte for byte.
- **Toolchain drift.** The workflow used `dtolnay/rust-toolchain@stable`,
  which suggested stable but was silently overridden by
  `rust-toolchain.toml`. Every job now runs `rustup toolchain install`,
  which installs exactly the pinned toolchain and its components.

Checked and found already deterministic: `build.rs` embeds no time,
hostname or path (only the sha, `-dirty` and `unknown`); zig's lld emits no
`.note.gnu.build-id`; `codegen-units = 1` and `lto = true` in the release
profile and no incremental compilation in release builds; no C/C++ build
dependencies in the CLI's graph; the checkout path and target directory do
not reach the binary.

### Rebuilding a release

On x86_64 Linux with rustup, Python 3, git, GNU tar and gzip:

```sh
git clone https://github.com/RhizoNymph/gauntlet && cd gauntlet
git checkout vX.Y.Z                   # a clean tree: edits embed "-dirty"
rustup toolchain install              # the rust-toolchain.toml pin
python3 -m pip install --target "$HOME/.local/zig-0.16.0" ziglang==0.16.0
export PYTHONPATH="$HOME/.local/zig-0.16.0"
cargo install cargo-zigbuild --version =0.23.4 --locked
cargo install cargo-deb --version =3.7.0 --locked

export CLI_TARGET=x86_64-unknown-linux-gnu GLIBC_FLOOR=2.31
scripts/build-release-cli.sh
scripts/package-release-cli.sh X.Y.Z out

curl -LO https://github.com/RhizoNymph/gauntlet/releases/download/vX.Y.Z/SHA256SUMS
(cd out && sha256sum --check --ignore-missing ../SHA256SUMS)
```

Use the tool versions from the tag's `.github/workflows/release.yml` `env`
(and `GLIBC_FLOOR`) if they differ from the ones above. Build from a git
checkout, not a source tarball: without git metadata the binary embeds
`unknown` instead of the sha. `--ignore-missing` skips the viewer's
entries, which are not reproducible.
