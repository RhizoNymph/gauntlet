# Release CI

## Scope

Turning a `vX.Y.Z` tag into published artifacts:

- Two Debian packages, `gauntlet` (CLI) and `gauntlet-view` (GUI viewer).
- Publication of those packages into the existing reprepro-managed APT
  repository on the `apt-repo` branch of `RhizoNymph/sysdui`, served over
  GitHub Pages at `https://rhizonymph.github.io/sysdui`.
- A GitHub release carrying both `.deb` files and both `x86_64` tarballs.
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

## Control flow

```
git tag vX.Y.Z && git push --tags
        |
        v
   [verify]  resolve tag -> version, check both manifests match the tag,
        |    cargo test --workspace --locked
        v
   [build]   cargo deb -p gauntlet-bench
        |    cargo deb -p gauntlet-view
        |    tarballs from target/release
        |    -> uploads the `packages` artifact
        |
        +--------------------+--------------------+
        v                    v                    v
  [publish-apt]      [github-release]      [publish-crates]
  reprepro into      gh release with       gauntlet-bench, then
  sysdui@apt-repo    dist/*                gauntlet-view
```

`verify` is a hard gate: a tag whose version disagrees with the manifests
fails before anything is pushed anywhere. The three terminal jobs run in
parallel because they write to disjoint destinations.

## Files

| File | Role |
| --- | --- |
| `.github/workflows/release.yml` | The whole pipeline. Triggered by `push` on `v*` tags, plus `workflow_dispatch` with a `tag` input for re-running a release. |
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
- **The runner image sets the glibc floor.** Binaries are dynamically linked
  against the builder's glibc, so `dpkg-shlibdeps` stamps a `libc6 (>= X)`
  dependency taken from whatever `ubuntu-latest` currently is (24.04 →
  glibc 2.39). Nodes older than the runner cannot install the package. If
  the fleet ever runs an older distribution than the runner, pin
  `runs-on:` to the oldest supported image rather than relaxing the
  dependency. The same floor applies to the orchestrator's ssh deploy,
  which uploads the operator's own binary to each node.
