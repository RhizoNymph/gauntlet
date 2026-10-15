# Gauntlet

Cluster pre-flight benchmark and health check.

Gauntlet reads a fleet of hosts from a TOML config, deploys itself to each node
over ssh, and runs a phased suite of correctness and performance tests — CPU,
DRAM, disk, GPU, intra/inter-node network, and NCCL collectives. It emits a
schema-versioned JSON document plus a human-readable table, using
fleet-relative outlier detection (median + MAD) as the primary straggler signal
with optional absolute thresholds as a secondary overlay.

Results feed simulation calibration: per-node roofline numbers and alpha/beta
network fits per link class.

## Install

### From APT (Debian/Ubuntu)

```sh
curl -fsSL https://rhizonymph.github.io/sysdui/gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/sysdui.gpg
echo "deb [signed-by=/usr/share/keyrings/sysdui.gpg] https://rhizonymph.github.io/sysdui stable main" | sudo tee /etc/apt/sources.list.d/sysdui.list
sudo apt update
sudo apt install gauntlet
```

The desktop viewer is a separate package, so fleet nodes do not pull GUI
runtime dependencies:

```sh
sudo apt install gauntlet-view
```

### From crates.io

`gauntlet` was already taken on crates.io, so the crate publishes as
`gauntlet-bench`. The installed binary is still `gauntlet`.

```sh
cargo install gauntlet-bench     # CLI
cargo install gauntlet-view      # GUI viewer
```

### From source

Requires Rust 1.85+ (edition 2024).

```sh
cargo install --path .                 # CLI
cargo install --path viewer            # GUI viewer
```

No CUDA toolchain is needed to build: `libcuda`, `libcublas`, and `libnccl` are
dlopened at runtime, so only the driver stack has to be present on GPU nodes.
Build the CLI with `--no-default-features` to drop the GPU phases entirely.

## Usage

Describe the fleet in `gauntlet.toml` (see `gauntlet.example.toml`), then:

```sh
gauntlet bootstrap          # connectivity, arch check, agent deploy, capability probe
gauntlet run                # run the phase suite across the fleet
gauntlet report             # re-render a saved run
gauntlet-view               # browse, live-tail, and diff runs in the GUI
```

`gauntlet agent` is the node-side mode; the orchestrator invokes it over ssh,
and you do not normally run it by hand.

## Releases

Tagging `vX.Y.Z` builds both `.deb` packages, publishes them to the APT
repository, attaches them to a GitHub release, and publishes both crates to
crates.io. See `docs/features/release_ci.md`.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
