# gauntlet-view

Native desktop viewer for [gauntlet](https://github.com/RhizoNymph/gauntlet)
cluster benchmark runs, built with [gpui](https://www.gpui.rs/).

Renders a run as a fully connected fleet graph — node color is host health,
edge color is pairwise-path health — alongside per-direction edge details,
roofline cards, link fits, and a filterable quantitative metric table.

- Sidebar lists all runs; any run can be selected or pinned as a baseline.
- A 1s poll loop live-tails `<run_id>.partial.json` while a run is executing.
- ▶ launches `gauntlet run` directly; ✕ cancels a launched child.
- ⚙ runs `bootstrap --json` and renders the readiness matrix.
- Diff mode recolors the graph by regression against the baseline run
  (>5% warn, >15% bad, unit-aware direction of goodness).

## Install

```sh
sudo apt install gauntlet-view      # see the main README for repo setup
cargo install gauntlet-view
```

Building links `libxkbcommon` and `libxkbcommon-x11`; Wayland, Vulkan, and
fontconfig are dlopened at runtime.

```sh
sudo apt install libxkbcommon-dev libxkbcommon-x11-dev
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
