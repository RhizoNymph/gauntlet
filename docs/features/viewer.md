# Viewer (`gauntlet-view`)

Native GPUI desktop app that renders a saved gauntlet run as a fully
connected fleet graph plus a quantitative metric table.

## Scope

- Load one `RunResults` JSON (a file, or the newest run in a directory —
  default `runs/`).
- Fleet graph: nodes on a ring, one edge per host pair. Node color encodes
  host health (green ok / amber outlier-or-skew / red failed); edge color
  encodes pairwise-path health the same way. Edge midpoints show worst-case
  direction bandwidth.
- Click a node or edge to see its findings (issues, roofline digest,
  per-direction bandwidth/RTT) and filter the metric table to that subject.
- Metric table: every fleet-comparable metric with per-sample deviation from
  the group median in MADs; outlier-flagged rows amber, threshold-violating
  rows red. Link alpha-beta fits shown in the fleet overview card.

Non-scope: running tests, live/streaming updates (load-at-startup only),
run-to-run diffing, editing config. No new analysis: the viewer projects the
findings the report pipeline already computed (it re-derives per-row
deviations for display, but flags come solely from `fleet.outliers` /
`fleet.threshold_violations`).

## Data / control flow

```
runs/<run_id>.json
  └─ gauntlet::report::history::load        (main.rs resolve_input)
       └─ model::ViewModel::new(&RunResults)     [pure, tested]
            ├─ nodes:  severity ⇐ failed_hosts / Failed outcomes (non-pair
            │          scope) = Bad; outliers, threshold violations,
            │          consistency dissent = Warn. Roofline digest from
            │          calibration.rooflines.
            ├─ edges:  built from HostPair-scoped metrics (net_bandwidth
            │          gib_per_sec, net_latency rtt_p50/p99), kept per
            │          direction (from_a = measured by the lexicographically
            │          smaller endpoint). Pair-scoped findings and Failed
            │          pair outcomes attach here, not to nodes.
            ├─ rows:   metric groups keyed "<test>.<metric>"; groups with
            │          repeated sample keys (NCCL sweeps) are excluded and
            │          surface as calibration link fits instead.
            └─ links:  calibration.links (alpha µs, GiB/s, r²).
       └─ ui::RootView (gpui)
            ├─ graph pane: canvas paints edges (PathBuilder stroke); node
            │   chips are absolutely positioned clickable divs. The canvas
            │   prepaint records its window bounds into RootView via
            │   cx.defer (one-frame settle) — chips and hit-testing need
            │   pixel geometry before paint time.
            └─ side panel: selection card + scrollable metric table.
```

Sample-key attribution (`model::attribute`): keys are `host`,
`host:<scope>`, or `host:pair:<peer>`; longest-host-prefix match so scope
labels containing ':' (e.g. `disk:/tmp`) cannot misattribute.

## Files

- `viewer/Cargo.toml` — workspace member; depends on `gauntlet`
  (`default-features = false`, so no cudarc) and `gpui =0.2.2`. Root
  workspace has `default-members = ["."]`: plain `cargo build` still builds
  only the CLI; use `-p gauntlet-view` or `--workspace` for the viewer.
- `viewer/build.rs` — links-shim for Debian/Ubuntu without
  `libxkbcommon-x11-dev`: symlinks `libxkbcommon-x11.so.0` into OUT_DIR and
  adds it to the link search path.
- `viewer/src/model.rs` — pure `RunResults -> ViewModel` projection.
  Exports `ViewModel`, `NodeView`, `EdgeView` (with per-direction
  `Directional`), `MetricRow`, `LinkRow`, `Severity`, `format_value`.
- `viewer/src/layout.rs` — ring layout in unit space, letterboxed pixel
  mapping, point-segment distance, node/edge hit-testing. Pure math.
- `viewer/src/ui/mod.rs` — theme constants, `Selection`, `RootView`, header.
- `viewer/src/ui/graph.rs` — graph pane (canvas edges + node chips +
  labels + legend; edge labels suppressed above 24 edges).
- `viewer/src/ui/table.rs` — detail cards (fleet / node / edge) and the
  metric table with selection filtering.
- `viewer/tests/model_tests.rs` — projection contract, driven through the
  real `report::build` pipeline.
- `viewer/tests/layout_tests.rs` — geometry and hit-testing.

## Invariants and constraints

- `model` and `layout` must stay gpui-free (headless testability).
- Severity ordering `Ok < Warn < Bad`; merging uses `max`.
- `EdgeView.a` is always the lexicographically smaller endpoint; the
  `Directional.from_a` reading was measured by `a`.
- Pair-scoped findings never color a node; node findings never color an
  edge.
- The viewer never recomputes verdicts or flags — `report` is the single
  source of truth (`verdict()`, `fleet.outliers`, ...).
- gpui's `Pixels` inner field is private: convert with `f32::from(px)`.
- Edge selection indices refer to `ViewModel::edges` order; the graph pane
  keeps its `edge_indices()` vector aligned with that order (unknown hosts
  map to `usize::MAX`, skipped by hit-testing).
