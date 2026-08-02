# Viewer (`gauntlet-view`)

Native GPUI desktop app that renders a saved gauntlet run as a fully
connected fleet graph plus a quantitative metric table.

## Scope

- Browse every run in the runs directory (default `runs/`) in a sidebar:
  newest first, verdict-colored, with live runs badged. Click to open.
- Live tailing: a 1s poll loop rescans the directory; a run in flight
  (`<run_id>.partial.json`, written by `gauntlet run` every ~2s) reloads as
  it grows and swaps to the final JSON when the run completes.
- Launch runs from the GUI: the ▶ button spawns the `gauntlet` binary
  (sibling of the viewer binary, else $PATH) with `run --config <config>`,
  logs to `runs/gauntlet-run.log`, follows the new live run, and reports
  the exit verdict. ✕ cancel kills the child; a cancelled run's orphaned
  partial snapshot is deleted and the view falls back to the newest
  finished run. Live runs whose partial stops updating for >15s show
  "stalled" in the sidebar.
- Bootstrap from the GUI: the ⚙ button spawns `gauntlet bootstrap --json`,
  captures the readiness report to `runs/.bootstrap.json` (dot-prefixed so
  the run scanner ignores it), and renders a check × host matrix in the
  center pane (✓/!/✗ cells, details for anything non-ok). The report is
  reloaded on startup and reachable via "view last bootstrap".
- Runs produced by a debug (unoptimized) build carry `debug_build: true`
  and get a red "debug build" chip in the header — their numbers are not
  comparable to release-build runs.
- Diff mode: toggle in the header. Colors nodes/edges by regression vs a
  baseline run (pinned via the sidebar, else the previous finished run);
  the table's deviation column becomes Δ% vs baseline.
- Fleet graph: nodes on a ring, one edge per host pair. Node color encodes
  host health (green ok / amber outlier-or-skew / red failed); edge color
  encodes pairwise-path health the same way. Edge midpoints show worst-case
  direction bandwidth.
- Click a node or edge to see its findings (issues, roofline digest,
  per-direction bandwidth/RTT) and filter the metric table to that subject.
- Metric table: every fleet-comparable metric with per-sample deviation from
  the group median in MADs; outlier-flagged rows amber, threshold-violating
  rows red. Link alpha-beta fits shown in the fleet overview card.

Non-scope: editing config, cancelling a launched run, run scheduling. No
new absolute-mode analysis: the viewer projects the findings the report
pipeline computed (it re-derives per-row deviations for display, but flags
come solely from `fleet.outliers` / `fleet.threshold_violations`). Diff
mode's regression math lives in the viewer (`diff.rs`): unit-derived
direction of goodness, warn at >5% worse, bad at >15% worse.

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
- `viewer/src/diff.rs` — pure run-to-run diff: `DiffView::new(current,
  baseline)` produces per-row `RowDelta` (Δ fraction, regression severity,
  improved flag) plus node/edge regression severities and capped issue
  lists. `higher_is_better(unit)` is the direction-of-goodness oracle.
- `viewer/src/runs.rs` — run-list plumbing: `classify_file_name`
  (`.json` vs `.partial.json`), `order_and_dedupe` (newest first, finals
  shadow stale partials), `effective_baseline`, `scan` (directory scan
  with an (mtime, len) parse cache), `is_stalled`, `format_epoch_utc`.
- `viewer/src/bootstrap.rs` — pure readiness-matrix projection
  (`check_columns` union, `status_of`, `worst_of`); rendered by
  `viewer/src/ui/bootstrap.rs`.
- `viewer/src/layout.rs` — ring layout in unit space, letterboxed pixel
  mapping, point-segment distance, node/edge hit-testing. Pure math.
- `viewer/src/ui/mod.rs` — theme constants, `Selection` (keyed by host /
  host-pair so it survives live reloads), `RootView` state (current run,
  baseline cache, diff, child process, scan cache), the 1s poll loop
  (`cx.spawn` + background timer), run spawning, and the header (verdict,
  live badge, diff toggle).
- `viewer/src/ui/sidebar.rs` — run button, status note, run list rows
  (verdict dot, UTC timestamp, live badge, set/unset baseline).
- `viewer/src/ui/graph.rs` — graph pane (canvas edges + node chips +
  labels + legend; edge labels suppressed above 24 edges).
- `viewer/src/ui/table.rs` — detail cards (fleet / node / edge) and the
  metric table with selection filtering.
- `viewer/tests/model_tests.rs` — projection contract, driven through the
  real `report::build` pipeline.
- `viewer/tests/layout_tests.rs` — geometry and hit-testing.
- `viewer/tests/diff_tests.rs` — regression directions, thresholds,
  node/edge attribution, issue capping.
- `viewer/tests/runs_tests.rs` — classification, ordering, baseline
  resolution, scanning, timestamp math.
- `viewer/tests/bootstrap_tests.rs` — matrix projection and stalled-run
  detection.

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
- `diff` and `runs` stay gpui-free, like `model` and `layout`.
- Selection is keyed by identity (host, host pair), never by index — live
  reloads rebuild the ViewModel every ~1s and indices are not stable.
- The viewer only ever *reads* the runs directory plus writes
  `gauntlet-run.log`; partial files are produced and cleaned up by
  `gauntlet run` (see docs/features/reporting.md for the contract).
- Diff mode never recolors from absolute findings: with a baseline
  resolved, node/edge colors come exclusively from `DiffView`.
