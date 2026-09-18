# Error-Counter Deltas Across Load Phases

## Scope
Detect marginal hardware that passes throughput tests while silently
accumulating errors: snapshot every available hardware error counter before
the load phases, snapshot again after the last one, and flag any counter
that incremented. Non-scope: interpreting counter semantics beyond
"monotonic error tally" (no severity model), fixing anything, and device
presence/absence questions (those belong to inventory consistency).

## Data / control flow
1. **Baseline** — `gauntlet run` (orchestrator/mod.rs): in repeat 0, after
   any leading `inventory` phase and immediately before the first phase
   that loads the hardware (`cpu_mem`/`gpu`/`network`),
   `counter_baseline_pass` runs `agent run` on every host with a spec of
   `{phases: [], counters: {mode: "baseline"}}`. The agent's
   `counters::collect_snapshot()` gathers a `CounterSnapshot` and emits
   `AgentEvent::CounterBaseline`; the orchestrator intercepts and holds it
   per host (baselines never reach the collector).
2. **Load phases** run exactly as before (all `--repeat` iterations
   included; the window covers every repeat).
3. **Delta** — after the repeat loop, `counter_delta_pass` sends each
   baselined host `{phases: [], counters: {mode: "delta", baseline: ...}}`.
   The agent re-snapshots, computes `counters::diff_snapshots(baseline,
   now)` **on the agent**, and emits `AgentEvent::CounterDeltas` with the
   full delta list (zero and negative deltas included). The orchestrator
   forwards only that event into the collector.
4. **Collector** (`orchestrator/collect.rs`) stores the deltas as
   `HostObservations.counter_deltas`.
5. **Report** (`report/mod.rs`): `counter_findings()` keeps only deltas
   with `after > before` and lands them in
   `fleet.counter_findings: host -> Vec<CounterFinding>`. Any finding makes
   the verdict at least `Stragglers` (exit 1). The terminal table renders a
   section "error-counter deltas (across load phases)" with host, domain,
   device, counter, before, after, +increment — omitted entirely when
   there are no findings. The complete delta list (zeros included) stays in
   the JSON under `hosts.*.counter_deltas`.

A run whose phase list is only `inventory` never takes a baseline and
therefore never runs the delta pass.

## Counter identity
A counter is `(domain, device, counter)`:
- `pcie_aer` / `0000:65:00.0` / `aer_dev_correctable.BadTLP` — every field
  of the sysfs `aer_dev_correctable|aer_dev_nonfatal|aer_dev_fatal` files
  under `/sys/bus/pci/devices/*` (correctable includes the replay-relevant
  BadTLP/BadDLLP/Rollover tallies).
- `gpu_ecc` / `gpu0` / `ecc_{corrected,uncorrected}_{volatile,aggregate}`,
  `remapped_rows_{correctable,uncorrectable}` — one `nvidia-smi
  --query-gpu` call, same CSV/N-A conventions as phase 0
  (`inventory::csv_field` reused).
- `gpu_xid` / `dmesg` / `xid_lines` — count (not dedup) of kernel-log
  `NVRM: Xid` lines via journalctl / kern.log (`inventory::parse_xid_line`
  reused); a positive delta means new Xids under load. The reading is
  omitted (not zero) when no kernel log is readable, so a permission
  problem cannot fake a clean diff.
- `nvlink` / `gpu0/link1` / `replay_errors|recovery_errors|crc_errors|...`
  — parsed from `nvidia-smi nvlink -e`; any "Link N: <Name>: <count>" line
  is kept with the name normalized, so driver-generation naming variants
  survive. (The inventory does not expose NVML, so nvidia-smi is the
  source here too.)
- `edac` / `mc0`, `mc0/dimm1` / `ce_count|ue_count|*_noinfo_count`,
  `dimm_ce_count|dimm_ue_count` — `/sys/devices/system/edac/mc/`.
- `ib_port` / `mlx5_0/1` / `symbol_error|link_error_recovery|
  port_rcv_errors|port_xmit_discards|port_xmit_wait|link_downed` —
  `/sys/class/infiniband/<hca>/ports/<port>/counters/`.
- `nvme` / `nvme0` / `media_errors|num_err_log_entries` — `smartctl -j -A`
  first, `nvme smart-log -o json` fallback; both JSON shapes accepted.

## Files
- `src/agent/counters.rs` — snapshot collection (per-domain collectors
  with injectable sysfs roots, pure text parsers for tool output),
  `diff_snapshots`, `run` (event emission). Reuses
  `inventory::{run_capture, csv_field, parse_xid_line, PROBE_TIMEOUT}`.
- `src/proto.rs` — `CounterDomain`, `CounterReading`, `CounterSnapshot`,
  `CounterDelta` (+`increment()`), `CounterDeltas`, `CounterRequest`,
  events `CounterBaseline`/`CounterDeltas`, `AgentTaskSpec.counters`.
  PROTO_VERSION 3.
- `src/agent/mod.rs` — runs the counter pass after the listed phases when
  `spec.counters` is set.
- `src/orchestrator/mod.rs` — `counter_baseline_pass`,
  `counter_delta_pass`, scheduling around the repeat loop,
  `COUNTER_PASS_TIMEOUT` (60s).
- `src/orchestrator/collect.rs` — `HostObservations.counter_deltas`.
- `src/report/mod.rs` — `CounterFinding`,
  `FleetAnalysis.counter_findings`, verdict inclusion,
  `render_counter_findings`. SCHEMA_VERSION 4.
- `tests/counter_tests.rs` — fixture sysfs trees + tool-output parsing,
  delta logic, event round-trips, collector/report/render integration.

## Invariants
- Collection is infallible and best effort: a missing subsystem, tool, or
  permission yields absent readings, never an error, and never a host
  failure. Everything compiles and passes tests on machines with no
  GPU/IB/NVMe/EDAC hardware.
- A snapshot never carries the same (domain, device, counter) twice, and
  its readings are sorted.
- Deltas exist only for counters present in **both** snapshots; appearing
  or vanishing counters are dropped (presence is an inventory question).
- `increment() > 0` is the only condition that creates a finding; zero
  deltas are quiet in the table but always present in the JSON, and
  negative deltas (counter reset / log rotation) are recorded but never
  findings.
- Counter passes are advisory at the transport level: a failed or timed
  out pass logs a warning and leaves `counter_deltas` absent for that
  host; it never marks the host failed.
- The baseline window opens once per run (repeat 0) and closes once, after
  the final repeat's last load phase.
- Baselines travel orchestrator-side only (`CounterBaseline` events are
  intercepted; a stray one reaching the collector is ignored).
