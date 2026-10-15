# NCCL Transport Capture

## Scope
Records, for every NCCL communicator gauntlet creates, which transports
NCCL actually used:
- the network transport for inter-node traffic: NCCL's internal IB-verbs
  transport (with the HCA ports and their link layer, InfiniBand or
  RoCE), its TCP socket transport (with the interfaces), or an external
  net plugin (by name, with any IB ports it reports);
- the external net plugin NCCL loaded, if any;
- per-transport counts of peer connections (P2P, SHM, NET with the
  GPUDirect-RDMA subset, COLLNET);
- the NCCL version.

A multi-host communicator whose network transport is the socket fallback
is a finding (`fleet.socket_fallbacks`, verdict Stragglers) when the host
had an ACTIVE IB/RoCE port to fall back from and the level's env did not
ask for sockets on purpose. A fleet with no IB at all (e.g. an Ethernet
bond) runs NCCL over sockets by necessity, and that is never flagged. The
point is diagnosis: a silent
fallback from IB to TCP (a wrong NCCL_IB_HCA, a missing GID index, a down
port) is about 10x slower and otherwise just looks like low bandwidth.

Non-scope:
- An NCCL API for any of this. There is none (cudarc exposes nothing
  either), so the record comes from NCCL's own INFO log.
- Per-channel device mapping (which HCA carried which ring). The
  `NET/IB : Using` list is the device set NCCL selected after
  NCCL_IB_HCA filtering, which is what a misconfiguration changes.
- Judging plugin transports. A plugin (Libfabric/EFA, IBext) is recorded,
  never flagged.
- Log rotation of the managed log files beyond "one file per host and
  level, overwritten by the next run".

## Data / control flow
1. **Spawn env** (`orchestrator/session.rs::spawn_env_words`,
   `nccl_transport/debug.rs::capture_env`). Every NCCL-hosting spawn
   (`AgentEnv::Nccl(level)`) gets, on its `env` command line, the level's
   effective env overlaid with the debug-log variables the config does
   not already set (an `NcclEnv`, quoted by the same `agent_env_words`
   path as every other word; computed once per host at connect,
   `SpawnWords`):
   - NCCL_DEBUG unset -> `NCCL_DEBUG=INFO`, and if NCCL_DEBUG_SUBSYS is
     unset too, `NCCL_DEBUG_SUBSYS=INIT,NET` (INIT carries every line the
     parser reads; NET adds plugin detail; the log stays small).
   - NCCL_DEBUG_FILE unset and the effective level is INFO/TRACE ->
     `NCCL_DEBUG_FILE=<remote_dir>/nccl-logs/<level>.%h.log`. NCCL expands
     `%h` to the short hostname, so hosts sharing an NFS remote_dir never
     share a file; one file per host and level, truncated by the next run.
     `resolve_remote_dir` creates `nccl-logs` (NCCL cannot create
     directories).

   Precedence: **a variable set in the config (global or level env) is
   never overridden.** A user NCCL_DEBUG=INFO keeps NCCL's default
   subsystem mask (which includes INIT) and gains only the file, so their
   log moves from the agent's stderr into that file. A user NCCL_DEBUG
   below INFO (VERSION, WARN, ...) or a user subsystem mask without INIT
   (even with NCCL_DEBUG unset) gets nothing added at all — no logging
   nobody asked for — and the transport is recorded as unknown with that
   reason. A user NCCL_DEBUG_FILE is parsed in place (with `%h`/`%p`
   expanded the way NCCL does). Values set by gauntlet override values the
   node's login shell exports for the same keys.

   NCCL never writes to fd 1 in this setup: the log goes to the file, and
   `agent::channel::isolate_stdout` already points fd 1 at stderr.

2. **Agent** (`agent/transport.rs`). `CaptureWindow::open()` before the
   first NCCL call of the process, then `window.run(sink, level, span,
   body)` around every NCCL call of the communicator. `run` emits the
   report exactly once on every exit of the body — success, a failed
   `ncclGetUniqueId` / `ncclCommInitRank` / `ncclCommInitAll`, a
   collective error — and then returns the body's result unchanged. A
   failed init is exactly when the diagnosis matters (wrong NCCL_IB_HCA or
   GID index), and NCCL has logged its network selection by then. NCCL
   connects peers lazily (at the first collective, since 2.22), so the
   bodies end after traffic:
   - `agent nccl`: id mint, communicator init and the workload (sweep,
     barrier or fleet overlap). Stage 1 (device checks, buffers) runs
     before the body and makes no NCCL call, so a failure there emits
     nothing. Level from `NcclWorkload::level()`, span
     `CommSpan::from_world(block count, world size)` (one host owns one
     contiguous block, so the world spans hosts exactly when it is larger
     than the block).
   - `agent run` network phase (intra-node sweep): `ncclCommInitAll`
     through the sweep loop; `Intranode`, single host.
   - `agent run` overlap phase: `ncclCommInitAll` through the warmup
     rounds, before the GEMM load; `OverlapIntranode`, single host.

   `open` reads NCCL_DEBUG, NCCL_DEBUG_SUBSYS and NCCL_DEBUG_FILE from the
   process env (reading is sound; the agent never sets env), decides with
   the pure `debug::log_source` where NCCL will write its log (or why it
   will write none), and **deletes whatever is at that path**. What
   exists there after the body ran was written by this process's NCCL by
   construction; no timestamps are compared, so an NFS server's clock
   cannot matter. (NCCL truncates the file at open anyway, so nothing is
   lost.) A path that cannot be cleared is `Unknown::Uncleared`; a file
   NCCL never wrote is `Unknown::Unreadable`. The text is parsed with the
   pure `parse::parse_log`; NCCL opens its debug file unbuffered, so every
   line is on disk. A process killed outright (the fleet-overlap
   watchdog's hard exit) emits nothing.

3. **Wire** (PROTO_VERSION 11). `AgentEvent::NcclTransport { report:
   Box<NcclTransportReport> }`, one per NCCL-hosting process:
   `{ level, span, capture: Captured { info } | Unknown { reason } }`.
   Forwarded to the collector unchanged (the fleet drivers' intercepts
   pass it through).

4. **Collector** (`orchestrator/collect.rs`). Appended to
   `HostObservations.nccl_transports` (serde-defaulted), in arrival order
   across repeats.

5. **Report** (`report/nccl_transport.rs`, SCHEMA_VERSION 13).
   `socket_fallbacks(observations, nccl_level_env)` is the one pure
   decision. It yields one `SocketFallback { level, ifaces }` per (host,
   level), however many repeats saw it, when all of these hold:
   - the report is `MultiHost` and its captured network is `Socket`;
   - the host's phase-0 inventory shows at least one IB/RoCE port in state
     ACTIVE (`IbAvailability::of` == `ActivePort`). Without one, sockets
     are the only transport the host has, not a fallback;
   - the level's recorded effective env does not make sockets deliberate
     (`socket_is_deliberate`: NCCL_IB_DISABLE a non-zero integer, or
     NCCL_NET=Socket, case-insensitive).

   `IbAvailability` is `ActivePort`, `NoActivePort` (no ports, or none
   ACTIVE), `NoDeviceSeenByNccl` (no inventory, but NCCL logged `NET/IB :
   No device found` while the level left NCCL_IB_HCA unset, i.e. no IB
   device exists; recorded by the parser as `ib_no_device`) or `Unknown`
   (no inventory and no such signal). Only `ActivePort` can flag. With
   NCCL_IB_HCA set, "no device" may be the misconfiguration itself, so it
   is not taken as proof of absence. Unknown captures and single-host
   communicators never flag either. The transport record itself is kept
   in every case. `fleet.socket_fallbacks` non-empty makes the verdict at
   least Stragglers.

6. **Terminal** (`report::render_table`). A "nccl transports" section, one
   row per (host, level) with the latest report: span, network
   ("IB mlx5_0:1,mlx5_1:1", "RoCE ...", "Socket bond0(10.1.1.10)",
   "plugin Libfabric", or "unknown: <reason>"), peer channel counts
   ("P2P 4 NET 4 (GDR 4)") and NCCL version; plus a "nccl socket
   fallback" findings section when any.

## The parser (`nccl_transport/parse.rs`)
Pure, never fails, ignores what it does not recognize. Only text after
`NCCL INFO ` counts (so line prefixes and WARN lines are irrelevant);
the `NCCL version` banner is matched anywhere. Recognized messages, as
emitted by NCCL 2.18 through 2.30:
- `NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/RoCE [RO]; OOB ...` — ports;
  tokens without a numeric `[index]` prefix (`[RO]`) and the OOB part are
  skipped.
- `NET/Socket : Using [0]bond0:10.1.1.10<0>` — interfaces (the name ends
  at the first colon, so IPv6 addresses survive).
- `NET/Plugin: Loaded net plugin <name> (vN)` — the plugin name.
- `Using network <name>` — authoritative: `IB` -> `Ib`, `Socket` ->
  `Socket`, anything else -> `Plugin { name }` with any IB ports. Without
  it the device lines decide (plugin, then IB, then sockets: NCCL's own
  order).
- `Channel NN/N : a[x] -> b[y] [send] via <T>/...` and `CollNet ... via
  COLLNET/...` — counted by the first path segment; `GDRDMA` marks
  GPUDirect RDMA.

`None` (-> `Unknown::NoTransportLines`) when the log names no network and
no connection. Fixtures in `src/nccl_transport/fixtures/`: IB (2.18),
RoCE (2.21), socket fallback after a wrong NCCL_IB_HCA (2.23), Libfabric
plugin (2.19), IBext plugin with IB ports (2.27), single-node
`ncclCommInitAll` with P2P/SHM (2.30). They are representative
reconstructions of NCCL INFO output, not captures from this fleet; real
logs should be added as fixtures when hardware validation produces them.

## Files
- `src/nccl_transport/mod.rs` — `IbLinkLayer`, `IbPort`, `SocketIface`,
  `NetTransport` (`is_socket`, `summary`), `ChannelTransports`,
  `NcclTransportInfo`, `UnknownTransport` (Display), `TransportCapture`,
  `CommSpan` (`from_world`), `NcclTransportReport`,
  `socket_is_deliberate`.
- `src/nccl_transport/parse.rs` — `parse_log`.
- `src/nccl_transport/debug.rs` — `DebugSettings`, `capture_additions`,
  `capture_env`, `log_source`, `expand_debug_file`, `short_hostname`,
  `managed_log_path`, `level_logs_info`, `subsys_admits_init`,
  `CAPTURE_LEVEL`, `CAPTURE_SUBSYS`, `LOG_DIR`.
- `src/agent/transport.rs` — `CaptureWindow` (`open`, `open_with`,
  `run`).
- `src/agent/nccl/mod.rs`, `src/agent/gpu/intranode.rs`,
  `src/agent/gpu/overlap.rs` — the three capture points.
- `src/orchestrator/session.rs` — `spawn_env_words` adds the variables,
  `SpawnWords` precomputes them per host; `resolve_remote_dir` creates
  `nccl-logs`.
- `src/orchestrator/collect.rs` — `HostObservations.nccl_transports`.
- `src/report/nccl_transport.rs` — `SocketFallback`, `IbAvailability`,
  `socket_fallbacks`, `net_cell`, `render`.
- `src/report/mod.rs` — `FleetAnalysis.socket_fallbacks`, verdict.

## Invariants
- Gauntlet never overrides an NCCL debug variable the config sets; it
  only adds the ones missing, and only when the result is a parsable log.
- The agent decides where the log is from the variables in its own
  environment, so it parses exactly the file NCCL was told to write.
- Every NCCL-hosting process that reached its first NCCL call emits
  exactly one report, whether the body succeeded or failed (init failures
  included); its level comes from `Phase::nccl_level` /
  `NcclWorkload::level`, the same mapping that chose its spawn env.
- The log path is cleared before NCCL starts, so a previous run's log is
  never parsed; freshness never depends on clocks.
- Only a captured `Socket` network on a `MultiHost` communicator, on a
  host whose inventory shows an ACTIVE IB/RoCE port, at a level whose env
  does not disable IB, is a finding.
- Parsing is pure and total: any text yields a value or `None`.
