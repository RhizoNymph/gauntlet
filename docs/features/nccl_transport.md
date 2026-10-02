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
is a finding (`fleet.socket_fallbacks`, verdict Stragglers) unless the
level's env asked for sockets on purpose. The point is diagnosis: a silent
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
   `nccl_transport/debug.rs::capture_additions`). Every NCCL-hosting
   spawn (`AgentEnv::Nccl(level)`) gets, on its `env` command line, the
   level's effective env plus the debug-log variables the config does not
   already set:
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
   below INFO (VERSION, WARN, ...) or a subsystem mask without INIT gets
   nothing added, and the transport is recorded as unknown with that
   reason. A user NCCL_DEBUG_FILE is parsed in place (with `%h`/`%p`
   expanded the way NCCL does). Values set by gauntlet override values the
   node's login shell exports for the same keys.

   NCCL never writes to fd 1 in this setup: the log goes to the file, and
   `agent::channel::isolate_stdout` already points fd 1 at stderr.

2. **Agent** (`agent/transport.rs`). `CaptureWindow::open()` before the
   first NCCL call of the process; `CaptureWindow::emit(sink, level,
   span)` after the communicator has carried traffic (NCCL connects peers
   lazily, at the first collective, since 2.22):
   - `agent nccl`: after the workload (sweep, barrier or fleet overlap),
     whether or not it succeeded, before any error propagates — a socket
     fallback is exactly what makes a workload crawl or time out. Level
     from `NcclWorkload::level()`, span `CommSpan::from_world(block count,
     world size)` (one host owns one contiguous block, so the world spans
     hosts exactly when it is larger than the block).
   - `agent run` network phase (intra-node sweep): after the sweep loop;
     `Intranode`, single host.
   - `agent run` overlap phase: after the warmup rounds, before the GEMM
     load; `OverlapIntranode`, single host.

   The window reads NCCL_DEBUG, NCCL_DEBUG_SUBSYS and NCCL_DEBUG_FILE from
   the process env (reading is sound; the agent never sets env), decides
   with the pure `debug::log_source` where NCCL wrote its log (or why it
   wrote none), checks the file was modified after the window opened
   (else `Stale`: NCCL could not open it and a previous run's file is
   still there), reads it and parses it with the pure `parse::parse_log`.
   NCCL opens its debug file unbuffered, so every line is on disk.

3. **Wire** (PROTO_VERSION 11). `AgentEvent::NcclTransport { report:
   Box<NcclTransportReport> }`, one per NCCL-hosting process:
   `{ level, span, capture: Captured { info } | Unknown { reason } }`.
   Forwarded to the collector unchanged (the fleet drivers' intercepts
   pass it through).

4. **Collector** (`orchestrator/collect.rs`). Appended to
   `HostObservations.nccl_transports` (serde-defaulted), in arrival order
   across repeats.

5. **Report** (`report/nccl_transport.rs`, SCHEMA_VERSION 13).
   `socket_fallbacks(observations, nccl_level_env)`: for every host, every
   `MultiHost` report whose captured network is `Socket`, one
   `SocketFallback { level, ifaces }` per (host, level) however many
   repeats saw it — unless that level's recorded effective env makes
   sockets deliberate (`socket_is_deliberate`: NCCL_IB_DISABLE a non-zero
   integer, or NCCL_NET=Socket, case-insensitive). Unknown captures and
   single-host communicators never flag. `fleet.socket_fallbacks`
   non-empty makes the verdict at least Stragglers.

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
  `log_source`, `expand_debug_file`, `short_hostname`,
  `managed_log_path`, `level_logs_info`, `subsys_admits_init`,
  `CAPTURE_LEVEL`, `CAPTURE_SUBSYS`, `LOG_DIR`.
- `src/agent/transport.rs` — `CaptureWindow` (`open`, `emit`).
- `src/agent/nccl/mod.rs`, `src/agent/gpu/intranode.rs`,
  `src/agent/gpu/overlap.rs` — the three capture points.
- `src/orchestrator/session.rs` — `spawn_env_words` adds the variables;
  `resolve_remote_dir` creates `nccl-logs`.
- `src/orchestrator/collect.rs` — `HostObservations.nccl_transports`.
- `src/report/nccl_transport.rs` — `SocketFallback`, `socket_fallbacks`,
  `net_cell`, `render`.
- `src/report/mod.rs` — `FleetAnalysis.socket_fallbacks`, verdict.

## Invariants
- Gauntlet never overrides an NCCL debug variable the config sets; it
  only adds the ones missing, and only when the result is a parsable log.
- The agent decides where the log is from the variables in its own
  environment, so it parses exactly the file NCCL was told to write.
- A report is emitted only by a process whose communicator initialized;
  its level comes from `Phase::nccl_level` / `NcclWorkload::level`, the
  same mapping that chose its spawn env.
- A log not modified since the process's NCCL init began is never parsed.
- Only a captured `Socket` network on a `MultiHost` communicator, at a
  level whose env does not disable IB, is a finding.
- Parsing is pure and total: any text yields a value or `None`.
