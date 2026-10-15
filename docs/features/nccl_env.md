# NCCL Environment Passthrough

## Scope
Lets `[nccl]` in the fleet config hand NCCL the same tuning knobs a
training job uses (NCCL_IB_HCA, NCCL_IB_GID_INDEX, NCCL_NET_GDR_LEVEL,
NCCL_P2P_LEVEL, NCCL_ALGO / NCCL_PROTO, NCCL_DEBUG / NCCL_DEBUG_SUBSYS,
NCCL_CROSS_NIC, ...). Every agent process gauntlet starts is started with
the resolved env, so every NCCL communicator is created under it, and the
run records it for reproducibility and run-to-run diffing.

```toml
[nccl]
socket_ifname = "bond0"   # typed first-class knob -> NCCL_SOCKET_IFNAME
env = { NCCL_IB_HCA = "mlx5_0,mlx5_1", NCCL_IB_GID_INDEX = 3, NCCL_IB_DISABLE = false, NCCL_DEBUG = "WARN" }
```

Values may be TOML strings, integers or booleans. Integers are written in
decimal and booleans as `1` / `0`, which is NCCL's convention. Floats,
arrays and tables are rejected with `NcclEnvError::UnsupportedValueType`.
A TOML datetime reaches serde as its string form and is passed through
verbatim.

Per-level overrides (proto v11 / schema v13) layer a map on top of the
global env for one NCCL call site only, e.g. force Ring on the NVLink
sweep, or disable P2P/SHM for the fleet sweep so every hop crosses the
NICs:

```toml
[nccl.levels.intranode]
env = { NCCL_ALGO = "Ring" }
[nccl.levels.fleet]
env = { NCCL_P2P_DISABLE = 1, NCCL_SHM_DISABLE = 1 }
```

Non-scope:
- Unsetting a global key for one level. Values must be non-empty, so an
  override can replace a global value but not remove it.
- Non-NCCL variables (LD_PRELOAD, LD_LIBRARY_PATH, CUDA_VISIBLE_DEVICES,
  UCX_*, ...). Rejected by design; see the allowlist rationale.
- Per-host env. One map per level is resolved from one config and used
  for every host; per-host tuning would make the fleet-relative MAD
  comparison compare differently-configured nodes.
- Capturing NCCL_* variables a node would otherwise *inherit*. Keys not in
  the map are left as the remote shell provides them and are not recorded.
- Validating values against NCCL's own grammar (e.g. that NCCL_ALGO names
  a real algorithm). NCCL owns its value syntax; an invalid value surfaces
  as NCCL behavior or an NCCL error on the rank.

## Why the env is set before process start
NCCL reads its knobs with `getenv` at communicator init, so they must be
in the agent's environment by then. The agent cannot safely put them
there itself: `main` builds a multi-threaded tokio runtime before
dispatching any subcommand (its worker threads start at build time), and
`agent run`, which hosts the `ncclCommInitAll` of both the intra-node
overlap phase and the intra-node NCCL sweep, executes on that runtime. `std::env::set_var` while
other threads exist is unsound (it is `unsafe` in edition 2024 for
exactly this reason). The pre-existing `set_socket_ifname` path in
`agent nccl` had the same flaw — the runtime already existed when it ran.

So the agent never mutates its environment. The orchestrator places the
resolved map on the remote command line of every agent spawn — `env
LD_LIBRARY_PATH='…/lib' NCCL_A='…' NCCL_B='…' <agent> agent <mode> …` —
and the variables exist before the agent process starts. This is the
same mechanism already used for LD_LIBRARY_PATH (which dlopen likewise
captures at process start). The map does not travel on the wire.

### Remote shell assumption
sshd runs the command through the remote user's login shell. The
command line single-quotes every value (`'…'`, with `'` spelled `'\''`),
and so does the existing LD_LIBRARY_PATH word. That quoting is correct in
any POSIX shell (sh, bash, dash, zsh, ksh). It also holds in csh/tcsh as
long as a value stays on one line: a newline inside single quotes breaks
those shells. That is why values may not contain ASCII control
characters (see below). Gauntlet assumes a POSIX-compatible login shell
on nodes, as it already did for LD_LIBRARY_PATH.

## Per-level overrides

### Levels
`NcclLevel` (`src/nccl_level.rs`) names every NCCL call site, and each
level has its own agent process:

| level | process | communicator |
|---|---|---|
| `intranode` | `agent run`, network phase | intra-node sweep (`ncclCommInitAll`) |
| `fleet` | `agent nccl`, `NcclWorkload::Sweep` | rank-per-GPU fleet sweep |
| `barrier` | `agent nccl` | NCCL barrier-skew probe |
| `overlap_intranode` | `agent run`, overlap phase | node-local overlap step |
| `overlap_fleet` | `agent nccl`, `NcclWorkload::Overlap` | fleet overlap step |

The mapping has one source of truth each side reads:
`Phase::nccl_level()` (network -> intranode, overlap ->
overlap_intranode, other phases none) and `NcclWorkload::level()`
(sweep -> fleet, overlap -> overlap_fleet, barrier -> barrier). The
orchestrator picks the spawn env from it and the agent labels its
transport record with it.

### Resolution
`[nccl.levels.<level>] env` (`NcclLevelsConfig` / `NcclLevelConfig`,
`deny_unknown_fields`, so an unknown level name is a parse error) goes
through the same stringify + `NcclEnv::from_map` validation as the global
map; a failure is `ConfigError::NcclLevel { level, source }`. The effective
env of a level is `global.overlay(override)`: override keys replace or add,
every other global key survives. `NcclLevelEnvs` holds the global env and
one effective env per level (a struct with one field per level, so a
lookup cannot miss). An override of NCCL_SOCKET_IFNAME is allowed even
when `[nccl] socket_ifname` is set: the both-places conflict rule applies
to the global section only, and layering is the point of a level.

`FleetConfig::nccl_levels()` resolves and caches the whole section once
(`validate` runs it); `nccl_env()` returns its global part, so a config
with an invalid level override yields no env at all.

### Co-hosted levels
Per-level env is only possible because no agent process hosts two levels
with different envs:
- `agent run`: the orchestrator sends exactly one phase per spawn
  (`node_phase` with `[phase]`), so the intra-node sweep (network phase)
  and the node-local overlap (overlap phase) are separate processes.
  A hand-run `agent run --phases network,overlap` would host both under
  whatever env it was started with; the orchestrator never does that.
- `agent nccl`: the barrier probe historically rides the fleet sweep's
  communicator (`NcclWorkload::Sweep { barrier }`). When the barrier and
  fleet effective envs are identical (`barrier_shares_fleet_comm`) it
  still does. When they differ, the orchestrator splits the spawn: the
  sweep runs without the probe, then a second fleet job runs
  `NcclWorkload::Barrier(spec)` on its own communicator (same rank
  layout) under the barrier env (`BarrierPlacement` in
  `orchestrator/nccl/barrier_job.rs`). The split costs one extra
  communicator init; results land in the same `nccl_barrier` metric
  groups. The separate job runs only after a clean sweep: if the sweep
  attributed a failure to a host, the probe is recorded Skipped on every
  member (`nccl_barrier`, node scope, naming the failed host) instead of
  re-forming a world that includes the culprit. The comparison reads
  `HostSession::nccl_levels()`, the levels resolved and validated when
  the config loaded, so it cannot fail at use time.

### Spawn env selection
`HostSession` holds the resolved `NcclLevelEnvs` and every spawn names its
`AgentEnv`:
- `AgentEnv::Base` (inventory, cpu/mem, gpu, counter passes, probe, peer,
  TCP barrier): LD_LIBRARY_PATH plus the global env, exactly the words
  `agent_env_words` produced before.
- `AgentEnv::Nccl(level)` (`node_phase` for network/overlap, every fleet
  `agent nccl` job via `NcclJob::spawn_env`): LD_LIBRARY_PATH, the level's
  effective env overlaid with the NCCL debug-log variables of transport
  capture as one `NcclEnv` (`nccl_transport::debug::capture_env`;
  docs/features/nccl_transport.md). Capture never names a key the level
  env sets, so no word is duplicated.

`spawn_env_words` builds both forms through `agent_env_words`, the single
quoting path: still single-quoted, still on the `env` command line, still
never `set_var`. `HostSession::connect` precomputes every list once
(`SpawnWords`: `Base` plus one per level, held in a `PerLevel`), so a
spawn only picks its list; a remote_dir that cannot form a valid debug
log path fails the connect, not a later spawn.

## Library stdout isolation
Setting NCCL_DEBUG makes NCCL write its logs to fd 1, and CUDA and other
native libraries can do the same. fd 1 is also the agent's protocol
channel (JSON-lines events, or the single JSON document from
probe/peer/barrier). If a partial library line is flushed just before an
event write, the two merge into one undecodable line. A lost `NcclId`
stalls the rendezvous, and a lost report loses its results.

`main` therefore calls `agent::channel::isolate_stdout()` for every agent
subcommand before building the tokio runtime, while no other thread
exists. It makes two changes:
- It duplicates fd 1 to a private close-on-exec fd, which becomes the
  protocol channel.
- It runs `dup2(2, 1)`, so all library output on "stdout" goes to the
  stderr log the orchestrator already drains.

`EventSink::stdout()` and `channel::write_line` (probe, peer, barrier)
write only to the protocol channel. `EventSink::new(writer)` stays
injectable for in-process tests. A test (`tests/stdout_isolation_tests.rs`)
runs the real binary's hidden `agent stdout-isolation-check` mode. That
mode puts raw `write(1, …)` calls between events: a partial line right
before an event, a full line, and a `println!`. The test then decodes
stdout exactly as the orchestrator does.

## Allowlist rationale
Keys must match `^NCCL_[A-Z0-9_]+$`. The map exists to tune NCCL, and a
general env passthrough has a much larger blast radius: LD_PRELOAD changes
what code runs in every phase, CUDA_VISIBLE_DEVICES silently changes which
GPUs phase 2 and the overlap phase measure, LD_LIBRARY_PATH changes which
libcuda/libnccl gets dlopened (and would fight gauntlet's own shim path).
None of that would be visible in the results document, so numbers would
stop meaning what they say. The prefix check keeps the knob's effect
inside NCCL, and its charset makes every key a literal shell word. Values
must be non-empty, because an empty value is almost always a templating
mistake. They must also be free of ASCII control characters
(`NcclEnvValueError::ControlCharacter`). An environment string cannot
carry a NUL, and a newline or other control character would break
quoting on non-POSIX login shells. NCCL_SOCKET_IFNAME
set in both `socket_ifname` and `env` is a conflict error, not a
precedence rule — even when the two agree.

`socket_ifname` stays a typed first-class field: existing configs set it
(and the config is `deny_unknown_fields`), and the management-vs-data
plane story (docs/features/phase3_network.md) is documented around it.

## Data / control flow
1. **Config load** (`config.rs`). `[nccl]` deserializes as the raw file
   shape, `NcclConfig { socket_ifname, env: BTreeMap<String,
   RawNcclEnvValue> }` (`deny_unknown_fields`). The config file is parsed
   once. `FleetConfig::validate()`, which `load` always runs, calls
   `FleetConfig::nccl_env()`, and that resolves the section exactly once
   with `NcclConfig::resolve`: stringify the values, validate keys and
   values, then fold in `socket_ifname`. The result is cached in a
   private write-once field (`OnceLock<NcclEnv>`, `#[serde(skip)]`).
   Failures are typed `ConfigError::Nccl { source: NcclEnvError }`.
   Callers borrow the result as `&NcclEnv`.

   **Why an invalid env can never be observed:** the only way to obtain
   an `NcclEnv` from a config is `nccl_env()`. That accessor either
   returns the cached value, which only a successful `resolve` can have
   written, or resolves (and so validates) on first use. After `load`,
   `validate` has already filled the cache, so the accessor cannot fail.
   On a config that skipped `validate` (e.g. a bare `toml::from_str`), it
   returns the typed error instead of an env.
2. **Session setup** (`orchestrator/mod.rs::connect_fleet`,
   `orchestrator/bootstrap.rs::prepare_host`). `config.nccl_env()?` is
   passed to `HostSession::connect`, which precomputes
   `agent_env_words(remote_dir, &nccl_env)`: `LD_LIBRARY_PATH=<quoted>`
   then `KEY=<single-quoted value>` per entry, in key order. An empty map
   adds no words.
3. **Spawn** (`orchestrator/session.rs`). `run_agent`, `run_agent_capture`
   and `spawn_agent` — the only three ways an agent is started — put those
   words between `env` and the agent binary via `raw_args` (already
   quoted; openssh does not re-escape raw args). That covers every
   communicator-creating invocation. That is `agent run`, which runs the
   intra-node NCCL sweep, the intra-node overlap and any future node-local
   NCCL. It is also `agent nccl` Lead/Participate, which runs the
   rank-per-GPU phase-3 sweep, the NCCL barrier-skew probe on the same
   communicator, and the fleet overlap. Finally it covers the TCP barrier / peer / probe modes (harmless there —
   NCCL_* only affects NCCL — and uniform, so no spawn path can be
   forgotten).
4. **Agent**. Does nothing: the variables are simply in its environment
   when NCCL initializes. `agent/nccl/` no longer touches the env.
   `agent_kill_pattern` (`[g]auntlet-agent agent <args>`, used by `pkill
   -f`) still matches, because `env` execs the binary and the `KEY=value`
   words never appear in the agent's own argv. A test builds the full
   spawn command with a non-empty env and checks this.
5. **Wire** (`proto/mod.rs`, PROTO_VERSION 9). `socket_ifname` is removed
   from `NcclDirective::{Lead, Participate}`, and nothing related to the
   NCCL env is on the wire. `NcclDirective` is now
   `deny_unknown_fields`, so a stale `socket_ifname` fails loudly instead
   of being ignored.
6. **Reporting** (`report/mod.rs`, `report/nccl_env.rs`, SCHEMA_VERSION 10;
   per-level fields since 13).
   `report::build` sets the run-level field `RunResults.nccl_env:
   Option<BTreeMap<String, String>>` from `config.nccl_env()`. It is
   recorded orchestrator-side because the orchestrator knows exactly what
   it put on every command line. `None` means not recorded: pre-v10
   documents get this through the serde default, so old history still
   loads. `Some(empty)` means an untuned run. The terminal table prints
   `nccl env: K=V ...`, `(none)` or `(not recorded)` under the header
   line. Since schema v13 `RunResults.nccl_level_env:
   Option<BTreeMap<NcclLevel, BTreeMap<String, String>>>` records every
   level's effective env (`NcclLevelEnvs::to_record`; the capture
   variables are not part of it, their path is per host). The table adds
   `nccl env [<level>]: ...` for every level whose env differs from the
   global one (`format_level_overrides`).
7. **Viewer** (`viewer/src/model.rs`, `diff.rs`, `ui/table.rs`).
   `ViewModel.nccl_env` mirrors the document. `DiffView.nccl_env_drift =
   report::nccl_env::nccl_env_drift(baseline, current)` lists
   added/removed/changed keys. It is `None`, and nothing is shown, when
   either run did not record its env, so a pre-v10 baseline never produces
   invented `+NCCL_SOCKET_IFNAME` drift. Diff mode shows an "nccl env drift" chip
   plus one line per change; the overview shows the run's env. Drift is
   context, never a per-node regression. `DiffView.nccl_level_env_drift =
   nccl_level_env_drift(baseline, current, global_drift)` adds per-level
   changes the global drift does not already list (a changed global key
   is shown once, not once per level), as `[level] change` lines; it is
   `None` when either run predates v13. The overview lists level
   overrides under the global env.

### Why run-level, not per-host
The env is fleet-uniform by construction: one config, one resolved map,
identical words on every host's command lines. A per-host inventory or
consistency field would be N copies of one value that can never dissent.
What does vary is tuning between runs — exactly what makes two runs' NCCL
numbers incomparable — so the field lives on `RunResults` and the
baseline diff compares it.

## Files
- `src/nccl_env.rs` — `NcclEnvKey` (`^NCCL_[A-Z0-9_]+$`), `NcclEnvValue`
  (non-empty, no ASCII control characters), `RawNcclEnvValue` (TOML
  string/integer/boolean, anything else rejected) + `stringify_raw`,
  `NcclEnv` (`from_map`, `resolve`, `get`, `iter`, `len`,
  `to_string_map`), `NcclEnvError` / `NcclEnvValueError`, `SOCKET_IFNAME`.
- `src/nccl_env.rs` also has `NcclEnv::overlay` (level layering).
- `src/nccl_level.rs` — `NcclLevel` (`ALL`, `as_str`, serde snake_case),
  `NcclLevelEnvs` (`resolve`, `uniform`, `global`, `level`,
  `barrier_shares_fleet_comm`, `to_record`).
- `src/config.rs` — `NcclConfig` (raw section, `resolve`,
  `resolve_levels`), `NcclLevelsConfig` / `NcclLevelConfig`,
  `ConfigError::{Nccl, NcclLevel}`, `FleetConfig::from_toml_str`,
  `FleetConfig::nccl_levels` (cached) and `FleetConfig::nccl_env` (its
  global part).
- `src/agent/channel.rs` — `isolate_stdout`, `protocol_writer`,
  `write_line`, the hidden `isolation_check` mode; `src/main.rs` calls
  `isolate_stdout` for agent subcommands before building the runtime.
- `tests/stdout_isolation_tests.rs` — end-to-end protocol isolation.
- `src/orchestrator/session.rs` — `AgentEnv`, `HostSession::connect(host,
  ssh, Arc<NcclLevelEnvs>)`, `spawn_env_words`, `agent_env_words`,
  `nccl_env_words`, the three spawn paths (`run_agent` takes an
  `AgentEnv`; `run_agent_capture` and `spawn_agent` are always `Base`).
- `src/orchestrator/nccl/mod.rs` — `NcclJob::spawn_env`,
  `BarrierPlacement` (barrier on the sweep communicator vs its own world).
- `src/proto/mod.rs` — `Phase::nccl_level`, `NcclWorkload::{Barrier,
  level}`; PROTO_VERSION 11.
- `src/agent/nccl/` — runs `NcclWorkload::Barrier` (`sweep::run_barrier`).
- `src/orchestrator/mod.rs`, `src/orchestrator/bootstrap.rs` — pass
  `config.nccl_env()` into every session.
- `src/proto/mod.rs` — `socket_ifname` removed from the directives
  (which now carry only `RankAssignment` plus the workload, and the
  rendezvous id for participants); PROTO_VERSION 9.
- `src/agent/nccl/mod.rs`, `src/orchestrator/nccl/mod.rs` —
  `set_socket_ifname` and `NcclJob.socket_ifname` removed.
- `src/report/nccl_env.rs` — `NcclEnvChange`, `nccl_env_drift`,
  `format_nccl_env`, `LevelEnvMap`, `nccl_level_env_drift`,
  `format_level_overrides`.
- `src/report/mod.rs` — `RunResults.nccl_env`, table line; SCHEMA_VERSION 10.
- `viewer/src/{model,diff}.rs`, `viewer/src/ui/table.rs` — display + drift.

## Invariants
- An `NcclEnv` only ever contains keys matching `^NCCL_[A-Z0-9_]+$` and
  non-empty values free of ASCII control characters. Every constructor
  validates.
- `FleetConfig::nccl_env()` only ever returns an env that passed
  `NcclConfig::resolve`. After `load` it cannot fail.
- In every agent process, only the protocol channel reaches the
  orchestrator's decoder. fd 1 is stderr from before the first thread
  exists.
- Gauntlet assumes a POSIX-compatible login shell on nodes.
  Single-quoted one-line values are also safe under csh/tcsh.
- The agent never calls `std::env::set_var`. NCCL env reaches it only via
  the spawn command line, so it is present before any thread exists.
- Every agent spawn picks its words through `spawn_env_words` by its
  `AgentEnv`; there is no spawn path that bypasses it. Spawns of the same
  `AgentEnv` carry the same words on every host (except the per-host
  remote_dir inside LD_LIBRARY_PATH and NCCL_DEBUG_FILE).
- A level's effective env is the global env overlaid with that level's
  override, and nothing else (plus capture variables the config does not
  set).
- No agent process the orchestrator starts hosts two NCCL levels with
  different effective envs: one phase per `agent run`, one workload per
  `agent nccl`, and the barrier probe rides the fleet communicator only
  when both levels' envs are identical.
- Values are single-quoted; a POSIX shell hands them to the process byte
  for byte (tested against a real `sh` with adversarial values: quotes,
  `$`, `$(…)`, backticks, globs, newlines, `;|&<>`).
- Keys absent from the map are never unset; inherited values survive.
- `RunResults.nccl_env` is `Some` of the map placed on the command
  lines. Pre-v10 documents decode with `None` ("not recorded"), and drift
  against them is never shown.
