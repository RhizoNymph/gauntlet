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
env = { NCCL_IB_HCA = "mlx5_0,mlx5_1", NCCL_DEBUG = "WARN" }
```

Non-scope:
- Non-NCCL variables (LD_PRELOAD, LD_LIBRARY_PATH, CUDA_VISIBLE_DEVICES,
  UCX_*, ...). Rejected by design; see the allowlist rationale.
- Per-host env. One map is resolved from one config and used for every
  host; per-host tuning would make the fleet-relative MAD comparison
  compare differently-configured nodes.
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
`agent run` — which hosts the intra-node overlap phase's
`ncclCommInitAll` — executes on that runtime. `std::env::set_var` while
other threads exist is unsound (it is `unsafe` in edition 2024 for
exactly this reason). The pre-existing `set_socket_ifname` path in
`agent nccl` had the same flaw — the runtime already existed when it ran.

So the agent never mutates its environment. The orchestrator places the
resolved map on the remote command line of every agent spawn — `env
LD_LIBRARY_PATH='…/lib' NCCL_A='…' NCCL_B='…' <agent> agent <mode> …` —
and the variables exist before the agent process starts. This is the
same mechanism already used for LD_LIBRARY_PATH (which dlopen likewise
captures at process start). The map does not travel on the wire.

## Allowlist rationale
Keys must match `^NCCL_[A-Z0-9_]+$`. The map exists to tune NCCL, and a
general env passthrough has a much larger blast radius: LD_PRELOAD changes
what code runs in every phase, CUDA_VISIBLE_DEVICES silently changes which
GPUs phase 2 and the overlap phase measure, LD_LIBRARY_PATH changes which
libcuda/libnccl gets dlopened (and would fight gauntlet's own shim path).
None of that would be visible in the results document, so numbers would
stop meaning what they say. The prefix check keeps the knob's effect
inside NCCL, and its charset makes every key a literal shell word. Values
must be non-empty (an empty value is almost always a templating mistake)
and NUL-free (an environment string cannot carry one). NCCL_SOCKET_IFNAME
set in both `socket_ifname` and `env` is a conflict error, not a
precedence rule — even when the two agree.

`socket_ifname` stays a typed first-class field: existing configs set it
(and the config is `deny_unknown_fields`), and the management-vs-data
plane story (docs/features/phase3_network.md) is documented around it.

## Data / control flow
1. **Config load** (`config.rs`). `[nccl]` deserializes as
   `RawNcclConfig { socket_ifname, env }` (file shape,
   `deny_unknown_fields`) and converts through `TryFrom` into
   `NcclConfig { raw, resolved }`, where `resolved =
   NcclEnv::resolve(socket_ifname, env)`. `NcclConfig` is
   `#[serde(try_from = "RawNcclConfig")]`, so a `FleetConfig` can never
   hold a disallowed env however it was deserialized. Because serde can
   only carry a message, `FleetConfig::from_toml_str` (used by `load`)
   first probes just the `[nccl]` section leniently (`nccl_policy_error`)
   and returns a typed `ConfigError::Nccl { source: NcclEnvError }`;
   syntax and shape errors fall through to the full parse and stay
   `ConfigError::Parse`.
2. **Session setup** (`orchestrator/mod.rs::connect_fleet`,
   `orchestrator/bootstrap.rs::prepare_host`). `config.nccl_env()` is
   passed to `HostSession::connect`, which precomputes
   `agent_env_words(remote_dir, &nccl_env)`: `LD_LIBRARY_PATH=<quoted>`
   then `KEY=<single-quoted value>` per entry, in key order. An empty map
   adds no words.
3. **Spawn** (`orchestrator/session.rs`). `run_agent`, `run_agent_capture`
   and `spawn_agent` — the only three ways an agent is started — put those
   words between `env` and the agent binary via `raw_args` (already
   quoted; openssh does not re-escape raw args). That covers every
   communicator-creating invocation: `agent run` (intra-node overlap and
   any future node-local NCCL), `agent nccl` Lead/Participate (phase-3
   sweep, the NCCL barrier-skew probe on the same communicator, fleet
   overlap), and the TCP barrier / peer / probe modes (harmless there —
   NCCL_* only affects NCCL — and uniform, so no spawn path can be
   forgotten).
4. **Agent**. Does nothing: the variables are simply in its environment
   when NCCL initializes. `agent/nccl.rs` no longer touches the env.
5. **Wire** (`proto.rs`, PROTO_VERSION 7). `socket_ifname` is removed from
   `NcclDirective::{Lead, Participate}`; nothing NCCL-env related is on
   the wire.
6. **Reporting** (`report/mod.rs`, `report/nccl_env.rs`, SCHEMA_VERSION 8).
   `report::build` sets run-level `RunResults.nccl_env` (plain
   `BTreeMap<String, String>`, `#[serde(default)]`) from
   `config.nccl_env()` — orchestrator-side, since the orchestrator knows
   exactly what it put on every command line. The terminal table prints
   `nccl env: K=V ...` (or `(none)`) under the header line.
7. **Viewer** (`viewer/src/model.rs`, `diff.rs`, `ui/table.rs`).
   `ViewModel.nccl_env` mirrors the document; `DiffView.nccl_env_drift =
   report::nccl_env::nccl_env_drift(baseline, current)` lists
   added/removed/changed keys. Diff mode shows an "nccl env drift" chip
   plus one line per change; the overview shows the run's env. Drift is
   context, never a per-node regression.

### Why run-level, not per-host
The env is fleet-uniform by construction: one config, one resolved map,
identical words on every host's command lines. A per-host inventory or
consistency field would be N copies of one value that can never dissent.
What does vary is tuning between runs — exactly what makes two runs' NCCL
numbers incomparable — so the field lives on `RunResults` and the
baseline diff compares it.

## Files
- `src/nccl_env.rs` — `NcclEnvKey` (`^NCCL_[A-Z0-9_]+$`), `NcclEnvValue`
  (non-empty, NUL-free), `NcclEnv` (`from_map`, `resolve`, `get`, `iter`,
  `len`, `to_string_map`), `NcclEnvError` / `NcclEnvValueError`,
  `SOCKET_IFNAME`.
- `src/config.rs` — `RawNcclConfig`, `NcclConfig` (`socket_ifname()`,
  `resolved_env()`), `ConfigError::Nccl`, `FleetConfig::from_toml_str`,
  `FleetConfig::nccl_env`.
- `src/orchestrator/session.rs` — `HostSession::connect(host, ssh,
  nccl_env)`, `agent_env_words`, `nccl_env_words`, the three spawn paths.
- `src/orchestrator/mod.rs`, `src/orchestrator/bootstrap.rs` — pass
  `config.nccl_env()` into every session.
- `src/proto.rs` — `socket_ifname` removed from the directives;
  PROTO_VERSION 7.
- `src/agent/nccl.rs`, `src/orchestrator/nccl.rs` — `set_socket_ifname`
  and `NcclJob.socket_ifname` removed.
- `src/report/nccl_env.rs` — `NcclEnvChange`, `nccl_env_drift`,
  `format_nccl_env`.
- `src/report/mod.rs` — `RunResults.nccl_env`, table line; SCHEMA_VERSION 8.
- `viewer/src/{model,diff}.rs`, `viewer/src/ui/table.rs` — display + drift.

## Invariants
- An `NcclEnv` only ever contains keys matching `^NCCL_[A-Z0-9_]+$` and
  non-empty, NUL-free values; every constructor validates.
- A loaded `FleetConfig` always holds a valid, conflict-free `[nccl]`
  section; `nccl_env()` is infallible because of it.
- The agent never calls `std::env::set_var`. NCCL env reaches it only via
  the spawn command line, so it is present before any thread exists.
- Every agent spawn of a session carries the same env words; there is no
  spawn path that bypasses `env_words`.
- Values are single-quoted; a POSIX shell hands them to the process byte
  for byte (tested against a real `sh` with adversarial values: quotes,
  `$`, `$(…)`, backticks, globs, newlines, `;|&<>`).
- Keys absent from the map are never unset; inherited values survive.
- `RunResults.nccl_env` equals the map placed on the command lines;
  pre-v8 documents decode with it empty (read as untuned).
