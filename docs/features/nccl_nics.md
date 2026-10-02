# NCCL NIC Selection and the NIC Bandwidth Ceiling

## Scope
Answer, per host and without guessing: which InfiniBand/RoCE ports would
NCCL's IB transport use under the run's resolved `NCCL_IB_HCA`, what link
layer and rate are they, and what is their summed line rate (the "nccl
nic ceiling"). Make that a fleet-comparable node metric plus consistency
fields, and make "ports exist but NCCL would select none" a per-host
finding (NCCL would silently fall back to sockets there).

This replaces the downstream guess "active ports at the highest rate" and
the manual links-per-node override that guess needed.

Non-scope:
- Measuring NIC bandwidth (phase 3 does that; this is the line-rate
  ceiling the measurements should approach).
- Per-rank NIC assignment. NCCL maps each GPU to NICs by topology; the
  ceiling is the node total. GPU<->NIC PCI locality is recorded so a
  consumer can reason about it, but no rank mapping is modelled.
- `NCCL_NET` / external net plugins, `NCCL_IB_MERGE_NICS`, and
  `NCCL_IB_HCA` values a node supplies itself (shell profile,
  `/etc/nccl.conf`, `~/.nccl.conf`): only the `[nccl] env` map is known to
  the orchestrator (see nccl_env.md, which never captures inherited
  variables).

## What NCCL does, and what is modelled
NCCL's IB transport init (`net_ib.cc`) scans every RDMA device and port
and keeps a port only if, in order:
1. the transport is enabled: `NCCL_IB_DISABLE` unset or 0. Modelled: an
   integer value != 0 disables (NCCL's `ncclParam` reads integers with
   `strtoll`; a non-integer keeps the default 0).
2. the port's logical state is ACTIVE.
3. the link layer is InfiniBand or Ethernet (RoCE).
4. the port passes the `NCCL_IB_HCA` filter (unset = everything passes).

`NCCL_IB_HCA` grammar (`src/nccl_ib/hca.rs`, after NCCL's
`parseStringList` / `matchIfList`):
- leading `^` = exclude list; then leading `=` = exact names instead of
  prefixes. `^=mlx5_0` excludes exactly mlx5_0; `=^x` is an exact include
  of a device literally named `^x`.
- comma-separated entries `name` or `name:port`; empty entries skipped;
  no trimming (`a, b` has an entry " b" that matches nothing).
- prefix matching by default: `mlx5_1` matches mlx5_1 *and* mlx5_10.
- the port is read with C `atoi` (no digits = 0, trailing junk ignored);
  -1 means any port; 0 or junk matches no real port.
- `:1` (no name) is dropped. An empty list matches everything, so `^`
  alone excludes everything.
- NCCL's 32-entry and 64-byte caps are not modelled.

## Data flow
1. Agent (phase 0, `agent/ib.rs`, `agent/pci.rs`): facts only, from
   `/sys/class/infiniband` (injectable root). Per device: PCI placement
   via the `device` link (canonical path → host bridge, upstream bridges,
   address; `numa_node`, -1 → None). Per port: `state` (symbolic),
   `phys_state` (typed `PhysState`), `rate` (typed `IbRate` → rate_gbps,
   lanes, speed), `link_layer` (typed `LinkLayer`; on Ethernet the RoCE
   versions of populated GIDs from `gid_attrs/types/*`), `netdevs`
   (`device/net/*` filtered by `dev_port == port-1`, else the
   `gid_attrs/ndevs/*` names). GPUs get `pci` from the nvidia-smi bus id
   resolved against `/sys/bus/pci/devices`.
2. Orchestrator (`orchestrator::node_phase`): as each `Inventory` event
   arrives, `nccl_ib::nccl_nic_events(snapshot, NcclIbConfig)` emits one
   Node-scope `nccl_nics` outcome and, when it applies, the
   `nccl_nics.ceiling_gib_per_sec` metric (Unit GibPerSec). Same pattern
   as gpu_idle: the env is orchestrator config, the agent stays
   fact-only, and partial snapshots show the outcome immediately.
3. Report (`report::build`): `calibration.nccl_nics[host]` =
   `nccl_ib::summarize(inventory, config)` for every host with an
   inventory; the summary's consistency fields join the inventory
   majority vote; `report/nccl_nics.rs` renders the "nccl nics" section.
   Outcome, metric and summary all come from `summarize`, so they cannot
   disagree.

## Outcome policy (`NcclNicSummary::outcome`)
- No IB/RoCE ports at all → Skipped ("NCCL uses sockets"). Not a finding:
  an Ethernet-only fleet is a legitimate configuration. A host lacking
  IB in an IB fleet still dissents on the consistency fields.
- `NCCL_IB_DISABLE` set → Skipped.
- Ports exist, none selected → **Failed**, naming the HCA value and every
  port's exclusion reason. Feeds the verdict (Stragglers).
- Otherwise Passed.

## Ceiling math
Per selected port, payload Gb/s = sysfs rate × encoding efficiency
(`proto::payload_gbps`):
- InfiniBand SDR/DDR/QDR: the kernel prints the 8b/10b signalling rate
  (2.5/5/10 Gb/s per lane) → × 0.8. Older kernels print SDR as a bare
  width ("10 Gb/sec (4X)"), parsed as SDR.
- InfiniBand FDR: printed at 14 Gb/s per lane, 64b/66b → × 64/66.
- FDR10, EDR, HDR, NDR, XDR: printed at the data rate → × 1.
- Ethernet (RoCE): never corrected. mlx5 maps Ethernet speeds onto IB
  names (40GbE prints "40 Gb/sec (4X QDR)") but the number is already the
  Ethernet data rate.
- Unknown speed: taken at face value.
Ceiling GiB/s = Σ payload Gb/s × 1e9 / 8 / 2^30 (200 Gb/s = 23.28 GiB/s).
It is a line-rate ceiling, not achievable NCCL bus bandwidth (headers,
protocol overhead and PCIe limits come off it). `None` when any selected
port's rate is unknown (a partial sum would understate); `Some(0.0)` when
nothing is selected.

The metric is emitted only when the host has IB/RoCE ports, the transport
is enabled and the ceiling is known; a socket-fallback host therefore
reads 0.

## Fleet comparison
- Metric `nccl_nics.ceiling_gib_per_sec` (node scope) rides the normal
  MAD/aggregate/threshold machinery (`thresholds.absolute` can bound it).
- MAD is zero on an otherwise uniform fleet (8 hosts at 400 and one at
  200), so the summary also contributes exact-match consistency fields:
  `nccl_ib_ports` (selected count), `nccl_ib_link_layer`
  (`none`/`infiniband`/`ethernet`/`mixed`) and `nccl_ib_ceiling_gbps`
  (payload Gb/s, `unknown` when not computable).

## GPU<->NIC locality
`proto::pci_locality(a, b)`, recorded per selected port as
`gpu_locality: {gpu index: locality}`, nearest first:
- `pcie_switch`: the two upstream paths share a bridge below the host
  bridge (a PCIe switch, or a root port with a switch under it) — NCCL's
  PIX/PXB, where GPUDirect RDMA is effective.
- `host_bridge`: same host bridge only (PHB).
- `same_numa`: different host bridges (or unresolved paths), same NUMA
  node (NODE).
- `cross_numa`: different NUMA nodes (SYS).
- `unknown`: neither paths nor NUMA nodes available.
Shared bridges decide first; NUMA decides otherwise.

## Types and files
- `src/proto/ib.rs` — `PortState` (typed reading of the wire string),
  `PhysState`, `LinkLayer { Infiniband, Ethernet { roce_versions },
  Unknown }` (internally tagged `kind`), `RoceVersion`, `IbSpeed`,
  `IbRate` / `IbRateError`, `payload_gbps`, `gbps_to_gib_per_sec`,
  `PciLocation`, `PciLocality`, `pci_locality`, `IbDeviceInventory`.
- `src/proto/mod.rs` — `IbPortInventory` (+ lanes, speed, link_layer,
  phys_state, netdevs; `logical_state()`, `payload_gbps()`),
  `InventorySnapshot.ib_devices`, `GpuInventory.pci`, `TestId::NcclNics`.
- `src/agent/ib.rs` — `INFINIBAND_ROOT`, `IbInventory`, `probe(root)`.
- `src/agent/pci.rs` — `PCI_DEVICES_ROOT`, `locate(link)`,
  `resolve(root, &mut PciLocation)`.
- `src/nccl_ib/hca.rs` — `HcaFilter` (`unset`, `parse`, `matches`),
  `HcaEntry`, `HcaPort`.
- `src/nccl_ib/mod.rs` — `NcclIbConfig` (`new`, `from_env`),
  `PortExclusion`, `SelectedPort`, `ExcludedPort`, `SelectionLinkLayer`,
  `NcclNicSummary` (`outcome`, `ceiling_metric`, `consistency_fields`,
  `ceiling_gbps`, `link_layer`), `classify_port`, `summarize`,
  `nccl_nic_events`, `CEILING_METRIC`, `consistency::*`.
- `src/orchestrator/mod.rs` (`node_phase`) — emits the derived events.
- `src/report/nccl_nics.rs` — `summaries`, `render`.
- `tests/ib_inventory_tests.rs` — fixture sysfs trees (IB HDR NIC, RoCE
  NIC, storage NIC excluded via `^`, a down port, mixed rates), selection,
  outcome/metric, report integration, serde compatibility.

## Invariants
- The agent never decides selection; `TestId::NcclNics` is never emitted
  by an agent.
- A selected port is ACTIVE with an InfiniBand or Ethernet link layer.
- `HcaFilter::parse` is total: every string has NCCL's meaning.
- An unknown link layer (pre-v11 inventories) is never selected.
- All new wire/document fields are serde-defaulted (proto v11, schema
  v13).
