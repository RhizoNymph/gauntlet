# NCCL NIC Selection and the NIC Bandwidth Ceiling

## Scope
Answer, per host and without guessing: which InfiniBand/RoCE ports would
NCCL's IB transport use under the run's resolved env, what link layer and
rate are they, and what is their summed line rate (the "nccl nic
ceiling")? Make that a fleet-comparable node metric plus consistency
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
- External net plugins' own device selection, `NCCL_IB_MERGE_NICS`, and
  NCCL_* values a node supplies itself (shell profile, `/etc/nccl.conf`,
  `~/.nccl.conf`): only the `[nccl] env` map is known to the orchestrator
  (see nccl_env.md, which never captures inherited variables).

## What NCCL does, and what is modelled
In order (`crate::nccl_ib::classify_port`; the first that applies is the
recorded exclusion):
1. `NCCL_NET` — NCCL compares it with `strcasecmp` against each network's
   name: an external plugin first, then the built-in "IB" and "Socket".
   Modelled: "Socket" (any case) selects sockets, so every port is
   excluded with `net_socket`. Unset or "IB" keep the IB transport. Any
   other value names a plugin (`NetChoice::Plugin`): its behaviour is not
   modelled. The IB selection is reported as if IB ran, and the summary
   carries a caveat that the report shows.
2. `NCCL_IB_DISABLE` — read like NCCL's `ncclLoadParam`:
   `strtoll(str, &end, 0)`, falling back to the default 0 when
   `errno != 0` (out of range) or `end == str` (no digits). So C
   whitespace, a sign, `0x` hex and leading-`0` octal are accepted, and
   trailing text is ignored: "1", "0x1", "010" and "1x" all disable;
   "0", "0x0", "08", "yes" and "x1" do not.
3. Logical port state is ACTIVE.
4. Link layer is InfiniBand or Ethernet (RoCE).
5. The port passes the `NCCL_IB_HCA` filter (unset = everything passes).

`NCCL_IB_HCA` grammar (`src/nccl_ib/hca.rs`, after NCCL's
`parseStringList` / `matchIfList`):
- A leading `^` makes it an exclude list; after that, a leading `=` makes
  names exact instead of prefixes. `^=mlx5_0` excludes exactly mlx5_0;
  `=^x` is an exact include of a device literally named `^x`.
- Entries are comma-separated, `name` or `name:port`. Empty entries are
  skipped and nothing is trimmed (`a, b` has an entry " b" that matches
  nothing).
- Prefix matching is the default: `mlx5_1` matches mlx5_1 *and* mlx5_10.
- The port is read with C `atoi` (no digits means 0, trailing junk is
  ignored). -1 means any port; 0 or junk matches no real port.
- `:1` (no name) is dropped. An empty list matches everything, so `^`
  alone excludes everything.
- NCCL's 32-entry and 64-byte caps are not modelled.

## Data flow
1. **Agent** (phase 0, `agent/ib.rs`, `agent/pci.rs`, `agent/sysfs.rs`):
   facts only, from `/sys/class/infiniband` (injectable root).
   - Per device: PCI placement via the `device` link. The path is
     canonicalized and anchored at the *innermost* host bridge (Intel VMD
     nests `pci10000:e0` under `pci0000:00/0000:00:0e.0`). It yields the
     upstream bridges, the address and `numa_node` (-1 → None).
   - Per port: `state` (symbolic), `phys_state` (typed), `rate` (typed
     `PortRate`: `known {gbps, lanes, speed}`, `unparseable {error}` with
     the typed `IbRateError` plus a warn log, or `unreadable`) and
     `link_layer` (typed; on Ethernet, the RoCE versions of the populated
     GIDs).
   - Per port, `netdevs`: on Ethernet, the GID table's
     `gid_attrs/ndevs/*` (deduped, non-empty), so RoCE LAG reports
     `bond0` and switchdev never reports a VF representor. Otherwise
     (IPoIB, or an empty GID table), `device/net/*` whose `dev_port` is
     port-1, skipping VF/SF representors (`phys_port_name` like `pf0vf3`
     / `pf0sf1`). An entry without `dev_port` is not attributed.
   - GPUs get `pci` from the nvidia-smi bus id, resolved against
     `/sys/bus/pci/devices`.
2. **Orchestrator** (`orchestrator/derive.rs`): `node_phase` routes every
   inventory through `derive_inventory_events(snapshot,
   &InventoryDerivation)`, the one hook for inventory-derived tests. It
   emits, behind the snapshot:
   - `gpu_idle` outcomes, one per GPU;
   - then, when the env is known, the Node-scope `nccl_nics` outcome;
   - the `nccl_nics.ceiling_gib_per_sec` metric, when it applies;
   - an `AgentEvent::NcclNics { summary }`.

   The collector records the summary as `hosts.<h>.nccl_nics`. Partial
   snapshots show all of it immediately.
3. **Report**: reads the recorded results, never re-derives:
   - the consistency vote chains each host's
     `NcclNicSummary::consistency_fields` (`report/consistency.rs`);
   - `report/nccl_nics.rs` renders the "nccl nics" section from
     `hosts.*.nccl_nics` and the recorded outcome.

   A document whose hosts carry no summary (pre-v13, or no derivation)
   shows no section and no NIC consistency fields.

## The summary (`proto::nccl_nics::NcclNicSummary`)
Records the env inputs it was derived under:
- `hca` (`NCCL_IB_HCA` value or null);
- `net` (`NetChoice`: `default` / `ib` / `socket` / `plugin {name}`);
- `ib_disabled`.

And the result:
- `selected` ports: device, port, link_layer, typed `rate`, netdevs,
  numa_node, `gpu_locality`;
- `excluded` ports with a tagged `reason`: `net_socket`, `ib_disabled`,
  `not_active {state}`, `unsupported_link_layer` or `filtered_by_hca`.

The ceiling is not stored. Everything else is a method of this one value,
so the outcome, metric, consistency fields and table cannot drift:
- `ceiling_gbps()` / `ceiling_gib_per_sec()`;
- `ceiling_unknown_reason()`, `transport_off()`, `net_caveat()`;
- `outcome()`, `ceiling_metric()`, `consistency_fields()`, `link_layer()`;
- `ExcludedPort::describe()`, which the outcome reason and the table
  share, and `describe_hca()`, the one spelling of the HCA label.

## Outcome policy (`NcclNicSummary::outcome`)
- No IB/RoCE ports at all → Skipped ("NCCL uses sockets"). Not a finding:
  an Ethernet-only fleet is a legitimate configuration. A host lacking
  IB in an IB fleet still dissents on the consistency fields.
- Transport off by `NCCL_NET=Socket` or `NCCL_IB_DISABLE` → Skipped, with
  that reason.
- Ports exist, none selected → **Failed**. The reason names the HCA value
  and every port's exclusion, and the failure feeds the verdict
  (Stragglers).
- Otherwise Passed (including under an unmodelled plugin; the caveat
  shows in the report note).

## Ceiling math
Per selected port, payload Gb/s = the printed rate × the encoding
efficiency (`proto::payload_gbps`):
- InfiniBand SDR/DDR/QDR: the kernel prints the 8b/10b signalling rate
  (2.5/5/10 Gb/s per lane), so the factor is 0.8. Older kernels print SDR
  as a bare width ("10 Gb/sec (4X)"), which is parsed as SDR.
- InfiniBand FDR: printed at 14 Gb/s per lane with 64b/66b encoding, so
  the factor is 64/66.
- FDR10, EDR, HDR, NDR, XDR: printed at the data rate, factor 1.
- Ethernet (RoCE): never corrected. mlx5 maps Ethernet speeds onto IB
  names (40GbE prints "40 Gb/sec (4X QDR)"), but the number is already
  the Ethernet data rate.
- Unknown speed: taken at face value.

Ceiling GiB/s = Σ payload Gb/s × 1e9 / 8 / 2^30 (200 Gb/s = 23.28 GiB/s).
It is a line-rate ceiling, not achievable NCCL bus bandwidth: headers,
protocol overhead and PCIe limits come off it.
- `None` when any selected port's rate is not `known`.
  `ceiling_unknown_reason()` then names each such port with its typed
  reason (e.g. "mlx5_0:1 unparseable rate: ib rate "fast" does not start
  with a number"), and the report note shows it.
- `Some(0.0)` when nothing is selected.

The metric is emitted only when the host has IB/RoCE ports, the transport
is on and the ceiling is known; a socket-fallback host therefore reads 0.

## Fleet comparison
- Metric `nccl_nics.ceiling_gib_per_sec` (node scope) rides the normal
  MAD/aggregate/threshold machinery (`thresholds.absolute` can bound it).
- MAD is zero on an otherwise uniform fleet (8 hosts at 400 and one at
  200), so the summary also contributes exact-match consistency fields:
  - `nccl_ib_ports` (the selected count);
  - `nccl_ib_link_layer` (`none` / `infiniband` / `ethernet` / `mixed`);
  - `nccl_ib_ceiling_gbps` (payload Gb/s, or `unknown`).

## GPU<->NIC locality
`proto::pci_locality(a, b)`, recorded per selected port as
`gpu_locality: {gpu index: locality}`, nearest first:
- `pcie_switch`: the two upstream paths (from the innermost host bridge)
  share a bridge below it, i.e. a PCIe switch, or a root port with a
  switch under it. This is NCCL's PIX/PXB, where GPUDirect RDMA is
  effective.
- `host_bridge`: same host bridge only (PHB). Two root ports of one VMD
  domain land here, never in `pcie_switch`.
- `same_numa`: different host bridges (or unresolved paths), same NUMA
  node (NODE).
- `cross_numa`: different NUMA nodes (SYS).
- `unknown`: neither paths nor NUMA nodes available.

Shared bridges decide first; NUMA decides otherwise.

## Types and files
- `src/proto/ib.rs`:
  - port-level types: `IbPortInventory` (+ the private compatibility
    decoder `IbPortWire`, which lifts a v1-v10 `rate_gbps` into
    `PortRate::Known`), `PortRate`, `IbRate` / `IbRateError` (serde,
    tagged `kind`), `IbSpeed`, `PortState`, `PhysState`, `LinkLayer`,
    `RoceVersion`;
  - placement: `PciLocation`, `PciLocality`, `pci_locality`,
    `IbDeviceInventory`;
  - rate helpers: `payload_gbps`, `gbps_to_gib_per_sec`.
- `src/proto/nccl_nics.rs`:
  - summary types: `NcclNicSummary`, `SelectedPort`, `ExcludedPort`,
    `PortExclusion`, `NetChoice`, `SelectionLinkLayer`;
  - `describe_hca`;
  - constants: `IB_HCA`, `IB_DISABLE`, `NET`, `CEILING_METRIC`,
    `consistency::*`.
- `src/proto/mod.rs` — `InventorySnapshot.ib_devices`, `GpuInventory.pci`,
  `TestId::NcclNics`, `AgentEvent::NcclNics`.
- `src/agent/ib.rs` — `INFINIBAND_ROOT`, `IbInventory`, `probe(root)`.
- `src/agent/pci.rs` — `PCI_DEVICES_ROOT`, `locate(link)`,
  `resolve(root, &mut PciLocation)`.
- `src/agent/sysfs.rs` — `read_trimmed`, shared with `inventory.rs`.
  Empty-after-trim is `None`: sysfs uses an empty read for an unset
  attribute.
- `src/nccl_ib/hca.rs` — `HcaFilter` (`unset`, `parse`, `matches`),
  `HcaEntry`, `HcaPort`.
- `src/nccl_ib/mod.rs` — `NcclIbConfig` (`new(hca, ib_disable, net)`,
  `from_env`), the `strtoll(.., 0)` model, `classify_port`, `summarize`.
- `src/orchestrator/derive.rs` — `InventoryDerivation`,
  `derive_inventory_events`.
- `src/orchestrator/collect.rs` — `HostObservations.nccl_nics`.
- `src/report/consistency.rs` — the majority vote, inventory + NIC fields.
- `src/report/nccl_nics.rs` — `render`.
- `tests/ib_inventory_tests.rs` — the fixture sysfs trees: an IB HDR NIC,
  a RoCE NIC, a storage NIC excluded via `^`, a down port, mixed rates,
  VMD, RoCE LAG, switchdev, and unparseable/empty rates. The tests cover
  selection under NCCL_IB_HCA / NCCL_NET / NCCL_IB_DISABLE, the derived
  events, report integration through the real collector, and serde
  compatibility.

## Invariants
- The agent never decides selection. `TestId::NcclNics` and
  `AgentEvent::NcclNics` are never emitted by an agent.
- The report never re-derives NIC results from an inventory.
- A selected port is ACTIVE with an InfiniBand or Ethernet link layer.
- `HcaFilter::parse` is total: every string has NCCL's meaning.
- An unknown link layer (pre-v11 inventories) is never selected.
- All new wire and document fields are serde-defaulted (proto v11,
  schema v13).
