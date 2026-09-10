# DeepScaleMD integration-boundary audit

Classification: A belongs in dsDecompX, B belongs in the MD application, and C
is duplicate legacy code.

| Area | Occurrence | Class | Resolution |
|---|---|---:|---|
| MPI | `dsDecompX/src/backends/*` topology, counts, payload, stats | A | Hidden behind the backend/runtime API |
| MPI | DeepScaleMD lifecycle, model/output/thermodynamic reductions | B | Simulation policy, not decomposition |
| NCCL | `dsDecompX/src/backends/nccl` | A | Sparse GPU payload data plane |
| Halo | `dsDecompX/src/halo.cpp` | A | Canonical membership/pack/exchange implementation |
| Halo | `MDSystem` exchange scheduling | B | Calls the single synchronous `Decomp` API |
| Halo | old application `nvshmem_halo` implementation | C | Removed; backend calls now exist only in dsDecompX |
| Migration | `dsDecompX/src/migration.cpp` | A | Canonical wrap/classify/exchange/compact operation |
| Migration | `MDSystem::migrate_particles_check` | B | Scheduling; dsDecompX replaces storage |
| Domain | `dsDecompX/src/topology.cpp`, `grid.cpp` | A | Global/local bounds and grid selection |
| Domain | copying local bounds into `SimulationBox` | B | Adapter for force/integrator consumers |
| Neighbor rank | dsDecompX Cartesian topology | A | Canonical calculation |
| Neighbor rank | cached `MDSystem::neighbor_ranks` | B | Read-only compatibility cache populated from dsDecompX |
| Buffers | dsDecompX host/device/pinned/symmetric communication storage | A | Persistent and capacity-managed |
| Buffers | force, PME, neighbor-list, MD particle/ghost arrays | B | Application numerical/storage state |

DeepScaleMD constructs one `dsdecompx::Decomp`. Host storage derives from
`ParticleInfo`; CUDA storage uses `ParticleInfo`. It does not
access backend, plan, operation-handle, or low-level decomposition APIs.

A source audit under `dsDecompX/include` and `dsDecompX/src` finds no DeepScaleMD
header, type, namespace, compile definition, or global variable. The dependency
is one-way:

```text
DeepScaleMD -> dsDecompX -> MPI + optional CUDA/NCCL/NVSHMEM
```
