# Architecture

```text
Particle application -> Decomp
                     -> ParticleInfo (host | external/device)
                     -> domain + Cartesian topology
                     -> CUDA execution: classify, scan, pack, compact, unpack
                     -> data plane: MPI | NCCL | NVSHMEM
```

CUDA is an execution layer, not a communication backend. MPI owns bootstrap,
Cartesian topology, count metadata, and global statistics. The selected data
plane moves packed `ParticleRecord` payloads. `Decomp` retains topology metadata,
buffers, CUDA events, and backend state across timesteps.

Host execution uses MPI. CUDA execution uses device-resident SoA views and
persistent device buffers. NCCL groups sparse point-to-point operations.
NVSHMEM uses symmetric destination slots, one-sided puts, and peer signals.

The application decides when to migrate, exchange halos, rebuild a neighbor
list, and launch forces/integration. `Decomp` performs storage sizing, ghost
append/update, migration replacement, unpacking, and count bookkeeping through
the `ParticleInfo` interface.

With `GridPolicy::Dynamic`, the Cartesian connectivity remains fixed while the
x/y/z cut planes may become nonuniform. Before selected migrations, the runtime
collects one workload value per rank, reduces it into axis slabs, and places
equal-work quantiles inside the previous slabs. Relaxation and a minimum cell
width prevent abrupt or degenerate domains. Updating cuts refreshes each rank's
`LocalDomain` and invalidates the halo plan; the immediately following migration
establishes the new ownership. CPU `owner_rank()` and the CUDA classification
kernel both search the same cut arrays.

`TopologyAware` 0.3 discovery distinguishes hostname/node membership. Detailed
NVLink/NVSwitch/NIC discovery is intentionally a separate future provider;
callers may already inject a richer `TopologyInfo`. Discovery failure selects
the Balanced grid heuristic and never fails initialization.

## Call-path mapping

`exchange_halo()` is handled by the following function chain:

- User API entry point: `dsdecompx::Decomp::exchange_halo` (`src/runtime.cpp`)
- Core start/finish: `dsdecompx::Decomposition::begin_halo_exchange`, `dsdecompx::Decomposition::end_halo_exchange` (`src/halo.cpp`)
- Host boundary selection/packing: `dsdecompx::Decomposition::rebuild_halo_plan`, `dsdecompx::Decomposition::refresh_halo_buffers` (`src/halo.cpp`)
- Host neighbor record exchange: `dsdecompx::detail::MpiBackend::begin_exchange_records`, `dsdecompx::detail::MpiBackend::end_exchange_records` (`src/backends/mpi_backend.cpp`)
- Device path start/finish: `dsdecompx::detail::Backend::begin_device_exchange`, `dsdecompx::detail::Backend::end_device_exchange` implementations (`src/backends/*`)

The per-backend `exchange_halo` data-path functions are:

- MPI (host): `MpiBackend::begin_exchange_records` / `MpiBackend::end_exchange_records`
- CUDA-aware or staging MPI (device): `MpiBackend::begin_device_exchange` / `MpiBackend::end_device_exchange`
- NCCL (device): `NcclBackend::begin_device_exchange` / `NcclBackend::end_device_exchange` (`src/backends/nccl/nccl_backend.cpp`)
- NVSHMEM (device): `NvshmemBackend::begin_device_exchange` / `NvshmemBackend::end_device_exchange` (`src/backends/nvshmem/nvshmem_backend.cpp`)

The core implementation functions behind `migrate()` and `reverse_halo()` are:

- `migrate()`: `dsdecompx::Decomp::migrate` (`src/runtime.cpp`), `dsdecompx::Decomposition::migrate_particles` (`src/migration.cpp`)
- `rebalance()`: collective cut calculation and public orchestration in
  `dsdecompx::Decomp::rebalance` (`src/runtime.cpp`), followed by `migrate()`
- `reverse_halo()`: `dsdecompx::Decomp::reverse_halo` (`src/runtime.cpp`), `dsdecompx::Decomposition::reverse_halo` (`src/halo.cpp`)
- Host force reduction path: `dsdecompx::Decomposition::reverse_exchange` (`src/halo.cpp`)
