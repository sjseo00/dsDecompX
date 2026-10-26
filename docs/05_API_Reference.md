# 5. API Reference

This page answers one question: **what are the exact types, functions, arguments and defaults?**
For how to use them together, see [04_Using_in_Your_Code.md](04_Using_in_Your_Code.md).

Everything is in namespace `dsdecompx` and comes from one header:

```cpp
#include <dsdecompx.hpp>
```

Contents: [Conventions](#conventions) · [Config](#config) · [ParticleInfo](#particleinfo) ·
[Field and memory types](#field-and-memory-types) · [Decomp](#decomp) ·
[Topology](#topology) · [Process-grid functions](#process-grid-functions) ·
[Performance counters](#performance-counters) · [Version and build macros](#version-and-build-macros) ·
[Errors](#errors)

## Conventions

- **Collective.** Calls marked *collective* must be made by every rank of the communicator, in the same order.
- **Stream.** `ExecutionStream` is `void*`. Pass a `cudaStream_t`; use `nullptr` (the default) for host memory.
  CUDA memory needs a non-null stream.
- **Counts.** `owned_count()` is the number of owned particles, `ghost_count()` the number of ghosts, and
  `size()` their sum.
- **Vectors.** `Vec3d` is `std::array<double, 3>`, `Vec3i` is `std::array<int, 3>`.

## Config

```cpp
dsdecompx::Config config;   // every field has a default
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `domain` | `GlobalDomain` | see below | The global box. |
| `process_grid` | `ProcessGrid` | see below | Manual process-grid shape. |
| `halo` | `HaloConfig` | see below | Halo width and neighbor set. |
| `grid_policy` | `GridPolicy` | `Manual` | How the process grid is chosen. |
| `topology_info` | `TopologyInfo` | see below | Node information for `TopologyAware`. |
| `dynamic` | `DynamicLoadBalanceConfig` | see below | Load-balancing settings (used with `Dynamic`). |
| `execution` | `ExecutionMode` | `Cuda` in a CUDA build, otherwise `Host` | Host or GPU particle memory. |
| `advanced` | `AdvancedBackendConfig` | see below | Backend and GPU selection. |

### GlobalDomain

| Field | Type | Default | Meaning |
|---|---|---|---|
| `lower` | `Vec3d` | `{0, 0, 0}` | Lower corner of the box. |
| `upper` | `Vec3d` | `{1, 1, 1}` | Upper corner. Must be greater than `lower` on every axis. |
| `periodic` | `std::array<bool, 3>` | `{true, true, true}` | Periodic boundary per axis. Cannot be changed after construction. |

### ProcessGrid

| Field | Type | Default | Meaning |
|---|---|---|---|
| `dimensions` | `Vec3i` | `{0, 0, 0}` | Process-grid shape. Used when `grid_policy` is `Manual`; zero entries are filled in with `MPI_Dims_create`. The product must equal the number of ranks. |
| `allow_rank_reorder` | `bool` | `false` | Allow MPI to reorder ranks in the Cartesian communicator. Must stay `false` with `ExecutionMode::Cuda`. |

### HaloConfig

| Field | Type | Default | Meaning |
|---|---|---|---|
| `width` | `double` | `0.0` | A particle within this distance of a subdomain face is copied to the neighbor across that face. Must not be negative. |
| `include_edges_and_corners` | `bool` | `true` | `true`: all 26 neighbors. `false`: only the 6 face neighbors. |

### GridPolicy

| Value | Meaning |
|---|---|
| `Manual` | Use `process_grid.dimensions`. |
| `Balanced` | Choose the factorization of the rank count with the best surface-to-volume ratio for the box. |
| `TopologyAware` | Like `Balanced`, but minimizes weighted communication, counting inter-node faces more heavily (`topology_info`). Falls back to `Balanced` when node information is unavailable. |
| `Dynamic` | Start from the `Balanced` grid and let cut planes move at run time. |

### TopologyInfo

Only used with `TopologyAware`. If `available` is false, the `Decomp` constructor detects nodes from
MPI processor names.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `available` | `bool` | `false` | Set `true` when you fill in `rank_to_node` yourself. |
| `rank_to_node` | `std::vector<int>` | empty | Node index of each rank. |
| `intra_node_weight` | `double` | `1.0` | Relative cost of a face shared inside a node. |
| `inter_node_weight` | `double` | `4.0` | Relative cost of a face shared between nodes. |
| `local_rank`, `local_size`, `node_count` | `int` | `0`, `1`, `1` | Filled by detection. |

### DynamicLoadBalanceConfig

| Field | Type | Default | Meaning |
|---|---|---|---|
| `interval` | `std::size_t` | `1` | Check balance inside every N-th `migrate()` call. `0` disables the automatic path; `rebalance()` still works. |
| `imbalance_threshold` | `double` | `1.10` | Rebalance when `max(work) / mean(work)` reaches this. Must be at least 1. |
| `relaxation` | `double` | `0.5` | Fraction of the computed shift applied per rebalance. Must be in (0, 1]. |
| `minimum_cell_fraction` | `double` | `0.10` | Smallest cell width as a fraction of a uniform cell. Must be in (0, 1]. Cells are also kept at least a halo width wide where the uniform cell allows it. |

### ExecutionMode

| Value | Meaning |
|---|---|
| `Host` | Particle arrays are in CPU memory. Transport is MPI. |
| `Cuda` | Particle arrays are in GPU memory (`CudaDevice` or `CudaManaged`). Requires a CUDA build. |

### AdvancedBackendConfig

| Field | Type | Default | Meaning |
|---|---|---|---|
| `selection` | `BackendSelection` | `Auto` | `Auto`, `Mpi`, `Nccl` or `Nvshmem`. `Auto` in CUDA execution tries NVSHMEM, then NCCL, then MPI. |
| `read_environment` | `bool` | `true` | When `selection` is `Auto`, read the environment variable `BACKEND` (`auto`, `mpi`, `nccl`, `nvshmem`). |
| `allow_host_staging` | `bool` | `true` | Let the MPI backend stage GPU buffers through pinned CPU memory when MPI cannot take device pointers. |
| `cuda_device` | `int` | `kCudaDeviceCurrent` | `kCudaDeviceCurrent` (`-1`): keep the device already selected. `kCudaDeviceByLocalRank` (`-2`): one GPU per rank on a node. `>= 0`: that device ordinal. |

`BackendKind` (`Mpi`, `Nccl`, `Nvshmem`) is what `Decomp::backend_kind()` reports.
Helpers: `parse_backend_selection(std::string_view)` (case-insensitive, throws
`std::invalid_argument` on unknown text), `backend_selection_name(BackendSelection)`, and
`backend_compiled(BackendSelection)`, which says whether that backend was built in.

## ParticleInfo

The container every `Decomp` call works on. It holds owned particles in slots
`0 ... owned_count()-1` and ghosts after them.

### Construction

| Call | Meaning |
|---|---|
| `ParticleInfo()` | Empty. |
| `ParticleInfo(std::size_t owned)` | Library-owned host arrays for `owned` particles (all fields zero-initialized). |
| `ParticleInfo::allocate(MemorySpace space, std::size_t owned)` | `Host`: same as above. `CudaDevice` / `CudaManaged`: library-owned GPU arrays (CUDA build only; otherwise throws `std::runtime_error`). They grow automatically (by at least 1.5x) when migrants or ghosts need room. |
| `ParticleInfo::external(MemorySpace space, std::size_t capacity, std::size_t owned, std::size_t ghosts = 0)` | Attach your own arrays. `capacity` is the number of particles each array can hold. Set the pointers with the setters below. |
| `ParticleInfo(std::size_t owned, std::size_t ghosts, ParticleView storage)` | Attach arrays described by a `ParticleView`. Throws if `storage.count < owned + ghosts`. |
| `ParticleInfo(std::size_t owned, std::size_t ghosts, Provider& provider)` | Like the previous, with a provider that can supply larger storage on request. |

`ParticleInfo` is movable and not copyable.

### Setters for attached arrays

Each returns `ParticleInfo&`, so calls chain.

| Call | Sets |
|---|---|
| `positions(double* x, double* y, double* z)` | Position arrays. |
| `velocities(double* vx, double* vy, double* vz)` | Velocity arrays. |
| `forces(double* fx, double* fy, double* fz)` | Force arrays. |
| `properties(double* mass, double* charge, int* type, int* global_id)` | Mass, charge, type and unique id. |
| `add_field<T>(std::string name, T* data, std::size_t capacity = 0, std::size_t components = 1)` | Registers an application field (see [04](04_Using_in_Your_Code.md#6-application-defined-fields)). `T` is `int32_t`, `int64_t`, `uint32_t`, `uint64_t`, `float` or `double`. `capacity` counts particles; `0` means the current capacity of the container. Throws `std::invalid_argument` for an empty name, null data, zero components, a duplicate name, or a capacity smaller than `size()`. |

### Access

| Call | Returns |
|---|---|
| `owned_count()`, `ghost_count()`, `size()` | Counts. |
| `capacity()` | Number of particles the storage can hold. |
| `memory_space()` | `MemorySpace::Host`, `CudaDevice` or `CudaManaged`. |
| `uses_external_memory()` | `true` for borrowed (not library-owned) arrays. |
| `x_data()`, `y_data()`, `z_data()`, `vx_data()` ... `vz_data()`, `fx_data()` ... `fz_data()`, `mass_data()`, `charge_data()`, `type_data()`, `global_id_data()` | The array addresses, for CUDA kernels and external functions. |
| `x`, `y`, `z`, `vx`, `vy`, `vz`, `fx`, `fy`, `fz`, `mass`, `charge` (`std::vector<double>`), `type`, `global_id` (`std::vector<int>`) | The arrays themselves, for library-owned host storage only. |

### Counts and storage

| Call | Meaning |
|---|---|
| `resize_owned(std::size_t count)` | Set the owned count (library-owned storage), discarding ghosts. Checks application-field capacity. Throws `std::logic_error` for borrowed memory. |
| `reset_counts(std::size_t owned, std::size_t ghosts = 0)` | Declare how many owned and ghost particles your own arrays hold. Throws if the storage is smaller than `owned + ghosts`. Use it after you change the arrays yourself. |
| `reset_view(ParticleView storage)` | Replace the attached storage (for example after you reallocated your arrays). The new storage must hold at least the current `size()`. The registered application fields are replaced by the ones listed in `storage.fields` (none if it is null). |

### Provider

A provider lets dsDecompX ask for larger storage while it runs:

```cpp
class ParticleInfo::Provider {
public:
    virtual ParticleView particle_info_view(std::size_t minimum) = 0;
    virtual StorageResizeResult resize_particle_info(
        std::size_t required, std::size_t preserve) = 0;
};
```

`resize_particle_info` must return a view with room for at least `required` particles that keeps the
first `preserve` particles' data; `StorageResizeResult` is `{ParticleView view; bool reallocated;}`.

## Field and memory types

| Type | Values / members |
|---|---|
| `MemorySpace` | `Host`, `CudaDevice`, `CudaManaged` |
| `FieldType` | `Int32`, `Int64`, `UInt32`, `UInt64`, `Float32`, `Float64` (`field_type_v<T>` maps a C++ type) |
| `ReduceOp` | `Sum`, `Min`, `Max`, `Replace` |
| `ReverseExchangeField` | `{std::string name; ReduceOp operation = ReduceOp::Sum;}` |
| `ParticleFieldView` | `{name, type, components, capacity, data}`: a non-owning application field |
| `ParticleView` | `{memory_space, count, x ... global_id, fields, field_count}`: pointers to all arrays; `count` is the capacity |
| `ParticleRecord` | The built-in wire record: `x y z vx vy vz fx fy fz mass charge type global_id` |

## Decomp

```cpp
dsdecompx::Decomp decomp(MPI_Comm communicator, dsdecompx::Config config = {});
```

Collective. Creates the process grid, assigns subdomains and selects the backend. Throws
`std::invalid_argument` for a null communicator, a non-positive box extent, a negative halo width,
invalid `dynamic` settings, a manual grid that does not match the rank count, or
`allow_rank_reorder` with CUDA execution; `std::runtime_error` if MPI is not initialized, the
thread level is below `MPI_THREAD_FUNNELED`, CUDA execution is requested without a CUDA build, or a requested
backend was not built. Movable, not copyable.

### Status

| Call | Returns |
|---|---|
| `topology()` | `const Topology&` for this rank (see [Topology](#topology)). |
| `backend_kind()` | `BackendKind` actually in use. |
| `mpi_takes_device_pointers()` | `true` when the MPI backend is in use and the MPI library accepts device pointers. |
| `owner_rank(const Vec3d& position)` | The rank that owns the position. Periodic coordinates are wrapped first; on a non-periodic axis an outside position goes to the nearest boundary rank. |
| `any_rank_true(bool local_flag)` | *Collective.* `true` on every rank if any rank passed `true`. |
| `has_out_of_domain_particles(const ParticleInfo&)` | *Collective.* `true` if any rank has an owned particle that belongs to another rank. Host memory only. |

### Box and halo width

| Call | Meaning |
|---|---|
| `update_domain(const GlobalDomain& domain)` | Set a new box; cut planes scale proportionally; invalidates the halo plan. Periodicity must not change (throws `std::invalid_argument`). |
| `set_halo_width(double width)` | Set the halo width (not negative). Invalidates the halo plan if the value changed. |
| `halo_width()` | Current halo width. |
| `invalidate_halo()` | Force the next `exchange_halo()` to rebuild the ghost list. |

### migrate

```cpp
MigrationResult migrate(ParticleInfo& particles, ExecutionStream stream = nullptr);
```

*Collective.* Sends each owned particle to the rank that owns its (wrapped) position, directly to
the destination rank, so a particle may cross several subdomains in one call. Wraps periodic
coordinates into the box, discards ghosts, resizes the owned set, and invalidates the halo plan.
With `GridPolicy::Dynamic` it may first move the cut planes (see `dynamic.interval`). Application
fields move with the particles. Needs `x y z global_id` (host) or all arrays (CUDA).

`MigrationResult`:

| Field | Meaning |
|---|---|
| `ownership_changed` | Any particle was sent or received on this rank. |
| `previous_owned_count` | Owned count before the call. |
| `retained_count` | Particles that stayed. |
| `sent_count` | Particles sent away. |
| `received_count` | Particles received. |
| `new_owned_count` | Owned count after the call. |

### exchange_halo

```cpp
HaloResult exchange_halo(ParticleInfo& particles, HaloExchangeOptions options = {});
```

*Collective.* Appends ghosts to `particles` (after the owned slots) and sets `ghost_count()`. Each
ghost has the owner's velocity, mass, charge, type, id and application fields, a position shifted
across periodic boundaries, and zero force. The ghost set is every owned particle within
`halo.width` of a face, sent to each neighbor that face borders. Along an axis with one process,
and for a run on one rank, no ghosts are produced. The array storage grows when needed.

`HaloExchangeOptions`:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `rebuild` | `bool` | `false` | `true`: rebuild the ghost list from the current positions. `false`: reuse the previous list (it is rebuilt automatically when none is valid, for example after `migrate()`). Pass the same value on every rank (required for CUDA memory). |
| `stream` | `ExecutionStream` | `nullptr` | CUDA stream for GPU memory (must be non-null). |
| `positions_only` | `bool` | `false` | CUDA memory only: on a reused plan with no application fields, send only positions (see [04](04_Using_in_Your_Code.md#reusing-the-ghost-list)). |

`HaloResult`:

| Field | Meaning |
|---|---|
| `ghost_count` | Ghosts held after the call. |
| `layout_changed` | Ghost count or order differs from the previous call (also `true` after any rebuild on CUDA). Ghost-indexed neighbor lists are invalid. |
| `storage_reallocated` | The particle storage had to grow, so array addresses may have changed. |
| `plan_reused` | CUDA memory: the previous plan was refreshed in place. Identical on all ranks. |

With a reused list on host memory, ghost positions and application fields are refreshed; the other
ghost fields keep the values of the last rebuild.

### Split-phase exchange

```cpp
void begin_halo_exchange(ParticleInfo& particles, HaloExchangeOptions options = {});
HaloResult end_halo_exchange(ParticleInfo& particles);
bool can_refresh_halo(const ParticleInfo& particles) const;
```

`exchange_halo()` is `begin_halo_exchange()` followed by `end_halo_exchange()`. Between the two,
do not resize or exchange `particles` and do not start another halo, migration or reverse
operation. `can_refresh_halo()` is `true` when a non-rebuild exchange of CUDA memory would reuse
the existing plan; it is always `false` for host memory.

### reverse_halo

```cpp
void reverse_halo(ParticleInfo& particles, ExecutionStream stream = nullptr);
```

*Collective.* Adds `fx fy fz` of each ghost to the owned particle it copies, on the owning rank.
Follows the ghost list of the latest `exchange_halo()`; needs force arrays and storage for
`owned + ghosts`.

### reverse_exchange

```cpp
void reverse_exchange(ParticleInfo& particles,
                      const std::vector<ReverseExchangeField>& fields,
                      ExecutionStream stream = nullptr);
```

*Collective.* For each named application field, reduces the ghost values into the owners with the
given `ReduceOp`. The owner's existing value takes part in the reduction (for `Sum`, the ghost
value is added to it). Fields must be registered with `add_field()`. Names in `fields` must be
unique. On CUDA memory at most 16 fields per call.

### forward_exchange

```cpp
void forward_exchange(ParticleInfo& particles,
                      const std::vector<std::string>& fields,
                      ExecutionStream stream = nullptr);
```

*Collective.* Copies the named application fields from the owned particles into the ghost slots
that hold their copies, along the ghost list of the latest CUDA `exchange_halo()`, without rebuilding
it. It is the inverse of `reverse_exchange()` with `ReduceOp::Sum`. CUDA memory only, with a
non-null stream; at most 16 fields; each field's per-particle size must be a multiple of 4 bytes.
With no valid device ghost list (one rank, or after a `migrate()`) it does nothing.

### rebalance

```cpp
RebalanceResult rebalance(ParticleInfo& particles,
                          RebalanceOptions options = {},
                          ExecutionStream stream = nullptr);
```

*Collective.* Moves the cut planes using each rank's work value, then migrates, so ghosts are
discarded.

| `RebalanceOptions` field | Default | Meaning |
|---|---|---|
| `local_work` | `-1.0` | This rank's work; must be finite and non-negative. Negative means "use the owned count". |
| `force` | `false` | Ignore `dynamic.imbalance_threshold`. |

| `RebalanceResult` field | Meaning |
|---|---|
| `attempted` | The balance check ran. |
| `geometry_changed` | The cut planes moved. |
| `imbalance_before` | `max(work) / mean(work)` before the call. |
| `migration` | The `MigrationResult` of the migrate that followed. |

## Topology

`decomp.topology()` returns a `const Topology&`:

| Member | Returns |
|---|---|
| `rank()`, `size()` | This rank's index and the number of ranks. |
| `dimensions()` | `Vec3i` process-grid shape. |
| `coordinates()` | `Vec3i` position of this rank in the grid. |
| `local_domain()` | `LocalDomain{lower, upper}`, the subdomain of this rank. |
| `axis_cuts()` | `std::array<std::vector<double>, 3>`: the cut positions per axis (`dimensions()[axis] + 1` values). |
| `neighbors()` | `std::vector<Neighbor>`, up to 26 entries. |

`Neighbor`: `rank` (`-1` when the direction leaves a non-periodic box), `direction` (`Vec3i` with entries in -1..1),
`periodic_shift` (`Vec3d` added to positions sent that way), `direction_slot` (an index 0-25 of the direction), `is_self` (the neighbor is this rank).
Directions along an axis with a single process are omitted.

## Process-grid functions

```cpp
GridShape choose_process_grid(int nranks, const DomainShape& domain,
                              const TopologyInfo& topology, GridPolicy policy,
                              GridShape manual = {});
```

The planner the `Decomp` constructor uses, available standalone (it needs no MPI). `GridShape`
has `px`, `py`, `pz` and `as_array()`; `DomainShape` has `lengths` and `periodic`. With `Manual` it
returns `manual` after checking that `px * py * pz == nranks`; otherwise it picks the best
factorization. Throws `std::invalid_argument` for non-positive `nranks` or lengths.

## Performance counters

```cpp
decomp.set_performance_counters_enabled(true, /*neighbor_detail=*/false);
PerfStats local  = decomp.get_stats();         // this rank
PerfStats global = decomp.get_global_stats();  // collective, aggregated over all ranks
decomp.reset_stats();
```

Counters are off until enabled and compiled in unless the library was built with
`-DENABLE_PERF_COUNTERS=OFF` (then the getters return empty stats). `neighbor_detail = true`
also records per-neighbor byte counts in `local.neighbor_bytes` (rank-local only).

`PerfStats` fields: `halo_calls`, `migration_calls`, `bytes_sent`, `bytes_received`,
`buffer_reallocations`, `peak_communication_buffer_capacity`, and the timing aggregates
`pack_ms`, `communication_ms`, `wait_ms`, `unpack_ms`, `migration_classify_ms`,
`migration_compact_ms`. A timing aggregate (`SampleAggregate`) has `count`, `sum`, `min`, `max`.
In `get_global_stats()` the counts are summed over ranks, `peak_communication_buffer_capacity` is
the maximum, and each timing aggregate combines all ranks' samples (`sum` over ranks, `min`/`max` over ranks).

## Version and build macros

| Name | Meaning |
|---|---|
| `dsdecompx::version_major`, `version_minor`, `version_patch`, `version_string` | Version (`1.0.0`). |
| `dsdecompx::header_path()` | Path of the `dsdecompx.hpp` your translation unit included. |
| `HAS_CUDA`, `HAS_NCCL`, `HAS_NVSHMEM`, `ENABLE_PERF_COUNTERS` | Defined by the CMake target to match how the library was built. |
| `DSDECOMPX_HALO_POSITIONS_ONLY` | Defined when `HaloExchangeOptions::positions_only` exists. |

## Errors

dsDecompX reports problems with standard exceptions. The messages begin with `dsDecompX:`.

| Exception | Raised for |
|---|---|
| `std::invalid_argument` | Bad arguments or configuration: missing or incomplete particle arrays, undersized or mismatched application fields, a null stream for CUDA memory, an unknown backend name, invalid `Config` values. |
| `std::length_error` | Fixed external storage or an application field too small for the particles that must fit. |
| `std::logic_error` | Calls in the wrong order: a second `begin_halo_exchange()`, `end_halo_exchange()` without `begin`, a migration or reverse call while an exchange is in flight, `resize_owned()` on borrowed memory. |
| `std::runtime_error` | MPI not initialized or thread level too low, CUDA or backend failures, a backend that was requested but not built, an application-field schema that differs between ranks, duplicate particle ids (debug builds). |

## Limitations

- 3-D Cartesian decomposition only; ghosts come from at most the 26 adjacent subdomains.
- CUDA memory: at most 16 application fields per exchange, and a CUDA halo supports at most 26 neighbors.
- `forward_exchange()` is CUDA only; its NVSHMEM path has not been run.
- `has_out_of_domain_particles()` reads particle positions on the host, so use it with host
  memory.
