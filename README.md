# dsDecomp 0.3

English | [한국어](README.ko.md)

dsDecomp is a domain-decomposition and particle-communication library for
simulations in which multiple CPUs or GPUs divide the simulation space. It
keeps MD convenience fields while supporting application-defined fields for
DEM, SPH, PIC, and Lagrangian particle tracking. Normal users only need to
include `<runtime.hpp>`.

Core operations are:

- direct-destination particle migration;
- 26-neighbor halo exchange;
- application-defined scalar and fixed-width vector fields; and
- generic reverse ghost-contribution reduction.

## Build

CPU build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix /desired/prefix
```

CUDA build:

```sh
cmake -S . -B build -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

For a CUDA build, dsDecomp automatically detects installed NCCL and NVSHMEM libraries and includes the available backends.

For the standalone CPU/GPU strong- and weak-scaling benchmark, see [examples/benchmark/README.md](examples/benchmark/README.md).

```cmake
find_package(dsDecomp 0.3 REQUIRED CONFIG)
target_link_libraries(my_particles PRIVATE dsdecomp::dsdecomp)
```

## Basic call order

```cpp
dsdecomp::Config config;
config.domain.upper = {100.0, 100.0, 100.0};
config.halo.width = cutoff + skin;
config.execution = dsdecomp::ExecutionMode::Host; // or Cuda

dsdecomp::Decomp decomp(MPI_COMM_WORLD, config);
constexpr std::size_t capacity = 2000;
dsdecomp::ParticleInfo particles(1000);

std::vector<double> temperature(capacity);
std::vector<int> species(capacity);
particles
    .add_field<double>("temperature", temperature.data(), capacity)
    .add_field<int>("species", species.data(), capacity);

decomp.migrate(particles);
decomp.exchange_halo(particles);

// Calculate particle interactions here. Registered fields are available on
// both owned and ghost particles.

decomp.reverse_halo(particles);
// Or reduce an application-defined ghost contribution:
decomp.reverse_exchange(
    particles, {{"temperature", dsdecomp::ReduceOp::Sum}});
```

- `migrate()` sends particles that left the local region to their new rank/GPU.
- `exchange_halo()` exchanges neighboring Particle data needed for force calculations near region boundaries.
- `reverse_halo()` returns forces calculated using neighboring particles to the original particles.
- `reverse_exchange()` reduces selected ghost fields into their owners.

## `ParticleInfo`: the object that contains Particle data

`ParticleInfo` describes non-owning or library-owned particle arrays. Position
and global ID are intrinsic to host decomposition. Velocity, force, mass,
charge, and type remain available as MD convenience fields; arbitrary
application data is registered with `add_field()`.

### Create Particle data (CPU memory)

```cpp
dsdecomp::ParticleInfo particles(1000);

particles.x[0] = 1.0;
particles.y[0] = 2.0;
particles.z[0] = 3.0;
particles.mass[0] = 1.0;
particles.global_id[0] = 0;
```

`ParticleInfo(1000)` directly creates Particle data (CPU memory) for 1000 particles.

### Create Particle data (CUDA memory)

```cpp
auto particles = dsdecomp::ParticleInfo::allocate(
    dsdecomp::MemorySpace::CudaDevice, 1000);

double* d_x = particles.x_data();
double* d_y = particles.y_data();
double* d_z = particles.z_data();
```

`ParticleInfo::allocate()` creates all required Particle data in the selected memory and releases that memory automatically when the object is destroyed.

### Use existing Particle data (CPU/CUDA memory)

```cpp
auto particles = dsdecomp::ParticleInfo::external(
    dsdecomp::MemorySpace::CudaDevice,
    capacity, owned_count, ghost_count);

particles
    .positions(d_x, d_y, d_z)
    .velocities(d_vx, d_vy, d_vz)
    .forces(d_fx, d_fy, d_fz)
    .properties(d_mass, d_charge, d_type, d_global_id);
```

This method tells `ParticleInfo` where the application's existing Particle data (CPU/CUDA memory) is stored. The application that created the memory must release it.

### Application-defined fields

Host applications can attach scalar or fixed-width vector fields that follow
particles through halo exchange and migration:

```cpp
particles
    .add_field<double>("temperature", temperature, capacity)
    .add_field<int>("species", species, capacity)
    .add_field<double>("angular_velocity", omega, capacity, 3);

decomp.reverse_exchange(
    particles, {{"temperature", dsdecomp::ReduceOp::Sum}});
```

This supports DEM, SPH, PIC, and Lagrangian particle data without changing the
MD convenience fields. See [the custom-field guide](docs/user-guide/custom-fields.md)
for storage, schema, reduction, and current CUDA limitations.

### `ParticleInfo` functions

| Function | Input arguments | Return value | What it does |
|---|---|---|---|
| `ParticleInfo()` | None | A new `ParticleInfo` object | Creates empty Particle data (CPU memory). |
| `ParticleInfo(count)` | `count`: number of owned particles | A new `ParticleInfo` object | Creates Particle data (CPU memory) for `count` particles. |
| `allocate(memory, count)` | `memory`: memory location, `count`: number of owned particles | A new `ParticleInfo` object | Creates Particle data directly in the selected CPU or CUDA memory. |
| `external(memory, capacity, owned, ghosts)` | `memory`: memory location, `capacity`: maximum number that fits, `owned`: owned particle count, `ghosts`: received particle count | A new `ParticleInfo` object | Prepares an object that will use existing Particle data (CPU/CUDA memory). |
| `positions(x, y, z)` | Memory addresses containing x/y/z positions | Current `ParticleInfo&` | Registers the position information. Other registration functions can be chained after it. |
| `velocities(vx, vy, vz)` | Memory addresses containing vx/vy/vz velocities | Current `ParticleInfo&` | Registers the velocity information. |
| `forces(fx, fy, fz)` | Memory addresses containing fx/fy/fz forces | Current `ParticleInfo&` | Registers the force information. |
| `properties(mass, charge, type, global_id)` | Memory addresses containing mass/charge/type/ID | Current `ParticleInfo&` | Registers the remaining particle properties. |
| `add_field<T>(name, data, capacity, components)` | Field name, storage, particle capacity, optional component count | Current `ParticleInfo&` | Registers an application-defined field for halo exchange and migration. |
| `resize(count)` | `count`: new total particle count | Nothing | Changes the size of directly created Particle data (CPU memory). |
| `resize_owned(count)` | `count`: new owned particle count | Nothing | Changes the owned count and grows the Particle data memory when needed. |
| `reset_counts(owned, ghosts)` | `owned`: owned particle count, `ghosts`: received particle count | Nothing | Updates how many particles are currently stored in external Particle data. |
| `owned_count()` | None | Number of owned particles | Returns the number of particles assigned to this rank. |
| `ghost_count()` | None | Number of ghost particles | Returns the number of particles received from neighboring ranks. |
| `size()` | None | Total number of particles currently in use | Returns owned count plus ghost count. |
| `capacity()` | None | Maximum number of particles that fit | Returns how many particles the current memory can hold. |
| `memory_space()` | None | `MemorySpace` | Reports whether the Particle data is in CPU, CUDA device, or CUDA managed memory. |
| `uses_external_memory()` | None | `bool` | Returns `true` when using memory created by the application. |
| `x_data()` through `global_id_data()` | None | Memory address of the corresponding information | Returns a Particle data address for a CUDA kernel or external function. |

Directly created Particle data (CPU memory) is accessed as `particles.x`, `particles.vx`, `particles.fx`, `particles.mass`, `particles.type`, and `particles.global_id`. The same form applies to `y/z`, `vy/vz`, `fy/fz`, and `charge`.

## `Decomp`: space division and particle communication

### Construction and status

| Function | Input arguments | Return value | What it does |
|---|---|---|---|
| `Decomp(communicator, config)` | `communicator`: MPI communicator to use, `config`: space and execution settings | A new `Decomp` object | Assigns a region to each MPI rank and prepares particle communication. |
| `topology()` | None | `const Topology&` | Returns this rank's number, region, and neighboring rank information. |
| `backend_kind()` | None | `BackendKind` | Returns the communication method actually selected: MPI, NCCL, or NVSHMEM. |
| `owner_rank(position)` | `position`: particle `{x,y,z}` position | Owning rank number | Finds the rank responsible for that position. Use it for the initial distribution of particles. |
| `any_rank_true(local_flag)` | `local_flag`: this rank's `bool` value | The same `bool` value on every rank | Checks whether at least one rank passed `true`. Every rank must call it. |
| `has_out_of_domain_particles(particles)` | `particles`: Particle data to inspect | The same `bool` value on every rank | Checks whether any rank has an owned particle outside its assigned region. |

```cpp
dsdecomp::Decomp decomp(MPI_COMM_WORLD, config);
int my_rank = decomp.topology().rank();
int particle_rank = decomp.owner_rank({x, y, z});
bool need_migration = decomp.has_out_of_domain_particles(particles);
```

### Change the region and boundary distance

| Function | Input arguments | Return value | What it does |
|---|---|---|---|
| `update_domain(domain)` | `domain`: new global bounds and periodic settings | Nothing | Recalculates the global space and each rank's assigned region. |
| `set_halo_width(width)` | `width`: distance to check from a region boundary | Nothing | Makes `exchange_halo()` exchange Particle data within this distance of a boundary. |
| `halo_width()` | None | Current boundary distance | Returns the current setting. |
| `invalidate_halo()` | None | Nothing | Makes the next `exchange_halo()` search for boundary particles again from the beginning. |

```cpp
domain.upper = {new_lx, new_ly, new_lz};
decomp.update_domain(domain);
decomp.set_halo_width(cutoff + skin);
```

### `migrate()`

```cpp
auto result = decomp.migrate(particles, stream);
```

| Input arguments | Return value |
|---|---|
| `particles`: Particle data (CPU/CUDA memory) to move, `stream`: CUDA stream (`nullptr` on CPU) | `MigrationResult`: result describing how many particles were sent and received |

Calling it performs these actions:

1. Finds particles that left this rank's region.
2. Finds the new rank/GPU responsible for each particle.
3. Sends those particles to their new rank/GPU.
4. Receives particles that entered this rank's region from other ranks/GPUs.
5. Adjusts positions of particles that crossed a periodic boundary.
6. Removes previously received ghost particles.

If no particle left its region, no particle is moved and `ownership_changed` is `false`.

| Result field | Meaning |
|---|---|
| `ownership_changed` | Whether any particle changed its assigned rank. |
| `previous_owned_count` | Number of owned particles before the call. |
| `retained_count` | Number of particles that remained on this rank. |
| `sent_count` | Number of particles sent to other ranks. |
| `received_count` | Number of particles received from other ranks. |
| `new_owned_count` | Number of particles owned by this rank after the call. |

CUDA `Auto` selects an available method in this order: NVSHMEM, NCCL, CUDA-aware MPI, then regular MPI. NVSHMEM, NCCL, and CUDA-aware MPI transfer Particle data directly from GPU memory. Regular MPI transfers through pinned CPU memory when staging is allowed.

### `exchange_halo()`

```cpp
auto result = decomp.exchange_halo(
    particles,
    dsdecomp::HaloExchangeOptions{rebuild, stream});
```

| Input arguments | Return value |
|---|---|
| `particles`: Particle data (CPU/CUDA memory) whose boundary information will be exchanged, `options`: whether to search again and which CUDA stream to use | `HaloResult`: received particle count and memory-change information |

Calling it performs these actions:

1. Finds particles close to the boundary of this rank's region.
2. Sends their built-in particle data and registered application fields to neighboring ranks/GPUs.
3. Receives boundary Particle data from neighboring regions.
4. Adds the received Particle data to the ghost section of `particles`.
5. Makes it possible to calculate forces between local particles and particles in neighboring regions.

`migrate()` changes the rank responsible for a particle. `exchange_halo()` does not change the responsible rank.

| Option | Meaning |
|---|---|
| `rebuild` | When `true`, searches for particles near the boundary again from the beginning. |
| `stream` | CUDA stream used for the operation. Use `nullptr` on CPU. |

| Result field | Meaning |
|---|---|
| `ghost_count` | Number of ghost particles received. |
| `layout_changed` | Whether the ghost particle count or order differs from the previous call. |
| `storage_reallocated` | Whether larger memory had to be created because the Particle data did not fit. |

### `reverse_halo()`

```cpp
decomp.reverse_halo(particles, stream);
```

| Input arguments | Return value |
|---|---|
| `particles`: Particle data (CPU/CUDA memory) containing calculated forces, `stream`: CUDA stream (`nullptr` on CPU) | Nothing |

Calling it performs these actions:

1. Collects `fx`, `fy`, and `fz` calculated using ghost particles.
2. Sends those forces back to the rank/GPU responsible for the original particles.
3. Adds the returned forces to the original particles' `fx`, `fy`, and `fz`.

With CUDA, NCCL or CUDA-aware MPI sends force data directly from GPU memory. Regular MPI with pinned CPU-memory staging can be used when necessary and permitted.

The NVSHMEM backend does not currently support `reverse_halo()`. If `Cuda + Auto` selected NVSHMEM, calling this function fails.

### `reverse_exchange()`

```cpp
decomp.reverse_exchange(
    particles,
    {{"density_contribution", dsdecomp::ReduceOp::Sum},
     {"contact_temperature", dsdecomp::ReduceOp::Max}});
```

`reverse_exchange()` sends values calculated on ghost particles back to the
rank that owns the original particle and applies the requested reduction.

| Reduction | Owner-side result |
|---|---|
| `ReduceOp::Sum` | Adds every ghost contribution. |
| `ReduceOp::Min` | Keeps the minimum owner/ghost value. |
| `ReduceOp::Max` | Keeps the maximum owner/ghost value. |
| `ReduceOp::Replace` | Replaces the owner value in deterministic receive order. |

The selected fields must first be registered with `add_field()`. Field names,
scalar types, component counts, and registration order must match on every MPI
rank; dsDecomp validates this collectively. Registered memory must remain valid
and have capacity for `owned_count + ghost_count` particles.

Application-defined field communication and generic reverse reduction
currently use host memory and the MPI host data plane. Built-in particle fields
retain the existing CUDA-aware MPI, NCCL, and NVSHMEM paths. CUDA custom fields
are rejected explicitly rather than being silently omitted.

## Calling the API with CUDA

CPU and CUDA use the same function names. CUDA calls only add a CUDA stream.

```cpp
config.execution = dsdecomp::ExecutionMode::Cuda;
dsdecomp::Decomp decomp(MPI_COMM_WORLD, config);

decomp.migrate(particles, stream);
decomp.exchange_halo(particles, {true, stream});
decomp.reverse_halo(particles, stream);
```

The default communication setting for a CUDA build is `Auto`. Normal users do not need to choose NCCL, NVSHMEM, or MPI. Use `decomp.backend_kind()` to inspect the actual selection.

## `Config`: execution settings

| Setting | What changing it does |
|---|---|
| `domain.lower`, `domain.upper` | Sets the beginning and end of the global simulation space. |
| `domain.periodic` | Enables or disables periodic boundaries separately for x, y, and z. |
| `process_grid.dimensions` | Sets how many ranks are placed along x, y, and z. `{0,0,0}` chooses automatically. |
| `process_grid.allow_rank_reorder` | Allows MPI to change the rank placement order. |
| `halo.width` | Exchanges Particle data for particles within this distance of a boundary. |
| `halo.include_edges_and_corners` | When `true`, also communicates with edge and corner neighbors, not only face neighbors. |
| `grid_policy` | Chooses whether rank placement is manual, based on domain dimensions, or based on node connectivity. |
| `topology_info` | Supplies node information used by `TopologyAware` placement. |
| `execution` | Use `Host` for Particle data (CPU memory) and `Cuda` for Particle data (CUDA memory). |

A CUDA-enabled library defaults to `Cuda`. A CPU-only library defaults to `Host`.

### Expert backend settings

Normal users do not need to change these settings.

| Setting | What changing it does |
|---|---|
| `advanced.selection` | Forces MPI, NCCL, or NVSHMEM instead of `Auto`. If initialization fails, it reports an error instead of selecting another method. |
| `advanced.read_environment` | Controls whether `DSDECOMP_BACKEND` is read. |
| `advanced.allow_host_staging` | Allows GPU data to be moved through pinned CPU memory when CUDA-aware MPI is unavailable. |

## `Topology`: placement information for the current rank

```cpp
const auto& topology = decomp.topology();
```

| Function | Input arguments | Return value | Meaning |
|---|---|---|---|
| `rank()` | None | `int` | Current rank number. |
| `size()` | None | `int` | Total number of ranks. |
| `dimensions()` | None | `{x count, y count, z count}` | Number of ranks placed along each direction. |
| `coordinates()` | None | `{x position, y position, z position}` | Position of this rank in the rank placement. |
| `local_domain()` | None | Beginning and end of this rank's region | Space assigned to this rank. |
| `neighbors()` | None | List of `Neighbor` objects | Ranks that share a boundary with this rank. |

```cpp
auto local_box = topology.local_domain();
for (const auto& neighbor : topology.neighbors()) {
    int neighbor_rank = neighbor.rank;
}
```

## `choose_process_grid()`

```cpp
auto grid = dsdecomp::choose_process_grid(
    mpi_size, domain_shape, topology_info,
    dsdecomp::GridPolicy::Balanced);
```

`choose_process_grid()` calculates how many MPI ranks should be placed along x, y, and z and returns a `GridShape`. Use it to inspect the expected placement before constructing `Decomp`.

| Input arguments | Return value |
|---|---|
| `nranks`: total rank count, `domain`: domain dimensions, `topology`: node placement information, `policy`: calculation method, `manual`: manually specified placement | `GridShape`: `px`, `py`, and `pz`, the rank counts along x/y/z |

- `Manual`: uses user-provided `px`, `py`, and `pz`.
- `Balanced`: makes region sizes similar based on the x, y, and z lengths.
- `TopologyAware`: also considers the cost of within-node and between-node communication.

## Performance measurement functions

```cpp
decomp.set_performance_counters_enabled(true, true);
auto local = decomp.get_stats();
auto global = decomp.get_global_stats();
decomp.reset_stats();
```

| Function | Input arguments | Return value | What it does |
|---|---|---|---|
| `set_performance_counters_enabled(enabled, neighbor_detail)` | `enabled`: whether to record, `neighbor_detail`: whether to record each neighbor | Nothing | Enables or disables time and communication-volume recording. |
| `get_stats()` | None | This rank's `PerfStats` | Returns this rank's call counts, elapsed times, and sent/received bytes. |
| `get_global_stats()` | None | Combined `PerfStats` for all ranks | Combines records from every rank. Every rank must call it. |
| `reset_stats()` | None | Nothing | Resets all recorded performance values to zero. |

`PerfStats` contains halo/migration call counts, sent/received bytes, Particle data memory growth counts, data-preparation time, communication time, waiting time, and time spent placing received data into Particle data.

## Other public functions

| Function/API | Input arguments | Return value | What it does |
|---|---|---|---|
| `parse_backend_selection(text)` | `text`: `"auto"`, `"mpi"`, `"nccl"`, or `"nvshmem"` | `BackendSelection` | Converts text into a backend selection value. |
| `backend_selection_name(value)` | `value`: `BackendSelection` | Backend name string | Converts a backend selection value into text. |
| `backend_compiled(value)` | `value`: `BackendSelection` to inspect | `bool` | Checks whether that backend is included in this dsDecomp build. |
| `GridShape::as_array()` | None | `{px, py, pz}` | Returns the x/y/z rank counts together. |
| `SampleAggregate::add(value)` | `value`: measurement to add | Nothing | Updates the measurement count, sum, minimum, and maximum. |
| `version_string` | None | Version string | Provides the current dsDecomp version. |

These functions are used when building configuration parsers, diagnostics, or performance comparison tools.

## Low-level API for library integration

Normal users do not need this section. Use it only when an external library automatically grows Particle data (CUDA memory) and changes its memory addresses while running.

| API | Input arguments | Return value | What it does |
|---|---|---|---|
| `ParticleInfo(owned, ghosts, view)` | `owned`: owned count, `ghosts`: received count, `view`: Particle data memory addresses and size | A new `ParticleInfo` object | Registers low-level memory information at once. |
| `ParticleInfo(owned, ghosts, provider)` | Owned/ghost counts and an external memory-management object | A new `ParticleInfo` object | Connects an external object that can grow Particle data memory automatically. |
| `reset_view(view)` | `view`: changed memory addresses and size | Nothing | Updates the addresses of external Particle data. |
| `Provider::particle_info_view(minimum)` | `minimum`: minimum required particle count | `ParticleView` | Gives dsDecomp the current Particle data addresses and size. |
| `Provider::resize_particle_info(required, preserve)` | `required`: required size, `preserve`: number of particles to keep | `StorageResizeResult` | Grows memory and returns the new addresses and whether reallocation occurred. |

## Code examples

The examples below assume that MPI has been initialized and `Decomp decomp(...)` has already been constructed.

### CPU example 1: create Particle data (CPU memory)

```cpp
dsdecomp::ParticleInfo particles(1000);

for (std::size_t i = 0; i < particles.owned_count(); ++i) {
    particles.x[i] = initial_x(i);
    particles.y[i] = initial_y(i);
    particles.z[i] = initial_z(i);
    particles.mass[i] = 1.0;
    particles.global_id[i] = static_cast<int>(i);
}

decomp.migrate(particles);
decomp.exchange_halo(particles);
decomp.reverse_halo(particles);
```

### CPU example 2: pass addresses of existing Particle data (CPU memory)

```cpp
const std::size_t owned = 1000;
const std::size_t capacity = 2000;
std::array<std::vector<double>, 11> values;
std::array<std::vector<int>, 2> integers;
for (auto& field : values) field.resize(capacity);
for (auto& field : integers) field.resize(capacity);

auto particles = dsdecomp::ParticleInfo::external(
    dsdecomp::MemorySpace::Host, capacity, owned);
particles
    .positions(values[0].data(), values[1].data(), values[2].data())
    .velocities(values[3].data(), values[4].data(), values[5].data())
    .forces(values[6].data(), values[7].data(), values[8].data())
    .properties(values[9].data(), values[10].data(),
                integers[0].data(), integers[1].data());

decomp.migrate(particles);
decomp.exchange_halo(particles);
```

### CUDA example 1: create Particle data (CUDA memory)

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

auto particles = dsdecomp::ParticleInfo::allocate(
    dsdecomp::MemorySpace::CudaDevice, 1000);

initialize_particles_on_gpu(particles, stream);

decomp.migrate(particles, stream);
decomp.exchange_halo(particles, {false, stream});
decomp.reverse_halo(particles, stream);

cudaStreamDestroy(stream);
```

### CUDA example 2: pass addresses of existing Particle data (CUDA memory)

```cpp
const std::size_t owned = 1000;
const std::size_t capacity = 2000;
double* values[11]{};
int* integers[2]{};

for (auto& field : values)
    cudaMalloc(&field, capacity * sizeof(double));
for (auto& field : integers)
    cudaMalloc(&field, capacity * sizeof(int));

auto particles = dsdecomp::ParticleInfo::external(
    dsdecomp::MemorySpace::CudaDevice, capacity, owned);
particles
    .positions(values[0], values[1], values[2])
    .velocities(values[3], values[4], values[5])
    .forces(values[6], values[7], values[8])
    .properties(values[9], values[10], integers[0], integers[1]);

decomp.migrate(particles, stream);
decomp.exchange_halo(particles, {false, stream});

for (auto* field : integers) cudaFree(field);
for (auto* field : values) cudaFree(field);
```

When external Particle data addresses are provided, do not release that memory before destroying `ParticleInfo`.

## Current limitations

- The communicated particle fields are fixed to position, velocity, force, mass, charge, type, and ID.
- Only 3D rectangular spaces divided along x, y, and z are supported.
- NCCL and NVSHMEM still require MPI when ranks are initially connected.
- NVSHMEM does not support `reverse_halo()`.
- One `Decomp` object cannot run two particle communication operations at the same time.

GPL-3.0-only; see [LICENSE](LICENSE).
