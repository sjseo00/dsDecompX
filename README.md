# dsDecompX 1.0

dsDecompX is a domain-decomposition and particle-communication library for simulations in which multiple CPUs or GPUs divide the simulation space. It keeps the fields molecular dynamics needs and also supports application-defined per-particle fields, so the same calls serve DEM, SPH,particle-in-cell and Lagrangian particle-tracking codes. You include one header,`<dsdecompx.hpp>`.  


## What it does

The simulation box is cut into a Cartesian grid, one subdomain per MPI rank (one GPU per rank onGPU runs). dsDecompX keeps each rank's particle data complete with three operations:

- `migrate()` sends particles that left a rank's subdomain to the rank that now owns them.
- `exchange_halo()` copies particles near a subdomain face to the neighboring ranks as ghosts, so forces across the face can be computed.
- `reverse_halo()` returns the forces computed on ghosts to the particles they are copies of.

It also provides application-defined fields (`add_field()`, `reverse_exchange()`,`forward_exchange()`), dynamic load balancing, and a split-phase halo exchange for overlapping communication with computation. It does not compute forces or build neighbor lists; that stays in your code.

## Features

- CPU (MPI) and GPU execution with the same function names; GPU calls add a CUDA stream
- GPU transports: CUDA-aware or staged MPI, NCCL, NVSHMEM, chosen automatically or by the user
- Periodic and non-periodic boxes, balanced, topology-aware and dynamic process grids
- Application-defined scalar and vector fields that move with particles and can be reduced back to owners
- Optional performance counters

## Requirements

- C++17 compiler, CMake 3.18 or newer, MPI with C++ bindings
- For GPU builds: the CUDA Toolkit; NCCL and NVSHMEM are optional

## Build and install

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release  
cmake --build build -j  
cmake --install build --prefix /desired/prefix
```

For GPU memory, add `-DENABLE_CUDA=ON` to the first command. Build options and verification are in[docs/02_Installation.md](docs/02_Installation.md).

## Minimal usage

In your `CMakeLists.txt`:

```cmake
find_package(dsDecompX 1.0 REQUIRED CONFIG)  
target_link_libraries(my_particles PRIVATE dsdecompx::dsdecompx)
```

In your code:

```cpp
#include <dsdecompx.hpp>  
  
int provided = MPI_THREAD_SINGLE;  
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);  
{  
    dsdecompx::Config config;  
    config.execution = dsdecompx::ExecutionMode::Host;   // ExecutionMode::Cuda for GPU memory  
    config.domain.upper = {100.0, 100.0, 100.0};  
    config.halo.width = cutoff + skin;  
    dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);  
  
    dsdecompx::ParticleInfo particles(n_local);  
    // ... fill positions, masses and unique global_id values ...  
  
    decomp.migrate(particles);          // move particles to their owners  
    decomp.exchange_halo(particles);    // add ghost particles after the owned ones  
    // ... compute forces on owned and ghost particles ...  
    decomp.reverse_halo(particles);     // return ghost forces to the owners  
}  
MPI_Finalize();
```

A complete, runnable program is in [docs/03_First_Program.md](docs/03_First_Program.md).

## Examples and sanity check

Small examples, a sanity check of your installation, and the programs of the dsDecompX paper are in [examples/](examples/README.md). To confirm that an installation works, build that folder against it and run `ctest`.

## Directory structure

| Path | Contents |
|---|---|
| `CMakeLists.txt` | Build and install of the library (options `ENABLE_CUDA`, `ENABLE_NCCL`, `ENABLE_NVSHMEM`) |
| `VERSION`, `CHANGELOG.md` | Version number and list of changes |
| `LICENSE` | GNU General Public License version 3 |
| `CONTRIBUTING.md` | Notes for contributors |
| `README.md`, `README.ko.md` | This file, and a short version in Korean |
| `cmake/` | `dsDecompXConfig.cmake.in` for `find_package(dsDecompX)`, and `FindNCCL.cmake`, `FindNVSHMEM.cmake` |
| `include/` | Public headers. Applications include only `dsdecompx.hpp` |
| `include/dsdecompx.hpp` | Main entry header: `Decomp`, `ParticleInfo`, and the communication functions |
| `include/config.hpp`, `domain.hpp`, `grid.hpp` | `Config`, the global and local domain, the halo width, and the process-grid policies (`Manual`, `Balanced`, `TopologyAware`, `Dynamic`) |
| `include/field.hpp` | Application-defined per-particle fields (`add_field`) |
| `include/topology.hpp` | Cartesian process grid and neighbor ranks |
| `include/decomposition.hpp`, `halo.hpp`, `migration.hpp` | Internal decomposition class, halo plan, and device buffers used by the library |
| `include/backend.hpp`, `backend_selection.hpp`, `backends/` | Communication backend interface and selection; MPI, NCCL, and NVSHMEM backends |
| `include/perf.hpp`, `performance.hpp` | Optional performance counters |
| `include/version.hpp` | Version constants |
| `src/` | Library sources |
| `src/runtime.cpp`, `decomp.cpp`, `topology.cpp`, `grid.cpp` | Set-up, process grid, neighbor ranks, and dynamic boundary adjustment |
| `src/migration.cpp`, `halo.cpp`, `buffer.cpp`, `device_fields.cpp` | Particle migration, halo exchange and reverse exchange, communication buffers, and field packing |
| `src/backend_selection.cpp`, `src/backends/` | Backend selection; MPI, NCCL, and NVSHMEM transfer code |
| `src/cuda/` | CUDA path: classification, packing, unpacking, and reduction kernels, device buffers, and events |
| `src/internal/` | Internal headers not installed |
| `docs/` | User manual: overview, installation, first program, use in an application, API reference, recipes |
| `examples/` | Example programs built against an installed dsDecompX; see [examples/README.md](examples/README.md) |
| `examples/getting_started`, `registered_field`, `domain_decomposition`, `halo_exchange`, `particle_migration` | Small CPU examples (the complete example of the paper is `getting_started`) |
| `examples/sanity_check/` | Unit tests, an MPI regression test, and optional CUDA tests, run with `ctest` |
| `examples/applications/` | MD, DEM, and SPH applications with dsDecompX, Cabana, and AMReX (paper, Section 4.3) |
| `examples/communication_test/` | Communication test with a fixed number of particles per rank (paper, Section 4.2) |
| `examples/load_balancing/` | Dynamic load balancing with a moving particle cluster (paper, Section 4.4) |

## Documentation


| Page                                                   | Contents                                                              |
|--------------------------------------------------------|-----------------------------------------------------------------------|
| [Overview](docs/01_Overview.md)                        | Concepts, the three operations, what the library does and what you do |
| [Installation](docs/02_Installation.md)                | Requirements, CMake options, install, linking from your project       |
| [First program](docs/03_First_Program.md)              | A small program, step by step                                         |
| [Using it in your code](docs/04_Using_in_Your_Code.md) | Particle data, step loop, custom fields, load balancing, CUDA         |
| [API reference](docs/05_API_Reference.md)              | Types, functions, arguments and defaults                              |
| [Recipes](docs/06_Recipes.md)                          | MD, per-layer feature exchange, DEM, SPH, particle tracking           |


## License

GPL-3.0-only; see [LICENSE](LICENSE).

 
