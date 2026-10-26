# dsDecompX 1.0

dsDecompX is a domain-decomposition and particle-communication library for simulations in whichmultiple CPUs or GPUs divide the simulation space. It keeps the fields molecular dynamics needsand also supports application-defined per-particle fields, so the same calls serve DEM, SPH,particle-in-cell and Lagrangian particle-tracking codes. You include one header,`<dsdecompx.hpp>`.  


## What it does

The simulation box is cut into a Cartesian grid, one subdomain per MPI rank (one GPU per rank onGPU runs). dsDecompX keeps each rank's particle data complete with three operations:

- `migrate()` sends particles that left a rank's subdomain to the rank that now owns them.
- `exchange_halo()` copies particles near a subdomain face to the neighboring ranks as ghosts, soforces across the face can be computed.
- `reverse_halo()` returns the forces computed on ghosts to the particles they are copies of.

It also provides application-defined fields (`add_field()`, `reverse_exchange()`,`forward_exchange()`), dynamic load balancing, and a split-phase halo exchange for overlappingcommunication with computation. It does not compute forces or build neighbor lists; that stays inyour code.

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

## Documentation


| Page                                                   | Contents                                                              |
|--------------------------------------------------------|-----------------------------------------------------------------------|
| [Overview](docs/01_Overview.md)                        | Concepts, the three operations, what the library does and what you do |
| [Installation](docs/02_Installation.md)                | Requirements, CMake options, install, linking from your project       |
| [First program](docs/03_First_Program.md)              | A small program, step by step                                         |
| [Using it in your code](docs/04_Using_in_Your_Code.md) | Particle data, step loop, custom fields, load balancing, CUDA         |
| [API reference](docs/05_API_Reference.md)              | Types, functions, arguments and defaults                              |
| [Recipes](docs/06_Recipes.md)                          | MD, per-layer feature exchange, DEM, SPH, particle tracking           |


##   
License

GPL-3.0-only; see [LICENSE](LICENSE).

 
