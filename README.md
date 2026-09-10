# dsDecompX 0.3

English | [한국어](README.ko.md)

dsDecompX is a domain-decomposition and particle-communication library for
simulations in which multiple CPUs or GPUs divide the simulation space. It
keeps MD convenience fields while supporting application-defined fields for
DEM, SPH, PIC, and Lagrangian particle tracking. Normal users only need to
include `<dsdecompx.hpp>`.

Core operations are:

- direct-destination particle migration;
- 26-neighbor halo exchange;
- application-defined scalar and fixed-width vector fields; and
- generic reverse ghost-contribution reduction.

## Quick start

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix /desired/prefix
```

```cmake
find_package(dsDecompX 0.3 REQUIRED CONFIG)
target_link_libraries(my_particles PRIVATE dsdecompx::dsdecompx)
```

```cpp
dsdecompx::Config config;
config.domain.upper = {100.0, 100.0, 100.0};
config.halo.width = cutoff + skin;
config.execution = dsdecompx::ExecutionMode::Host; // or Cuda

dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);
dsdecompx::ParticleInfo particles(1000);

decomp.migrate(particles);
decomp.exchange_halo(particles);

// Calculate particle interactions here. Registered fields are available on
// both owned and ghost particles.

decomp.reverse_halo(particles);
```

- `migrate()` sends particles that left the local region to their new rank/GPU.
- `exchange_halo()` exchanges neighboring Particle data needed for force calculations near region boundaries.
- `reverse_halo()` returns forces calculated using neighboring particles to the original particles.

CUDA builds use the same function names, only adding a CUDA stream argument.
CPU and GPU support the same set of options.

## Documentation

- [Installation and build options](docs/user-guide/installation.md)
- [Architecture](docs/architecture.md)
- [API reference](docs/api-reference.md) — full `ParticleInfo`/`Decomp`/`Config` reference, code examples, and current limitations
- [Application-defined particle fields](docs/user-guide/custom-fields.md)
- [Backend selection](docs/user-guide/backend-selection.md), [MPI](docs/user-guide/mpi.md), [CUDA](docs/user-guide/cuda.md), [NCCL](docs/user-guide/nccl.md), [NVSHMEM](docs/user-guide/nvshmem.md)
- [Performance measurement](docs/user-guide/performance.md)
- [Troubleshooting](docs/user-guide/troubleshooting.md)
- [Benchmark suite](examples/benchmark/README.md)

GPL-3.0-only; see [LICENSE](LICENSE).
