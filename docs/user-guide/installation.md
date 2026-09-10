# Installation

Requirements are a C++17 compiler and MPI. Enable CUDA with
`ENABLE_CUDA`; NCCL and NVSHMEM each imply CUDA. Configure with CMake,
build, install, then use `find_package(dsDecompX 0.3 CONFIG REQUIRED)` and link
`dsdecompx::dsdecompx`. Optional roots are found through `NCCL_ROOT`/`NCCL_HOME`
and `NVSHMEM_ROOT`/`NVSHMEM_HOME`. The installed package re-discovers all
transitive dependencies.

Start with MPI-only, then enable one GPU backend at a time on the target system.
The application must initialize MPI before constructing `Decomp` and destroy it
before finalizing MPI.

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

For a CUDA build, dsDecompX automatically detects installed NCCL and NVSHMEM libraries and includes the available backends.

Build options use the same unprefixed names in standalone and parent-project
builds:

| Option | Default | Purpose |
|---|---:|---|
| `ENABLE_CUDA` | `OFF` | Build the CUDA execution layer |
| `ENABLE_NCCL` | `OFF` | Build the NCCL backend; also enables CUDA |
| `ENABLE_NVSHMEM` | `OFF` | Build the NVSHMEM backend; also enables CUDA |
| `AUTO_DETECT_GPU_BACKENDS` | `ON` | Detect installed NCCL/NVSHMEM libraries in CUDA builds |
| `ENABLE_PERF_COUNTERS` | `ON` | Compile performance counters |
| `BUILD_TESTS` | `OFF` | Build CPU/MPI unit and regression tests |
| `RUN_GPU_TESTS` | `OFF` | Register CUDA backend tests with CTest |
| `BUILD_EXAMPLES` | `OFF` | Build the standalone examples |
| `BUILD_BENCHMARKS` | `OFF` | Build the scaling benchmark |

Runtime overrides are `BACKEND=auto|mpi|nccl|nvshmem` and
`CUDA_AWARE_MPI=0|1`. See the [backend-selection guide](backend-selection.md)
for details.

For the standalone CPU/GPU strong- and weak-scaling benchmark, see
[examples/benchmark/README.md](../../examples/benchmark/README.md).

## Use it from another CMake project

```cmake
find_package(dsDecompX 0.3 REQUIRED CONFIG)
target_link_libraries(my_particles PRIVATE dsdecompx::dsdecompx)
```
