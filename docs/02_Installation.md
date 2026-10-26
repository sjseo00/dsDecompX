# 2. Installation

## Contents

- [Requirements](#requirements)
- [Build and install (CPU)](#build-and-install-cpu)
- [Build and install (CUDA)](#build-and-install-cuda)
- [Check the installation](#check-the-installation)
- [Use it from your own project](#use-it-from-your-own-project)
- [Limitations](#limitations)

This page answers one question: **how do I build dsDecompX, install it, and link it into my
project?**

## Requirements

| Requirement | Notes |
|---|---|
| C++17 compiler | The library target requires `cxx_std_17`. |
| CMake 3.18 or newer | |
| MPI with C++ bindings | The only mandatory dependency. Your program must initialize MPI with at least `MPI_THREAD_FUNNELED` (see [03_First_Program.md](03_First_Program.md)). |
| CUDA Toolkit | Only for GPU builds (`-DENABLE_CUDA=ON`). |
| NCCL, NVSHMEM | Optional extras for GPU builds. Neither is needed for CUDA support itself. |

## Build and install (CPU)

From the dsDecompX source directory:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix /desired/prefix
```

## Build and install (CUDA)

```sh
cmake -S . -B build_cuda -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda -j
cmake --install build_cuda --prefix /desired/prefix_cuda
```

Use a separate build directory and install prefix for each variant. An application built against
the CPU install cannot use GPU memory, and the CUDA install changes the default of
`Config::execution` to `ExecutionMode::Cuda`.

NCCL and NVSHMEM are picked up automatically when `-DENABLE_CUDA=ON` and the libraries are
found. To point CMake at them, or to force or forbid them:

```sh
cmake -S . -B build_cuda -DENABLE_CUDA=ON \
  -DNCCL_ROOT=/path/to/nccl \
  -DNVSHMEM_ROOT=/path/to/nvshmem
```

If your CUDA toolkit does not accept the default host compiler, add for example
`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10`.

### CMake options

| Option | Default | Meaning |
|---|---|---|
| `ENABLE_CUDA` | `OFF` | Build the CUDA execution layer (GPU-resident particle memory). |
| `ENABLE_NCCL` | `AUTO` | `ON`: required, configuration fails if NCCL is missing. `OFF`: never built. `AUTO`: built in a CUDA build when found. `ON` also turns `ENABLE_CUDA` on. |
| `ENABLE_NVSHMEM` | `AUTO` | Same three states as `ENABLE_NCCL`. |
| `AUTO_DETECT_GPU_BACKENDS` | `ON` | Lets the `AUTO` settings above search for installed NCCL and NVSHMEM. Set `OFF` to build neither unless forced `ON`. |
| `NCCL_ROOT`, `NVSHMEM_ROOT` | unset | Install prefixes of NCCL and NVSHMEM. The environment variables of the same names (and `NCCL_HOME`, `NVSHMEM_HOME`, `CUDA_HOME`, `NVHPC_ROOT`) are also searched. |
| `CMAKE_CUDA_ARCHITECTURES` | `70;80;90` | GPU architectures to compile when you set none yourself (the default comes from `DSDECOMPX_CUDA_ARCHITECTURES`). Use e.g. `-DCMAKE_CUDA_ARCHITECTURES=80`, or `native` on a machine with the target GPU. |
| `ENABLE_PERF_COUNTERS` | `ON` | Compile the optional performance counters (see `get_stats()` in [05_API_Reference.md](05_API_Reference.md)). |

The library is built as a static library unless you pass the standard CMake option
`-DBUILD_SHARED_LIBS=ON`.

### What gets installed

| Location | Contents |
|---|---|
| `<prefix>/include/` | `dsdecompx.hpp` and the headers it includes (`config.hpp`, `domain.hpp`, `field.hpp`, `grid.hpp`, `perf.hpp`, `performance.hpp`, `topology.hpp`, `backend_selection.hpp`, `version.hpp`). Include `<dsdecompx.hpp>`; there is no `dsdecompx/` subdirectory. |
| `<prefix>/lib/` (or `lib64`) | The `dsdecompx` library. |
| `<prefix>/lib/cmake/dsDecompX/` | The CMake package (`dsDecompXConfig.cmake`, targets, version file, and the `FindNCCL` / `FindNVSHMEM` modules). |
| `<prefix>/share/doc/dsDecompX/` | License, README, changelog. |

## Check the installation

Build and run the first program in [03_First_Program.md](03_First_Program.md) against the
installed package. If it configures with `find_package(dsDecompX 1.0 REQUIRED CONFIG)`, links
and runs on four ranks, the installation works.

## Use it from your own project

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_particles LANGUAGES CXX)

find_package(dsDecompX 1.0 REQUIRED CONFIG)

add_executable(my_particles main.cpp)
target_link_libraries(my_particles PRIVATE dsdecompx::dsdecompx)
```

Tell CMake where the install is when configuring your project:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/desired/prefix
```

(`-DdsDecompX_DIR=/desired/prefix/lib/cmake/dsDecompX` also works.) In source files:

```cpp
#include <dsdecompx.hpp>
```

Link through the `dsdecompx::dsdecompx` target rather than adding include paths and libraries by
hand. The target brings MPI along, and for a CUDA install it also brings the CUDA runtime, NCCL
and NVSHMEM as they were built. It also defines the preprocessor macros `HAS_CUDA`, `HAS_NCCL`,
`HAS_NVSHMEM` and `ENABLE_PERF_COUNTERS`; the public headers depend on them (for example, the
default of `Config::execution`), so your code must be compiled with the same values the library
was built with.

If you compile your own `.cu` files, enable `CUDA` in your `project()` line. The installed
library is self-contained and does not require device linking in your project.

To check which headers a translation unit actually picked up (useful when several dsDecompX
installs exist on a machine), print `dsdecompx::header_path()` and `dsdecompx::version_string`.

## Limitations

- CMake is the only supported build system; the package is consumed through `find_package`.
- Only `dsdecompx.hpp` and the headers listed above are installed. Other headers in the source
  tree are internal.

## Next

Write and run a first program: [03_First_Program.md](03_First_Program.md).
