# Installation

Requirements are a C++17 compiler and MPI. Enable CUDA with
`ENABLE_CUDA`; NCCL and NVSHMEM each imply CUDA. Configure with CMake,
build, install, then use `find_package(dsDecomp 0.3 CONFIG REQUIRED)` and link
`dsdecomp::dsdecomp`. Optional roots are found through `NCCL_ROOT`/`NCCL_HOME`
and `NVSHMEM_ROOT`/`NVSHMEM_HOME`. The installed package re-discovers all
transitive dependencies.

Start with MPI-only, then enable one GPU backend at a time on the target system.
The application must initialize MPI before constructing `Decomp` and destroy it
before finalizing MPI.
