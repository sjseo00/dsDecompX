# Correctness checks

Automatic checks that dsDecompX returns correct results. They are built and registered with CTest by [`../CMakeLists.txt`](../CMakeLists.txt), so build the examples as described in [`../README.md`](../README.md) and run:

```sh
ctest --test-dir build --output-on-failure
```

The checks use `assert`. The build keeps assertions switched on in every build type, so a Release build checks as much as a Debug build.

## CPU checks (no GPU needed)

| CTest name | File | What it checks |
|---|---|---|
| `dsdecompx.grid.unit` | `unit/grid_planner.cpp` | The process grid chosen for a given number of ranks (`Balanced`, and the fallback of `TopologyAware` when the node layout is unknown) |
| `dsdecompx.backend-selection.unit` | `unit/backend_selection.cpp` | The names that `parse_backend_selection` accepts (`auto`, `mpi`, `nccl`, `nvshmem`, in any letter case) and that other names are rejected |
| `dsdecompx.mpi.1`, `.2`, `.4` | `mpi/regression.cpp` | Migration, halo exchange (rebuild and reuse), registered fields of several types and widths, and `reverse_exchange` with `ReduceOp::Sum`, on 1, 2, and 4 MPI ranks |

The small examples in `..` are also registered with CTest (`dsdecompx.example.<name>.<ranks>`); they must run without an error on 1, 2, and 4 ranks.

## CUDA checks (need GPUs)

The programs in [`cuda/`](cuda/) check the GPU path with two ranks and are described in [`cuda/README.md`](cuda/README.md). They are off by default. dsDecompX must be built with CUDA, and you configure the examples with

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=<dsDecompX install> -DENABLE_CUDA=ON -DRUN_GPU_TESTS=ON
```

Add `-DENABLE_NCCL=ON` or `-DENABLE_NVSHMEM=ON` to run the checks with these transports as well (the installed dsDecompX must include them).
