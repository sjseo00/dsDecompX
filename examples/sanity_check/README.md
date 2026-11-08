# Sanity check of your installation

Run these programs after you install dsDecompX to confirm that your build works with your MPI (and your GPUs, for a CUDA build). They are built and registered with CTest by [`../CMakeLists.txt`](../CMakeLists.txt), so build the examples as described in [`../README.md`](../README.md) and run:

```sh
ctest --test-dir build --output-on-failure
```

All tests should pass. A failed test means that the installation, the MPI library, or the compiler setup has a problem; the output of `--output-on-failure` shows which operation failed.

The programs use `assert`. The build keeps assertions switched on in every build type, so a Release build checks as much as a Debug build.

## CPU (no GPU needed)

| CTest name | File | What it confirms |
|---|---|---|
| `dsdecompx.grid.unit` | `unit/grid_planner.cpp` | The process grid chosen for a given number of ranks (`Balanced`, and the fallback of `TopologyAware` when the node layout is unknown) |
| `dsdecompx.backend-selection.unit` | `unit/backend_selection.cpp` | The names that `parse_backend_selection` accepts (`auto`, `mpi`, `nccl`, `nvshmem`, in any letter case) and that other names are rejected |
| `dsdecompx.mpi.1`, `.2`, `.4` | `mpi/regression.cpp` | Migration, halo exchange (rebuild and reuse), registered fields of several types and widths, and `reverse_exchange` with `ReduceOp::Sum`, on 1, 2, and 4 MPI ranks |

The small examples in `..` are also registered with CTest (`dsdecompx.example.<name>.<ranks>`); they must run without an error on 1, 2, and 4 ranks.

## CUDA (need GPUs)

The programs in [`cuda/`](cuda/) confirm that the GPU path works with two ranks and are described in [`cuda/README.md`](cuda/README.md). They are off by default. dsDecompX must be built with CUDA, and you configure the examples with

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=<dsDecompX install> -DENABLE_CUDA=ON -DRUN_GPU_TESTS=ON
```

Add `-DENABLE_NCCL=ON` or `-DENABLE_NVSHMEM=ON` to check these transports as well (the installed dsDecompX must include them).
