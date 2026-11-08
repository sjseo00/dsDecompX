# CUDA correctness checks

Three programs check the CUDA path with two MPI ranks, once for each transport (`mpi`, and `nccl` or `nvshmem` when enabled):

| Program | Checks |
|---|---|
| `backend_smoke.cu` | device migration, halo exchange, `reverse_halo`, and the reverse exchange of registered fields with the chosen transport |
| `halo_reuse.cu` | a refresh on a reused halo plan keeps the ghost count, order, and ids, and an invalidated plan is rebuilt and reported |
| `forward_exchange.cu` | `forward_exchange` copies registered fields from the owners into the ghosts and is the adjoint of `reverse_exchange` with `ReduceOp::Sum` |

They need GPUs, so they are registered only with `-DENABLE_CUDA=ON -DRUN_GPU_TESTS=ON`; see `examples/CMakeLists.txt`.
