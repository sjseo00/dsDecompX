# Synchronization audit

This is the source-level audit for dsDecompX 0.3. It excludes synchronization in
DeepScaleMD force/integrator code because that code is outside the library.

| Primitive | Location | Reason |
|---|---|---|
| `cudaDeviceSynchronize` | none in `dsDecompX/src` | prohibited |
| `cudaStreamSynchronize` | none in `dsDecompX/src` | stream/event ordering used |
| `cudaEventSynchronize` | `cuda/cuda_event.cpp`; `backends/mpi_backend.cpp` | host-visible variable counts; staged D2H readiness |
| `MPI_Waitall` | MPI payload/count paths and NCCL/NVSHMEM count control paths | completion of posted sparse messages/metadata |
| `MPI_Wait` | none in `dsDecompX/src` | not used |
| `MPI_Barrier` | none in `dsDecompX/src` | prohibited in timestep path |
| `ncclGroupStart/End` | `backends/nccl/nccl_backend.cpp` | batch sparse peer receives/sends |
| NVSHMEM barrier | none in `dsDecompX/src` | peer signaling replaces global barrier |
| `nvshmemx_quiet_on_stream` | `backends/nvshmem/nvshmem_backend.cpp` | remote completion before signaling |
| NVSHMEM signal/wait | same file | per-peer epoch completion on communication stream |

MPI bootstrap collectives occur at initialization. MPI count exchange and the
NVSHMEM global-capacity maximum are operation metadata collectives, not barriers.
