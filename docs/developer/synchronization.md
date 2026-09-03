# Synchronization

Required host synchronization is limited to GPU count metadata and MPI payload
completion. CUDA-aware MPI uses device pointers; staged MPI additionally waits
for D2H readiness before MPI can consume pinned memory. NCCL completion is
stream-ordered. NVSHMEM performs payload puts, stream quiet, remote signals, and
incoming per-peer stream waits.

Forbidden in the hot path: `cudaDeviceSynchronize`, `MPI_Barrier`, NVSHMEM global
barriers, per-step communicator construction, and per-step allocation when
capacity is unchanged.
