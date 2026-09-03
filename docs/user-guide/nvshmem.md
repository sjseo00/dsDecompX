# NVSHMEM backend

NVSHMEM requires MPI, CUDA, and NVSHMEM. MPI supplies bootstrap, Cartesian rank
mapping, and count metadata. Payload destinations and signal words are allocated
from symmetric memory and persist until capacity growth or decomposition
destruction. The 0.3 path is host-initiated but stream-ordered: puts, quiet,
signals, and waits are enqueued on the communication stream.

No timestep global barrier is used. Symmetric allocation/growth is collective;
all ranks must execute plans in the same order. Device-initiated NVSHMEM kernels
are future work.

The installation must provide `nvshmemx_putmem_on_stream`,
`nvshmemx_quiet_on_stream`, `nvshmemx_signal_op_on_stream`, and
`nvshmemx_signal_wait_until_on_stream`.
