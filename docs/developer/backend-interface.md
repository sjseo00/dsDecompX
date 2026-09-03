# Backend interface

The internal backend owns rank/topology control operations and payload movement.
`DeviceExchange` is a sparse list of peer, pointer, byte, tag, and optional
symmetric-signal descriptors. Implementations provide the same completed halo
and migration semantics, not an artificially identical transport algorithm.

MPI uses nonblocking messages, NCCL uses grouped stream operations, and NVSHMEM
uses symmetric one-sided writes. Public headers never expose NCCL/NVSHMEM types.
