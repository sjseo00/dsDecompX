# Troubleshooting

- “backend not compiled”: rebuild with the matching CMake option.
- NCCL/NVSHMEM discovery failure: set the package root and library path.
- CUDA-aware MPI unavailable: enable staging, or configure a CUDA-aware MPI.
- Hang: verify every rank constructs/destroys plans and invokes collective
  metadata phases in the same order.
- Stale ghosts: invalidate the halo plan after particle order, owner, domain, or
  halo-width changes.
- Wrong ownership: verify global bounds, periodic flags, and that only owned
  particles are passed to migration.
