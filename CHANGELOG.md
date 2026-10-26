# Changelog

## 1.0.0

- Added `Decomp::forward_exchange()` (CUDA memory; MPI, NCCL and NVSHMEM backends): copies application fields registered with
  `add_field()` from the owned particles into the ghost slots that hold their copies, along the halo plan of the last device halo
  exchange. It is the adjoint of `reverse_exchange()` with `ReduceOp::Sum`, for differentiable per-layer exchange of node features
  (message-passing MLIP models). New test `dsdecompx.cuda.forward-exchange.<backend>`; the NVSHMEM path has not been run yet.
- The MPI backend now asks Open MPI whether it is CUDA-aware (`MPIX_Query_cuda_support()`
  from `<mpi-ext.h>`, when the header exists at build time). The earlier
  `MPIX_CUDA_AWARE_SUPPORT` check never fired because the header was not included, so only
  the `CUDA_AWARE_MPI` / `OMPI_MCA_opal_cuda_support` environment variables could turn the
  detection on. Added `Decomp::mpi_takes_device_pointers()` (MPI backend in use and the MPI
  library takes device pointers).
- Added split-phase halo exchange (`Decomp::begin_halo_exchange()` /
  `end_halo_exchange()`) so applications can overlap the exchange with work on
  owned particles, plus `Decomp::can_refresh_halo()` and
  `HaloResult::plan_reused`. `exchange_halo()` is unchanged: it is begin
  followed by end.
- CUDA halo exchange now reuses its plan between rebuilds. A call with
  `rebuild = false` re-reads the owner of every ghost slot instead of
  reclassifying all owned particles, so ghost membership and order stay fixed
  until the next rebuild (neighbor lists index ghost slots), and each such call
  skips the count readback, the count exchange and the field-schema collective
  (which is now one reduction instead of two). `HaloResult::layout_changed` is also set when a new plan was
  built. All ranks must pass the same `rebuild` value on device views.
- Added `GridPolicy::Dynamic`, nonuniform Cartesian axis cut planes, automatic
  owned-particle load balancing during migration, and the collective
  `Decomp::rebalance()` API for user-provided workload metrics.
- Added matching host and CUDA owner classification for dynamic domains and
  exposed current cut positions through `Topology::axis_cuts()`.

## 0.3.0

- Added host-side application-defined scalar/vector fields that participate in
  halo exchange and migration, plus generic reverse reductions (`Sum`, `Min`,
  `Max`, and `Replace`).
- Kept `reverse_halo()` as an MD compatibility API backed by generic reverse
  reduction on the host path.
- Added backend-neutral `Decomp`, `HaloPlan`, `MigrationPlan`, and split-phase
  halo APIs.
- Added persistent CUDA packing, unpacking, migration layout, and memory reuse.
- Added CUDA-aware MPI device transfers and explicit pinned-staging fallback.
- Added sparse NCCL send/receive and NVSHMEM one-sided symmetric-buffer paths.
- Added deterministic runtime backend selection and install-time dependency
  discovery for optional GPU backends.
- Corrected periodic self-neighbor matching on process-grid axes of length one:
  incoming direction slots now consume the opposite outgoing slot consistently
  across MPI, NCCL, and NVSHMEM.

## 0.2.0

- Added a standalone MPI-backed Cartesian decomposition library target.
- Added 26-neighbor halo exchange with synchronous and begin/end APIs.
- Added direct-destination particle migration with debug ID/count validation.
- Added reusable communication buffers and optional performance counters.
- Added balanced and topology-aware process-grid heuristics as opt-in policies.
- Added install/export packaging, examples, tests, and integration documentation.

## 0.1.0

- Extracted Cartesian topology, neighbor discovery, and host MPI halo exchange
  from DeepScaleMD while preserving its tags and ordering.
