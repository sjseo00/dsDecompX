# Changelog

## Unreleased

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
