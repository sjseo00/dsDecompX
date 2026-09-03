# dsDecomp 0.3 release report

## Architecture

`Decomp` is the single public operation API and owns domain/topology and backend
lifecycle. `ParticleInfo` hides resizing, unpacking, ghost append, migration
replacement, and count bookkeeping. CUDA performs GPU memory,
classification, scans, packing, compaction, and unpacking. MPI, NCCL, or
NVSHMEM moves packed payloads; MPI remains bootstrap/count control.

## Files changed

- `include`: single `Decomp` API, storage adapters, configuration,
  topology, performance, and version API.
- `src/cuda`: persistent buffers/events and halo/migration kernels.
- `src/backends`: MPI direct/staged GPU path, grouped NCCL P2P, NVSHMEM
  symmetric one-sided path.
- `cmake`: optional dependency find modules and relocatable package config.
- `examples`, `tests`, `benchmarks`, `docs`: external integration and validation
  scaffolding.
- DeepScaleMD `md_system` adapter: host and GPU operations route through the same
  `Decomp` methods; duplicate application-side halo assembly was removed.

## CUDA execution model

The caller supplies the communication stream. Persistent geometrically-grown
buffers hold flags, offsets, scan storage, packed payloads, receives, and
migration results. A count-ready event is the one required host-visible metadata
point. Reuse events protect buffers; capacity growth may host-wait for prior use.
There is no device-wide synchronization.

## MPI implementation

MPI duplicates the communicator, creates the Cartesian topology, exchanges
counts, then posts sparse nonblocking receives/sends and waits for completion.
CUDA-aware MPI receives device pointers directly. The explicit fallback copies
through persistent pinned buffers and performs one event wait for send readiness.

## NCCL implementation

MPI broadcasts a unique ID and maps Cartesian ranks once. Nonzero receives and
sends are sorted by peer/tag and grouped on the communication stream. Self edges
use device copies. Communicators are not recreated per timestep.

## NVSHMEM implementation

Equal symmetric receive slots and epoch signal arrays persist per operation
type. MPI count metadata and a collective maximum prevent overflow. Stream puts
are followed by quiet, remote signals, and incoming peer waits. No timestep
global barrier is used. This is the host-initiated Phase 1 design.

## DeepScaleMD integration

DeepScaleMD retains MPI lifecycle, particle/ghost SoA storage, neighbor-list
updates, force scheduling, and physics reductions. It constructs `Decomp`
without using backend, plan, or low-level decomposition APIs. Host and device
storage both use `exchange_halo`, `migrate`, and `reverse_halo`.

## Correctness

Source tests cover grid selection, MPI synchronous/split halo equivalence,
migration global count/ID invariants, backend selection, and a CUDA backend smoke
case registered separately for MPI/NCCL/NVSHMEM. Periodic self-neighbor direction
matching is consistent across transports. Tests were not executed during this
edit-only task; no runtime pass result is claimed.

## Performance

Normal steps reuse allocations and communicators. The sparse transports do not
stage NCCL/NVSHMEM payloads or use all-to-all collectives. Local/global counters
and benchmark methodology are present. No performance benchmark was executed,
so no latency, bandwidth, overlap, or scaling result is claimed.

## Synchronization audit

See [synchronization-audit.md](developer/synchronization-audit.md). In summary:
no `cudaDeviceSynchronize`, `cudaStreamSynchronize`, MPI barrier, or NVSHMEM
global barrier exists in dsDecomp. MPI uses `MPI_Waitall`; NCCL uses group
start/end; NVSHMEM uses stream quiet and per-peer signal/wait. CUDA events handle
metadata readiness, consumer ordering, and safe reuse.

## Known limitations

- NVSHMEM device reverse halo is not implemented.
- Automatic discovery distinguishes nodes but not NVLink/NVSwitch/NIC layout.
- NVSHMEM GPU-initiated kernels are not implemented.
- Count metadata is host-visible for variable payload sizing.
- Aggregate timers are host/control-plane wall time; GPU kernel/overlap timing
  requires a CUDA profiler.
- The communicated particle record remains fixed in 0.3.
