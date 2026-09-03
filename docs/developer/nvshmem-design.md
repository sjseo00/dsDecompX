# NVSHMEM design

Each rank allocates equal-sized symmetric receive slots. Halo slots are indexed
by direction; migration slots are indexed by source rank. MPI count exchange and
a global maximum establish safe capacity before payload writes. A sender writes
directly to the receiver's designated slot, then signals completion. Allocations
are persistent and only replaced when the collective maximum grows.

Version 0.3 is Phase 1 host-initiated NVSHMEM. A later device-initiated path can
reuse the public plan semantics and symmetric layout.
