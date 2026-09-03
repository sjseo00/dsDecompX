# Adding a backend

1. Add a build option and dependency target without leaking its headers into the
   public API.
2. Implement bootstrap/topology/count control and the device data plane.
3. Preserve stable peer/result ordering and exact `ParticleRecord` semantics.
4. Make lifecycle, diagnostics, memory ownership, and completion explicit.
5. Add backend-vs-MPI halo/migration comparisons for single-node and multi-node.
6. Document execution, synchronization, limitations, and recommended runtime.

Do not translate a one-sided backend into fake blocking send/receive calls merely
to match another backend's internal shape.
