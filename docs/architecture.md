# Architecture

```text
Particle application -> Decomp
                     -> ParticleInfo (host | external/device)
                     -> domain + Cartesian topology
                     -> CUDA execution: classify, scan, pack, compact, unpack
                     -> data plane: MPI | NCCL | NVSHMEM
```

CUDA is an execution layer, not a communication backend. MPI owns bootstrap,
Cartesian topology, count metadata, and global statistics. The selected data
plane moves packed `ParticleRecord` payloads. `Decomp` retains topology metadata,
buffers, CUDA events, and backend state across timesteps.

Host execution uses MPI. CUDA execution uses device-resident SoA views and
persistent device buffers. NCCL groups sparse point-to-point operations.
NVSHMEM uses symmetric destination slots, one-sided puts, and peer signals.

The application decides when to migrate, exchange halos, rebuild a neighbor
list, and launch forces/integration. `Decomp` performs storage sizing, ghost
append/update, migration replacement, unpacking, and count bookkeeping through
the `ParticleInfo` interface.

`TopologyAware` 0.3 discovery distinguishes hostname/node membership. Detailed
NVLink/NVSwitch/NIC discovery is intentionally a separate future provider;
callers may already inject a richer `TopologyInfo`. Discovery failure selects
the Balanced grid heuristic and never fails initialization.
