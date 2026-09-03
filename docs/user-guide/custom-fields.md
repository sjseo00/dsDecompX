# Application-defined particle fields

`ParticleInfo` keeps the MD-oriented position, velocity, force, mass, charge,
type, and global-ID accessors for compatibility. Applications can additionally
register arbitrary scalar or fixed-width vector fields. Registered fields are
carried in the same stable particle order through halo exchange and migration.

```cpp
std::vector<double> temperature(capacity);
std::vector<int> species(capacity);
std::vector<double> angular_velocity(capacity * 3); // AoS components

particles
    .add_field<double>("temperature", temperature.data(), capacity)
    .add_field<int>("species", species.data(), capacity)
    .add_field<double>("angular_velocity",
                       angular_velocity.data(), capacity, 3);
```

The storage is non-owning. It must remain valid while `ParticleInfo` uses it,
and `capacity` must accommodate owned plus ghost particles. Every MPI rank must
register the same field names, scalar types, component counts, and order.
dsDecomp checks this schema collectively before communication.

Supported scalar types are 32- and 64-bit signed/unsigned integers, `float`,
and `double`. A multi-component field uses contiguous AoS layout:
`particle * components + component`.

Only positions and a global ID are intrinsic to host decomposition. The
velocity, force, mass, charge, and type arrays are optional compatibility
fields, so DEM, SPH, PIC, and Lagrangian tracking applications do not have to
invent values for unused MD properties.

## Reverse contribution reduction

`reverse_exchange()` reduces values computed on ghosts into their owners:

```cpp
decomp.reverse_exchange(
    particles,
    {{"pressure_contribution", dsdecomp::ReduceOp::Sum},
     {"contact_temperature", dsdecomp::ReduceOp::Max}});
```

`Sum`, `Min`, `Max`, and `Replace` are available. `reverse_halo()` remains as
the MD convenience API and, on the host path, delegates force x/y/z to the same
generic reduction primitive.

Application-defined fields and generic reverse reduction currently use host
particle memory and the MPI host data plane. CUDA-resident custom fields are
rejected explicitly rather than silently omitted. The existing built-in
particle fields continue to support CUDA-aware MPI, NCCL, and NVSHMEM paths.
