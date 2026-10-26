# 4. Using dsDecompX in Your Code

This page answers one question: **how do I connect my own particle data and my own time loop to
dsDecompX, on CPU or GPU?** [03_First_Program.md](03_First_Program.md) shows the shortest working
program; this page explains each decision in it. Function-by-function details are in
[05_API_Reference.md](05_API_Reference.md).

Contents: [Program structure](#1-program-structure) ·
[Configure](#2-configure-the-decomposition) · [Particle data](#3-give-it-your-particles) ·
[Owned and ghost slots](#4-owned-and-ghost-slots) · [The step loop](#5-the-step-loop) ·
[Custom fields](#6-application-defined-fields) · [Returning ghost data](#7-returning-ghost-data-to-owners) ·
[Load balancing](#8-load-balancing-and-box-changes) · [Overlapping](#9-overlapping-communication-with-computation) ·
[CUDA](#10-cuda)

## 1. Program structure

```cpp
int provided = MPI_THREAD_SINGLE;
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
{
    dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);
    dsdecompx::ParticleInfo particles(n_local);
    // ... time loop ...
}   // destroy dsDecompX objects before finalizing MPI
MPI_Finalize();
```

- Initialize MPI first, with thread level `MPI_THREAD_FUNNELED` or higher. The `Decomp` constructor
  throws if MPI is not initialized or the level is lower.
- Build one `Decomp` and keep it for the whole run. It duplicates the communicator, so you can
  pass `MPI_COMM_WORLD` or any sub-communicator. It can be moved but not copied.
- Every rank must make every collective call, in the same order (see
  [01_Overview.md](01_Overview.md#rules-that-apply-from-the-start)).
- Destroy `Decomp` and `ParticleInfo` before `MPI_Finalize()`.

## 2. Configure the decomposition

Fill a `dsdecompx::Config`. These are the fields most programs set; every field and its default is
in the [`Config` table of the API reference](05_API_Reference.md#config).

```cpp
dsdecompx::Config config;
config.domain.lower = {0.0, 0.0, 0.0};
config.domain.upper = {Lx, Ly, Lz};
config.domain.periodic = {true, true, true};
config.halo.width = cutoff + skin;
config.execution = dsdecompx::ExecutionMode::Host;       // or ExecutionMode::Cuda
config.grid_policy = dsdecompx::GridPolicy::Balanced;    // default is Manual
```

| Field | Choose |
|---|---|
| `domain` | The global box. Periodicity is per axis and cannot change after the `Decomp` is created. The box size can change later with `update_domain()`. |
| `halo.width` | Your interaction cutoff plus the neighbor-list skin. See below. |
| `execution` | `Host` for CPU arrays, `Cuda` for GPU arrays. In a CUDA build the default is `Cuda`, so a program that uses CPU arrays must set `Host` explicitly. |
| `grid_policy` | `Manual` (default): the grid comes from `process_grid.dimensions`, and zero entries are filled in by `MPI_Dims_create`. `Balanced`: the library picks the factorization of the rank count that gives the most compact subdomains for your box. `TopologyAware`: like `Balanced`, but prefers to keep neighboring subdomains on the same node (it detects nodes from MPI processor names). `Dynamic`: `Balanced` plus movable cut planes (section 8). |

To force a layout, set `config.process_grid.dimensions = {4, 2, 1};` with `Manual`; the product
must equal the number of ranks.

### Choosing the halo width

Set `halo.width` to the distance within which two particles interact, plus the skin you allow
for movement between neighbor-list rebuilds. A particle is copied as a ghost to a neighbor when it
lies within `halo.width` of the face that borders that neighbor.

Three constraints apply:

- **Reach.** Ghosts come only from the adjacent subdomains. Keep `halo.width` no larger than the
  narrowest subdomain width along any axis that has more than one process. The library does not
  check this, and a halo that is too wide silently drops interactions with ranks further away.
- **Two processes along an axis.** The lower and upper neighbors are then the same rank. A particle
  within `halo.width` of both faces of its own subdomain (a subdomain narrower than
  `2 x halo.width`) is sent twice, once as each periodic image. Keep such subdomains at least
  `2 x halo.width` wide unless your force code handles both images.
- **Skin.** A ghost list built at a rebuild stays fixed until the next one, so `halo.width` must cover
  the cutoff plus the distance particles can move between rebuilds.

## 3. Give it your particles

`ParticleInfo` is the container that every `Decomp` call works on. There are three ways to fill one.

### 3.1 Let ParticleInfo own host arrays

```cpp
dsdecompx::ParticleInfo particles(n_local);
particles.x[i] = ...;   // x y z vx vy vz fx fy fz mass charge are std::vector<double>
particles.global_id[i] = ...;  // type and global_id are std::vector<int>
```

The vectors grow and shrink as particles migrate and ghosts arrive, so do not keep raw pointers
across `migrate()` or `exchange_halo()`. `HaloResult::storage_reallocated` tells you when the
storage moved.

### 3.2 Attach your own host arrays

```cpp
auto particles = dsdecompx::ParticleInfo::external(
    dsdecompx::MemorySpace::Host, capacity, n_owned);
particles.positions(x.data(), y.data(), z.data())
         .velocities(vx.data(), vy.data(), vz.data())
         .forces(fx.data(), fy.data(), fz.data())
         .properties(mass.data(), charge.data(), type.data(), global_id.data());
```

`capacity` is the number of particles each array can hold. dsDecompX writes ghosts and incoming
migrants into your arrays, so `capacity` must be at least `owned + ghosts` after every exchange
and at least the new owned count after every `migrate()`. Fixed external storage cannot grow: a
call that needs more room throws `std::length_error`. Reserve headroom, or let dsDecompX ask you
to grow by passing a `ParticleInfo::Provider` (a small interface with
`particle_info_view(minimum)` and `resize_particle_info(required, preserve)`; see the API
reference).

Which arrays are required:

| Memory | Required | Optional |
|---|---|---|
| Host | `x y z global_id` | `vx vy vz fx fy fz mass charge type` (a null pointer is simply not communicated) |
| CUDA | all thirteen arrays for `migrate()`; all except `fx fy fz` for `exchange_halo()`; `fx fy fz` for `reverse_halo()` | none |

### 3.3 GPU arrays

See [section 10](#10-cuda).

### Particle identity and the first distribution

Every particle needs a `global_id` that is unique across all ranks. dsDecompX uses it to
identify ghosts and never changes it. Debug builds (without `NDEBUG`) additionally check, in each
host `migrate()`, that ids are unique and that no particle was lost or duplicated.

You do not have to place particles on the correct rank at the start. Load them any convenient
way (for example, rank 0 reads the file and deals particles out, or every rank reads a slice),
then call `migrate()` once; it sends each particle to its owner. To find an owner yourself, for
example while reading a file, use `decomp.owner_rank({x, y, z})`. On a non-periodic axis, a
position outside the box is assigned to the nearest boundary rank rather than removed.

## 4. Owned and ghost slots

After `exchange_halo()`:

| Slots | Contents |
|---|---|
| `0 ... owned_count()-1` | Your owned particles. dsDecompX does not reorder them during a halo exchange. |
| `owned_count() ... size()-1` | Ghosts, in an order fixed by the halo plan (stable until the plan is rebuilt). |

A ghost carries a copy of the owner's velocity, mass, charge, type, `global_id` and application
fields. Its position is the owner's position shifted by a box length when the ghost crossed a periodic
boundary, so you compute `x_i - x_ghost` directly with no minimum-image correction along divided axes.
Its force starts at zero.

`migrate()` discards all ghosts (`ghost_count()` becomes 0) and may reorder and resize the owned
particles. Call `exchange_halo()` again afterwards.

Do not integrate ghosts: their values are overwritten at the next exchange.

## 5. The step loop

```cpp
for (int step = 0; step < nsteps; ++step) {
    const bool rebuild = (step % rebuild_interval == 0);

    if (rebuild) decomp.migrate(particles);

    const auto halo = decomp.exchange_halo(
        particles, dsdecompx::HaloExchangeOptions{rebuild});

    if (halo.layout_changed || halo.storage_reallocated) {
        rebuild_neighbor_list(particles);   // your code
    }

    compute_forces(particles);              // owned + ghost slots; ghost forces start at zero

    decomp.reverse_halo(particles);         // ghost forces -> owners

    integrate(particles);                   // owned slots only
}
```

| Call | Frequency | Notes |
|---|---|---|
| `migrate()` | When particles may have left their subdomain, typically each neighbor-list rebuild | `decomp.has_out_of_domain_particles(particles)` is a collective check you can use as the trigger. |
| `exchange_halo()` | Every step | Pass `rebuild = true` on steps where you want a fresh ghost list. A `migrate()` always invalidates the old list, so the step right after it rebuilds regardless. |
| `reverse_halo()` | Every step with forces on ghosts | Must follow the `exchange_halo()` of the same step, with no `migrate()` between. |

**Rebuild and reuse.** With `rebuild = false`, the previous halo plan is reused. The same particles
are sent to the same neighbors and fill the same ghost slots, so a neighbor list that indexes ghost
slots stays valid until the next rebuild. On CPU memory a reused exchange refreshes ghost positions
and application fields; velocity, mass, charge, type and id keep the values from the last rebuild.
On CUDA memory a reused exchange refreshes the whole record, or only positions with
`positions_only` (section 10). Ghost forces are zero after every exchange.

**Results.** `HaloResult::ghost_count` is the number of ghosts. `layout_changed` is true when the ghost
count or order is not the same as after the previous call, which is when ghost-indexed neighbor
lists become invalid. `storage_reallocated` is true when the particle arrays moved. `plan_reused` is
true when a CUDA exchange refreshed the previous plan in place.

All ranks should pass the same `rebuild` value. On CUDA memory this is required. When each rank
decides locally whether to rebuild (for example, from its own largest particle displacement), agree on
one answer first with `rebuild = decomp.any_rank_true(local_wants_rebuild);`, which is collective.

## 6. Application-defined fields

Besides the built-in record, you can attach your own per-particle arrays. They move with their
particle in `migrate()`, are copied to the ghosts in `exchange_halo()`, and can be sent back with
`reverse_exchange()`.

```cpp
std::vector<double> temperature(capacity);
std::vector<double> momentum(3 * capacity);   // 3 components per particle, array-of-structures
std::vector<int>    species(capacity);

particles.add_field<double>("temperature", temperature.data(), capacity)
         .add_field<double>("momentum", momentum.data(), capacity, 3)
         .add_field<int>("species", species.data(), capacity);
```

`add_field<T>(name, data, capacity, components = 1)` registers a non-owning array. The rules:

- Supported element types: `int32_t`, `int64_t`, `uint32_t`, `uint64_t`, `float`, `double`.
- A field with several components is laid out per particle: `[p0c0, p0c1, p0c2, p1c0, ...]`.
- `capacity` counts particles (not elements) and must be at least `owned + ghosts` after every exchange. For a
  `ParticleInfo` that owns its host arrays, pass `capacity` explicitly: the default is the
  current number of owned particles, which has no room for ghosts.
- Names must be unique. The names, types, component counts and registration order must be identical
  on every rank; this is checked collectively and mismatches throw `std::invalid_argument`.
- On CUDA memory, at most 16 fields can be registered, and the pointers must be device or managed memory.
- Unlike forces, a field is copied to the ghosts, so ghost slots hold the owner's value after
  `exchange_halo()`.

## 7. Returning ghost data to owners

| Call | Sends back | Typical use |
|---|---|---|
| `reverse_halo(particles)` | `fx`, `fy`, `fz` of the ghosts, added to the owners | Forces computed on ghosts |
| `reverse_exchange(particles, {{"name", ReduceOp::Sum}, ...})` | The named application fields, reduced into the owners with `Sum`, `Min`, `Max` or `Replace` | Densities, torques, counts, any per-particle accumulation |
| `forward_exchange(particles, {"name", ...}, stream)` | Copies the named fields from owners into their ghost slots (CUDA memory only) | Refreshing a derived field on the ghosts without a full halo exchange |

All three follow the ghost list of the most recent `exchange_halo()`. Do not call `migrate()`
between the exchange and the reverse call.

**Zero ghost accumulators before you use `Sum`.** `reverse_halo()` is safe because ghost forces
start at zero. An application field is different: the exchange copies the owner's value into
the ghost slot, so a `Sum` would add that copy back to the owner. Before computing, set the
ghost entries of an accumulator field to zero (or assign them the contribution), and leave the owner entries
holding the owner's own part:

```cpp
decomp.exchange_halo(particles);
for (std::size_t g = particles.owned_count(); g < particles.size(); ++g) density[g] = 0.0;
// ... accumulate density on owned and ghost slots ...
decomp.reverse_exchange(particles, {{"density", dsdecompx::ReduceOp::Sum}});
```

**Which rank computes a cross-face pair.** An owned particle on one rank and its ghost on
another are both visible on two ranks. dsDecompX does not choose; your code must make sure the
pair contributes exactly once. Two valid designs:

- Compute the pair on exactly one rank by a rule both ranks agree on (for example, the smaller
  `global_id` of the owned particle, as in [03_First_Program.md](03_First_Program.md)), apply the
  force to both particles, and use `reverse_halo()`.
- Compute the pair on both ranks, apply the force only to the owned particle, halve the pair
  energy, and do not call `reverse_halo()`.

## 8. Load balancing and box changes

**Automatic.** Set the policy and, optionally, the tuning:

```cpp
config.grid_policy = dsdecompx::GridPolicy::Dynamic;
config.dynamic.interval = 10;               // rebalance check every 10th migrate() call (default 1; 0 turns it off)
config.dynamic.imbalance_threshold = 1.10;  // act when max(work)/mean(work) reaches this
config.dynamic.relaxation = 0.5;            // fraction of the computed shift applied each time
config.dynamic.minimum_cell_fraction = 0.10; // narrowest cell, as a fraction of a uniform cell
```

Work per rank is its owned particle count. The rank grid stays the same; only the cut planes
between subdomains move, inside `migrate()`, before particles are sent. A cell is never made
narrower than the halo width (unless a uniform cell is already narrower than that).

**Explicit.** To balance on your own measure of work (for example, measured force-loop time):

```cpp
dsdecompx::RebalanceOptions options;
options.local_work = my_seconds_this_rank;   // non-negative; default uses the owned count
options.force = false;                       // true ignores the imbalance threshold
const auto result = decomp.rebalance(particles, options);   // collective
// result.geometry_changed, result.imbalance_before, result.migration
```

`rebalance()` moves the cut planes and then migrates, so ghosts are gone afterwards. Call
`exchange_halo()` again. You can use it with `dynamic.interval = 0` to take over completely.

**Box changes (NPT).** After you change the box and rescale positions, call
`decomp.update_domain(new_domain)` and then `decomp.set_halo_width(new_width)` if the width changed,
then `migrate()`. The cut planes scale with the box. Periodicity cannot change.

## 9. Overlapping communication with computation

`exchange_halo()` is exactly `begin_halo_exchange()` followed by `end_halo_exchange()`. Anything
you do between the two that touches only owned particles runs while the transfers are in flight:

```cpp
decomp.begin_halo_exchange(particles, options);
compute_owned_only_part(particles);           // do not read or write ghost slots
const auto halo = decomp.end_halo_exchange(particles);
compute_ghost_part(particles);
```

Between the calls, `particles` must not be resized or exchanged, and no other halo, migration or
reverse operation may start. A second `begin`, or an `end` without a `begin`, throws
`std::logic_error`. On CUDA memory, `decomp.can_refresh_halo(particles)` tells you before `begin`
whether the exchange will reuse the existing plan (so ghost count and slot meaning are already
known).

## 10. CUDA

Build dsDecompX with CUDA ([02_Installation.md](02_Installation.md)) and set
`config.execution = dsdecompx::ExecutionMode::Cuda`. The function names are the same as on CPU.
Every call takes a CUDA stream as an extra argument, and the stream must not be null.

### Choosing the GPU

| `advanced.cuda_device` | Meaning |
|---|---|
| `kCudaDeviceCurrent` (default) | Use the device your program already selected with `cudaSetDevice()`. |
| `kCudaDeviceByLocalRank` | One GPU per rank on a node (node-local rank modulo the visible device count). |
| `0, 1, ...` | That device ordinal. |

On multi-GPU nodes, set `kCudaDeviceByLocalRank` unless you call `cudaSetDevice()` per rank
yourself. NCCL and NVSHMEM need one GPU per rank. When you request one of them explicitly and ranks on
a node share a GPU although enough GPUs are visible, the constructor throws.

### GPU particle arrays

Library-owned arrays:

```cpp
auto particles = dsdecompx::ParticleInfo::allocate(
    dsdecompx::MemorySpace::CudaDevice, n_owned);   // or MemorySpace::CudaManaged
double* d_x = particles.x_data();                   // also y_data(), vx_data(), ..., global_id_data()
```

The arrays grow when migrants or ghosts need more room, so take the pointers from the `*_data()`
accessors after each call rather than caching them. Application fields must be pre-sized for the
largest `owned + ghosts` (growth throws if a field is too small).

Your own arrays:

```cpp
auto particles = dsdecompx::ParticleInfo::external(
    dsdecompx::MemorySpace::CudaDevice, capacity, n_owned);
particles.positions(d_x, d_y, d_z)
         .velocities(d_vx, d_vy, d_vz)
         .forces(d_fx, d_fy, d_fz)
         .properties(d_mass, d_charge, d_type, d_global_id);
```

The same capacity rules as for host arrays apply (section 3.2). `MemorySpace::CudaManaged` works for
both forms.

### The step loop on GPU

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

decomp.migrate(particles, stream);                    // initial distribution
for (int step = 0; step < nsteps; ++step) {
    const bool rebuild = (step % rebuild_interval == 0);
    if (rebuild) decomp.migrate(particles, stream);

    const auto halo = decomp.exchange_halo(
        particles, dsdecompx::HaloExchangeOptions{rebuild, stream});

    my_force_kernel<<<blocks, threads, 0, stream>>>(/* particles.size() slots */);

    decomp.reverse_halo(particles, stream);
    my_integrate_kernel<<<blocks, threads, 0, stream>>>(/* particles.owned_count() slots */);
}
```

Return values (`ghost_count`, `new_owned_count`) are valid when the call returns. The data
itself is ordered on `stream`: enqueue your kernels on the same stream, or synchronize the stream
before reading from the host.

### Reusing the ghost list

On GPU, `rebuild = false` refreshes the existing plan in place without reclassifying the
particles or exchanging counts, so the call is cheap. `HaloExchangeOptions::positions_only = true`
goes further: it sends only the three position coordinates per ghost (24 bytes instead of about 96).
The ghosts then keep the velocity, mass, charge, type and id from the last full exchange and have
zero force. It applies only to a reused plan, only when no application fields are registered, and
not with the NVSHMEM backend; in the other cases it is ignored and a full record is sent. If the
storage had to be reallocated during such a refresh, the call throws, because the ghosts' other
fields would be lost; use a full exchange then. Pass the same value on every rank. The macro
`DSDECOMPX_HALO_POSITIONS_ONLY` is defined when your dsDecompX version has this option.

### Selecting the communication backend

With `ExecutionMode::Host` the transport is always MPI. With `ExecutionMode::Cuda`:

| Backend | Moves GPU buffers by | Needs |
|---|---|---|
| MPI | CUDA-aware MPI directly, or staging through pinned CPU memory when the MPI library cannot take device pointers | MPI only |
| NCCL | NCCL send/receive | dsDecompX built with NCCL; one GPU per rank |
| NVSHMEM | NVSHMEM one-sided puts | dsDecompX built with NVSHMEM; one GPU per rank |

The default (`BackendSelection::Auto`) tries NVSHMEM, then NCCL, then MPI, using the first one
that was built and starts successfully. To force one, set `config.advanced.selection`
(`BackendSelection::Mpi`, `Nccl`, `Nvshmem`) or, when `advanced.read_environment` is true (the
default), the environment variable `BACKEND` to `auto`, `mpi`, `nccl` or `nvshmem`. Asking for a
backend that was not built throws. Check the result with `decomp.backend_kind()`.

For the MPI backend, CUDA awareness is detected from the MPI library (Open MPI is asked
directly) and from the environment variables `CUDA_AWARE_MPI` (`1`/`true` or `0`/`false`) and
`OMPI_MCA_opal_cuda_support`. `decomp.mpi_takes_device_pointers()` reports the result. When MPI is
not CUDA-aware, buffers are staged through pinned host memory; set
`config.advanced.allow_host_staging = false` to make that an error instead. Do not force
`CUDA_AWARE_MPI=1` on an MPI library that is not CUDA-aware.

### Custom fields on GPU

Register device or managed arrays with `add_field`, exactly as in section 6. Up to 16 fields,
and `forward_exchange()` additionally needs each field's per-particle size
(`components x sizeof(type)`) to be a multiple of 4 bytes. At most 26 neighbors are supported in a CUDA halo.

## Limitations

- Host memory requires `ExecutionMode::Host` and CUDA memory requires `ExecutionMode::Cuda`;
  mixing them is not supported.
- `forward_exchange()` works only on CUDA memory, with MPI, NCCL and NVSHMEM. Its MPI path is
  tested on 2 to 4 ranks; its NVSHMEM path is implemented but has not been run.
- `ExecutionMode::Cuda` requires `process_grid.allow_rank_reorder = false` (the default).
- Positions-only halo refresh is unavailable with NVSHMEM.

## Next

Look up exact signatures and defaults: [05_API_Reference.md](05_API_Reference.md). See
complete usage patterns for MD, DEM, SPH and particle tracking:
[06_Recipes.md](06_Recipes.md).
