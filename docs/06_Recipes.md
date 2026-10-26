# 6. Recipes

This page answers one question: **what does my application's step look like for a given kind ofsimulation?** Each recipe lists what to register, the call sequence, and the points that arespecific to that method. The calls are described in[04_Using_in_Your_Code.md](04_Using_in_Your_Code.md) and [05_API_Reference.md](05_API_Reference.md).


| Recipe                                                                             | Use it for                                                                       |
|------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [1. Pairwise molecular dynamics](#1-pairwise-molecular-dynamics)                   | Short-range forces with a cutoff, neighbor lists, NVE/NVT/NPT                    |
| [2. Per-layer feature exchange (GPU)](#2-per-layer-feature-exchange-gpu)           | Message-passing machine-learning potentials                                      |
| [3. Discrete element method (DEM)](#3-discrete-element-method-dem)                 | Contacting grains with radius, spin and torque                                   |
| [4. Smoothed-particle hydrodynamics (SPH)](#4-smoothed-particle-hydrodynamics-sph) | Kernel summations such as density                                                |
| [5. Lagrangian particle tracking](#5-lagrangian-particle-tracking)                 | Particles that move through a fixed domain with no particle-particle interaction |


Recipes 1 and 2 correspond to how DeepScaleMD uses the library; the other recipes are patternsbuilt from the same calls. The repository's tests exercise the calls(application fields, reverse reductions, migration, load balancing) but not DEM, SPH or tracking codes.

## 1. Pairwise molecular dynamics

**Register.** Nothing beyond the built-in record: positions, velocities, forces, mass, charge,type and a unique `global_id`.

**Configure.**

```cpp
config.domain.upper = {Lx, Ly, Lz};  
config.halo.width   = cutoff + skin;  
config.grid_policy  = dsdecompx::GridPolicy::Balanced;   // or Dynamic for uneven systems
```

**Step loop.**

```cpp
if (rebuild_step) {  
decomp.migrate(particles);                   // owners change; ghosts are discarded  
}  
const auto halo = decomp.exchange_halo(  
particles, dsdecompx::HaloExchangeOptions{rebuild_step});  
if (halo.layout_changed || halo.storage_reallocated) {  
build_neighbor_list(particles);              // indexes owned + ghost slots  
}  
compute_forces(particles);                       // forces on owned and ghost slots  
decomp.reverse_halo(particles);                  // ghost forces -> owners  
integrate_and_thermostat(particles);             // owned slots only
```

**Points specific to MD.**

- **Rebuild together.** A neighbor list may index ghost slots until the next `rebuild = true`exchange: the same particles stay in the same ghost slots in between. Rebuild the list on exactlythe steps where you pass `rebuild = true` (or `migrate()`), and whenever `layout_changed` or`storage_reallocated` is set.
- **Skin.** Choose `skin` so that no pair that starts outside `cutoff + skin` can come within`cutoff` before the next rebuild. A common choice is twice the largest displacement per rebuild interval.
- **Counting each cross-face pair once.** Decide which rank computes a pair of an owned particleand a ghost (for example, the smaller `global_id`), then use `reverse_halo()`; or compute onboth ranks, update only the owned particle, and skip `reverse_halo()` (see[section 7 of 04](04_Using_in_Your_Code.md#7-returning-ghost-data-to-owners)).[03_First_Program.md](03_First_Program.md) is a runnable instance of the first design.
- **Minimum image along undivided axes.** Ghosts are only created across divided axes (and neveron one rank). Apply the minimum-image convention for distances along an axis whose process-griddimension is 1, as in [03_First_Program.md](03_First_Program.md).
- **Constant-pressure runs.** After you change the box and rescale positions:`decomp.update_domain(domain); decomp.set_halo_width(cutoff + skin); decomp.migrate(particles);`and rebuild the neighbor list.
- **Uneven density.** `GridPolicy::Dynamic` moves the cut planes during `migrate()`. To balance on measuredtime instead of particle count, use `decomp.rebalance(particles, options)` with `options.local_work`set to the time of your force loop.
- **GPU.** Use the same sequence with GPU arrays and a stream, and on non-rebuild steps pass`positions_only = true` to cut the traffic per ghost from about 96 to 24 bytes (see the CUDA section of[04](04_Using_in_Your_Code.md#reusing-the-ghost-list)). DeepScaleMD's own use of these calls is in`src/md_system.cpp` (`migrate_particles_on_device`, `exchange_halo_on_device`,`reverse_halo_on_device`).

## 2. Per-layer feature exchange (GPU)

A message-passing model computes node features layer by layer. Each layer needs the features ofthe neighbors, including those that are ghosts on this rank. The pair of calls below keeps ghostfeatures current in the forward pass and returns their gradients in the backward pass.

**Register.** One application field for the features, sized for `owned + ghosts` and`width` components per particle:

```cpp
particles.add_field<float>("features", d_features, capacity, width);   // device or managed memory
```

**Per layer.**

```cpp
// forward: copy the owners' features into the ghost slots  
decomp.forward_exchange(particles, {"features"}, stream);  
// ... run the layer on owned + ghost slots ...  

// backward: add the gradients held by ghost slots to their owners  
// (zero the ghost entries of the gradient field before accumulating, then:)  
decomp.reverse_exchange(particles, {{"features_grad", dsdecompx::ReduceOp::Sum}}, stream);
```

**Points specific to this recipe.**

- Both calls follow the ghost list of the latest GPU `exchange_halo()`; do not `migrate()` betweenthem and the exchange that built the list.
- The two operations are adjoint to each other: `forward_exchange()` is the transpose of`reverse_exchange()` with `ReduceOp::Sum`. This is what a differentiable layer needs.
- Limits: GPU memory only, a non-null stream, at most 16 fields per call, and each field'sper-particle size (`components x sizeof(type)`) must be a multiple of 4 bytes.
- Because ghost features are refreshed from their owners between layers, the ghost layer only hasto be one interaction cutoff deep. Without the exchange, an `L`-layer model needs ghosts `L`cutoffs deep. The halo still reaches only adjacent subdomains, so keep the halo width no largerthan a subdomain.
- DeepScaleMD's use of this pair is `MDSystem::exchange_field_on_device` in `src/md_system.cpp`(field name `mlip_features`).

## 3. Discrete element method (DEM)

**Register.** Per-particle quantities that the contact model needs on ghosts, and accumulatorsthat must come back:

```cpp
particles.add_field<double>("radius",  radius.data(),  capacity)  
         .add_field<double>("omega",   omega.data(),   capacity, 3)   // angular velocity  
         .add_field<double>("torque",  torque.data(),  capacity, 3);  // accumulator
```

**Step.**

```cpp
decomp.migrate(particles);                       // when particles may have changed owner  
decomp.exchange_halo(particles);                 // ghosts get radius and omega as copies  
zero_ghost_entries(torque);                      // before accumulating contributions  
compute_contacts(particles);                     // force on fx/fy/fz, moment on "torque"  
decomp.reverse_halo(particles);                  // contact forces on ghosts -> owners  
decomp.reverse_exchange(particles, {{"torque", dsdecompx::ReduceOp::Sum}});  
integrate(particles);
```

**Points specific to DEM.**

- Set `halo.width` to at least the largest contact range: the largest sum of two radii plus theskin. With a wide radius range the largest grain dictates the halo width.
- A contact between an owned and a ghost grain is visible on both ranks. Pick one rank to computeit (as in recipe 1) so force and torque are applied once.
- Because custom fields are copied to ghosts, zero the ghost entries of `torque` before accumulating(section 7 of [04](04_Using_in_Your_Code.md#7-returning-ghost-data-to-owners)).
- Contact history (for example, tangential springs) is per pair, not per particle. dsDecompX doesnot move per-pair data; keep it in your own structures, keyed by `global_id` pairs, and move it withyour particles after a `migrate()`.

## 4. Smoothed-particle hydrodynamics (SPH)

**Register.** The fields the kernel sums read on ghosts, and the accumulator:

```cpp
particles.add_field<double>("density", density.data(), capacity)  
         .add_field<double>("pressure", pressure.data(), capacity);
```

**Step (two passes).**

```cpp
decomp.migrate(particles);  
decomp.exchange_halo(particles);                 // positions and mass reach the ghosts  
zero_ghost_entries(density);  
accumulate_density(particles);                   // owned and ghost slots  
decomp.reverse_exchange(particles, {{"density", dsdecompx::ReduceOp::Sum}});  

compute_pressure(particles);                     // owned slots  
// push the final density and pressure back to the ghosts:  
//   CPU memory: decomp.exchange_halo(particles);  (a reused list refreshes application fields)  
//   GPU memory: decomp.forward_exchange(particles, {"density", "pressure"}, stream);  
compute_pressure_force(particles);               // owned and ghost slots  
decomp.reverse_halo(particles);  
integrate(particles);
```

**Points specific to SPH.**

- Set `halo.width` to the kernel support radius (for example, `2 h` for a cubic spline) plusthe skin. With variable smoothing lengths, use the largest.
- A second `exchange_halo()` on CPU memory resets ghost forces to zero and refreshes ghost positionsand application fields. If you have already accumulated ghost forces, do it before that pass,not between accumulation and `reverse_halo()`.
- Use `ReduceOp::Max` to share a quantity such as a signal speed for a time step limit, instead of `Sum`.

## 5. Lagrangian particle tracking

Particles are advected through a domain whose velocity field is yours. They do not interact, so no ghosts are needed.

**Register.** Any per-particle attributes (age, species, source id) as application fields. Theytravel with the particle.

**Setup.** Seed the particles on any rank, then distribute them:

```cpp
decomp.migrate(particles);          // each particle goes to the rank owning its position  
// or, while seeding: if (decomp.owner_rank({x, y, z}) == my_rank) add_particle(...);
```

**Step.**

```cpp
advance_positions(particles);                         // your velocity field, your integrator  
if (decomp.has_out_of_domain_particles(particles)) {  // collective: true on every rank if any rank needs it  
decomp.migrate(particles);  
}
```

**Points specific to tracking.**

- `migrate()` sends particles directly to their owner, so a fast particle that crosses severalsubdomains in one step is handled; you do not need a halo for that.
- On a periodic axis `migrate()` wraps positions back into the box. On a non-periodic axis a particlethat leaves the box is assigned to the nearest boundary rank; it is not deleted. Removesuch particles yourself.
- dsDecompX moves particles, not grid data. Storing and interpolating the flow field near subdomainfaces is your code's job.
- To keep ranks balanced as particles cluster, use `GridPolicy::Dynamic` or `rebalance()`.

##   

 
