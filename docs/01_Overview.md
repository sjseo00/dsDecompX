# 1. Overview

This page answers one question: **what does dsDecompX do for my code, and what stays my job?**
Read it first if you have never used a domain-decomposition library. If you already know the
vocabulary, go straight to [02_Installation.md](02_Installation.md).

## What dsDecompX is

dsDecompX divides a simulation box among several MPI ranks (CPUs, or one GPU per rank) and moves
particle data between them. It is a communication library: it does not compute forces, build
neighbor lists, or integrate anything. You keep your own physics code and call dsDecompX three
times per step to keep every rank's data complete.

It works for any particle method (molecular dynamics, DEM, SPH, particle-in-cell,
Lagrangian tracking). The built-in particle record has the fields molecular dynamics needs
(position, velocity, force, mass, charge, type, id), and you can attach your own per-particle
fields.

## The picture

The box is cut into a Cartesian grid of subdomains, one per rank. Each rank **owns** the
particles inside its subdomain. Near a subdomain face, a rank also holds read-only **ghost**
copies of particles that belong to the neighboring ranks, so it can compute interactions that
cross the face.

```text
   rank 0 subdomain          rank 1 subdomain
 +---------------------+---------------------+
 |                     |                     |
 |   owned  . . . .  ::|::  . . . .  owned   |
 |   particles       ::|::      particles    |
 |                   ::|::                   |
 +---------------------+---------------------+
                   <-->  halo width
        ::  = ghost copies held by the other rank
```

The grid is three-dimensional. A rank has up to 26 neighbors (faces, edges and corners), and
ghosts only ever come from those adjacent subdomains.

## Vocabulary

| Term | Meaning |
|---|---|
| Subdomain | The box region owned by one rank. |
| Owned particle | A particle inside this rank's subdomain. Your code integrates it. |
| Ghost particle | A copy of a neighbor's owned particle, placed within the halo width of this rank's faces. Your code reads it for force calculation but does not integrate it. |
| Halo width | How far from a face a particle must be to be copied as a ghost. Normally your force cutoff plus any neighbor-list skin. |
| Process grid | The `px x py x pz` arrangement of ranks. |
| Cut planes | The positions of the grid boundaries along each axis. Uniform by default; they can move for load balancing. |
| Halo plan | The list of owned particles sent as ghosts. It is kept between calls and reused until you ask for a rebuild. |

Owned and ghost particles share one set of arrays. Slots `0 ... owned_count()-1` are owned, and
slots `owned_count() ... size()-1` are ghosts.

## The three operations

| Call | What it does | When you call it |
|---|---|---|
| `migrate()` | Sends every particle that left this rank's subdomain to the rank that now owns it, and wraps positions across periodic boundaries. Ghosts are discarded. | Initially, to distribute particles. Then whenever particles may have crossed a subdomain face (typically each time you rebuild your neighbor list). |
| `exchange_halo()` | Fills the ghost slots with copies of nearby particles from the neighboring ranks. Ghost positions are already shifted across periodic boundaries. | Every step, before computing forces. |
| `reverse_halo()` | Adds the forces you accumulated on ghost slots to the owned particles they are copies of, on the owning rank. | Every step, after computing forces. |

Around these three there are a few related calls:

- **Application-defined fields** travel with the particles. `add_field()` registers your own
  array; `migrate()` moves it, `exchange_halo()` copies it to ghosts, and
  `reverse_exchange()` returns values from ghosts to owners with a reduction (`Sum`, `Min`,
  `Max`, `Replace`). `forward_exchange()` refreshes ghost copies of a field from the owners
  (CUDA memory only).
- **Load balancing** moves the cut planes so ranks hold similar amounts of work
  (`GridPolicy::Dynamic`, `rebalance()`).
- **Split-phase exchange** (`begin_halo_exchange()` / `end_halo_exchange()`) lets you compute
  on owned particles while the exchange is in flight.

## One timestep

```text
migrate()            only when particles may have left their subdomain
   |
exchange_halo()      ghosts appear after the owned particles
   |
your force code      loops over owned + ghost slots, accumulates forces on both
   |
reverse_halo()       ghost forces are added to their owners
   |
your integrator      updates owned particles only
```

## What dsDecompX does and what your code does

| dsDecompX | Your code |
|---|---|
| Chooses or applies the process grid and assigns subdomains | Chooses the box, the halo width and the execution mode |
| Moves particles between ranks and wraps periodic positions | Creates the initial particles and gives each a unique `global_id` |
| Copies ghosts, including periodic shifts of their positions | Builds neighbor lists and computes forces over owned and ghost particles |
| Returns ghost contributions to owners | Decides which rank computes a pair that crosses a face (see [06_Recipes.md](06_Recipes.md)) |
| Moves cut planes when you ask for load balancing | Integrates owned particles and applies thermostats and barostats |

## Where it runs

| `Config::execution` | Particle memory | Communication |
|---|---|---|
| `ExecutionMode::Host` | CPU memory | MPI |
| `ExecutionMode::Cuda` | GPU memory (`CudaDevice` or `CudaManaged`) | MPI (CUDA-aware, or staged through pinned CPU memory), NCCL, or NVSHMEM |

The function names are identical in both modes. CUDA calls add a CUDA stream argument. NCCL and
NVSHMEM are optional build-time add-ons; see [02_Installation.md](02_Installation.md) and the
CUDA section of [04_Using_in_Your_Code.md](04_Using_in_Your_Code.md).

## Rules that apply from the start

- **Collective calls.** `migrate()`, `exchange_halo()`, `reverse_halo()`, `reverse_exchange()`,
  `forward_exchange()`, `rebalance()` and the `Decomp` constructor must be called by every rank
  of the communicator, in the same order. Do not put them under `if (rank == 0)`.
- **Ghosts exist only across divided axes.** On an axis where the process grid has a single
  process, no ghosts are created, so a run with one rank has no ghosts at all. Apply the
  minimum-image convention in your own code along such axes.
- **Ghosts reach only adjacent subdomains.** Keep the halo width no larger than the narrowest
  subdomain width along any divided axis; the library does not check this.

## Limitations

- Only a 3-D Cartesian decomposition is supported; there is no 2-D or 1-D mode, and no
  non-Cartesian decomposition.
- `GridPolicy::Dynamic` moves cut planes only along the three axes of the Cartesian grid.

## Next

Install the library: [02_Installation.md](02_Installation.md). Then write your first program:
[03_First_Program.md](03_First_Program.md).
