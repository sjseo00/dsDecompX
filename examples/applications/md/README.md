# Molecular dynamics (Lennard-Jones fluid) -- dsDecompX, Cabana, and AMReX

One program for each library, with the same physics, the same initial state, the same neighbor list, and the same force. They differ only in
how the particles are stored and in the communication: migration of particles to their new owners and exchange of the ghost (halo) particles.
This is the comparison of the application tests of the dsDecompX paper.

## Problem

A Lennard-Jones fluid in reduced units (density 0.8442, temperature 0.728, cutoff 2.5, time step 0.005). The particles start on an FCC lattice in a box with reflecting walls. The forces come from a Verlet neighbor list with a skin of 0.3.

The force uses only the positions of the neighbors, so the halo refresh can send only the positions.

## Communication

| Step | dsDecompX | Cabana | AMReX |
|---|---|---|---|
| Rebuild (every 10th step) | `migrate`, `exchange_halo(rebuild=true)` | `Distributor` + `migrate`, `Halo` + `Gather` | `Redistribute`, `fillNeighbors` |
| Other steps | `exchange_halo(rebuild=false)` (GPU: positions only) | `Gather` of the position slice | `updateNeighbors` (whole particle) |

Cabana does not cut the domain, find the new owner of a particle, or select the boundary particles. That part is application code in
`common.h` and is included in the times. AMReX keeps one box of cells per rank, with a cell width of at least the halo width.

## Files

| File | Content |
|---|---|
| `main.cpp` | The time loop, the timing, and the output. The library is chosen with `USE_DSDECOMPX`, `USE_CABANA`, or `USE_AMREX` (set by the CMake files). |
| `common.h` | Physics, initial state, cell list, neighbor list, force, the rank grid for Cabana and AMReX, and the result line. |
| `backend_dsdecompx.h` | The dsDecompX part (CPU and GPU in the same code). |
| `backend_cabana.h` | The Cabana part. |
| `backend_amrex.h` | The AMReX part. |
| `random.h` | Hash-based random numbers (the initial velocities are the same on every rank and in every library). |
| `dsdecompx/`, `cabana/`, `amrex/` | One CMake project for each program. |

## Build

dsDecompX must be built with CUDA and installed. The CPU run uses the same program (`gpu=0`).

```sh
cmake -S dsdecompx -B build_dsdecompx -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<dsDecompX install> -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build_dsdecompx -j
```

Cabana: Kokkos built with CUDA gives the GPU program and Kokkos built with OpenMP gives the CPU program. For the GPU, use `nvcc_wrapper` of Kokkos as
the C++ compiler.

```sh
cmake -S cabana -B build_cabana -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="<Kokkos install>;<Cabana install>" \
      -DCMAKE_CXX_COMPILER=<nvcc_wrapper or an MPI C++ compiler>
cmake --build build_cabana -j
```

AMReX: `AMReX_GPU_BACKEND=CUDA` gives the GPU program and `NONE` gives the CPU program.

```sh
cmake -S amrex -B build_amrex -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<AMReX install> -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build_amrex -j
```

## Run

```sh
srun -n 16 ./md_dsdecompx gpu=1 particles_total=10000000      # GPU, one rank per GPU
srun -n 16 ./md_cabana --kokkos-map-device-id-by=mpi_rank particles_total=10000000
srun -n 16 ./md_amrex particles_total=10000000
srun -n 288 ./md_dsdecompx gpu=0 particles_total=1000000     # CPU
```

On the GPU use a GPU-aware MPI (`CUDA_AWARE_MPI=1` tells dsDecompX that the MPI library is GPU-aware). Cabana creates new device buffers in every
rebuild, which can make the GPU-to-GPU transfers of Cray MPICH very slow (see the comment in `backend_cabana.h`); compare with
`MPICH_GPU_IPC_ENABLED=0`.

| Setting | Default | Meaning |
|---|---|---|
| `particles_total` | 0 | Total number of particles (strong scaling). `particles_total` is rounded to an FCC lattice of 4 n^3 particles. |
| `particles_per_rank` | 0 | Particles per rank (weak scaling), used if `particles_total` is 0. |
| `rebuild_every` | 10 | Migration, halo build, and neighbor list are rebuilt at this interval. |
| `warmup`, `steps` | 100, 1000 | Warm-up steps and timed steps. |
| `breakdown_steps` | 100 | Steps used to time each section with a barrier around it (0: skip). |
| `gpu` | 1 | dsDecompX only: 1 for GPU, 0 for CPU. |

## Output

Rank 0 prints one line that starts with `RESULT`. All values are `key=value`: the library, the number of ranks, the particles, `step_ms` (the whole loop timed
at once, no barriers inside), the time of each section per step (`integrate_ms`, `migrate_ms`, `halo_build_ms`, `halo_refresh_ms`, `neighbor_ms`,
`force_ms`; the slowest rank, timed with a barrier around each section), `ghost_per_owned`, and the energies. `pe0`, `ke0`, `e0`, `pe1`, `ke1`, `e1`: potential, kinetic, and total energy per particle before and after the timed steps (the total energy is nearly conserved).
