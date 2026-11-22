# dsDecompX examples

Every program in this folder is built against an installed dsDecompX. Install it first as described in
[docs/02_Installation.md](../docs/02_Installation.md).

## Small examples

These programs run on any computer with MPI. Each rank has one particle, so the output is easy to check.

| Folder | What it does | Output on 2 ranks |
|---|---|---|
| [`getting_started`](getting_started/) | Migrates particles, builds the halo, and returns the forces of the ghosts to their owners with `reverse_halo`. This is the example in the dsDecompX paper | `rank 0 ghosts=1 returned_fx=1` |
| [`registered_field`](registered_field/) | The same with an application field, registered with `add_field` and returned with `reverse_exchange` | the same as above |
| [`domain_decomposition`](domain_decomposition/) | Prints the process grid, the coordinate of each rank, and its neighbors | `rank 0 grid 2x1x1 coordinate 0,0,0 neighbors 2` |
| [`halo_exchange`](halo_exchange/) | Prints how many ghost particles each rank receives | `rank 0 received 1 ghosts` |
| [`particle_migration`](particle_migration/) | Moves a particle that left its subdomain to its new owner | `rank 0 now owns 1 particles` |

Each rank prints one line.

## Sanity check

[`sanity_check`](sanity_check/) checks that an installation works: process-grid selection, backend names, and migration,
halo exchange, and reverse exchange on several ranks. The CUDA tests are off by default. See
[sanity_check/README.md](sanity_check/README.md).

## Build and run

Run these commands in this folder:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=<dsDecompX install>
cmake --build build -j
ctest --test-dir build --output-on-failure        # 20 tests on 1, 2, and 4 ranks
mpiexec -n 2 build/dsdecompx_getting_started      # one example
```

## Programs of the dsDecompX paper

These programs produced the results of the paper. They need CUDA and MPI. Each folder has a README with the build and
run commands.

| Folder | What it does |
|---|---|
| [`applications`](applications/) | Molecular dynamics, discrete element method, and smoothed particle hydrodynamics. Each solves the same problem with dsDecompX, Cabana, and AMReX |
| [`communication_test`](communication_test/) | Measures the time of migration, halo rebuild, halo refresh, and reverse exchange with a fixed number of particles per rank |
| [`load_balancing`](load_balancing/) | Compares fixed and dynamic subdomain boundaries for a cluster of particles that moves through the box |

`communication_test` and `load_balancing` run on the CPU with `gpu=0`.
