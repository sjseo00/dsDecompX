# dsDecompX examples

This folder has three kinds of programs. Install dsDecompX first (see `docs/02_Installation.md`); every program here is built against the installed library.

## 1. Small examples and tests (CPU, a few seconds)

Start here. These programs run on a laptop with any MPI and use one particle per rank, so the output is easy to check.

| Folder | What it shows | Output on 2 ranks |
|---|---|---|
| [`getting_started`](getting_started/) | The complete example of the paper (Section 3): migrate, build the halo, add the ghosts' forces back to their owners with `reverse_halo` | `rank 0 ghosts=1 returned_fx=1` (one line per rank) |
| [`registered_field`](registered_field/) | The same, with an application field registered by `add_field` and returned by `reverse_exchange` | like the one above; the program asserts the result |
| [`domain_decomposition`](domain_decomposition/) | The process grid, the coordinate of each rank, and its neighbors | `rank 0 grid 2x1x1 coordinate 0,0,0 neighbors 2` |
| [`halo_exchange`](halo_exchange/) | A halo exchange: how many ghost particles each rank receives | `rank 0 received 1 ghosts` |
| [`particle_migration`](particle_migration/) | Migration: a particle that left its subdomain moves to its new owner | `rank 0 now owns 1 particles` |
| [`tests`](tests/) | Automatic checks of the library (see [`tests/README.md`](tests/README.md)) | `ctest` reports passed or failed |

Build and run all of them, including the tests:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=<dsDecompX install>
cmake --build build -j
ctest --test-dir build --output-on-failure        # 20 tests on 1, 2, and 4 ranks
mpiexec -n 2 build/dsdecompx_getting_started      # run one example by hand
```

Run the commands in this folder. The CUDA tests are off by default; see [`tests/README.md`](tests/README.md).

## 2. Application comparison (the paper, Section 4.3)

[`applications`](applications/) has three applications (molecular dynamics, discrete element method, and smoothed particle hydrodynamics). Each solves the same problem with dsDecompX, Cabana, and AMReX, so the step time of the three libraries can be compared. It needs CUDA, MPI, and the other libraries. Each folder has its own README.

## 3. Measurements of dsDecompX itself (the paper, Sections 4.2 and 4.4)

| Folder | What it measures |
|---|---|
| [`communication_test`](communication_test/) | The time of each dsDecompX operation (migration, halo rebuild, halo refresh, reverse reduction) with a fixed number of particles per rank, while the number of ranks increases |
| [`load_balancing`](load_balancing/) | The effect of dynamic subdomain boundaries (`GridPolicy::Dynamic`) on a clustered particle distribution that moves through the box |

Both need CUDA and MPI (the CPU runs use the same program with `gpu=0`). Each folder has its own README with the build and run commands.
