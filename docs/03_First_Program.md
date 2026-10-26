# 3. First Program

This page answers one question: **how do I get a small program that uses dsDecompX working?**
You will write one file that scatters particles over the ranks, builds ghosts, computes a toy
pair force with Newton's third law, returns the ghost forces to their owners, and checks that the
total force is zero. It uses CPU memory only. The same steps for GPU memory are in the CUDA
section of [04_Using_in_Your_Code.md](04_Using_in_Your_Code.md).

Prerequisite: dsDecompX is installed ([02_Installation.md](02_Installation.md)).

## The program

Save as `main.cpp`.

```cpp
#include <dsdecompx.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <random>

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    {   // Decomp and ParticleInfo must be destroyed before MPI_Finalize()
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        const double box = 10.0;
        const double cutoff = 1.0;
        const std::size_t per_rank = 2000;

        // 1. Describe the box and the decomposition.
        dsdecompx::Config config;
        config.execution = dsdecompx::ExecutionMode::Host;   // particle data in CPU memory
        config.domain.upper = {box, box, box};                // lower defaults to {0, 0, 0}
        config.grid_policy = dsdecompx::GridPolicy::Balanced; // let the library pick px x py x pz
        config.halo.width = cutoff;                           // ghosts reach this far past a face
        dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);

        // 2. Create particles. Every rank starts with particles anywhere in the box;
        //    migrate() will move each one to the rank that owns its position.
        std::mt19937_64 rng(12345 + rank);
        std::uniform_real_distribution<double> uniform(0.0, box);
        dsdecompx::ParticleInfo particles(per_rank);
        for (std::size_t i = 0; i < per_rank; ++i) {
            particles.x[i] = uniform(rng);
            particles.y[i] = uniform(rng);
            particles.z[i] = uniform(rng);
            particles.mass[i] = 1.0;
            particles.global_id[i] = rank * static_cast<int>(per_rank) + static_cast<int>(i);
        }

        for (int step = 0; step < 5; ++step) {
            // 3. Move particles to their owners, then build the ghosts.
            decomp.migrate(particles);
            const dsdecompx::HaloResult halo = decomp.exchange_halo(particles);
            const std::size_t owned = particles.owned_count();
            const std::size_t total = particles.size();   // owned + ghosts

            // 4. Your force code. Owned slots are 0..owned-1, ghost slots owned..total-1.
            //    Ghost forces start at zero after exchange_halo(); owned forces are yours to reset.
            for (std::size_t i = 0; i < owned; ++i) {
                particles.fx[i] = 0.0;
                particles.fy[i] = 0.0;
                particles.fz[i] = 0.0;
            }
            for (std::size_t i = 0; i < owned; ++i) {
                for (std::size_t j = i + 1; j < total; ++j) {
                    // A pair of an owned and a ghost particle is visible on both ranks;
                    // compute it on exactly one: the rank whose owned particle has the smaller id.
                    if (j >= owned && particles.global_id[i] >= particles.global_id[j]) continue;

                    double dx = particles.x[i] - particles.x[j];
                    double dy = particles.y[i] - particles.y[j];
                    double dz = particles.z[i] - particles.z[j];
                    dx -= box * std::round(dx / box);   // minimum image (covers undivided axes)
                    dy -= box * std::round(dy / box);
                    dz -= box * std::round(dz / box);
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= cutoff * cutoff || r2 < 1.0e-24) continue;

                    const double r = std::sqrt(r2);
                    const double s = (cutoff - r) / r;   // soft repulsion
                    particles.fx[i] += s * dx;  particles.fx[j] -= s * dx;
                    particles.fy[i] += s * dy;  particles.fy[j] -= s * dy;
                    particles.fz[i] += s * dz;  particles.fz[j] -= s * dz;
                }
            }

            // 5. Add the forces accumulated on ghost slots to their owners.
            decomp.reverse_halo(particles);

            // Check: the total force over all ranks is zero (up to rounding).
            double net[3] = {0.0, 0.0, 0.0};
            for (std::size_t i = 0; i < owned; ++i) {
                net[0] += particles.fx[i];
                net[1] += particles.fy[i];
                net[2] += particles.fz[i];
            }
            double global_net[3];
            MPI_Allreduce(net, global_net, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            unsigned long long counts[2] = {owned, halo.ghost_count};
            unsigned long long totals[2];
            MPI_Allreduce(counts, totals, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            if (rank == 0) {
                std::printf("step %d: owned %llu, ghosts %llu, net force (%.2e, %.2e, %.2e)\n",
                            step, totals[0], totals[1],
                            global_net[0], global_net[1], global_net[2]);
            }

            // 6. Your integrator: move owned particles only.
            for (std::size_t i = 0; i < owned; ++i) {
                particles.x[i] += 0.001 * particles.fx[i];
                particles.y[i] += 0.001 * particles.fy[i];
                particles.z[i] += 0.001 * particles.fz[i];
            }
        }
    }
    MPI_Finalize();
    return 0;
}
```

## Build and run

Next to `main.cpp`, create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(first_program LANGUAGES CXX)

find_package(dsDecompX 1.0 REQUIRED CONFIG)

add_executable(first_program main.cpp)
target_link_libraries(first_program PRIVATE dsdecompx::dsdecompx)
```

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/desired/prefix
cmake --build build
mpirun -n 4 ./build/first_program
```

Try 1, 2, 4 and 8 ranks. On every run the `owned` count printed by rank 0 equals
`2000 x ranks`, because particles are moved, never created or lost. The ghost count grows with
the number of ranks, and a single rank has no ghosts. The net force is zero up to rounding, far
smaller than the individual forces. The program is meant for small rank counts: keep each
subdomain at least `2 x cutoff` wide, which for this box (10) and cutoff (1) allows up to five
ranks along each axis.

## What each call did

| Step | Call | Effect |
|---|---|---|
| 1 | `Config` fields, `Decomp` constructor | Created the Cartesian rank grid and assigned each rank a subdomain. `GridPolicy::Balanced` chose the grid shape from the box and rank count. |
| 2 | `ParticleInfo particles(n)` | Allocated host arrays for `n` owned particles: `x y z vx vy vz fx fy fz mass charge type global_id`. |
| 3 | `migrate()` | Sent every particle to the rank owning its position and wrapped coordinates into the box. Ghosts from earlier steps were discarded. |
| 3 | `exchange_halo()` | Appended ghost particles after the owned ones. `HaloResult::ghost_count` is how many. Because `migrate()` invalidates the previous ghost list, this call rebuilt it. |
| 5 | `reverse_halo()` | Added the forces you put on ghost slots to the owned particles they are copies of. |

Three habits from this program carry over to every real code:

- Particle ids (`global_id`) must be unique across all ranks. dsDecompX never creates ids.
- Forces on ghosts start at zero after each `exchange_halo()`. You accumulate onto owned and
  ghost slots, and `reverse_halo()` adds the ghost part to the owners.
- A pair of an owned and a ghost particle is seen by both ranks. Your code decides which rank
  computes it; here, the one whose owned particle has the smaller id.

## Next

Connect your own data and loop, switch to GPU memory, and add custom fields:
[04_Using_in_Your_Code.md](04_Using_in_Your_Code.md).
