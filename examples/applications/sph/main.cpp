// main.cpp -- the SPH program shared by dsDecompX, Cabana, and AMReX (weakly compressible fluid, see common.h).
// The time loop, the timing, and the output are here. What differs between the libraries is in a Backend
// (backend_dsdecompx.h, backend_cabana.h, backend_amrex.h). The library is chosen at compile time with one of
// USE_DSDECOMPX, USE_CABANA, USE_AMREX. The neighbor list and the force are the same code (common.h) for all of them.
//
// What a Backend provides:
//   Backend::name(), add_settings(), before_mpi()   its name, its own settings, and what to do before MPI_Init (select the GPU)
//   Backend(case, argc, argv)                       creates the domain decomposition and the particle storage
//   bounds()                                        the region of this rank
//   place(host)                                     stores the initial particles
//   kick_drift(), kick()                            half-step velocity update and position update (with wall reflection)
//   migrate()                                       sends particles to their new owners
//   halo_build()                                    builds the ghost particles
//   halo_refresh()                                  updates the ghost positions; returns true if the neighbor list must be rebuilt
//   build_neighbors(), compute_density(), compute_forces()   neighbor list, density, and force (functions of common.h on the library's
//                                                   particle storage)
//   density_exchange()                              sends the density to the ghosts
//   observe(), ghost_count(), sync(), on_gpu()
//
// Every step: 1. half-step velocity update and position update.
//             2. Every rebuild_every steps: migrate, halo, and neighbor list are rebuilt. In the other steps only the ghost positions
//                are refreshed.
//             3. Density, second halo transfer (the density), force, and the second half-step velocity update.
// step_ms is the whole loop timed at once. The times of the sections are measured afterwards over breakdown_steps steps, with a
// barrier around each section.
//
// Run:  srun -n 16 ./sph_dsdecompx gpu=1 particles_total=10000000
//       srun -n 288 ./sph_dsdecompx gpu=0 particles_total=1000000
#if defined(USE_DSDECOMPX)
#include "backend_dsdecompx.h"
#elif defined(USE_CABANA)
#include "backend_cabana.h"
#elif defined(USE_AMREX)
#include "backend_amrex.h"
#else
#error "define one of USE_DSDECOMPX, USE_CABANA, USE_AMREX"
#endif

int main(int argc, char** argv)
{
    Backend::before_mpi(argc, argv);
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    {
        Settings setting = {
            {"particles_per_rank", 0},   // weak scaling: particles per rank (approximate, because of the lattice)
            {"particles_total", 0},      // strong scaling: total number of particles (used if not 0)
            {"rebuild_every", 10},       // migrate, halo, and neighbor list are rebuilt at this interval
            {"warmup", 100},
            {"steps", 1000},
            {"breakdown_steps", 100},    // steps used to time the sections (0: do not time them)
        };
        Backend::add_settings(setting);
        parse_settings(argc, argv, setting, rank);
        int warmup = (int)setting["warmup"], steps = (int)setting["steps"];
        int rebuild_every = (int)setting["rebuild_every"], breakdown_steps = (int)setting["breakdown_steps"];

        SimCase problem;
        problem.setting = &setting;
        problem.rank = rank;
        problem.ranks = ranks;
        long long target = (long long)setting["particles_total"];
        if (target == 0) target = (long long)setting["particles_per_rank"] * ranks;
        problem.lattice = make_lattice(target);
        if (problem.lattice.total >= 2147483647LL) {            // the particle id is an int
            if (rank == 0) printf("too many particles: %lld (the id is an int)\n", problem.lattice.total);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        problem.dims[0] = problem.dims[1] = problem.dims[2] = 0;
        MPI_Dims_create(ranks, 3, problem.dims);
        problem.params = sim_params();
        const long long total = problem.lattice.total;

        Backend backend(problem, argc, argv);
        auto sync = [&] { backend.sync(); };

        // ---- Create the particles (each rank only the lattice points of its own region)
        {
            double lower[3], upper[3];
            backend.bounds(lower, upper);
            HostParticles host;
            generate_block(problem.lattice, lower, upper, host);
            long long placed = host.id.size(), placed_total = 0;
            MPI_Allreduce(&placed, &placed_total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            if (placed_total != total) {
                if (rank == 0) printf("wrong number of particles: %lld / %lld\n", placed_total, total);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            backend.place(host);
        }

        // ---- First halo, neighbor list, density, and force
        backend.halo_build();
        backend.build_neighbors();
        backend.compute_density();
        backend.density_exchange();
        backend.compute_forces();
        SimResult result = {};
        result.library = Backend::name();
        result.gpu = backend.on_gpu();
        result.ranks = ranks;
        for (int a = 0; a < 3; a++) result.dims[a] = problem.dims[a];
        result.particles = total;
        result.rebuild_every = rebuild_every;
        result.warmup = warmup;
        result.steps = steps;
        result.breakdown_steps = breakdown_steps;
        backend.observe(result.potential0, result.kinetic0);

        // ---- One step
        long long step = 0, neighbor_builds = 0;
        SectionTimer timer;
        auto one_step = [&](bool detail) {
            auto section = [&](int k, auto&& work) {
                if (!detail) {
                    work();
                    return;
                }
                sync();
                MPI_Barrier(MPI_COMM_WORLD);
                double time0 = MPI_Wtime();
                work();
                sync();
                timer.local_ms[k] += (MPI_Wtime() - time0) * 1000;
            };
            bool rebuild = (step % rebuild_every == 0);
            bool need_neighbors = rebuild;
            section(SEC_INTEGRATE, [&] { backend.kick_drift(); });
            if (rebuild) {
                section(SEC_MIGRATE, [&] { backend.migrate(); });
                section(SEC_HALO_BUILD, [&] { backend.halo_build(); });
            } else {
                section(SEC_HALO_REFRESH, [&] { need_neighbors = backend.halo_refresh(); });
            }
            if (need_neighbors) {
                section(SEC_NEIGHBOR, [&] { backend.build_neighbors(); });
                neighbor_builds++;
            }
            section(SEC_DENSITY, [&] { backend.compute_density(); });
            section(SEC_DENSITY_HALO, [&] { backend.density_exchange(); });
            section(SEC_FORCE, [&] { backend.compute_forces(); });
            section(SEC_INTEGRATE, [&] { backend.kick(); });
            step++;
        };

        for (int s = 0; s < warmup; s++) one_step(false);
        sync();
        MPI_Barrier(MPI_COMM_WORLD);
        long long builds_before = neighbor_builds;
        double time0 = MPI_Wtime();
        for (int s = 0; s < steps; s++) one_step(false);
        sync();
        double local_elapsed = (MPI_Wtime() - time0) * 1000, elapsed = 0;
        MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        result.step_ms = elapsed / steps;
        result.neighbor_builds_per_step = (double)(neighbor_builds - builds_before) / steps;
        backend.observe(result.potential1, result.kinetic1);
        unsigned long long ghosts = backend.ghost_count(), ghosts_total = 0;
        MPI_Reduce(&ghosts, &ghosts_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        result.ghost_per_owned = (double)ghosts_total / total;

        // ---- Time of each section
        for (int k = 0; k < SEC_COUNT; k++) result.section_ms[k] = -1;
        if (breakdown_steps > 0) {
            for (int s = 0; s < breakdown_steps; s++) one_step(true);
            timer.finish(result.section_ms, breakdown_steps);
        }
        if (rank == 0) print_result(result);
    }
    MPI_Finalize();
    return 0;
}
