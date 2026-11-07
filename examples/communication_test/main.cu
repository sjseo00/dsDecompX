// Communication test of dsDecompX: fixed number of particles per rank, no force evaluation. The CPU and the GPU use the same program.
//
// Every step does (in the order of an application; step_ms is the mean of this part):
//   1. Move all particles randomly by a small amount.
//   2. Every rebuild_every steps, migrate the particles to their new owners and rebuild the halo.
//      The other steps only refresh the halo (reused plan on the GPU, rebuilt every time on the CPU).
//   3. Fill the forces and add the forces of the ghosts to their owners with reverse_halo.
// In the rebuild steps, the other public functions are also timed one by one (not part of step_ms):
//   rebalance (default options, force), refresh with positions only (GPU), begin/end_halo_exchange (GPU),
//   owner_rank, any_rank_true, has_out_of_domain_particles (CPU), update_domain, set_halo_width, invalidate_halo
// After all steps, on the GPU a field is registered and forward_exchange / reverse_exchange are timed.
// At the end the performance-counter functions (set_performance_counters_enabled, get_stats, get_global_stats, reset_stats)
// and the query functions (topology, backend_kind, mpi_takes_device_pointers, halo_width, can_refresh_halo) are called once.
// The time is the largest value over the ranks, and all ranks start every operation together after a barrier.
//
// Run:  srun -n 16 ./communication_test gpu=1 particles_per_rank=1000000
//       key=value arguments change the defaults of `setting` below.
#include <dsdecompx.hpp>
#include "random.h"

#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

__global__ void place_particles(double* x, double* y, double* z, double* mass, int* id, int count, int first_id,
                                double lower_x, double lower_y, double lower_z,
                                double size_x, double size_y, double size_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    long long n = first_id + i;
    x[i] = lower_x + size_x * random_start(n, 0);
    y[i] = lower_y + size_y * random_start(n, 1);
    z[i] = lower_z + size_z * random_start(n, 2);
    mass[i] = 1.0;
    id[i] = first_id + i;
}

__global__ void move_particles(double* x, double* y, double* z, const int* id, int count, double move, long long step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    x[i] += move * random_move(id[i], step, 0);
    y[i] += move * random_move(id[i], step, 1);
    z[i] += move * random_move(id[i], step, 2);
}

__global__ void fill_forces(double* fx, double* fy, double* fz, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    fx[i] = 1.0;
    fy[i] = 1.0;
    fz[i] = 1.0;
}

// Timed operations. A value of -1 means that the operation was not timed in this step.
enum Op {
    MOVE, MIGRATE, REBUILD, REFRESH, REVERSE,                       // every step (step_ms)
    REBALANCE, REBALANCE_FORCE, REFRESH_POS, SPLIT,                 // additionally timed in the rebuild steps
    ANY_RANK, OUT_OF_DOMAIN, OWNER_RANK, UPDATE_DOMAIN, SET_HALO_WIDTH, INVALIDATE,
    NUM_OPS
};

int main(int argc, char** argv)
{
    // A GPU-aware MPI (GTL) is initialized for the GPU that is current at MPI_Init. If every rank does not select its own GPU first,
    // all ranks initialize for GPU 0 and the direct GPU-to-GPU communication inside a node (IPC) is switched off.
    bool gpu_run = true;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "gpu=0") gpu_run = false;
    }
    if (gpu_run) {
        const char* local_id = std::getenv("SLURM_LOCALID");
        cudaSetDevice(local_id ? std::atoi(local_id) : 0);
    }

    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    // ---- Settings (defaults). Change them with key=value on the command line.
    std::map<std::string, double> setting = {
        {"gpu", 1},                  // 1: GPU, 0: CPU
        {"backend", 1},              // 0 auto, 1 mpi, 2 nccl, 3 nvshmem (GPU only)
        {"particles_per_rank", 0},   // number of particles of one rank (fixed per rank)
        {"particles_total", 0},      // total number of particles (used if not 0)
        {"halo", 2.5},               // halo width; the mean spacing of the particles is 1
        {"move", 0.002},             // largest move along each axis per step (in units of the particle spacing)
        {"rebuild_every", 10},       // migrate, rebuild the halo, and time the other public functions at this interval
        {"warmup", 100},             // steps that are not timed
        {"steps", 1000},             // timed steps
        {"field_components", 3},     // components (double) of the field sent by forward / reverse_exchange; 0 = not timed (GPU only)
    };
    for (int i = 1; i < argc; i++) {
        char name[64];
        double value;
        if (sscanf(argv[i], "%63[^=]=%lf", name, &value) != 2 || setting.count(name) == 0) {
            if (rank == 0) printf("unknown setting: %s\n", argv[i]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        setting[name] = value;
    }
    bool use_gpu = setting["gpu"] == 1;
    int warmup = (int)setting["warmup"];
    int steps = (int)setting["steps"];
    int rebuild_every = (int)setting["rebuild_every"];
    double move = setting["move"];

    // ---- Number of particles and box. The box edge is chosen so that the mean spacing of the particles is 1.
    long long total = (long long)setting["particles_total"];
    if (total == 0) total = (long long)setting["particles_per_rank"] * ranks;
    if (total >= 2147483647LL) {                    // the particle number (global_id) is an int
        if (rank == 0) printf("too many particles: %lld (global_id is an int)\n", total);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int owned = total / ranks + (rank < total % ranks ? 1 : 0);
    int first_id = rank * (total / ranks) + std::min<long long>(rank, total % ranks);
    double edge = std::cbrt((double)total);

    // ---- dsDecompX
    dsdecompx::Config config;
    config.domain.upper = {edge, edge, edge};
    config.halo.width = setting["halo"];
    config.advanced.read_environment = false;
    if (use_gpu) {
        const char* backend_names[] = {"auto", "mpi", "nccl", "nvshmem"};
        config.execution = dsdecompx::ExecutionMode::Cuda;
        config.advanced.selection = dsdecompx::parse_backend_selection(backend_names[(int)setting["backend"]]);
        config.advanced.cuda_device = dsdecompx::kCudaDeviceByLocalRank;
    } else {
        config.execution = dsdecompx::ExecutionMode::Host;
        config.advanced.selection = dsdecompx::BackendSelection::Mpi;
    }
    {
        dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);
        dsdecompx::LocalDomain local = decomp.topology().local_domain();
        dsdecompx::Vec3i grid = decomp.topology().dimensions();

        cudaStream_t stream = nullptr;
        if (use_gpu) cudaStreamCreate(&stream);

        // ---- Create the particles. Each rank places them inside its own subdomain.
        dsdecompx::ParticleInfo particles = dsdecompx::ParticleInfo::allocate(
            use_gpu ? dsdecompx::MemorySpace::CudaDevice : dsdecompx::MemorySpace::Host, owned);
        double size_x = local.upper[0] - local.lower[0];
        double size_y = local.upper[1] - local.lower[1];
        double size_z = local.upper[2] - local.lower[2];
        if (use_gpu) {
            place_particles<<<(owned + 255) / 256, 256>>>(particles.x_data(), particles.y_data(), particles.z_data(),
                                                          particles.mass_data(), particles.global_id_data(), owned, first_id,
                                                          local.lower[0], local.lower[1], local.lower[2], size_x, size_y, size_z);
            cudaDeviceSynchronize();
        } else {
            for (int i = 0; i < owned; i++) {
                long long n = first_id + i;
                particles.x_data()[i] = local.lower[0] + size_x * random_start(n, 0);
                particles.y_data()[i] = local.lower[1] + size_y * random_start(n, 1);
                particles.z_data()[i] = local.lower[2] + size_z * random_start(n, 2);
                particles.mass_data()[i] = 1.0;
                particles.global_id_data()[i] = first_id + i;
            }
        }

        // ---- Helpers for the timing
        // Time (ms) from the common start of all ranks until the work is finished
        auto timed = [&](auto&& work) {
            MPI_Barrier(MPI_COMM_WORLD);
            double time0 = MPI_Wtime();
            work();
            if (use_gpu) cudaStreamSynchronize(stream);
            return (MPI_Wtime() - time0) * 1000;
        };
        auto move_all = [&](long long step) {
            int count = particles.owned_count();
            if (use_gpu) {
                move_particles<<<(count + 255) / 256, 256, 0, stream>>>(particles.x_data(), particles.y_data(), particles.z_data(),
                                                                        particles.global_id_data(), count, move, step);
            } else {
                for (int i = 0; i < count; i++) {
                    long long id = particles.global_id_data()[i];
                    particles.x_data()[i] += move * random_move(id, step, 0);
                    particles.y_data()[i] += move * random_move(id, step, 1);
                    particles.z_data()[i] += move * random_move(id, step, 2);
                }
            }
        };
        auto exchange = [&](bool rebuild, bool positions_only) {
            dsdecompx::HaloExchangeOptions options;
            options.rebuild = rebuild;
            options.stream = stream;
            options.positions_only = positions_only;
            return decomp.exchange_halo(particles, options);
        };
        auto fill_all_forces = [&]() {
            int all = particles.size();                           // owned + ghost
            if (use_gpu) {
                fill_forces<<<(all + 255) / 256, 256, 0, stream>>>(particles.fx_data(), particles.fy_data(), particles.fz_data(), all);
                cudaStreamSynchronize(stream);
            } else {
                for (int i = 0; i < all; i++) particles.fx_data()[i] = particles.fy_data()[i] = particles.fz_data()[i] = 1.0;
            }
        };

        // ---- Time loop
        double sum[NUM_OPS] = {0};
        long long count_op[NUM_OPS] = {0};
        unsigned long long total_sent = 0, ghosts_at_end = 0;
        int count_rebuild = 0;
        bool can_refresh = false;
        const int owner_calls = 1000;
        double owner_sink = 0;

        for (int step = 0; step < warmup + steps; step++) {
            bool rebuild = (step % rebuild_every == 0);
            double local_ms[NUM_OPS];
            for (int k = 0; k < NUM_OPS; k++) local_ms[k] = -1;
            dsdecompx::MigrationResult moved;
            dsdecompx::HaloResult halo;

            // Every step: move, migrate (rebuild step), halo, reverse
            local_ms[MOVE] = timed([&] { move_all(step); });
            if (rebuild) {
                local_ms[MIGRATE] = timed([&] { moved = decomp.migrate(particles, stream); });
                local_ms[REBUILD] = timed([&] { halo = exchange(true, false); });
            } else {
                local_ms[REFRESH] = timed([&] { halo = exchange(false, false); });
            }
            can_refresh = decomp.can_refresh_halo(particles);
            fill_all_forces();
            local_ms[REVERSE] = timed([&] { decomp.reverse_halo(particles, stream); });

            // In the rebuild steps, time the other public functions one by one. The halo is rebuilt to restore the state (not timed).
            if (rebuild) {
                local_ms[OWNER_RANK] = timed([&] {
                    for (int i = 0; i < owner_calls; i++) {
                        dsdecompx::Vec3d position{edge * random01(7ULL * i + step), edge * random01(7ULL * i + step + 1),
                                                  edge * random01(7ULL * i + step + 2)};
                        owner_sink += decomp.owner_rank(position);
                    }
                }) / owner_calls;                                  // time of one call
                local_ms[ANY_RANK] = timed([&] { owner_sink += decomp.any_rank_true(false); });
                if (!use_gpu) local_ms[OUT_OF_DOMAIN] = timed([&] { owner_sink += decomp.has_out_of_domain_particles(particles); });
                if (use_gpu) {
                    local_ms[REFRESH_POS] = timed([&] { exchange(false, true); });
                    exchange(false, false);
                    cudaStreamSynchronize(stream);
                    local_ms[SPLIT] = timed([&] {
                        dsdecompx::HaloExchangeOptions options;
                        options.rebuild = false;
                        options.stream = stream;
                        decomp.begin_halo_exchange(particles, options);
                        decomp.end_halo_exchange(particles);
                    });
                }
                local_ms[REBALANCE] = timed([&] { decomp.rebalance(particles, dsdecompx::RebalanceOptions{}, stream); });
                exchange(true, false);
                dsdecompx::RebalanceOptions forced;
                forced.force = true;
                local_ms[REBALANCE_FORCE] = timed([&] { decomp.rebalance(particles, forced, stream); });
                exchange(true, false);
                local_ms[UPDATE_DOMAIN] = timed([&] { decomp.update_domain(config.domain); });
                exchange(true, false);
                local_ms[SET_HALO_WIDTH] = timed([&] { decomp.set_halo_width(setting["halo"]); });
                local_ms[INVALIDATE] = timed([&] { decomp.invalidate_halo(); });
                halo = exchange(true, false);
            }

            // Collect the time of the slowest rank and the total counts on rank 0
            double slowest_ms[NUM_OPS];
            MPI_Reduce(local_ms, slowest_ms, NUM_OPS, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            unsigned long long local_count[2] = {halo.ghost_count, moved.sent_count};
            unsigned long long sum_count[2];
            MPI_Reduce(local_count, sum_count, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0 && step >= warmup) {
                for (int k = 0; k < NUM_OPS; k++) {
                    if (slowest_ms[k] >= 0) {
                        sum[k] += slowest_ms[k];
                        count_op[k]++;
                    }
                }
                if (rebuild) {
                    count_rebuild++;
                    total_sent += sum_count[1];
                }
                ghosts_at_end = sum_count[0];
            }
        }

        // ---- forward / reverse_exchange: a registered field is sent by migrate and halo as well, so it is timed at the end
        double sum_forward = -1, sum_reverse_field = -1;
        int components = (int)setting["field_components"];
        if (use_gpu && components > 0) {
            dsdecompx::HaloResult built = exchange(true, false);
            cudaStreamSynchronize(stream);
            std::size_t capacity = (std::size_t)((particles.owned_count() + built.ghost_count) * 1.1) + 1024;
            double* feature = nullptr;
            cudaMalloc(&feature, capacity * components * sizeof(double));
            cudaMemset(feature, 0, capacity * components * sizeof(double));
            particles.add_field("feature", feature, capacity, components);
            exchange(true, false);                                // rebuild the plan with the field included
            cudaStreamSynchronize(stream);
            sum_forward = sum_reverse_field = 0;
            for (int it = 0; it < warmup + steps; it++) {
                double local_ms[2];
                local_ms[0] = timed([&] { decomp.forward_exchange(particles, {"feature"}, stream); });
                local_ms[1] = timed([&] { decomp.reverse_exchange(particles, {{"feature", dsdecompx::ReduceOp::Sum}}, stream); });
                double slowest_ms[2];
                MPI_Reduce(local_ms, slowest_ms, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (rank == 0 && it >= warmup) {
                    sum_forward += slowest_ms[0];
                    sum_reverse_field += slowest_ms[1];
                }
            }
        }

        // ---- Call the performance counters and the query functions once
        decomp.set_performance_counters_enabled(true, false);
        exchange(false, false);
        dsdecompx::PerfStats local_stats = decomp.get_stats();
        dsdecompx::PerfStats global_stats = decomp.get_global_stats();
        (void)local_stats;
        (void)global_stats;
        decomp.reset_stats();
        decomp.set_performance_counters_enabled(false, false);
        int backend = (int)decomp.backend_kind();
        int takes_device_pointers = decomp.mpi_takes_device_pointers() ? 1 : 0;
        double halo_width = decomp.halo_width();

        if (rank == 0) {
            auto mean = [&](int k) { return count_op[k] > 0 ? sum[k] / count_op[k] : -1.0; };
            double step_ms = (sum[MOVE] + sum[MIGRATE] + sum[REBUILD] + sum[REFRESH] + sum[REVERSE]) / steps;
            printf("RESULT gpu=%d backend=%d ranks=%d grid=%dx%dx%d particles=%lld particles_per_rank=%lld halo=%.2f move=%.4f "
                   "rebuild_every=%d warmup=%d steps=%d "
                   "step_ms=%.4f move_ms=%.4f migrate_ms=%.4f halo_rebuild_ms=%.4f halo_refresh_ms=%.4f reverse_ms=%.4f "
                   "rebalance_ms=%.4f rebalance_force_ms=%.4f halo_refresh_pos_ms=%.4f halo_split_ms=%.4f "
                   "any_rank_true_ms=%.5f out_of_domain_ms=%.4f owner_rank_us=%.4f update_domain_ms=%.4f "
                   "set_halo_width_ms=%.5f invalidate_halo_ms=%.5f forward_ms=%.4f reverse_field_ms=%.4f field_components=%d "
                   "ghost_per_owned=%.4f migrated_fraction=%.6f backend_kind=%d device_pointers=%d can_refresh=%d halo_width=%.2f sink=%.0f\n",
                   (int)use_gpu, (int)setting["backend"], ranks, grid[0], grid[1], grid[2], total, total / ranks, setting["halo"], move,
                   rebuild_every, warmup, steps,
                   step_ms, mean(MOVE), mean(MIGRATE), mean(REBUILD), mean(REFRESH), mean(REVERSE),
                   mean(REBALANCE), mean(REBALANCE_FORCE), mean(REFRESH_POS), mean(SPLIT),
                   mean(ANY_RANK), mean(OUT_OF_DOMAIN), count_op[OWNER_RANK] > 0 ? mean(OWNER_RANK) * 1000 : -1.0, mean(UPDATE_DOMAIN),
                   mean(SET_HALO_WIDTH), mean(INVALIDATE),
                   sum_forward < 0 ? -1.0 : sum_forward / steps, sum_reverse_field < 0 ? -1.0 : sum_reverse_field / steps,
                   use_gpu && components > 0 ? components : 0,
                   (double)ghosts_at_end / total, (double)total_sent / std::max(count_rebuild, 1) / total,
                   backend, takes_device_pointers, can_refresh ? 1 : 0, halo_width, owner_sink);
        }
    }
    MPI_Finalize();
    return 0;
}
