// Dynamic load balancing (GridPolicy::Dynamic). Particles are clustered on one side, and the run with fixed subdomains is compared with the run
// with moving subdomain boundaries.
//
// 70% of the particles form one cluster (Gaussian) that moves slowly along the diagonal of the box. The other 30% are uniformly distributed.
// With fixed subdomains, the rank that holds the cluster gets most of the work. With dynamic=1 the subdomain boundaries follow the cluster.
// Every particle performs the same amount of work (work), so the rank with the most particles is the slowest.
//
// Run:  srun -n 16 ./load_balancing gpu=1 particles_total=64000000 dynamic=0
//       srun -n 16 ./load_balancing gpu=1 particles_total=64000000 dynamic=1
#include <dsdecompx.hpp>
#include "workload.h"

#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

// Set the particle positions of this step. The cluster particles are around center, the others are at fixed positions.
__global__ void set_positions(double* x, double* y, double* z, const int* id, int count, double edge,
                              double center, double sigma, long long blob_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    long long n = id[i];
    x[i] = blob_position(n, 0, edge, center, sigma, blob_count);
    y[i] = blob_position(n, 1, edge, center, sigma, blob_count);
    z[i] = blob_position(n, 2, edge, center, sigma, blob_count);
}

__global__ void compute_work(double* result, int count, int work)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    double v = i;
    for (int k = 0; k < work; k++) v = v * 0.999999 + 0.5;
    result[i] = v;
}

int main(int argc, char** argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    std::map<std::string, double> setting = {
        {"gpu", 1},                  // 1: GPU, 0: CPU
        {"particles_total", 8000000},
        {"dynamic", 0},              // 0: fixed subdomains, 1: dynamic boundaries
        {"blob_fraction", 0.7},      // fraction of the particles in the cluster
        {"blob_sigma", 0.15},        // size of the cluster (fraction of the box edge). A small value puts the particles on few ranks, and the imbalance grows with
                                 // the number of ranks (0.06 gives 38 times the mean on 128 GPUs, and one halo message exceeds the 2 GB limit of an MPI int).
                                 // 0.15 gives about 4 times the mean independent of the number of ranks.
        {"work", 2000},              // work per particle; on the GPU use 1e5 or more so that the computation exceeds the communication
        {"halo", 2.5},
        {"rebuild_every", 10},
        {"interval", 1},             // with dynamic=1, check the imbalance every N migrations (0 = no automatic repartitioning)
        {"threshold", 1.10},         // move the boundaries when (busiest rank) / (mean) exceeds this value
        {"relaxation", 0.5},         // fraction of the way to the new boundaries that is moved at once
        {"steps", 400},
        {"print_every", 10},
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
    bool dynamic = setting["dynamic"] == 1;
    int steps = (int)setting["steps"];
    int rebuild_every = (int)setting["rebuild_every"];
    int work = (int)setting["work"];

    long long total = (long long)setting["particles_total"];
    long long blob_count = (long long)(setting["blob_fraction"] * total);
    int owned = total / ranks + (rank < total % ranks ? 1 : 0);
    int first_id = rank * (total / ranks) + std::min<long long>(rank, total % ranks);
    double edge = std::cbrt((double)total);
    double sigma = setting["blob_sigma"] * edge;

    dsdecompx::Config config;
    config.domain.upper = {edge, edge, edge};
    config.halo.width = setting["halo"];
    config.advanced.read_environment = false;
    config.grid_policy = dynamic ? dsdecompx::GridPolicy::Dynamic : dsdecompx::GridPolicy::Balanced;
    config.dynamic.interval = (std::size_t)setting["interval"];
    config.dynamic.imbalance_threshold = setting["threshold"];
    config.dynamic.relaxation = setting["relaxation"];
    if (use_gpu) {
        config.execution = dsdecompx::ExecutionMode::Cuda;
        config.advanced.cuda_device = dsdecompx::kCudaDeviceByLocalRank;
    } else {
        config.execution = dsdecompx::ExecutionMode::Host;
    }
    {
        dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);

        // At first the particles are divided among the ranks in the order of their id. The first migrate sends them to their subdomains.
        dsdecompx::ParticleInfo particles = dsdecompx::ParticleInfo::allocate(
            use_gpu ? dsdecompx::MemorySpace::CudaDevice : dsdecompx::MemorySpace::Host, owned);
        std::vector<int> ids(owned);
        for (int i = 0; i < owned; i++) ids[i] = first_id + i;
        cudaMemcpy(particles.global_id_data(), ids.data(), owned * sizeof(int), cudaMemcpyDefault);

        cudaStream_t stream = nullptr;                          // CUDA migrate/halo need an explicit stream
        if (use_gpu) cudaStreamCreate(&stream);

        double sum_step = 0, sum_compute = 0, sum_rebalance = 0, sum_halo = 0, sum_imbalance = 0;
        int measured = 0, measured_rebuild = 0;
        for (int step = 0; step < steps; step++) {
            bool rebuild = (step % rebuild_every == 0);
            double center = edge * (0.25 + 0.5 * step / steps);       // the center of the cluster moves along the diagonal from 0.25 to 0.75
            int count = particles.owned_count();
            if (use_gpu) {
                set_positions<<<(count + 255) / 256, 256>>>(particles.x_data(), particles.y_data(), particles.z_data(),
                                                            particles.global_id_data(), count, edge, center, sigma, blob_count);
                cudaDeviceSynchronize();
            } else {
                for (int i = 0; i < count; i++) {
                    long long n = particles.global_id_data()[i];
                    particles.x_data()[i] = blob_position(n, 0, edge, center, sigma, blob_count);
                    particles.y_data()[i] = blob_position(n, 1, edge, center, sigma, blob_count);
                    particles.z_data()[i] = blob_position(n, 2, edge, center, sigma, blob_count);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);

            // migrate (with dynamic=1 the subdomain boundaries are also moved when needed) and halo
            double time0 = MPI_Wtime();
            if (rebuild || step == 0) {
                decomp.migrate(particles, stream);
                if (use_gpu) cudaDeviceSynchronize();
            }
            double time1 = MPI_Wtime();
            dsdecompx::HaloExchangeOptions options;
            options.rebuild = rebuild || step == 0;
            options.stream = stream;
            decomp.exchange_halo(particles, options);
            if (use_gpu) cudaDeviceSynchronize();

            // The same amount of work for every particle
            double time2 = MPI_Wtime();
            count = particles.owned_count();
            if (use_gpu) {
                compute_work<<<(count + 255) / 256, 256>>>(particles.fx_data(), count, work);
                cudaDeviceSynchronize();
            } else {
                for (int i = 0; i < count; i++) {
                    double v = i;
                    for (int k = 0; k < work; k++) v = v * 0.999999 + 0.5;
                    particles.fx_data()[i] = v;
                }
            }
            double time3 = MPI_Wtime();

            // Slowest time over the ranks, and the number of particles of the rank that holds the most
            double local_ms[4] = {(time3 - time0) * 1000, (time3 - time2) * 1000, (time1 - time0) * 1000,   // step, compute, migrate (+ boundary move),
                                  (time2 - time1) * 1000};                                                   // halo
            double slowest_ms[4];
            MPI_Reduce(local_ms, slowest_ms, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            long long local_owned = particles.owned_count(), most_owned = 0;
            MPI_Reduce(&local_owned, &most_owned, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
            double imbalance = (double)most_owned / ((double)total / ranks);       // most loaded rank / mean

            if (rank == 0) {
                if (step % (int)setting["print_every"] == 0)
                    printf("STEP dynamic=%d ranks=%d step=%d imbalance=%.3f step_ms=%.3f compute_ms=%.3f rebalance_ms=%.3f\n",
                           (int)dynamic, ranks, step, imbalance, slowest_ms[0], slowest_ms[1], slowest_ms[2]);
                if (step >= steps / 2) {                     // average over the second half
                    sum_step += slowest_ms[0];
                    sum_compute += slowest_ms[1];
                    sum_halo += slowest_ms[3];
                    sum_imbalance += imbalance;
                    measured++;
                    if (rebuild) {                           // repartitioning happens only in the steps with migrate
                        sum_rebalance += slowest_ms[2];
                        measured_rebuild++;
                    }
                }
            }
        }
        if (rank == 0)
            printf("RESULT lib=dsdecompx dynamic=%d gpu=%d ranks=%d particles=%lld interval=%d threshold=%.2f relaxation=%.2f "
                   "imbalance=%.3f step_ms=%.3f compute_ms=%.3f rebalance_ms=%.3f halo_ms=%.3f\n",
                   (int)dynamic, (int)use_gpu, ranks, total, (int)setting["interval"], setting["threshold"], setting["relaxation"],
                   sum_imbalance / measured, sum_step / measured, sum_compute / measured, sum_rebalance / std::max(measured_rebuild, 1),
                   sum_halo / measured);
    }
    MPI_Finalize();
    return 0;
}
