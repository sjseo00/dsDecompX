// Device halo plan reuse: a non-rebuild exchange must refresh ghost records in
// place (same count, order and ids; positions follow the owners; no layout
// change reported), while an invalidated plan must be rebuilt and reported.
#include <dsdecompx.hpp>

#include <cuda_runtime_api.h>
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int failures = 0;
void expect(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", what);
        ++failures;
    }
}
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    {
        dsdecompx::Config config;
        config.advanced.selection =
            dsdecompx::parse_backend_selection(argc > 1 ? argv[1] : "mpi");
        config.execution = dsdecompx::ExecutionMode::Cuda;
        config.advanced.read_environment = false;
        config.advanced.cuda_device = dsdecompx::kCudaDeviceByLocalRank;
        config.domain.upper = {8.0, 8.0, 8.0};
        config.halo.width = 0.25;
        dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);

        constexpr std::size_t capacity = 256;
        double* values[11]{};
        int* integers[2]{};
        for (auto& field : values) {
            cuda_check(cudaMallocManaged(reinterpret_cast<void**>(&field),
                                         capacity * sizeof(double)));
            cuda_check(cudaMemset(field, 0, capacity * sizeof(double)));
        }
        for (auto& field : integers) {
            cuda_check(cudaMallocManaged(reinterpret_cast<void**>(&field),
                                         capacity * sizeof(int)));
            cuda_check(cudaMemset(field, 0, capacity * sizeof(int)));
        }
        const auto& local = decomp.topology().local_domain();
        const int rank = decomp.topology().rank();
        // A few particles hugging the lower corner (inside the halo slab of
        // several neighbors) and a few in the domain interior (never ghosts).
        constexpr std::size_t owned = 8;
        for (std::size_t k = 0; k < owned; ++k) {
            const bool near_corner = k < 4;
            const double offset = near_corner ? 0.05 + 0.01 * k : 1.0 + 0.1 * k;
            values[0][k] = local.lower[0] + offset;
            values[1][k] = local.lower[1] + offset;
            values[2][k] = local.lower[2] + offset;
            values[9][k] = 1.0;
            integers[0][k] = 0;
            integers[1][k] = rank * static_cast<int>(owned) + static_cast<int>(k);
        }

        auto particles = dsdecompx::ParticleInfo::external(
            dsdecompx::MemorySpace::CudaManaged, capacity, owned);
        particles.positions(values[0], values[1], values[2])
            .velocities(values[3], values[4], values[5])
            .forces(values[6], values[7], values[8])
            .properties(values[9], values[10], integers[0], integers[1]);

        cudaStream_t stream{};
        cuda_check(cudaStreamCreate(&stream));
        const auto sync = [&] { cuda_check(cudaStreamSynchronize(stream)); };

        const auto built = decomp.exchange_halo(
            particles, dsdecompx::HaloExchangeOptions{true, stream});
        sync();
        const std::size_t ghosts = built.ghost_count;
        std::uint64_t local_ghosts = ghosts, global_ghosts = 0;
        MPI_Allreduce(&local_ghosts, &global_ghosts, 1, MPI_UINT64_T, MPI_SUM,
                      MPI_COMM_WORLD);
        expect(global_ghosts > 0, "the plan build produced ghosts");
        expect(built.layout_changed, "a rebuild reports a layout change");

        std::vector<int> ids_before(ghosts);
        std::vector<double> x_before(ghosts), y_before(ghosts), z_before(ghosts);
        for (std::size_t g = 0; g < ghosts; ++g) {
            ids_before[g] = integers[1][owned + g];
            x_before[g] = values[0][owned + g];
            y_before[g] = values[1][owned + g];
            z_before[g] = values[2][owned + g];
        }

        // Several reused refreshes in a row; every owner moves a little each
        // time (including particles that leave the halo slab, whose ghost
        // images must stay in their slots until the next rebuild).
        constexpr double drift = 0.01;
        constexpr double jump = 1.0;
        for (int step = 1; step <= 3; ++step) {
            for (std::size_t k = 0; k < owned; ++k) {
                values[0][k] += drift;
                values[1][k] += drift;
                values[2][k] += drift;
            }
            // From step 2 on, the first particle of every rank jumps out of the
            // halo slab. A per-call reclassification would drop its ghosts and
            // shift every later ghost slot; a reused plan must not.
            if (step == 2) {
                for (int axis = 0; axis < 3; ++axis) values[axis][0] += jump;
            }
            const auto refreshed = decomp.exchange_halo(
                particles, dsdecompx::HaloExchangeOptions{false, stream});
            sync();
            expect(refreshed.ghost_count == ghosts, "reused plan keeps the ghost count");
            expect(!refreshed.layout_changed, "reused plan reports no layout change");
            expect(decomp.topology().size() == 1 || ghosts == 0 ||
                       particles.ghost_count() == ghosts,
                   "ghost count is committed");
            bool same = true, followed = true;
            for (std::size_t g = 0; g < ghosts; ++g) {
                same = same && integers[1][owned + g] == ids_before[g];
                const double expected =
                    step * drift +
                    (step >= 2 && ids_before[g] % static_cast<int>(owned) == 0 ? jump : 0.0);
                followed = followed &&
                    std::abs(values[0][owned + g] - x_before[g] - expected) < 1e-12 &&
                    std::abs(values[1][owned + g] - y_before[g] - expected) < 1e-12 &&
                    std::abs(values[2][owned + g] - z_before[g] - expected) < 1e-12;
            }
            expect(same, "reused plan keeps ghost order and ids");
            expect(followed, "ghost positions follow their owners");
        }

        // Split-phase exchange: begin/end around other work must give the same
        // ghosts as exchange_halo(), and misuse is reported rather than hanging.
        {
            expect(decomp.can_refresh_halo(particles),
                   "a built plan can be refreshed in place");
            for (std::size_t k = 0; k < owned; ++k) values[0][k] += drift;
            decomp.begin_halo_exchange(
                particles, dsdecompx::HaloExchangeOptions{false, stream});
            bool second_begin_rejected = false;
            try {
                decomp.begin_halo_exchange(
                    particles, dsdecompx::HaloExchangeOptions{false, stream});
            } catch (const std::logic_error&) {
                second_begin_rejected = true;
            }
            expect(second_begin_rejected, "a second begin while one is in flight is rejected");
            // Work on owned particles may run here (it must not touch ghost slots).
            const auto split = decomp.end_halo_exchange(particles);
            sync();
            expect(split.ghost_count == ghosts, "split exchange keeps the ghost count");
            expect(!split.layout_changed && split.plan_reused,
                   "split exchange refreshes the plan in place");
            bool same = true, followed = true;
            for (std::size_t g = 0; g < ghosts; ++g) {
                same = same && integers[1][owned + g] == ids_before[g];
                const double expected =
                    4 * drift +
                    (ids_before[g] % static_cast<int>(owned) == 0 ? jump : 0.0);
                followed = followed &&
                    std::abs(values[0][owned + g] - x_before[g] - expected) < 1e-12;
            }
            expect(same, "split exchange keeps ghost order and ids");
            expect(followed, "split exchange refreshes ghost positions");
            bool end_without_begin_rejected = false;
            try {
                decomp.end_halo_exchange(particles);
            } catch (const std::logic_error&) {
                end_without_begin_rejected = true;
            }
            expect(end_without_begin_rejected, "end without begin is rejected");
        }

        // Dropping the plan forces a new classification, which must be
        // reported even though the call asks for no rebuild.
        decomp.invalidate_halo();
        expect(!decomp.can_refresh_halo(particles), "an invalidated plan is not refreshable");
        const auto replanned = decomp.exchange_halo(
            particles, dsdecompx::HaloExchangeOptions{false, stream});
        sync();
        if (replanned.ghost_count != 0) {
            expect(replanned.layout_changed, "a fresh plan is reported as a layout change");
        }
        // The jumped particles left the slab, so the new classification must
        // have dropped their ghost images (this is what reuse deliberately
        // defers until the next rebuild).
        expect(replanned.ghost_count < ghosts, "a replan drops ghosts that left the slab");

        cuda_check(cudaStreamDestroy(stream));
        for (auto* field : integers) cuda_check(cudaFree(field));
        for (auto* field : values) cuda_check(cudaFree(field));
    }
    int failed_local = failures, failed_global = 0;
    MPI_Allreduce(&failed_local, &failed_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::printf("halo_reuse: %s\n", failed_global ? "FAILED" : "PASSED");
    MPI_Finalize();
    return failed_global ? EXIT_FAILURE : EXIT_SUCCESS;
}
