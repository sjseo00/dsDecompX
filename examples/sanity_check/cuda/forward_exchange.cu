// Forward field exchange (Decomp::forward_exchange): every ghost slot of an application field must hold the value of the owner
// it is a copy of, for fields of several widths and types, and the exchange must be the adjoint of reverse_exchange(Sum):
//     sum over ghosts of forward(x)[g] . y[g]  ==  sum over owners of x[k] . reverse(y)[k]
// which is what a differentiable per-layer exchange (forward: owner -> ghost copy, backward: ghost -> owner sum) needs.
// Run with two or more ranks: mpiexec -n 2 dsdecompx_cuda_forward_exchange [mpi|nccl]
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

// A deterministic value in [-1, 1) for (seed, index).
double pseudo(std::uint64_t seed, std::uint64_t index) {
    std::uint64_t z = seed * 0x9E3779B97F4A7C15ull + index * 0xBF58476D1CE4E5B9ull + 0x94D049BB133111EBull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) / static_cast<double>(1ull << 53) * 2.0 - 1.0;
}
}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    {
        dsdecompx::Config config;
        config.advanced.selection = dsdecompx::parse_backend_selection(argc > 1 ? argv[1] : "mpi");
        config.execution = dsdecompx::ExecutionMode::Cuda;
        config.advanced.read_environment = false;
        config.advanced.cuda_device = dsdecompx::kCudaDeviceByLocalRank;
        config.domain.upper = {10.0, 10.0, 10.0};
        config.halo.width = 2.0;
        dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);

        constexpr std::size_t capacity = 2048;
        constexpr std::size_t owned = 120;
        double* values[11]{};
        int* integers[2]{};
        for (auto& field : values) {
            cuda_check(cudaMallocManaged(reinterpret_cast<void**>(&field), capacity * sizeof(double)));
            cuda_check(cudaMemset(field, 0, capacity * sizeof(double)));
        }
        for (auto& field : integers) {
            cuda_check(cudaMallocManaged(reinterpret_cast<void**>(&field), capacity * sizeof(int)));
            cuda_check(cudaMemset(field, 0, capacity * sizeof(int)));
        }
        constexpr std::size_t wide = 7;      // components of the double field (not a power of two on purpose)
        constexpr std::size_t narrow = 3;    // components of the float field
        double* feat = nullptr;
        float* ffeat = nullptr;
        cuda_check(cudaMallocManaged(reinterpret_cast<void**>(&feat), capacity * wide * sizeof(double)));
        cuda_check(cudaMallocManaged(reinterpret_cast<void**>(&ffeat), capacity * narrow * sizeof(float)));

        const auto& local = decomp.topology().local_domain();
        const int rank = decomp.topology().rank();
        for (std::size_t k = 0; k < owned; ++k) {
            for (int axis = 0; axis < 3; ++axis) {
                const double u = 0.5 * (pseudo(11 + axis, rank * 1000 + k) + 1.0);   // [0, 1)
                values[axis][k] = local.lower[axis] + u * (local.upper[axis] - local.lower[axis]);
            }
            values[9][k] = 1.0;
            integers[1][k] = rank * static_cast<int>(owned) + static_cast<int>(k);
        }

        auto particles = dsdecompx::ParticleInfo::external(dsdecompx::MemorySpace::CudaManaged, capacity, owned);
        particles.positions(values[0], values[1], values[2])
            .velocities(values[3], values[4], values[5])
            .forces(values[6], values[7], values[8])
            .properties(values[9], values[10], integers[0], integers[1])
            .add_field<double>("feat", feat, capacity, wide)
            .add_field<float>("ffeat", ffeat, capacity, narrow);

        cudaStream_t stream{};
        cuda_check(cudaStreamCreate(&stream));
        const auto sync = [&] { cuda_check(cudaStreamSynchronize(stream)); };

        const auto built = decomp.exchange_halo(particles, dsdecompx::HaloExchangeOptions{true, stream});
        sync();
        const std::size_t ghosts = built.ghost_count;
        std::uint64_t local_ghosts = ghosts, global_ghosts = 0;
        MPI_Allreduce(&local_ghosts, &global_ghosts, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        expect(global_ghosts > 0, "the plan build produced ghosts");

        for (int round = 0; round < 3; ++round) {
            if (round > 0) {   // a reused plan: ghost slots keep their meaning
                const auto refreshed = decomp.exchange_halo(particles, dsdecompx::HaloExchangeOptions{false, stream});
                sync();
                expect(refreshed.ghost_count == ghosts, "reused plan keeps the ghost count");
            }
            auto global_id = [&](std::size_t slot) { return integers[1][slot]; };
            const std::uint64_t seed = 100 + round;

            // 1. forward: ghost slots hold the owners' values (exact copies)
            for (std::size_t slot = 0; slot < owned + ghosts; ++slot) {
                for (std::size_t c = 0; c < wide; ++c) feat[slot * wide + c] = -99.0;
                for (std::size_t c = 0; c < narrow; ++c) ffeat[slot * narrow + c] = -99.0f;
            }
            for (std::size_t k = 0; k < owned; ++k) {
                for (std::size_t c = 0; c < wide; ++c) feat[k * wide + c] = pseudo(seed, global_id(k) * 16 + c);
                for (std::size_t c = 0; c < narrow; ++c)
                    ffeat[k * narrow + c] = static_cast<float>(pseudo(seed + 7, global_id(k) * 16 + c));
            }
            decomp.forward_exchange(particles, {"feat", "ffeat"}, stream);
            sync();
            bool copies = true, owners_kept = true;
            for (std::size_t g = 0; g < ghosts; ++g) {
                const std::size_t slot = owned + g;
                const long long id = global_id(slot);
                for (std::size_t c = 0; c < wide; ++c)
                    copies = copies && feat[slot * wide + c] == pseudo(seed, id * 16 + c);
                for (std::size_t c = 0; c < narrow; ++c)
                    copies = copies && ffeat[slot * narrow + c] == static_cast<float>(pseudo(seed + 7, id * 16 + c));
            }
            for (std::size_t k = 0; k < owned; ++k) {
                for (std::size_t c = 0; c < wide; ++c)
                    owners_kept = owners_kept && feat[k * wide + c] == pseudo(seed, global_id(k) * 16 + c);
            }
            expect(copies, "every ghost slot holds the value of its owner (double and float fields)");
            expect(owners_kept, "the owners' values are untouched");

            // 2. adjoint: <forward(x), y> over the ghosts == <x, reverse(y)> over the owners
            std::vector<double> x(owned * wide), y(ghosts * wide);
            for (std::size_t k = 0; k < owned; ++k)
                for (std::size_t c = 0; c < wide; ++c) x[k * wide + c] = pseudo(seed + 1, global_id(k) * 16 + c);
            for (std::size_t g = 0; g < ghosts; ++g)
                for (std::size_t c = 0; c < wide; ++c) y[g * wide + c] = pseudo(seed + 2, global_id(owned + g) * 31 + c + 17 * rank + g);
            for (std::size_t slot = 0; slot < owned + ghosts; ++slot)
                for (std::size_t c = 0; c < wide; ++c) feat[slot * wide + c] = 0.0;
            for (std::size_t k = 0; k < owned; ++k)
                for (std::size_t c = 0; c < wide; ++c) feat[k * wide + c] = x[k * wide + c];
            decomp.forward_exchange(particles, {"feat"}, stream);
            sync();
            double s1 = 0.0;
            for (std::size_t g = 0; g < ghosts; ++g)
                for (std::size_t c = 0; c < wide; ++c) s1 += feat[(owned + g) * wide + c] * y[g * wide + c];
            for (std::size_t slot = 0; slot < owned; ++slot)
                for (std::size_t c = 0; c < wide; ++c) feat[slot * wide + c] = 0.0;
            for (std::size_t g = 0; g < ghosts; ++g)
                for (std::size_t c = 0; c < wide; ++c) feat[(owned + g) * wide + c] = y[g * wide + c];
            decomp.reverse_exchange(particles, {{"feat", dsdecompx::ReduceOp::Sum}}, stream);
            sync();
            double s2 = 0.0;
            for (std::size_t k = 0; k < owned; ++k)
                for (std::size_t c = 0; c < wide; ++c) s2 += x[k * wide + c] * feat[k * wide + c];
            double sums[2] = {s1, s2}, total[2] = {0.0, 0.0};
            MPI_Allreduce(sums, total, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            expect(std::abs(total[0] - total[1]) <= 1e-10 * (1.0 + std::abs(total[0])),
                   "forward and reverse(Sum) are adjoint to each other");
            if (rank == 0 && round == 0) {
                std::printf("round 0: ghosts on rank 0 = %zu, <fwd(x),y> = %.12g, <x,rev(y)> = %.12g\n", ghosts, total[0], total[1]);
            }
        }

        // An unknown field and a missing stream are errors, not silent no-ops.
        bool threw = false;
        try { decomp.forward_exchange(particles, {"no_such_field"}, stream); } catch (const std::exception&) { threw = true; }
        expect(threw, "an unknown field name is an error");
        threw = false;
        try { decomp.forward_exchange(particles, {"feat"}, nullptr); } catch (const std::exception&) { threw = true; }
        expect(threw, "a missing stream is an error");

        cudaStreamDestroy(stream);
        for (auto& field : values) cudaFree(field);
        for (auto& field : integers) cudaFree(field);
        cudaFree(feat);
        cudaFree(ffeat);
    }
    int all_failures = 0;
    MPI_Allreduce(&failures, &all_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::printf(all_failures == 0 ? "forward_exchange: all checks passed\n" : "forward_exchange: FAILED\n");
    MPI_Finalize();
    return all_failures == 0 ? 0 : 1;
}
