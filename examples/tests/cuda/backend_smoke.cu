#include <dsdecompx.hpp>

#include <cuda_runtime_api.h>
#include <mpi.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace {
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    {
        assert(argc == 2);
        dsdecompx::Config config;
        assert(config.execution == dsdecompx::ExecutionMode::Cuda);
        assert(config.advanced.selection == dsdecompx::BackendSelection::Auto);
        config.advanced.selection = dsdecompx::parse_backend_selection(argv[1]);
        config.execution = dsdecompx::ExecutionMode::Cuda;
        config.advanced.read_environment = false;
        // One GPU per rank on multi-GPU nodes (required by NCCL / NVSHMEM).
        config.advanced.cuda_device = dsdecompx::kCudaDeviceByLocalRank;
        config.domain.upper = {8.0, 8.0, 8.0};
        config.halo.width = 0.25;
        dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);

        {
            auto directly_allocated = dsdecompx::ParticleInfo::allocate(
                dsdecompx::MemorySpace::CudaManaged, 1);
            assert(directly_allocated.x_data() != nullptr);
            assert(directly_allocated.capacity() == 1);
        }

        constexpr std::size_t capacity = 64;
        double* values[11]{};
        int* integers[2]{};
        for (auto& field : values) {
            cuda_check(cudaMallocManaged(reinterpret_cast<void**>(&field),
                                         capacity*sizeof(double)));
            cuda_check(cudaMemset(field, 0, capacity*sizeof(double)));
        }
        for (auto& field : integers) {
            cuda_check(cudaMallocManaged(reinterpret_cast<void**>(&field),
                                         capacity*sizeof(int)));
            cuda_check(cudaMemset(field, 0, capacity*sizeof(int)));
        }
        const auto& local = decomp.topology().local_domain();
        // Several owned particles near the lower corner so multiple ghosts map
        // back onto distinct owners in the reverse halo.
        constexpr std::size_t owned = 4;
        for (std::size_t k=0;k<owned;++k) {
            values[0][k]=local.lower[0]+0.05+0.01*k;
            values[1][k]=local.lower[1]+0.05+0.01*k;
            values[2][k]=local.lower[2]+0.05+0.01*k;
            values[9][k]=1.0; integers[0][k]=0;
            integers[1][k]=decomp.topology().rank()*static_cast<int>(owned)+
                           static_cast<int>(k);
        }

        // Two-component custom field whose value encodes the particle id, so
        // ghosts and migrated particles can be checked after each exchange.
        double* tag = nullptr;
        cuda_check(cudaMallocManaged(reinterpret_cast<void**>(&tag),
                                     capacity*2*sizeof(double)));
        cuda_check(cudaMemset(tag, 0, capacity*2*sizeof(double)));
        for (std::size_t k=0;k<owned;++k) {
            tag[2*k]=static_cast<double>(integers[1][k]);
            tag[2*k+1]=-static_cast<double>(integers[1][k]);
        }

        // Scalar fields for the device reverse exchange: "acc" is summed back
        // onto owners, "peak" takes the maximum over an owner's ghost images.
        double* acc = nullptr;
        double* peak = nullptr;
        cuda_check(cudaMallocManaged(reinterpret_cast<void**>(&acc),
                                     capacity*sizeof(double)));
        cuda_check(cudaMallocManaged(reinterpret_cast<void**>(&peak),
                                     capacity*sizeof(double)));
        for (std::size_t i=0;i<capacity;++i) { acc[i]=0.0; peak[i]=-1.0; }

        auto particles = dsdecompx::ParticleInfo::external(
            dsdecompx::MemorySpace::CudaManaged, capacity, owned);
        particles
            .positions(values[0], values[1], values[2])
            .velocities(values[3], values[4], values[5])
            .forces(values[6], values[7], values[8])
            .properties(values[9], values[10], integers[0], integers[1]);
        particles.add_field_view({"tag", dsdecompx::FieldType::Float64, 2,
                                  capacity, tag});
        particles.add_field_view({"acc", dsdecompx::FieldType::Float64, 1,
                                  capacity, acc});
        particles.add_field_view({"peak", dsdecompx::FieldType::Float64, 1,
                                  capacity, peak});

        cudaStream_t stream{}; cuda_check(cudaStreamCreate(&stream));
        const auto halo = decomp.exchange_halo(
            particles, dsdecompx::HaloExchangeOptions{true, stream});
        cuda_check(cudaStreamSynchronize(stream));
        const std::uint64_t local_ghosts = halo.ghost_count;
        std::uint64_t global_ghosts=0;
        MPI_Allreduce(&local_ghosts,&global_ghosts,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
        assert(global_ghosts > 0);
        assert(owned + local_ghosts <= capacity);
        for (std::size_t i=0;i<owned+local_ghosts;++i) {
            assert(tag[2*i]==static_cast<double>(integers[1][i]));
            assert(tag[2*i+1]==-static_cast<double>(integers[1][i]));
        }

        for (std::size_t i=owned;i<owned+local_ghosts;++i) values[6][i]=1.0;
        decomp.reverse_halo(particles,stream);
        cuda_check(cudaStreamSynchronize(stream));
        double local_returned_force=0.0, global_returned_force=0.0;
        for (std::size_t i=0;i<owned;++i) local_returned_force+=values[6][i];
        MPI_Allreduce(&local_returned_force,&global_returned_force,1,MPI_DOUBLE,
                      MPI_SUM,MPI_COMM_WORLD);
        assert(std::abs(global_returned_force-static_cast<double>(global_ghosts)) < 1e-9);

        // Device reverse exchange of custom fields. Every ghost contributes 1
        // to "acc" and its global id to "peak".
        for (std::size_t i=0;i<owned;++i) { acc[i]=0.0; peak[i]=-1.0; }
        for (std::size_t i=owned;i<owned+local_ghosts;++i) {
            acc[i]=1.0;
            peak[i]=static_cast<double>(integers[1][i]);
        }
        decomp.reverse_exchange(
            particles,
            {{"acc", dsdecompx::ReduceOp::Sum}, {"peak", dsdecompx::ReduceOp::Max}},
            stream);
        cuda_check(cudaStreamSynchronize(stream));
        double local_acc=0.0, global_acc=0.0;
        for (std::size_t i=0;i<owned;++i) {
            local_acc+=acc[i];
            // A ghost image carries its owner's id, so the maximum is either
            // untouched (no ghosts) or exactly the owner's own id.
            assert(peak[i]==-1.0 || peak[i]==static_cast<double>(integers[1][i]));
        }
        MPI_Allreduce(&local_acc,&global_acc,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        assert(std::abs(global_acc-static_cast<double>(global_ghosts)) < 1e-9);

        const auto moved=decomp.migrate(particles,stream);
        cuda_check(cudaStreamSynchronize(stream));
        std::uint64_t local_owned=moved.new_owned_count, global_owned=0;
        MPI_Allreduce(&local_owned,&global_owned,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
        int size=1; MPI_Comm_size(MPI_COMM_WORLD,&size);
        assert(global_owned==static_cast<std::uint64_t>(size)*owned);
        for (std::size_t i=0;i<moved.new_owned_count;++i) {
            assert(tag[2*i]==static_cast<double>(integers[1][i]));
            assert(tag[2*i+1]==-static_cast<double>(integers[1][i]));
        }

        cuda_check(cudaStreamDestroy(stream));
        for (auto* field : integers) cuda_check(cudaFree(field));
        for (auto* field : values) cuda_check(cudaFree(field));
        cuda_check(cudaFree(tag));
        cuda_check(cudaFree(acc));
        cuda_check(cudaFree(peak));
    }
    MPI_Finalize();
}
