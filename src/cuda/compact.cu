#include "kernels.hpp"
#include "cuda_error.hpp"

#include <cub/device/device_scan.cuh>

namespace dsdecomp::cuda {
namespace {

__global__ void mark_retained_kernel(const int* destinations,
                                     std::size_t count, int local_rank,
                                     unsigned int* flags) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) flags[i] = destinations[i] == local_rank ? 1u : 0u;
}

__global__ void scatter_retained_kernel(ConstParticleView p,
                                        const unsigned int* flags,
                                        const unsigned int* offsets,
                                        ParticleRecord* retained) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.count || !flags[i]) return;
    ParticleRecord r;
    r.x=p.x[i]; r.y=p.y[i]; r.z=p.z[i];
    r.vx=p.vx[i]; r.vy=p.vy[i]; r.vz=p.vz[i];
    r.fx=p.fx[i]; r.fy=p.fy[i]; r.fz=p.fz[i];
    r.mass=p.mass[i]; r.charge=p.charge[i];
    r.type=p.type[i]; r.global_id=p.global_id[i];
    retained[offsets[i]] = r;
}

__global__ void finalize_count_kernel(const unsigned int* flags,
                                      const unsigned int* offsets,
                                      std::size_t count,
                                      unsigned int* retained_count) {
    if (threadIdx.x || blockIdx.x) return;
    *retained_count = count ? offsets[count - 1] + flags[count - 1] : 0u;
}

} // namespace

void launch_compact_records(const ConstParticleView& particles,
                            const int* destinations, int local_rank,
                            ParticleRecord* retained,
                            unsigned int* flags, unsigned int* offsets,
                            unsigned int* retained_count,
                            void* scan_temporary,
                            std::size_t scan_temporary_bytes,
                            cudaStream_t stream) {
    if (!particles.count) {
        detail::check(cudaMemsetAsync(retained_count, 0, sizeof(unsigned int), stream),
                      "clear empty retained count");
        return;
    }
    constexpr int threads = 256;
    const unsigned blocks =
        static_cast<unsigned>((particles.count + threads - 1) / threads);
    mark_retained_kernel<<<blocks,threads,0,stream>>>(
        destinations, particles.count, local_rank, flags);
    detail::check(cudaGetLastError(), "launch retained marking kernel");
    detail::check(cub::DeviceScan::ExclusiveSum(
                      scan_temporary, scan_temporary_bytes,
                      flags, offsets, particles.count, stream),
                  "stable compaction prefix scan");
    scatter_retained_kernel<<<blocks,threads,0,stream>>>(
        particles, flags, offsets, retained);
    detail::check(cudaGetLastError(), "launch stable compact scatter kernel");
    finalize_count_kernel<<<1,1,0,stream>>>(flags, offsets, particles.count,
                                            retained_count);
    detail::check(cudaGetLastError(), "launch compact count kernel");
}

} // namespace dsdecomp::cuda
