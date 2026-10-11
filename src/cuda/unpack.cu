#include "kernels.hpp"
#include "cuda_error.hpp"
#include "wire.cuh"

namespace dsdecompx::cuda {
namespace {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
#error "dsDecompX requires compute capability 6.0 or newer (double atomicAdd); set CMAKE_CUDA_ARCHITECTURES accordingly"
#endif

__device__ double reverse_atomic_add(double* address,double value) {
    return atomicAdd(address,value);
}

__global__ void unpack_kernel(const unsigned char* input, std::size_t count,
                              ParticleView p, std::size_t offset,
                              DeviceWireFields fields) {
    const std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const std::size_t i = offset + k;
    const unsigned char* record = input + k * fields.stride;
    const ParticleRecord r = *reinterpret_cast<const ParticleRecord*>(record);
    load_wire_fields(fields, record, i);
    if (p.x) p.x[i] = r.x; if (p.y) p.y[i] = r.y; if (p.z) p.z[i] = r.z;
    if (p.vx) p.vx[i] = r.vx; if (p.vy) p.vy[i] = r.vy; if (p.vz) p.vz[i] = r.vz;
    if (p.fx) p.fx[i] = r.fx; if (p.fy) p.fy[i] = r.fy; if (p.fz) p.fz[i] = r.fz;
    if (p.mass) p.mass[i] = r.mass; if (p.charge) p.charge[i] = r.charge;
    if (p.type) p.type[i] = r.type;
    if (p.global_id) p.global_id[i] = r.global_id;
}

__global__ void unpack_positions_kernel(const double* input, std::size_t count,
                                        ParticleView p, std::size_t offset) {
    const std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const std::size_t i = offset + k;
    const double* r = input + k * 3;
    if (p.x) p.x[i] = r[0]; if (p.y) p.y[i] = r[1]; if (p.z) p.z[i] = r[2];
    // Forward ghosts start with zero force, as in a full exchange.
    if (p.fx) p.fx[i] = 0.0; if (p.fy) p.fy[i] = 0.0; if (p.fz) p.fz[i] = 0.0;
}

// Record k came back from the neighbor it was originally sent to, in the same
// order it was packed, so source_indices[k] is the owned particle it maps to.
__global__ void accumulate_reverse_kernel(const ParticleRecord* input,
                                          std::size_t count,
                                          const unsigned int* source_indices,
                                          ParticleView p) {
    const std::size_t k=blockIdx.x*blockDim.x+threadIdx.x;
    if (k>=count) return;
    const ParticleRecord record=input[k];
    const std::size_t i=source_indices[k];
    reverse_atomic_add(p.fx+i,record.fx);
    reverse_atomic_add(p.fy+i,record.fy);
    reverse_atomic_add(p.fz+i,record.fz);
}

} // namespace

void launch_unpack_records(const unsigned char* records, std::size_t count,
                           ParticleView particles, std::size_t destination_offset,
                           const WireFields& fields, cudaStream_t stream) {
    if (!count) return;
    constexpr int threads = 256;
    unpack_kernel<<<static_cast<unsigned>((count + threads - 1) / threads),
                    threads, 0, stream>>>(records, count, particles,
                                          destination_offset,
                                          to_device_fields(fields));
    detail::check(cudaGetLastError(), "launch unpack kernel");
}

void launch_unpack_positions(const double* positions, std::size_t count,
                             ParticleView particles, std::size_t destination_offset,
                             cudaStream_t stream) {
    if (!count) return;
    constexpr int threads = 256;
    unpack_positions_kernel<<<static_cast<unsigned>((count + threads - 1) / threads),
                              threads, 0, stream>>>(positions, count, particles,
                                                    destination_offset);
    detail::check(cudaGetLastError(), "launch position unpack kernel");
}

void launch_accumulate_reverse_records(const ParticleRecord* records,
                                       std::size_t count,
                                       const unsigned int* source_indices,
                                       ParticleView particles,
                                       cudaStream_t stream) {
    if (!count) return;
    constexpr int threads=256;
    accumulate_reverse_kernel<<<static_cast<unsigned>((count+threads-1)/threads),
                                threads,0,stream>>>(
        records,count,source_indices,particles);
    detail::check(cudaGetLastError(),"launch reverse halo accumulation kernel");
}

} // namespace dsdecompx::cuda
