#include "kernels.hpp"
#include "cuda_error.hpp"

namespace dsdecomp::cuda {
namespace {

__device__ double reverse_atomic_add(double* address,double value) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address,value);
#else
    auto* integer=reinterpret_cast<unsigned long long*>(address);
    unsigned long long old=*integer,assumed;
    do {
        assumed=old;
        old=atomicCAS(integer,assumed,
            __double_as_longlong(value+__longlong_as_double(assumed)));
    } while (assumed!=old);
    return __longlong_as_double(old);
#endif
}

__global__ void unpack_kernel(const ParticleRecord* input, std::size_t count,
                              ParticleView p, std::size_t offset) {
    const std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const std::size_t i = offset + k;
    const ParticleRecord r = input[k];
    if (p.x) p.x[i] = r.x; if (p.y) p.y[i] = r.y; if (p.z) p.z[i] = r.z;
    if (p.vx) p.vx[i] = r.vx; if (p.vy) p.vy[i] = r.vy; if (p.vz) p.vz[i] = r.vz;
    if (p.fx) p.fx[i] = r.fx; if (p.fy) p.fy[i] = r.fy; if (p.fz) p.fz[i] = r.fz;
    if (p.mass) p.mass[i] = r.mass; if (p.charge) p.charge[i] = r.charge;
    if (p.type) p.type[i] = r.type;
    if (p.global_id) p.global_id[i] = r.global_id;
}

__global__ void accumulate_reverse_kernel(const ParticleRecord* input,
                                          std::size_t count,
                                          ParticleView p,
                                          std::size_t owned_count) {
    const std::size_t k=blockIdx.x*blockDim.x+threadIdx.x;
    if (k>=count) return;
    const ParticleRecord record=input[k];
    for (std::size_t i=0;i<owned_count;++i) {
        if (p.global_id[i]==record.global_id) {
            reverse_atomic_add(p.fx+i,record.fx);
            reverse_atomic_add(p.fy+i,record.fy);
            reverse_atomic_add(p.fz+i,record.fz);
            break;
        }
    }
}

} // namespace

void launch_unpack_records(const ParticleRecord* records, std::size_t count,
                           ParticleView particles, std::size_t destination_offset,
                           cudaStream_t stream) {
    if (!count) return;
    constexpr int threads = 256;
    unpack_kernel<<<static_cast<unsigned>((count + threads - 1) / threads),
                    threads, 0, stream>>>(records, count, particles,
                                          destination_offset);
    detail::check(cudaGetLastError(), "launch unpack kernel");
}

void launch_accumulate_reverse_records(const ParticleRecord* records,
                                       std::size_t count,
                                       ParticleView particles,
                                       std::size_t owned_count,
                                       cudaStream_t stream) {
    if (!count) return;
    constexpr int threads=256;
    accumulate_reverse_kernel<<<static_cast<unsigned>((count+threads-1)/threads),
                                threads,0,stream>>>(
        records,count,particles,owned_count);
    detail::check(cudaGetLastError(),"launch reverse halo accumulation kernel");
}

} // namespace dsdecomp::cuda
