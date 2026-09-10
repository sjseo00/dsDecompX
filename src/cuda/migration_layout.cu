#include "kernels.hpp"
#include "cuda_error.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cmath>

namespace dsdecompx::cuda {
namespace {

struct DeviceDomain { double lower[3]; double upper[3]; bool periodic[3]; };

__global__ void initialize_layout_kernel(const int* destinations,
                                         std::size_t count,
                                         unsigned int* indices,
                                         unsigned int* rank_counts) {
    const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    indices[i]=static_cast<unsigned int>(i);
    atomicAdd(rank_counts+destinations[i],1u);
}

__device__ double wrap(double value,double lower,double upper,bool periodic) {
    if (!periodic) return value;
    const double extent=upper-lower;
    value-=lower; value-=floor(value/extent)*extent; return value+lower;
}

__global__ void pack_sorted_kernel(ConstParticleView p, DeviceDomain domain,
                                   const unsigned int* sorted_indices,
                                   ParticleRecord* packed) {
    const std::size_t k=blockIdx.x*blockDim.x+threadIdx.x;
    if (k>=p.count) return;
    const std::size_t i=sorted_indices[k];
    ParticleRecord r;
    r.x=wrap(p.x[i],domain.lower[0],domain.upper[0],domain.periodic[0]);
    r.y=wrap(p.y[i],domain.lower[1],domain.upper[1],domain.periodic[1]);
    r.z=wrap(p.z[i],domain.lower[2],domain.upper[2],domain.periodic[2]);
    r.vx=p.vx[i]; r.vy=p.vy[i]; r.vz=p.vz[i];
    r.fx=p.fx[i]; r.fy=p.fy[i]; r.fz=p.fz[i];
    r.mass=p.mass[i]; r.charge=p.charge[i];
    r.type=p.type[i]; r.global_id=p.global_id[i];
    packed[k]=r;
}

} // namespace

std::size_t migration_layout_temporary_bytes(std::size_t particle_count,
                                             std::size_t rank_count) {
    std::size_t sort_bytes=0, scan_bytes=0;
    if (particle_count) {
        cub::DeviceRadixSort::SortPairs(
            nullptr,sort_bytes,static_cast<const int*>(nullptr),
            static_cast<int*>(nullptr),static_cast<const unsigned int*>(nullptr),
            static_cast<unsigned int*>(nullptr),particle_count);
    }
    cub::DeviceScan::ExclusiveSum(
        nullptr,scan_bytes,static_cast<const unsigned int*>(nullptr),
        static_cast<unsigned int*>(nullptr),rank_count);
    return std::max(sort_bytes,scan_bytes);
}

void launch_build_migration_layout(
    const ConstParticleView& particles, GlobalDomain domain,
    const int* destinations, int* sorted_destinations,
    unsigned int* indices, unsigned int* sorted_indices,
    unsigned int* counts, unsigned int* offsets,
    ParticleRecord* packed_records, std::size_t rank_count,
    void* temporary, std::size_t temporary_bytes,
    cudaStream_t stream) {
    detail::check(cudaMemsetAsync(counts,0,rank_count*sizeof(unsigned int),stream),
                  "clear migration rank counts");
    if (!particles.count) {
        detail::check(cudaMemsetAsync(offsets,0,rank_count*sizeof(unsigned int),stream),
                      "clear migration offsets");
        return;
    }
    constexpr int threads=256;
    const unsigned blocks=static_cast<unsigned>((particles.count+threads-1)/threads);
    initialize_layout_kernel<<<blocks,threads,0,stream>>>(
        destinations,particles.count,indices,counts);
    detail::check(cudaGetLastError(),"initialize migration layout");
    detail::check(cub::DeviceRadixSort::SortPairs(
                      temporary,temporary_bytes,destinations,sorted_destinations,
                      indices,sorted_indices,particles.count,0,sizeof(int)*8,stream),
                  "stable migration radix sort");
    detail::check(cub::DeviceScan::ExclusiveSum(
                      temporary,temporary_bytes,counts,offsets,rank_count,stream),
                  "migration rank-offset scan");
    const DeviceDomain device_domain{
        {domain.lower[0],domain.lower[1],domain.lower[2]},
        {domain.upper[0],domain.upper[1],domain.upper[2]},
        {domain.periodic[0],domain.periodic[1],domain.periodic[2]}};
    pack_sorted_kernel<<<blocks,threads,0,stream>>>(
        particles,device_domain,sorted_indices,packed_records);
    detail::check(cudaGetLastError(),"pack sorted migration records");
}

} // namespace dsdecompx::cuda
