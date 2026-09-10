#include "kernels.hpp"
#include "cuda_error.hpp"

#include <cub/device/device_scan.cuh>
#include <cmath>

namespace dsdecompx::cuda {
namespace {

struct DeviceShift { double value[3]; };
struct DeviceLocalDomain { double lower[3]; double upper[3]; };
struct DeviceDirection { int value[3]; };
struct DeviceDomain {
    double lower[3]; double upper[3]; bool periodic[3];
};

__global__ void pack_kernel(ConstParticleView p, const int* indices,
                            std::size_t count, DeviceShift shift,
                            ParticleRecord* output) {
    const std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const int i = indices[k];
    ParticleRecord record;
    record.x = p.x[i] + shift.value[0]; record.y = p.y[i] + shift.value[1];
    record.z = p.z[i] + shift.value[2];
    record.vx = p.vx[i]; record.vy = p.vy[i]; record.vz = p.vz[i];
    record.fx = 0.0; record.fy = 0.0; record.fz = 0.0;
    record.mass = p.mass[i]; record.charge = p.charge[i];
    record.type = p.type[i]; record.global_id = p.global_id[i];
    output[k] = record;
}

__global__ void pack_reverse_kernel(ParticleView p, std::size_t source_offset,
                                    std::size_t count, ParticleRecord* output) {
    const std::size_t k=blockIdx.x*blockDim.x+threadIdx.x;
    if (k>=count) return;
    const std::size_t i=source_offset+k;
    ParticleRecord record;
    record.fx=p.fx[i]; record.fy=p.fy[i]; record.fz=p.fz[i];
    record.global_id=p.global_id[i];
    output[k]=record;
}

__global__ void mark_halo_kernel(ConstParticleView p, DeviceLocalDomain local,
                                 DeviceDirection direction, double width,
                                 unsigned int* flags) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.count) return;
    bool selected = true;
    const double position[3] = {p.x[i], p.y[i], p.z[i]};
    for (int axis = 0; axis < 3; ++axis) {
        if (direction.value[axis] < 0)
            selected = selected && position[axis] < local.lower[axis] + width;
        else if (direction.value[axis] > 0)
            selected = selected && position[axis] >= local.upper[axis] - width;
    }
    flags[i] = selected ? 1u : 0u;
}

__global__ void scatter_halo_kernel(ConstParticleView p,
                                    const unsigned int* flags,
                                    const unsigned int* offsets,
                                    DeviceShift shift, ParticleRecord* output) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.count || !flags[i]) return;
    ParticleRecord r;
    r.x=p.x[i]+shift.value[0]; r.y=p.y[i]+shift.value[1];
    r.z=p.z[i]+shift.value[2];
    r.vx=p.vx[i]; r.vy=p.vy[i]; r.vz=p.vz[i];
    // Match the host reference semantics: forward ghosts start with zero force.
    r.fx=0.0; r.fy=0.0; r.fz=0.0;
    r.mass=p.mass[i]; r.charge=p.charge[i];
    r.type=p.type[i]; r.global_id=p.global_id[i];
    output[offsets[i]] = r;
}

__global__ void finalize_halo_count_kernel(const unsigned int* flags,
                                           const unsigned int* offsets,
                                           std::size_t count,
                                           unsigned int* output_count) {
    if (threadIdx.x || blockIdx.x) return;
    *output_count = count ? offsets[count-1] + flags[count-1] : 0u;
}

__global__ void mark_destination_kernel(const int* destinations,
                                        std::size_t count, int target,
                                        unsigned int* flags) {
    const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count) flags[i]=destinations[i]==target?1u:0u;
}

__device__ double wrap_coordinate(double value, double lower, double upper,
                                  bool periodic) {
    if (!periodic) return value;
    const double extent=upper-lower;
    value-=lower;
    value-=floor(value/extent)*extent;
    return value+lower;
}

__global__ void scatter_migration_kernel(ConstParticleView p,
                                         DeviceDomain domain,
                                         const unsigned int* flags,
                                         const unsigned int* offsets,
                                         ParticleRecord* output) {
    const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=p.count || !flags[i]) return;
    ParticleRecord r;
    r.x=wrap_coordinate(p.x[i],domain.lower[0],domain.upper[0],domain.periodic[0]);
    r.y=wrap_coordinate(p.y[i],domain.lower[1],domain.upper[1],domain.periodic[1]);
    r.z=wrap_coordinate(p.z[i],domain.lower[2],domain.upper[2],domain.periodic[2]);
    r.vx=p.vx[i]; r.vy=p.vy[i]; r.vz=p.vz[i];
    r.fx=p.fx[i]; r.fy=p.fy[i]; r.fz=p.fz[i];
    r.mass=p.mass[i]; r.charge=p.charge[i];
    r.type=p.type[i]; r.global_id=p.global_id[i];
    output[offsets[i]]=r;
}

} // namespace

void launch_pack_records(const ConstParticleView& particles,
                         const int* indices, std::size_t count,
                         Vec3d periodic_shift, ParticleRecord* output,
                         cudaStream_t stream) {
    if (!count) return;
    const DeviceShift shift{{periodic_shift[0], periodic_shift[1], periodic_shift[2]}};
    constexpr int threads = 256;
    pack_kernel<<<static_cast<unsigned>((count + threads - 1) / threads),
                  threads, 0, stream>>>(particles, indices, count, shift, output);
    detail::check(cudaGetLastError(), "launch pack kernel");
}

void launch_pack_reverse_records(const ParticleView& particles,
                                 std::size_t source_offset,
                                 std::size_t count,
                                 ParticleRecord* output,
                                 cudaStream_t stream) {
    if (!count) return;
    constexpr int threads=256;
    pack_reverse_kernel<<<static_cast<unsigned>((count+threads-1)/threads),
                          threads,0,stream>>>(
        particles,source_offset,count,output);
    detail::check(cudaGetLastError(),"launch reverse halo pack kernel");
}

void launch_pack_halo_neighbor(const ConstParticleView& particles,
                               LocalDomain local_domain, Vec3i direction,
                               Vec3d periodic_shift, double halo_width,
                               unsigned int* flags, unsigned int* offsets,
                               ParticleRecord* output,
                               unsigned int* output_count,
                               void* scan_temporary,
                               std::size_t scan_temporary_bytes,
                               cudaStream_t stream) {
    if (!particles.count) {
        detail::check(cudaMemsetAsync(output_count, 0, sizeof(unsigned int), stream),
                      "clear empty halo count");
        return;
    }
    constexpr int threads = 256;
    const unsigned blocks =
        static_cast<unsigned>((particles.count + threads - 1) / threads);
    const DeviceLocalDomain local{{local_domain.lower[0], local_domain.lower[1],
                                   local_domain.lower[2]},
                                  {local_domain.upper[0], local_domain.upper[1],
                                   local_domain.upper[2]}};
    const DeviceDirection device_direction{{direction[0], direction[1], direction[2]}};
    const DeviceShift shift{{periodic_shift[0], periodic_shift[1], periodic_shift[2]}};
    mark_halo_kernel<<<blocks,threads,0,stream>>>(
        particles, local, device_direction, halo_width, flags);
    detail::check(cudaGetLastError(), "launch halo classification kernel");
    detail::check(cub::DeviceScan::ExclusiveSum(
                      scan_temporary, scan_temporary_bytes,
                      flags, offsets, particles.count, stream),
                  "halo prefix scan");
    scatter_halo_kernel<<<blocks,threads,0,stream>>>(
        particles, flags, offsets, shift, output);
    detail::check(cudaGetLastError(), "launch stable halo packing kernel");
    finalize_halo_count_kernel<<<1,1,0,stream>>>(
        flags, offsets, particles.count, output_count);
    detail::check(cudaGetLastError(), "launch halo count kernel");
}

void launch_pack_migration_destination(
    const ConstParticleView& particles, GlobalDomain domain,
    const int* destinations, int target_rank,
    unsigned int* flags, unsigned int* offsets,
    ParticleRecord* output, unsigned int* output_count,
    void* scan_temporary, std::size_t scan_temporary_bytes,
    cudaStream_t stream) {
    if (!particles.count) {
        detail::check(cudaMemsetAsync(output_count,0,sizeof(unsigned int),stream),
                      "clear empty migration count");
        return;
    }
    constexpr int threads=256;
    const unsigned blocks=static_cast<unsigned>((particles.count+threads-1)/threads);
    mark_destination_kernel<<<blocks,threads,0,stream>>>(
        destinations,particles.count,target_rank,flags);
    detail::check(cudaGetLastError(),"launch migration destination marking");
    detail::check(cub::DeviceScan::ExclusiveSum(
                      scan_temporary,scan_temporary_bytes,
                      flags,offsets,particles.count,stream),
                  "migration destination prefix scan");
    const DeviceDomain device_domain{
        {domain.lower[0],domain.lower[1],domain.lower[2]},
        {domain.upper[0],domain.upper[1],domain.upper[2]},
        {domain.periodic[0],domain.periodic[1],domain.periodic[2]}};
    scatter_migration_kernel<<<blocks,threads,0,stream>>>(
        particles,device_domain,flags,offsets,output);
    detail::check(cudaGetLastError(),"launch migration destination packing");
    finalize_halo_count_kernel<<<1,1,0,stream>>>(
        flags,offsets,particles.count,output_count);
    detail::check(cudaGetLastError(),"launch migration destination count");
}

} // namespace dsdecompx::cuda
