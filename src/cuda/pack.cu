#include "kernels.hpp"
#include "cuda_error.hpp"
#include "wire.cuh"

namespace dsdecompx::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMax = static_cast<int>(kMaxHaloNeighbors);

struct DeviceLocalDomain { double lower[3]; double upper[3]; };

struct DeviceHaloSpec {
    int count;
    unsigned int active;
    int direction[kMax][3];
    double shift[kMax][3];
    std::size_t offset[kMax];
};

DeviceHaloSpec to_device(const HaloPackSpec& spec) {
    DeviceHaloSpec device{};
    device.count = static_cast<int>(spec.neighbor_count);
    for (int n = 0; n < device.count; ++n) {
        if (spec.active[n]) device.active |= (1u << n);
        for (int axis = 0; axis < 3; ++axis) {
            device.direction[n][axis] = spec.direction[n][axis];
            device.shift[n][axis] = spec.periodic_shift[n][axis];
        }
        device.offset[n] = spec.offset[n];
    }
    return device;
}

// Bit n is set when the particle lies in the halo slab sent to neighbor n.
__device__ unsigned int halo_mask(const ConstParticleView& p, std::size_t i,
                                  const DeviceLocalDomain& local,
                                  const DeviceHaloSpec& spec, double width) {
    const double position[3] = {p.x[i], p.y[i], p.z[i]};
    bool low[3], high[3];
    for (int axis = 0; axis < 3; ++axis) {
        low[axis] = position[axis] < local.lower[axis] + width;
        high[axis] = position[axis] >= local.upper[axis] - width;
    }
    unsigned int mask = 0;
    for (int n = 0; n < kMax; ++n) {
        if (n >= spec.count || !((spec.active >> n) & 1u)) continue;
        bool selected = true;
        for (int axis = 0; axis < 3; ++axis) {
            const int d = spec.direction[n][axis];
            if (d < 0) selected = selected && low[axis];
            else if (d > 0) selected = selected && high[axis];
        }
        if (selected) mask |= (1u << n);
    }
    return mask;
}

// Pass 1: one thread per particle classifies against all neighbors at once and
// each block reports how many of its particles go to every neighbor.
__global__ void halo_classify_kernel(ConstParticleView p, DeviceLocalDomain local,
                                     DeviceHaloSpec spec, double width,
                                     unsigned int* masks,
                                     unsigned int* block_counts) {
    __shared__ unsigned int warp_counts[kWarps][kMax];
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned int mask = 0;
    if (i < p.count) {
        mask = halo_mask(p, i, local, spec, width);
        masks[i] = mask;
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    // Every lane must reach the ballot, including those past the last particle.
#pragma unroll
    for (int n = 0; n < kMax; ++n) {
        const unsigned int ballot = __ballot_sync(0xffffffffu, (mask >> n) & 1u);
        if (lane == 0) warp_counts[warp][n] = __popc(ballot);
    }
    __syncthreads();
    if (threadIdx.x < kMax) {
        unsigned int sum = 0;
        for (int w = 0; w < kWarps; ++w) sum += warp_counts[w][threadIdx.x];
        block_counts[blockIdx.x * kMax + threadIdx.x] = sum;
    }
}

// Pass 2: one thread block per neighbor turns block counts into exclusive
// block offsets and reports the neighbor total.
__global__ void halo_scan_blocks_kernel(unsigned int* block_counts,
                                        unsigned int num_blocks,
                                        unsigned int* totals) {
    __shared__ unsigned int warp_sums[kWarps];
    __shared__ unsigned int carry;
    const unsigned int n = blockIdx.x;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    for (unsigned int base = 0; base < num_blocks; base += blockDim.x) {
        const unsigned int b = base + threadIdx.x;
        const unsigned int value = b < num_blocks ? block_counts[b * kMax + n] : 0u;
        unsigned int inclusive = value;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const unsigned int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
            if (lane >= static_cast<unsigned int>(offset)) inclusive += other;
        }
        if (lane == 31) warp_sums[warp] = inclusive;
        __syncthreads();
        unsigned int warp_base = 0, tile_total = 0;
        for (unsigned int w = 0; w < kWarps; ++w) {
            if (w < warp) warp_base += warp_sums[w];
            tile_total += warp_sums[w];
        }
        const unsigned int running = carry;
        if (b < num_blocks)
            block_counts[b * kMax + n] = running + warp_base + inclusive - value;
        __syncthreads();
        if (threadIdx.x == 0) carry = running + tile_total;
        __syncthreads();
    }
    if (threadIdx.x == 0) totals[n] = carry;
}

// Fills one forward halo wire record from owned particle `i`. Shared by the
// scatter pass (new plan) and the gather pass (reused plan) so both produce
// byte-identical records.
__device__ inline void store_halo_record(const ConstParticleView& p, std::size_t i,
                                         const double* shift,
                                         const DeviceWireFields& fields,
                                         unsigned char* record) {
    ParticleRecord r;
    r.x = p.x[i] + shift[0]; r.y = p.y[i] + shift[1]; r.z = p.z[i] + shift[2];
    r.vx = p.vx[i]; r.vy = p.vy[i]; r.vz = p.vz[i];
    // Forward ghosts start with zero force, matching the host path.
    r.fx = 0.0; r.fy = 0.0; r.fz = 0.0;
    r.mass = p.mass[i]; r.charge = p.charge[i];
    r.type = p.type[i]; r.global_id = p.global_id[i];
    *reinterpret_cast<ParticleRecord*>(record) = r;
    store_wire_fields(fields, record, i);
}

// Pass 3: stable scatter. Position inside a neighbor's block is
// block offset + earlier warps + earlier lanes, so ghosts keep owned order.
__global__ void halo_scatter_kernel(ConstParticleView p, DeviceHaloSpec spec,
                                    const unsigned int* masks,
                                    const unsigned int* block_offsets,
                                    unsigned char* output,
                                    unsigned int* source_indices,
                                    DeviceWireFields fields) {
    __shared__ unsigned int warp_counts[kWarps][kMax];
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int mask = i < p.count ? masks[i] : 0u;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const unsigned int lane_below = (1u << lane) - 1u;
    unsigned int rank_in_warp[kMax];
#pragma unroll
    for (int n = 0; n < kMax; ++n) {
        const unsigned int ballot = __ballot_sync(0xffffffffu, (mask >> n) & 1u);
        rank_in_warp[n] = __popc(ballot & lane_below);
        if (lane == 0) warp_counts[warp][n] = __popc(ballot);
    }
    __syncthreads();
    if (i >= p.count || !mask) return;
    for (int n = 0; n < kMax; ++n) {
        if (!((mask >> n) & 1u)) continue;
        unsigned int earlier_warps = 0;
        for (int w = 0; w < warp; ++w) earlier_warps += warp_counts[w][n];
        const std::size_t slot = spec.offset[n] + block_offsets[blockIdx.x * kMax + n] +
                                 earlier_warps + rank_in_warp[n];
        store_halo_record(p, i, spec.shift[n], fields,
                          output + slot * fields.stride);
        // Remember the owner of each ghost so the reverse halo can map forces
        // back by index instead of searching global IDs.
        source_indices[slot] = static_cast<unsigned int>(i);
    }
}

struct DeviceGatherSpec {
    int count;
    double shift[kMax][3];
    std::size_t end[kMax];  // one past the last slot of each neighbor block
};

// Reused-plan refresh: slot k re-reads owned particle source_indices[k], so the
// ghost list (membership and order) stays exactly as the plan fixed it.
__global__ void halo_gather_kernel(ConstParticleView p, DeviceGatherSpec spec,
                                   std::size_t total,
                                   const unsigned int* source_indices,
                                   unsigned char* output,
                                   DeviceWireFields fields) {
    const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= total) return;
    // Blocks are contiguous and may be empty; the first end beyond `slot` owns it.
    int n = 0;
    while (n + 1 < spec.count && slot >= spec.end[n]) ++n;
    store_halo_record(p, source_indices[slot], spec.shift[n], fields,
                      output + slot * fields.stride);
}

// Same slot walk as halo_gather_kernel, but only the position triple is written.
__global__ void halo_gather_positions_kernel(ConstParticleView p, DeviceGatherSpec spec,
                                             std::size_t total,
                                             const unsigned int* source_indices,
                                             double* output) {
    const std::size_t slot = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= total) return;
    int n = 0;
    while (n + 1 < spec.count && slot >= spec.end[n]) ++n;
    const std::size_t i = source_indices[slot];
    double* record = output + slot * 3;
    record[0] = p.x[i] + spec.shift[n][0];
    record[1] = p.y[i] + spec.shift[n][1];
    record[2] = p.z[i] + spec.shift[n][2];
}

__global__ void pack_reverse_kernel(ParticleView p, std::size_t source_offset,
                                    std::size_t count, ParticleRecord* output) {
    const std::size_t k=blockIdx.x*blockDim.x+threadIdx.x;
    if (k>=count) return;
    const std::size_t i=source_offset+k;
    ParticleRecord record;
    record.fx=p.fx[i]; record.fy=p.fy[i]; record.fz=p.fz[i];
    output[k]=record;
}

} // namespace

std::size_t halo_block_count(std::size_t particle_count) {
    return (particle_count + kThreads - 1) / kThreads;
}

void launch_halo_classify(const ConstParticleView& particles,
                          LocalDomain local_domain, double halo_width,
                          const HaloPackSpec& spec, unsigned int* masks,
                          unsigned int* block_counts, unsigned int* totals,
                          cudaStream_t stream) {
    if (!particles.count || !spec.neighbor_count) return;
    const unsigned blocks = static_cast<unsigned>(halo_block_count(particles.count));
    const DeviceLocalDomain local{{local_domain.lower[0], local_domain.lower[1],
                                   local_domain.lower[2]},
                                  {local_domain.upper[0], local_domain.upper[1],
                                   local_domain.upper[2]}};
    halo_classify_kernel<<<blocks, kThreads, 0, stream>>>(
        particles, local, to_device(spec), halo_width, masks, block_counts);
    detail::check(cudaGetLastError(), "launch halo classification kernel");
    halo_scan_blocks_kernel<<<static_cast<unsigned>(spec.neighbor_count), kThreads,
                              0, stream>>>(block_counts, blocks, totals);
    detail::check(cudaGetLastError(), "launch halo block scan kernel");
}

void launch_halo_scatter(const ConstParticleView& particles,
                         const HaloPackSpec& spec, const unsigned int* masks,
                         const unsigned int* block_offsets,
                         unsigned char* output, unsigned int* source_indices,
                         const WireFields& fields, cudaStream_t stream) {
    if (!particles.count || !spec.neighbor_count) return;
    halo_scatter_kernel<<<static_cast<unsigned>(halo_block_count(particles.count)),
                          kThreads, 0, stream>>>(
        particles, to_device(spec), masks, block_offsets, output, source_indices,
        to_device_fields(fields));
    detail::check(cudaGetLastError(), "launch halo scatter kernel");
}

void launch_halo_gather(const ConstParticleView& particles,
                        const HaloPackSpec& spec, const unsigned int* counts,
                        const unsigned int* source_indices,
                        unsigned char* output, const WireFields& fields,
                        cudaStream_t stream) {
    if (!spec.neighbor_count) return;
    DeviceGatherSpec device{};
    device.count = static_cast<int>(spec.neighbor_count);
    std::size_t total = 0;
    for (int n = 0; n < device.count; ++n) {
        for (int axis = 0; axis < 3; ++axis)
            device.shift[n][axis] = spec.periodic_shift[n][axis];
        total = spec.offset[n] + counts[n];
        device.end[n] = total;
    }
    if (!total) return;
    halo_gather_kernel<<<static_cast<unsigned>((total + kThreads - 1) / kThreads),
                         kThreads, 0, stream>>>(
        particles, device, total, source_indices, output, to_device_fields(fields));
    detail::check(cudaGetLastError(), "launch halo gather kernel");
}

void launch_halo_gather_positions(const ConstParticleView& particles,
                                  const HaloPackSpec& spec, const unsigned int* counts,
                                  const unsigned int* source_indices,
                                  unsigned char* output, cudaStream_t stream) {
    if (!spec.neighbor_count) return;
    DeviceGatherSpec device{};
    device.count = static_cast<int>(spec.neighbor_count);
    std::size_t total = 0;
    for (int n = 0; n < device.count; ++n) {
        for (int axis = 0; axis < 3; ++axis)
            device.shift[n][axis] = spec.periodic_shift[n][axis];
        total = spec.offset[n] + counts[n];
        device.end[n] = total;
    }
    if (!total) return;
    halo_gather_positions_kernel<<<static_cast<unsigned>((total + kThreads - 1) / kThreads),
                                   kThreads, 0, stream>>>(
        particles, device, total, source_indices, reinterpret_cast<double*>(output));
    detail::check(cudaGetLastError(), "launch halo position gather kernel");
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

} // namespace dsdecompx::cuda
