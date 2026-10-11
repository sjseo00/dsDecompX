#pragma once

#include "domain.hpp"
#include "field.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>

namespace dsdecompx::cuda {

constexpr std::size_t kMaxHaloNeighbors = 26;
constexpr std::size_t kMaxDeviceFields = 16;

// Layout of one wire record: the fixed ParticleRecord followed by the bytes of
// every application-defined field, `stride` bytes in total (multiple of 8).
// Without custom fields stride == sizeof(ParticleRecord). `data` is the device
// storage of each field: the source when packing, the destination when
// unpacking.
struct WireFields {
    std::size_t count = 0;
    std::size_t stride = sizeof(ParticleRecord);
    std::size_t bytes[kMaxDeviceFields]{};   // per-particle bytes of the field
    std::size_t offset[kMaxDeviceFields]{};  // byte offset inside the record
    void* data[kMaxDeviceFields]{};
};

// Describes every halo destination of one exchange so all neighbors are packed
// in a single pass over the owned particles.
struct HaloPackSpec {
    std::size_t neighbor_count = 0;
    Vec3i direction[kMaxHaloNeighbors]{};
    Vec3d periodic_shift[kMaxHaloNeighbors]{};
    bool active[kMaxHaloNeighbors]{};
    // Element offset of each neighbor's block in the packed send buffer.
    std::size_t offset[kMaxHaloNeighbors]{};
};

// Thread blocks used by the halo kernels for this many owned particles.
std::size_t halo_block_count(std::size_t particle_count);

// Custom fields returned from ghosts to their owners by a reverse exchange.
// A reverse record holds the selected fields back to back (each aligned to its
// element size), `stride` bytes in total. `data` is the field's device storage.
struct ReverseFields {
    std::size_t count = 0;
    std::size_t stride = 0;
    FieldType type[kMaxDeviceFields]{};
    ReduceOp op[kMaxDeviceFields]{};
    std::size_t components[kMaxDeviceFields]{};
    std::size_t offset[kMaxDeviceFields]{};
    void* data[kMaxDeviceFields]{};
};

// Copies ghost fields [ghost_offset, ghost_offset + count) into reverse records.
void launch_pack_reverse_fields(const ReverseFields& fields,
                                std::size_t ghost_offset, std::size_t count,
                                unsigned char* output, cudaStream_t stream);

// Reduces record k into owned particle sources[k] with each field's operation
// (Sum, Min, Max via atomics; Replace stores, last writer wins).
void launch_reduce_reverse_fields(const ReverseFields& fields,
                                  const unsigned char* records, std::size_t count,
                                  const unsigned int* sources, cudaStream_t stream);

// Pass 1+2: classifies every owned particle against all active neighbors and
// writes per-neighbor totals to `totals`. `masks` needs one uint per particle
// and `block_counts` halo_block_count(n) * kMaxHaloNeighbors uints; the latter
// is turned into exclusive block offsets for launch_halo_scatter.
void launch_halo_classify(const ConstParticleView& particles,
                          LocalDomain local_domain,
                          double halo_width,
                          const HaloPackSpec& spec,
                          unsigned int* masks,
                          unsigned int* block_counts,
                          unsigned int* totals,
                          cudaStream_t stream);

// Pass 3: writes each neighbor's records, in owned order, at spec.offset[n].
// source_indices[slot] receives the owned index of the record at that slot.
void launch_halo_scatter(const ConstParticleView& particles,
                         const HaloPackSpec& spec,
                         const unsigned int* masks,
                         const unsigned int* block_offsets,
                         unsigned char* output,
                         unsigned int* source_indices,
                         const WireFields& fields,
                         cudaStream_t stream);

// Refreshes the records of an existing halo plan without reclassifying.
// counts[n] records sit in neighbor n's block at spec.offset[n];
// source_indices (filled by launch_halo_scatter) names the owned particle behind
// every slot. Output is identical to what launch_halo_scatter would write for
// the same ghosts, so the receive side is unchanged.
void launch_halo_gather(const ConstParticleView& particles,
                        const HaloPackSpec& spec,
                        const unsigned int* counts,
                        const unsigned int* source_indices,
                        unsigned char* output,
                        const WireFields& fields,
                        cudaStream_t stream);

// Unpacks `count` wire records (fields.stride bytes apart) into particles
// starting at destination_offset, including any custom fields.
void launch_unpack_records(const unsigned char* records,
                           std::size_t count,
                           ParticleView particles,
                           std::size_t destination_offset,
                           const WireFields& fields,
                           cudaStream_t stream);

void launch_pack_reverse_records(const ParticleView& particles,
                                 std::size_t source_offset,
                                 std::size_t count,
                                 ParticleRecord* output,
                                 cudaStream_t stream);

// Adds records[k] forces onto owned particle source_indices[k].
void launch_accumulate_reverse_records(const ParticleRecord* records,
                                       std::size_t count,
                                       const unsigned int* source_indices,
                                       ParticleView particles,
                                       cudaStream_t stream);

void launch_classify_destinations(const ConstParticleView& particles,
                                  GlobalDomain domain,
                                  Vec3i process_grid,
                                  const double* axis_cuts,
                                  Vec3i cut_offsets,
                                  int* destinations,
                                  cudaStream_t stream);

std::size_t migration_layout_temporary_bytes(std::size_t particle_count,
                                             std::size_t rank_count);
void launch_build_migration_layout(
    const ConstParticleView& particles,
    GlobalDomain domain,
    const int* destinations,
    int* sorted_destinations,
    unsigned int* indices,
    unsigned int* sorted_indices,
    unsigned int* counts,
    unsigned int* offsets,
    unsigned char* packed_records,
    const WireFields& fields,
    std::size_t rank_count,
    void* temporary,
    std::size_t temporary_bytes,
    cudaStream_t stream);

} // namespace dsdecompx::cuda
