#pragma once

#include "domain.hpp"
#include "field.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>

namespace dsdecompx::cuda {

void launch_pack_records(const ConstParticleView& particles,
                         const int* indices,
                         std::size_t count,
                         Vec3d periodic_shift,
                         ParticleRecord* output,
                         cudaStream_t stream);

void launch_pack_halo_neighbor(const ConstParticleView& particles,
                               LocalDomain local_domain,
                               Vec3i direction,
                               Vec3d periodic_shift,
                               double halo_width,
                               unsigned int* flags,
                               unsigned int* offsets,
                               ParticleRecord* output,
                               unsigned int* output_count,
                               void* scan_temporary,
                               std::size_t scan_temporary_bytes,
                               cudaStream_t stream);

void launch_pack_migration_destination(
    const ConstParticleView& particles,
    GlobalDomain domain,
    const int* destinations,
    int target_rank,
    unsigned int* flags,
    unsigned int* offsets,
    ParticleRecord* output,
    unsigned int* output_count,
    void* scan_temporary,
    std::size_t scan_temporary_bytes,
    cudaStream_t stream);

void launch_unpack_records(const ParticleRecord* records,
                           std::size_t count,
                           ParticleView particles,
                           std::size_t destination_offset,
                           cudaStream_t stream);

void launch_pack_reverse_records(const ParticleView& particles,
                                 std::size_t source_offset,
                                 std::size_t count,
                                 ParticleRecord* output,
                                 cudaStream_t stream);

void launch_accumulate_reverse_records(const ParticleRecord* records,
                                       std::size_t count,
                                       ParticleView particles,
                                       std::size_t owned_count,
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
    ParticleRecord* packed_records,
    std::size_t rank_count,
    void* temporary,
    std::size_t temporary_bytes,
    cudaStream_t stream);

void launch_compact_records(const ConstParticleView& particles,
                            const int* destinations,
                            int local_rank,
                            ParticleRecord* retained,
                            unsigned int* flags,
                            unsigned int* offsets,
                            unsigned int* retained_count,
                            void* scan_temporary,
                            std::size_t scan_temporary_bytes,
                            cudaStream_t stream);

std::size_t prefix_scan_temporary_bytes(std::size_t count);
void launch_exclusive_scan(const unsigned int* counts,
                           unsigned int* offsets,
                           std::size_t count,
                           void* temporary,
                           std::size_t temporary_bytes,
                           cudaStream_t stream);

} // namespace dsdecompx::cuda
