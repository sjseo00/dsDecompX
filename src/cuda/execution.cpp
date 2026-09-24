#include "execution.hpp"
#include "kernels.hpp"

#include "field.hpp"

#include <algorithm>

namespace dsdecompx::cuda {

ExecutionState::ExecutionState()
    : pack_ready(cudaEventDisableTiming),
      receive_ready(cudaEventDisableTiming),
      unpack_ready(cudaEventDisableTiming) {}

void ExecutionState::reserve_halo(std::size_t send_records,
                                  std::size_t receive_records,
                                  std::size_t neighbor_count) {
    device_send.reserve(send_records * sizeof(ParticleRecord));
    device_receive.reserve(receive_records * sizeof(ParticleRecord));
    device_counts.reserve(neighbor_count * sizeof(unsigned int));
    device_offsets.reserve((neighbor_count + 1) * sizeof(unsigned int));
    host_counts.reserve(neighbor_count * sizeof(unsigned int));
}

void ExecutionState::reserve_migration(std::size_t particle_count,
                                       std::size_t rank_count) {
    migration_destinations.reserve(particle_count * sizeof(int));
    migration_flags.reserve(particle_count * sizeof(unsigned int));
    migration_offsets.reserve(particle_count * sizeof(unsigned int));
    migration_compacted.reserve(particle_count * sizeof(ParticleRecord));
    device_counts.reserve(rank_count * sizeof(unsigned int));
    device_offsets.reserve((rank_count + 1) * sizeof(unsigned int));
    const std::size_t scan_bytes = prefix_scan_temporary_bytes(
        std::max(particle_count, rank_count));
    scan_temporary.reserve(scan_bytes);
}

std::uint64_t ExecutionState::reallocations() const noexcept {
    return device_send.reallocations()+device_receive.reallocations()+
        device_counts.reallocations()+device_offsets.reallocations()+
        device_indices.reallocations()+halo_flags.reallocations()+
        halo_offsets.reallocations()+scan_temporary.reallocations()+
        migration_destinations.reallocations()+
        migration_sorted_destinations.reallocations()+
        migration_indices.reallocations()+migration_sorted_indices.reallocations()+
        migration_flags.reallocations()+migration_offsets.reallocations()+
        migration_compacted.reallocations()+migration_temporary.reallocations()+
        domain_cuts.reallocations()+
        host_counts.reallocations()+host_staging.reallocations();
}

std::size_t ExecutionState::communication_capacity() const noexcept {
    return device_send.capacity()+device_receive.capacity()+
        device_counts.capacity()+device_offsets.capacity()+device_indices.capacity()+
        halo_flags.capacity()+halo_offsets.capacity()+scan_temporary.capacity()+
        migration_destinations.capacity()+migration_sorted_destinations.capacity()+
        migration_indices.capacity()+migration_sorted_indices.capacity()+
        migration_flags.capacity()+migration_offsets.capacity()+
        migration_compacted.capacity()+migration_temporary.capacity()+
        domain_cuts.capacity()+
        host_counts.capacity()+host_staging.capacity();
}

} // namespace dsdecompx::cuda
