#include "execution.hpp"

namespace dsdecompx::cuda {

ExecutionState::ExecutionState()
    : pack_ready(cudaEventDisableTiming),
      unpack_ready(cudaEventDisableTiming) {}

std::uint64_t ExecutionState::reallocations() const noexcept {
    return device_send.reallocations()+device_receive.reallocations()+
        device_counts.reallocations()+device_offsets.reallocations()+
        reverse_send.reallocations()+reverse_receive.reallocations()+
        halo_source_indices.reallocations()+halo_masks.reallocations()+
        halo_block_offsets.reallocations()+scan_temporary.reallocations()+
        migration_destinations.reallocations()+
        migration_sorted_destinations.reallocations()+
        migration_indices.reallocations()+migration_sorted_indices.reallocations()+
        migration_temporary.reallocations()+
        domain_cuts.reallocations()+host_counts.reallocations();
}

std::size_t ExecutionState::communication_capacity() const noexcept {
    return device_send.capacity()+device_receive.capacity()+
        device_counts.capacity()+device_offsets.capacity()+
        reverse_send.capacity()+reverse_receive.capacity()+
        halo_source_indices.capacity()+halo_masks.capacity()+
        halo_block_offsets.capacity()+scan_temporary.capacity()+
        migration_destinations.capacity()+migration_sorted_destinations.capacity()+
        migration_indices.capacity()+migration_sorted_indices.capacity()+
        migration_temporary.capacity()+
        domain_cuts.capacity()+host_counts.capacity();
}

} // namespace dsdecompx::cuda
