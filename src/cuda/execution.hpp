#pragma once

#include "cuda_buffer.hpp"
#include "cuda_event.hpp"

#include "field.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>

namespace dsdecomp::cuda {

class ExecutionState {
public:
    ExecutionState();

    void reserve_halo(std::size_t send_records,
                      std::size_t receive_records,
                      std::size_t neighbor_count);
    void reserve_migration(std::size_t particle_count,
                           std::size_t rank_count);
    std::uint64_t reallocations() const noexcept;
    std::size_t communication_capacity() const noexcept;

    DeviceBuffer device_send;
    DeviceBuffer device_receive;
    DeviceBuffer device_counts;
    DeviceBuffer device_offsets;
    DeviceBuffer device_indices;
    DeviceBuffer halo_flags;
    DeviceBuffer halo_offsets;
    DeviceBuffer scan_temporary;
    DeviceBuffer migration_destinations;
    DeviceBuffer migration_sorted_destinations;
    DeviceBuffer migration_indices;
    DeviceBuffer migration_sorted_indices;
    DeviceBuffer migration_flags;
    DeviceBuffer migration_offsets;
    DeviceBuffer migration_compacted;
    DeviceBuffer migration_temporary;
    PinnedBuffer host_counts;
    PinnedBuffer host_staging;

    Event pack_ready;
    Event receive_ready;
    Event unpack_ready;

    bool operation_active = false;
    bool reuse_event_recorded = false;
};

} // namespace dsdecomp::cuda
