#pragma once

#include "cuda_buffer.hpp"
#include "cuda_event.hpp"

#include "field.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>

namespace dsdecompx::cuda {

class ExecutionState {
public:
    ExecutionState();

    std::uint64_t reallocations() const noexcept;
    std::size_t communication_capacity() const noexcept;

    // Forward halo / migration payloads.
    DeviceBuffer device_send;
    DeviceBuffer device_receive;
    DeviceBuffer device_counts;
    DeviceBuffer device_offsets;
    // Reverse halo payloads. Kept apart from the forward buffers so the ghost
    // records returned by received_device_halos() stay valid across a reverse.
    DeviceBuffer reverse_send;
    DeviceBuffer reverse_receive;
    // Forward field payloads (Decomposition::forward_exchange).
    DeviceBuffer field_send;
    DeviceBuffer field_receive;
    // Owned-particle index of every forward halo record, laid out like the
    // packed send buffer (neighbor blocks back to back).
    DeviceBuffer halo_source_indices;
    DeviceBuffer halo_masks;          // per-particle neighbor bitmask
    DeviceBuffer halo_block_offsets;  // per-block, per-neighbor pack offsets
    DeviceBuffer scan_temporary;
    DeviceBuffer migration_destinations;
    DeviceBuffer migration_sorted_destinations;
    DeviceBuffer migration_indices;
    DeviceBuffer migration_sorted_indices;
    DeviceBuffer migration_temporary;
    DeviceBuffer domain_cuts;
    PinnedBuffer host_counts;

    Event pack_ready;
    Event unpack_ready;

    bool operation_active = false;
    bool reuse_event_recorded = false;
};

} // namespace dsdecompx::cuda
