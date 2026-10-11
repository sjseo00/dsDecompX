#pragma once

#include "decomposition.hpp"
#include "internal/backend.hpp"
#include "internal/buffer_pool.hpp"

#include <memory>
#include <chrono>
#include <cstddef>
#include <string>
#include <vector>
#if HAS_CUDA
#include "cuda/execution.hpp"
#include "internal/device_fields.hpp"
#endif

namespace dsdecompx {

struct FieldTransferBuffers {
    std::string name;
    FieldType type = FieldType::Float64;
    std::size_t components = 1;
    std::size_t record_bytes = 0;
    std::vector<std::vector<std::byte>> send;
    std::vector<std::vector<std::byte>> receive;
    std::vector<std::byte> result;
};

struct Decomposition::Impl {
    explicit Impl(Config input_config,
                  std::unique_ptr<detail::Backend> input_backend)
        : config(std::move(input_config)),
          backend(std::move(input_backend)) {}

    void initialize_topology();
    void refresh_domain_geometry();
    void validate_owned_view(const ConstParticleView& particles) const;

    Config config;
    std::unique_ptr<detail::Backend> backend;
    Topology topology;

    bool halo_plan_valid = false;
    std::vector<HaloPlanEntry> halo_plan;
    std::vector<std::vector<ParticleRecord>> halo_send_buffers;
    std::vector<std::vector<ParticleRecord>> halo_receive_buffers;
    std::vector<std::vector<ParticleRecord>> reverse_send_buffers;
    std::vector<std::vector<ParticleRecord>> reverse_receive_buffers;
    std::vector<FieldTransferBuffers> halo_fields;
    std::vector<FieldTransferBuffers> migration_fields;

    std::vector<std::vector<ParticleRecord>> migration_send_buffers;
    std::vector<std::vector<ParticleRecord>> migration_receive_buffers;
    std::vector<ParticleRecord> migrated_owned;
    std::vector<std::size_t> migration_destination;
    std::vector<std::size_t> migration_counts;

#if HAS_CUDA
    std::unique_ptr<cuda::ExecutionState> cuda_execution;
    std::vector<unsigned int> device_halo_send_counts;
    std::vector<unsigned int> device_halo_receive_counts;
    // True while the device halo plan (per-neighbor counts and the owned index
    // behind every ghost slot) may be refreshed in place instead of rebuilt.
    // Cleared together with halo_plan_valid by every invalidation; kept apart
    // from it because the host plan lives in different buffers.
    bool device_halo_plan_valid = false;
    // Whether the most recent device halo exchange reused the plan.
    bool device_halo_plan_reused = false;
    // Owned count at the last device halo pack (validates reverse halo).
    std::size_t device_halo_owned_count = 0;
    // Wire layout (record stride + custom fields) of the last device halo /
    // migration, needed to interpret the received records when unpacking.
    detail::DeviceWireLayout device_halo_layout;
    detail::DeviceWireLayout device_migration_layout;
    std::size_t nvshmem_halo_record_bytes = 0;
    std::size_t nvshmem_migration_record_bytes = 0;
    std::vector<detail::DeviceTransfer> device_transfers;
    std::vector<DeviceHaloBuffer> device_halo_results;
    void* nvshmem_halo_receive = nullptr;
    std::uint64_t* nvshmem_halo_signals = nullptr;
    std::size_t nvshmem_halo_slot_capacity = 0;
    std::uint64_t nvshmem_halo_epoch = 0;
    // Symmetric receive buffers of the device reverse exchange. Set 0 carries
    // forces (reverse_halo), set 1 the custom fields (reverse_exchange).
    struct SymmetricReverseBuffers {
        void* receive = nullptr;
        std::uint64_t* signals = nullptr;
        std::size_t slot_capacity = 0;  // records per slot
        std::size_t record_bytes = 0;
        std::uint64_t epoch = 0;
    };
    SymmetricReverseBuffers nvshmem_reverse[2];
    bool device_operation_active = false;
    ExecutionStream device_communication_stream = nullptr;
    DeviceParticleBuffer device_migration_result;
    void* nvshmem_migration_receive = nullptr;
    std::uint64_t* nvshmem_migration_signals = nullptr;
    std::size_t nvshmem_migration_slot_capacity = 0;
    std::uint64_t nvshmem_migration_epoch = 0;
#endif

    bool operation_active = false;
    std::uint64_t operation_generation = 0;
    bool timing_enabled = false;
    bool perf_enabled = false;
    bool neighbor_detail_enabled = false;
    PerfStats perf_stats{};
    std::uint64_t backend_reallocation_baseline = 0;
#if HAS_CUDA
    std::uint64_t cuda_reallocation_baseline = 0;
#endif
    HaloExchangeTimings halo_timings{};
    std::chrono::steady_clock::time_point exchange_start{};

    template<class T>
    void reserve_buffer(std::vector<T>& buffer, std::size_t required) {
        const std::size_t before = buffer.capacity();
        detail::reserve_geometric(buffer, required);
        if (perf_enabled && buffer.capacity() != before) {
            ++perf_stats.buffer_reallocations;
        }
    }
};

} // namespace dsdecompx
