#pragma once

#include "dsdecomp.hpp"
#include "internal/backend.hpp"
#include "internal/buffer_pool.hpp"

#include <memory>
#include <chrono>
#include <cstddef>
#include <string>
#include <vector>
#if DSDECOMP_HAS_CUDA
#include "cuda/execution.hpp"
#endif

namespace dsdecomp {

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

#if DSDECOMP_HAS_CUDA
    std::unique_ptr<cuda::ExecutionState> cuda_execution;
    std::vector<unsigned int> device_halo_send_counts;
    std::vector<unsigned int> device_halo_receive_counts;
    std::vector<detail::DeviceTransfer> device_transfers;
    std::vector<DeviceHaloBuffer> device_halo_results;
    void* nvshmem_halo_receive = nullptr;
    std::uint64_t* nvshmem_halo_signals = nullptr;
    std::size_t nvshmem_halo_slot_capacity = 0;
    std::uint64_t nvshmem_halo_epoch = 0;
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
#if DSDECOMP_HAS_CUDA
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

} // namespace dsdecomp
