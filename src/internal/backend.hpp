#pragma once

#include "backend.hpp"
#include "domain.hpp"
#include "field.hpp"
#include "topology.hpp"
#include "perf.hpp"
#include "device_exchange.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace dsdecomp::detail {

class Backend {
public:
    virtual ~Backend() = default;

    virtual BackendKind kind() const noexcept = 0;
    virtual int rank() const noexcept = 0;
    virtual int size() const noexcept = 0;

    virtual void create_cartesian(const ProcessGrid& grid,
                                  const std::array<bool, 3>& periodic) = 0;
    virtual Vec3i cartesian_dimensions() const noexcept = 0;
    virtual Vec3i cartesian_coordinates() const noexcept = 0;
    virtual int cartesian_rank(const Vec3i& coordinates) const = 0;
    virtual void exchange_neighbor_counts(
        const std::vector<Neighbor>& neighbors,
        const std::vector<unsigned int>& send_counts,
        std::vector<unsigned int>& receive_counts,
        int tag_base) = 0;
    virtual void exchange_rank_counts(
        const std::vector<unsigned int>& send_counts,
        std::vector<unsigned int>& receive_counts) = 0;

    virtual bool supports_device_data_plane() const noexcept { return false; }
    virtual void begin_device_exchange(const DeviceExchange&) {
        throw std::runtime_error(
            "dsDecomp: selected backend has no GPU-resident data plane");
    }
    virtual void end_device_exchange(const DeviceExchange&) {
        throw std::runtime_error(
            "dsDecomp: selected backend has no GPU-resident data plane");
    }
    virtual void* allocate_symmetric(std::size_t) {
        throw std::runtime_error(
            "dsDecomp: selected backend does not provide symmetric memory");
    }
    virtual void* allocate_symmetric_zeroed(std::size_t count,
                                             std::size_t element_size) {
        (void)count; (void)element_size;
        throw std::runtime_error(
            "dsDecomp: selected backend does not provide symmetric memory");
    }
    virtual void free_symmetric(void*) {
        throw std::runtime_error(
            "dsDecomp: selected backend does not provide symmetric memory");
    }
    virtual std::size_t maximum_size(std::size_t local) const = 0;

    virtual void begin_exchange_records(
        const std::vector<Neighbor>& neighbors,
        const std::vector<std::vector<ParticleRecord>>& send_buffers,
        std::vector<std::vector<ParticleRecord>>& receive_buffers,
        int count_tag_base,
        int data_tag_base,
        bool direct_self_copy) = 0;
    virtual void end_exchange_records() = 0;

    void exchange_records(
        const std::vector<Neighbor>& neighbors,
        const std::vector<std::vector<ParticleRecord>>& send_buffers,
        std::vector<std::vector<ParticleRecord>>& receive_buffers,
        int count_tag_base,
        int data_tag_base,
        bool direct_self_copy) {
        begin_exchange_records(neighbors, send_buffers, receive_buffers,
                               count_tag_base, data_tag_base, direct_self_copy);
        end_exchange_records();
    }

    virtual void exchange_all_to_all_records(
        const std::vector<std::vector<ParticleRecord>>& send_by_rank,
        std::vector<std::vector<ParticleRecord>>& receive_by_rank) = 0;

    virtual void exchange_neighbor_bytes(
        const std::vector<Neighbor>&,
        const std::vector<std::vector<std::byte>>&,
        std::vector<std::vector<std::byte>>&,
        int, int, bool) {
        throw std::runtime_error(
            "dsDecomp: selected backend does not support host custom fields");
    }
    virtual void exchange_all_to_all_bytes(
        const std::vector<std::vector<std::byte>>&,
        std::vector<std::vector<std::byte>>&) {
        throw std::runtime_error(
            "dsDecomp: selected backend does not support host custom fields");
    }

#ifndef NDEBUG
    virtual std::vector<int> allgather_ids(const int* local_ids,
                                           std::size_t count) = 0;
#endif

    virtual bool cuda_aware_enabled() const noexcept = 0;
    virtual PerfStats aggregate_stats(const PerfStats& local) const = 0;
    virtual std::uint64_t buffer_reallocations() const noexcept = 0;
    virtual std::uint64_t peak_buffer_capacity() const noexcept = 0;
};

} // namespace dsdecomp::detail
