#pragma once

#include "backend.hpp"
#include "backend_selection.hpp"
#include "config.hpp"
#include "domain.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "perf.hpp"
#include "topology.hpp"
#include "version.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <cstdint>
#include <vector>

namespace dsdecompx {

struct HaloPlanEntry {
    int peer_rank = -1;
    Vec3i direction{0, 0, 0};
    Vec3d periodic_shift{0.0, 0.0, 0.0};
    std::vector<int> source_indices;
    std::size_t receive_count = 0;
};

struct HaloExchangeTimings {
    double pack_seconds = 0.0;
    double communication_seconds = 0.0;
    double interior_compute_seconds = 0.0;
    double wait_seconds = 0.0;
    double unpack_seconds = 0.0;
    double boundary_compute_seconds = 0.0;
    double total_seconds = 0.0;
};

struct DeviceHaloBuffer {
    int peer_rank = -1;
    const ParticleRecord* data = nullptr;
    std::size_t count = 0;
};

struct DeviceParticleBuffer {
    const ParticleRecord* data = nullptr;
    std::size_t count = 0;
};

class Decomposition;

class HaloExchangeHandle {
public:
    HaloExchangeHandle() noexcept = default;
    ~HaloExchangeHandle();

    HaloExchangeHandle(HaloExchangeHandle&& other) noexcept;
    HaloExchangeHandle& operator=(HaloExchangeHandle&& other) noexcept;

    HaloExchangeHandle(const HaloExchangeHandle&) = delete;
    HaloExchangeHandle& operator=(const HaloExchangeHandle&) = delete;

    bool active() const noexcept { return active_; }

private:
    friend class Decomposition;
    HaloExchangeHandle(Decomposition* owner, std::uint64_t generation) noexcept
        : owner_(owner), generation_(generation), active_(true) {}

    Decomposition* owner_ = nullptr;
    std::uint64_t generation_ = 0;
    bool active_ = false;
};

class Decomposition {
public:
    Decomposition(Config config, Backend backend);
    ~Decomposition();

    Decomposition(Decomposition&&) noexcept;
    Decomposition& operator=(Decomposition&&) noexcept;

    Decomposition(const Decomposition&) = delete;
    Decomposition& operator=(const Decomposition&) = delete;

    const Topology& topology() const noexcept;
    const GlobalDomain& global_domain() const noexcept;

    void update_domain(const GlobalDomain& domain);
    void set_halo_width(double width);

    int rank_at(const Vec3i& coordinates) const;
    int owner_rank(const Vec3d& position) const;

    void invalidate_halo_plan() noexcept;
    bool halo_plan_valid() const noexcept;

    void rebuild_halo_plan(const ConstParticleView& owned_particles);
    void refresh_halo_buffers(const ConstParticleView& owned_particles);
    // One Decomposition supports one in-flight operation. The handle must be
    // completed explicitly; destroying an active handle never synchronizes and
    // terminates the process to prevent use-after-free of MPI/CUDA resources.
    HaloExchangeHandle begin_halo_exchange(
        const ConstParticleView& owned_particles,
        ExecutionStream stream = nullptr);
    void end_halo_exchange(HaloExchangeHandle& handle,
                           ExecutionStream stream = nullptr);
    void exchange_halo(const ConstParticleView& owned_particles);

    void reverse_halo(ParticleView particles, std::size_t owned_count,
                      ExecutionStream stream = nullptr);
    void reverse_exchange(
        ParticleView particles, std::size_t owned_count,
        const std::vector<ReverseExchangeField>& fields,
        ExecutionStream stream = nullptr);

    MigrationResult migrate_particles(const ConstParticleView& owned_particles,
                                      ExecutionStream stream = nullptr);
    const std::vector<ParticleRecord>& migrated_owned_particles() const noexcept;
    DeviceParticleBuffer migrated_owned_device_particles() const noexcept;
    void unpack_migrated_device_particles(ParticleView destination,
                                           ExecutionStream stream);
    void unpack_migrated_fields(ParticleView destination);

    std::size_t total_received_halo_particles() const noexcept;

    const std::vector<std::vector<ParticleRecord>>& received_halos() const noexcept;
    const std::vector<DeviceHaloBuffer>& received_device_halos() const noexcept;
    void unpack_device_halos(ParticleView destination,
                             std::size_t destination_offset,
                             ExecutionStream stream);
    void unpack_halo_fields(ParticleView destination,
                            std::size_t destination_offset);
    const std::vector<HaloPlanEntry>& halo_plan() const noexcept;

    bool cuda_aware_mpi_enabled() const noexcept;
    BackendKind backend_kind() const noexcept;

    void set_timing_enabled(bool enabled) noexcept;
    void record_overlap_compute_times(double interior_seconds,
                                      double boundary_seconds) noexcept;
    const HaloExchangeTimings& last_halo_timings() const noexcept;

    void set_performance_counters_enabled(bool enabled,
                                          bool neighbor_detail = false) noexcept;
    PerfStats get_stats() const;
    PerfStats get_global_stats() const;
    void reset_stats() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dsdecompx
