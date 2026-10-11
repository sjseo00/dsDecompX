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
#include <functional>
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

// Records received on the device. Without custom fields they are a plain
// ParticleRecord array. With custom fields each record is followed by the
// field bytes, so consecutive records are further apart than
// sizeof(ParticleRecord); use unpack_device_halos() / the migration unpack to
// consume them rather than indexing the array directly.
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
    void set_axis_cuts(const std::array<std::vector<double>, 3>& cuts);
    GridPolicy grid_policy() const noexcept;

    int rank_at(const Vec3i& coordinates) const;
    int owner_rank(const Vec3d& position) const;

    void invalidate_halo_plan() noexcept;
    bool halo_plan_valid() const noexcept;
    // True when the last CUDA halo exchange refreshed the existing plan in
    // place, i.e. ghost membership and slot order are unchanged. Always false
    // for host particle memory (the host path reports changes via global IDs).
    bool halo_plan_reused() const noexcept;
    // True when the last CUDA halo exchange moved positions only (see
    // HaloExchangeOptions::positions_only).
    bool halo_positions_only() const noexcept;
    // True when a non-rebuild device exchange of this view would refresh the
    // existing plan in place (same ghost list, no count exchange). Identical on
    // every rank as long as the ranks make the same collective calls.
    bool can_refresh_halo_plan(const ConstParticleView& owned_particles) const;

    void rebuild_halo_plan(const ConstParticleView& owned_particles);
    void refresh_halo_buffers(const ConstParticleView& owned_particles);
    // One Decomposition supports one in-flight operation. The handle must be
    // completed explicitly; destroying an active handle never synchronizes and
    // terminates the process to prevent use-after-free of MPI/CUDA resources.
    // positions_only: see HaloExchangeOptions::positions_only. With it, the
    // received device halo buffers (received_device_halos) hold 3 doubles per
    // ghost, not ParticleRecords; use unpack_device_halos() to consume them.
    HaloExchangeHandle begin_halo_exchange(
        const ConstParticleView& owned_particles,
        ExecutionStream stream = nullptr,
        bool positions_only = false);
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

#if HAS_CUDA
    // Shared device reverse path: sends `record_bytes`-sized records for every
    // received ghost back to the rank that owns it and hands the returned
    // records to `reduce` together with the owning particle indices.
    void reverse_device_records(
        std::size_t owned_count, ExecutionStream stream, std::size_t record_bytes,
        int buffer_set,
        const std::function<void(std::size_t, std::size_t, unsigned char*)>& pack,
        const std::function<void(const unsigned char*, std::size_t,
                                 const unsigned int*)>& reduce,
        const char* operation);
#endif
};

} // namespace dsdecompx
