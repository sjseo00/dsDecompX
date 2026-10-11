#pragma once

#include "config.hpp"
#include "field.hpp"
#include "perf.hpp"
#include "topology.hpp"

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dsdecompx {

struct HaloResult {
    std::size_t ghost_count = 0;
    bool layout_changed = false;
    bool storage_reallocated = false;
    // CUDA memory only: the previous plan was refreshed in place. Identical on
    // every rank for a collective call, so it can gate collectives.
    bool plan_reused = false;
};

struct HaloExchangeOptions {
    bool rebuild = false;
    ExecutionStream stream = nullptr;
    // CUDA views only. When the exchange refreshes an existing halo plan in
    // place (rebuild == false and the plan is reusable), send just the ghost
    // positions (3 doubles per ghost) instead of the whole particle record
    // (about 96 bytes). The ghosts then keep the velocity, mass, charge, type
    // and id of the last full exchange and get zero forces; positions are
    // exactly what a full exchange would deliver. Ignored for a rebuilt plan,
    // with custom wire fields and with the NVSHMEM backend. The same value must
    // be passed on every rank.
    bool positions_only = false;
};
// Defined when HaloExchangeOptions has positions_only (lets a client build against older releases).
#define DSDECOMPX_HALO_POSITIONS_ONLY 1
// Where the translation unit that includes this header found it (to diagnose mixed installs).
inline const char* header_path() noexcept { return __FILE__; }

struct StorageResizeResult {
    ParticleView view;
    bool reallocated = false;
};

struct RebalanceOptions {
    // Negative means use the current owned-particle count as the work metric.
    double local_work = -1.0;
    // Ignore the configured imbalance threshold.
    bool force = false;
};

struct RebalanceResult {
    bool attempted = false;
    bool geometry_changed = false;
    double imbalance_before = 1.0;
    MigrationResult migration;
};

// Particle storage used by every Decomp operation. Constructing with a count
// creates host arrays. Constructing with a view/provider uses existing arrays.
class ParticleInfo {
public:
    class Provider {
    public:
        virtual ~Provider() = default;
        virtual ParticleView particle_info_view(std::size_t minimum) = 0;
        virtual StorageResizeResult resize_particle_info(
            std::size_t required, std::size_t preserve) = 0;
    };

    ParticleInfo() = default;
    explicit ParticleInfo(std::size_t owned) { resize_owned(owned); }
    ~ParticleInfo();
    ParticleInfo(ParticleInfo&& other) noexcept;
    ParticleInfo& operator=(ParticleInfo&& other) noexcept;
    ParticleInfo(const ParticleInfo&) = delete;
    ParticleInfo& operator=(const ParticleInfo&) = delete;

    ParticleInfo(std::size_t owned, std::size_t ghosts, ParticleView storage);
    ParticleInfo(std::size_t owned, std::size_t ghosts, Provider& provider);

    static ParticleInfo allocate(MemorySpace memory_space,
                                 std::size_t owned);
    static ParticleInfo external(MemorySpace memory_space,
                                 std::size_t capacity,
                                 std::size_t owned,
                                 std::size_t ghosts = 0);
    ParticleInfo& positions(double* x, double* y, double* z) noexcept;
    ParticleInfo& velocities(double* vx, double* vy, double* vz) noexcept;
    ParticleInfo& forces(double* fx, double* fy, double* fz) noexcept;
    ParticleInfo& properties(double* mass, double* charge,
                             int* type, int* global_id) noexcept;
    // Register a non-owning application field. Storage must remain valid and
    // large enough for capacity particles through communication operations.
    template<class T>
    ParticleInfo& add_field(std::string name, T* data,
                            std::size_t capacity = 0,
                            std::size_t components = 1) {
        ParticleFieldView field{std::move(name), field_type_v<T>, components,
                                capacity ? capacity : this->capacity(), data};
        return add_field_view(std::move(field));
    }

    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;
    std::vector<double> fx, fy, fz;
    std::vector<double> mass, charge;
    std::vector<int> type, global_id;
    std::size_t num_particles = 0;

    void resize(std::size_t count);

    std::size_t owned_count() const noexcept { return owned_count_; }
    std::size_t ghost_count() const noexcept { return ghost_count_; }
    std::size_t size() const noexcept { return owned_count_ + ghost_count_; }
    std::size_t capacity() const noexcept;
    MemorySpace memory_space() const noexcept;
    bool uses_external_memory() const noexcept {
        return external_ && !owns_device_memory_;
    }

    double* x_data() noexcept;
    double* y_data() noexcept;
    double* z_data() noexcept;
    double* vx_data() noexcept;
    double* vy_data() noexcept;
    double* vz_data() noexcept;
    double* fx_data() noexcept;
    double* fy_data() noexcept;
    double* fz_data() noexcept;
    double* mass_data() noexcept;
    double* charge_data() noexcept;
    int* type_data() noexcept;
    int* global_id_data() noexcept;

    void resize_owned(std::size_t count);
    void reset_counts(std::size_t owned, std::size_t ghosts = 0);

    void reset_view(ParticleView storage);

private:
    friend class Decomp;
    ConstParticleView read_view(std::size_t count) const;
    ConstParticleView host_const_view(std::size_t count) const;
    ParticleView host_view(std::size_t count);
    StorageResizeResult resize_storage(std::size_t required,
                                       std::size_t preserve);
    void commit_counts(std::size_t owned, std::size_t ghosts) noexcept;
    void release_device_memory() noexcept;
    ParticleInfo& add_field_view(ParticleFieldView field);

    bool external_ = false;
    bool owns_device_memory_ = false;
    std::size_t owned_count_ = 0;
    std::size_t ghost_count_ = 0;
    std::vector<ParticleFieldView> fields_;
    ParticleView storage_;
    Provider* provider_ = nullptr;
};

// The sole normal user entry point. Host and device storage use the same
// methods; device callers only add their execution stream.
class Decomp {
public:
    Decomp(MPI_Comm communicator, Config config = {});
    ~Decomp();

    Decomp(Decomp&&) noexcept;
    Decomp& operator=(Decomp&&) noexcept;
    Decomp(const Decomp&) = delete;
    Decomp& operator=(const Decomp&) = delete;

    const Topology& topology() const noexcept;
    BackendKind backend_kind() const noexcept;
    int owner_rank(const Vec3d& position) const;
    bool any_rank_true(bool local_flag) const;
    bool has_out_of_domain_particles(const ParticleInfo& particles) const;

    void update_domain(const GlobalDomain& domain);
    void set_halo_width(double width);
    double halo_width() const noexcept;
    void invalidate_halo() noexcept;

    HaloResult exchange_halo(ParticleInfo& particles,
                             HaloExchangeOptions options = {});
    // Split-phase form of exchange_halo() for overlapping the exchange with work
    // that touches only owned particles. begin_halo_exchange() posts the
    // exchange (device memory: packs and starts the transfers on options.stream);
    // end_halo_exchange() completes it, unpacks the ghosts and returns the same
    // result exchange_halo() would. Between the calls `particles` must not be
    // resized or exchanged again, and no other halo/migration/reverse operation
    // may start. exchange_halo() is exactly begin followed by end.
    void begin_halo_exchange(ParticleInfo& particles,
                             HaloExchangeOptions options = {});
    HaloResult end_halo_exchange(ParticleInfo& particles);
    // True when a non-rebuild device exchange of `particles` would refresh the
    // existing halo plan in place, i.e. ghost slots keep their meaning and the
    // ghost count is already known. Always false for host memory.
    bool can_refresh_halo(const ParticleInfo& particles) const;
    MigrationResult migrate(ParticleInfo& particles,
                            ExecutionStream stream = nullptr);
    // Collective operation. Moves nonuniform Cartesian cut planes using the
    // supplied per-rank work metric, then migrates particles to their new
    // owners. All ranks must call it with the same options.
    RebalanceResult rebalance(ParticleInfo& particles,
                              RebalanceOptions options = {},
                              ExecutionStream stream = nullptr);
    void reverse_halo(ParticleInfo& particles,
                      ExecutionStream stream = nullptr);
    void reverse_exchange(
        ParticleInfo& particles,
        const std::vector<ReverseExchangeField>& fields,
        ExecutionStream stream = nullptr);

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
