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
};

struct HaloExchangeOptions {
    bool rebuild = false;
    ExecutionStream stream = nullptr;
};

struct StorageResizeResult {
    ParticleView view;
    bool reallocated = false;
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
    MigrationResult migrate(ParticleInfo& particles,
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
