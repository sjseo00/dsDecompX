#include "decomposition.hpp"

#include "internal/backend.hpp"
#include "internal/decomp_impl.hpp"
#include "internal/topology_access.hpp"

#include <stdexcept>
#include <algorithm>
#include <exception>
#include <utility>

namespace dsdecompx {

Backend::Backend() noexcept = default;
Backend::~Backend() = default;
Backend::Backend(Backend&&) noexcept = default;
Backend& Backend::operator=(Backend&&) noexcept = default;

Backend::Backend(std::unique_ptr<detail::Backend> implementation)
    : implementation_(std::move(implementation)) {}

BackendKind Backend::kind() const noexcept {
    return implementation_ ? implementation_->kind() : BackendKind::Mpi;
}

Backend::operator bool() const noexcept {
    return static_cast<bool>(implementation_);
}

Decomposition::Decomposition(Config config, Backend backend) {
    if (!backend.implementation_) {
        throw std::invalid_argument("dsDecompX: backend is not initialized");
    }
    impl_ = std::make_unique<Impl>(std::move(config),
                                  std::move(backend.implementation_));
    impl_->initialize_topology();
}

HaloExchangeHandle::~HaloExchangeHandle() {
    if (active_) std::terminate();
}

HaloExchangeHandle::HaloExchangeHandle(HaloExchangeHandle&& other) noexcept
    : owner_(other.owner_), generation_(other.generation_), active_(other.active_) {
    other.owner_ = nullptr;
    other.generation_ = 0;
    other.active_ = false;
}

HaloExchangeHandle& HaloExchangeHandle::operator=(HaloExchangeHandle&& other) noexcept {
    if (this == &other) return *this;
    if (active_) std::terminate();
    owner_ = other.owner_;
    generation_ = other.generation_;
    active_ = other.active_;
    other.owner_ = nullptr;
    other.generation_ = 0;
    other.active_ = false;
    return *this;
}

Decomposition::~Decomposition() {
    if (impl_ && impl_->operation_active) std::terminate();
#if HAS_CUDA
    if (impl_) {
        if (impl_->nvshmem_halo_receive)
            impl_->backend->free_symmetric(impl_->nvshmem_halo_receive);
        if (impl_->nvshmem_halo_signals)
            impl_->backend->free_symmetric(impl_->nvshmem_halo_signals);
        if (impl_->nvshmem_migration_receive)
            impl_->backend->free_symmetric(impl_->nvshmem_migration_receive);
        if (impl_->nvshmem_migration_signals)
            impl_->backend->free_symmetric(impl_->nvshmem_migration_signals);
        impl_->nvshmem_halo_receive = nullptr;
        impl_->nvshmem_halo_signals = nullptr;
        impl_->nvshmem_migration_receive = nullptr;
        impl_->nvshmem_migration_signals = nullptr;
    }
#endif
}
Decomposition::Decomposition(Decomposition&& other) noexcept {
    if (other.impl_ && other.impl_->operation_active) std::terminate();
    impl_ = std::move(other.impl_);
}

Decomposition& Decomposition::operator=(Decomposition&& other) noexcept {
    if (this == &other) return *this;
    if ((impl_ && impl_->operation_active) ||
        (other.impl_ && other.impl_->operation_active)) {
        std::terminate();
    }
    impl_ = std::move(other.impl_);
    return *this;
}

const Topology& Decomposition::topology() const noexcept {
    return impl_->topology;
}

const GlobalDomain& Decomposition::global_domain() const noexcept {
    return impl_->config.domain;
}

void Decomposition::update_domain(const GlobalDomain& domain) {
    if (domain.periodic != impl_->config.domain.periodic) {
        throw std::invalid_argument(
            "dsDecompX: changing periodicity after topology creation requires reconstructing Decomp");
    }
    auto& cuts = detail::TopologyAccess::axis_cuts(impl_->topology);
    const GlobalDomain previous = impl_->config.domain;
    for (int axis = 0; axis < 3; ++axis) {
        const double old_extent = previous.upper[axis] - previous.lower[axis];
        const double new_extent = domain.upper[axis] - domain.lower[axis];
        for (double& cut : cuts[axis]) {
            const double fraction = (cut - previous.lower[axis]) / old_extent;
            cut = domain.lower[axis] + fraction * new_extent;
        }
    }
    impl_->config.domain = domain;
    impl_->refresh_domain_geometry();
    invalidate_halo_plan();
}

void Decomposition::set_axis_cuts(
    const std::array<std::vector<double>, 3>& cuts) {
    const Vec3i& dimensions = impl_->topology.dimensions();
    for (int axis = 0; axis < 3; ++axis) {
        if (cuts[axis].size() != static_cast<std::size_t>(dimensions[axis] + 1)) {
            throw std::invalid_argument("dsDecompX: dynamic cut count does not match process grid");
        }
        if (cuts[axis].front() != impl_->config.domain.lower[axis] ||
            cuts[axis].back() != impl_->config.domain.upper[axis] ||
            !std::is_sorted(cuts[axis].begin(), cuts[axis].end()) ||
            std::adjacent_find(cuts[axis].begin(), cuts[axis].end()) != cuts[axis].end()) {
            throw std::invalid_argument("dsDecompX: dynamic cuts must be strictly increasing and span the domain");
        }
    }
    detail::TopologyAccess::axis_cuts(impl_->topology) = cuts;
    impl_->refresh_domain_geometry();
    invalidate_halo_plan();
}

GridPolicy Decomposition::grid_policy() const noexcept {
    return impl_->config.grid_policy;
}

void Decomposition::set_halo_width(double width) {
    if (width < 0.0) {
        throw std::invalid_argument("dsDecompX: halo width cannot be negative");
    }
    if (impl_->config.halo.width != width) {
        impl_->config.halo.width = width;
        invalidate_halo_plan();
    }
}

int Decomposition::rank_at(const Vec3i& coordinates) const {
    return impl_->backend->cartesian_rank(coordinates);
}

void Decomposition::invalidate_halo_plan() noexcept {
    impl_->halo_plan_valid = false;
#if HAS_CUDA
    impl_->device_halo_plan_valid = false;
#endif
}

bool Decomposition::halo_plan_valid() const noexcept {
    return impl_->halo_plan_valid;
}

bool Decomposition::halo_plan_reused() const noexcept {
#if HAS_CUDA
    return impl_->device_halo_plan_reused;
#else
    return false;
#endif
}

bool Decomposition::halo_positions_only() const noexcept {
#if HAS_CUDA
    return impl_->device_halo_positions_only;
#else
    return false;
#endif
}

std::size_t Decomposition::total_received_halo_particles() const noexcept {
    std::size_t total = 0;
#if HAS_CUDA
    if (!impl_->device_halo_results.empty()) {
        for (const auto& buffer : impl_->device_halo_results) total+=buffer.count;
        return total;
    }
#endif
    for (const auto& buffer : impl_->halo_receive_buffers) {
        total += buffer.size();
    }
    return total;
}

const std::vector<std::vector<ParticleRecord>>&
Decomposition::received_halos() const noexcept {
    return impl_->halo_receive_buffers;
}

const std::vector<HaloPlanEntry>& Decomposition::halo_plan() const noexcept {
    return impl_->halo_plan;
}

bool Decomposition::cuda_aware_mpi_enabled() const noexcept {
    return impl_->backend->cuda_aware_enabled();
}

BackendKind Decomposition::backend_kind() const noexcept {
    return impl_->backend->kind();
}

const std::vector<ParticleRecord>&
Decomposition::migrated_owned_particles() const noexcept {
    return impl_->migrated_owned;
}

DeviceParticleBuffer
Decomposition::migrated_owned_device_particles() const noexcept {
#if HAS_CUDA
    return impl_->device_migration_result;
#else
    return {};
#endif
}

void Decomposition::set_timing_enabled(bool enabled) noexcept {
    impl_->timing_enabled = enabled;
}

void Decomposition::record_overlap_compute_times(
    double interior_seconds, double boundary_seconds) noexcept {
    if (!impl_->timing_enabled) return;
    impl_->halo_timings.interior_compute_seconds = interior_seconds;
    impl_->halo_timings.boundary_compute_seconds = boundary_seconds;
}

const HaloExchangeTimings& Decomposition::last_halo_timings() const noexcept {
    return impl_->halo_timings;
}

void Decomposition::set_performance_counters_enabled(
    bool enabled, bool neighbor_detail) noexcept {
#if ENABLE_PERF_COUNTERS
    if (enabled && !impl_->perf_enabled) {
        impl_->backend_reallocation_baseline =
            impl_->backend->buffer_reallocations();
#if HAS_CUDA
        impl_->cuda_reallocation_baseline = impl_->cuda_execution
            ? impl_->cuda_execution->reallocations() : 0;
#endif
    }
    impl_->perf_enabled = enabled;
    impl_->neighbor_detail_enabled = enabled && neighbor_detail;
#else
    (void)enabled;
    (void)neighbor_detail;
#endif
}

PerfStats Decomposition::get_stats() const {
#if ENABLE_PERF_COUNTERS
    PerfStats result = impl_->perf_stats;
    const std::uint64_t backend_reallocations =
        impl_->backend->buffer_reallocations();
    if (backend_reallocations >= impl_->backend_reallocation_baseline) {
        result.buffer_reallocations +=
            backend_reallocations - impl_->backend_reallocation_baseline;
    }
    result.peak_communication_buffer_capacity = std::max(
        result.peak_communication_buffer_capacity,
        impl_->backend->peak_buffer_capacity());
#if HAS_CUDA
    if (impl_->cuda_execution) {
        const auto reallocations=impl_->cuda_execution->reallocations();
        if (reallocations>=impl_->cuda_reallocation_baseline)
            result.buffer_reallocations +=
                reallocations-impl_->cuda_reallocation_baseline;
        result.peak_communication_buffer_capacity = std::max<std::uint64_t>(
            result.peak_communication_buffer_capacity,
            impl_->cuda_execution->communication_capacity());
    }
#endif
    return result;
#else
    return {};
#endif
}

PerfStats Decomposition::get_global_stats() const {
#if ENABLE_PERF_COUNTERS
    return impl_->backend->aggregate_stats(get_stats());
#else
    return {};
#endif
}

void Decomposition::reset_stats() noexcept {
#if ENABLE_PERF_COUNTERS
    impl_->perf_stats = {};
    impl_->backend_reallocation_baseline =
        impl_->backend->buffer_reallocations();
#if HAS_CUDA
    impl_->cuda_reallocation_baseline = impl_->cuda_execution
        ? impl_->cuda_execution->reallocations() : 0;
#endif
#endif
}

} // namespace dsdecompx
