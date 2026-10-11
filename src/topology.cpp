#include "internal/decomp_impl.hpp"
#include "internal/topology_access.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dsdecompx {
namespace {

int direction_slot(int dx, int dy, int dz) {
    const int raw = (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1);
    return raw < 13 ? raw : raw - 1;
}

} // namespace

void Decomposition::Impl::initialize_topology() {
    if (!backend) {
        throw std::invalid_argument("dsDecompX: missing backend");
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (!(config.domain.upper[axis] > config.domain.lower[axis])) {
            throw std::invalid_argument("dsDecompX: global-domain extent must be positive");
        }
    }
    if (config.halo.width < 0.0) {
        throw std::invalid_argument("dsDecompX: halo width cannot be negative");
    }
    if (config.execution == ExecutionMode::Cuda &&
        config.process_grid.allow_rank_reorder) {
        throw std::invalid_argument(
            "dsDecompX CUDA execution currently requires allow_rank_reorder=false");
    }

    ProcessGrid selected_grid = config.process_grid;
    if (config.grid_policy != GridPolicy::Manual) {
        DomainShape domain_shape;
        for (int axis = 0; axis < 3; ++axis) {
            domain_shape.lengths[axis] = config.domain.upper[axis] -
                                         config.domain.lower[axis];
        }
        domain_shape.periodic = config.domain.periodic;
        const GridShape shape = choose_process_grid(
            backend->size(), domain_shape, config.topology_info,
            config.grid_policy);
        selected_grid.dimensions = shape.as_array();
    }
    backend->create_cartesian(selected_grid, config.domain.periodic);
    detail::TopologyAccess::rank(topology) = backend->rank();
    detail::TopologyAccess::size(topology) = backend->size();
    detail::TopologyAccess::dimensions(topology) = backend->cartesian_dimensions();
    detail::TopologyAccess::coordinates(topology) = backend->cartesian_coordinates();
    refresh_domain_geometry();
}

void Decomposition::Impl::refresh_domain_geometry() {
    auto& dimensions = detail::TopologyAccess::dimensions(topology);
    auto& coordinates = detail::TopologyAccess::coordinates(topology);
    auto& local_domain = detail::TopologyAccess::local_domain(topology);
    auto& neighbors = detail::TopologyAccess::neighbors(topology);
    auto& axis_cuts = detail::TopologyAccess::axis_cuts(topology);
    for (int axis = 0; axis < 3; ++axis) {
        const double global_lower = config.domain.lower[axis];
        const double global_upper = config.domain.upper[axis];
        const double extent = global_upper - global_lower;
        if (axis_cuts[axis].size() !=
            static_cast<std::size_t>(dimensions[axis] + 1)) {
            axis_cuts[axis].resize(static_cast<std::size_t>(dimensions[axis] + 1));
            for (int i = 0; i <= dimensions[axis]; ++i) {
                axis_cuts[axis][static_cast<std::size_t>(i)] =
                    global_lower + extent * static_cast<double>(i) /
                                       static_cast<double>(dimensions[axis]);
            }
        }
        local_domain.lower[axis] =
            axis_cuts[axis][static_cast<std::size_t>(coordinates[axis])];
        local_domain.upper[axis] =
            axis_cuts[axis][static_cast<std::size_t>(coordinates[axis] + 1)];
    }

    neighbors.clear();
    neighbors.reserve(26);

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                if (!config.halo.include_edges_and_corners &&
                    std::abs(dx) + std::abs(dy) + std::abs(dz) != 1) {
                    continue;
                }

                const Vec3i direction{dx, dy, dz};
                // An axis with dimensions[axis] == 1 has a single rank
                // spanning its entire (periodic) extent, so stepping along
                // it never leaves this rank. If this direction moves along
                // any such axis at all, skip it entirely rather than
                // collapsing just that axis and keeping the rest: a
                // single-axis (face) direction along an unsplit axis
                // degenerates to this same rank with zero shift, which is
                // not a valid halo neighbor and would round-trip this
                // rank's own atoms through the halo pipeline as
                // self-ghosts -- double-counted against the very same
                // (already local) atoms by the CPU force kernels'
                // unconditional minimum-image convention. A multi-axis
                // (edge/corner) direction that also moves along an unsplit
                // axis would, after dropping that axis's contribution,
                // duplicate a lower-dimensional direction that this same
                // triple loop already enumerates on its own (e.g. (1,1,0)
                // with y unsplit reduces to (1,0,0), which is generated
                // separately) -- and the local rank's ghost-selection for
                // that lower-dimensional neighbor already spans this rank's
                // full extent along the unsplit axis, so no coverage is
                // lost by skipping the redundant corner/edge case. This
                // mirrors the identical dims[axis] == 1 skip in grid.cpp's
                // process-grid cost estimator.
                bool touches_unsplit_axis = false;
                for (int axis = 0; axis < 3; ++axis) {
                    if (direction[axis] != 0 && dimensions[axis] == 1) {
                        touches_unsplit_axis = true;
                        break;
                    }
                }
                if (touches_unsplit_axis) {
                    continue;
                }

                Vec3i neighbor_coordinates{
                    coordinates[0] + dx,
                    coordinates[1] + dy,
                    coordinates[2] + dz
                };
                Vec3d shift{0.0, 0.0, 0.0};
                bool outside_nonperiodic = false;

                for (int axis = 0; axis < 3; ++axis) {
                    if (neighbor_coordinates[axis] < 0 ||
                        neighbor_coordinates[axis] >= dimensions[axis]) {
                        if (!config.domain.periodic[axis]) {
                            outside_nonperiodic = true;
                            break;
                        }
                        const double extent = config.domain.upper[axis] -
                                              config.domain.lower[axis];
                        if (neighbor_coordinates[axis] < 0) {
                            neighbor_coordinates[axis] += dimensions[axis];
                            shift[axis] = extent;
                        } else {
                            neighbor_coordinates[axis] -= dimensions[axis];
                            shift[axis] = -extent;
                        }
                    }
                }

                Neighbor neighbor;
                neighbor.direction = direction;
                neighbor.periodic_shift = shift;
                neighbor.direction_slot = direction_slot(dx, dy, dz);
                neighbor.rank = outside_nonperiodic
                    ? -1
                    : backend->cartesian_rank(neighbor_coordinates);
                neighbor.is_self = neighbor.rank == topology.rank();
                neighbors.push_back(neighbor);
            }
        }
    }

    halo_plan.resize(neighbors.size());
    halo_send_buffers.resize(neighbors.size());
    halo_receive_buffers.resize(neighbors.size());
    halo_plan_valid = false;
#if HAS_CUDA
    device_halo_plan_valid = false;
#endif
}

} // namespace dsdecompx
