#include "grid.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace dsdecomp {
namespace {

double balanced_score(const GridShape& grid, const DomainShape& domain) {
    const double local_x = domain.lengths[0] / grid.px;
    const double local_y = domain.lengths[1] / grid.py;
    const double local_z = domain.lengths[2] / grid.pz;
    const double surface_to_volume =
        2.0 * (1.0 / local_x + 1.0 / local_y + 1.0 / local_z);
    const double longest = std::max({local_x, local_y, local_z});
    const double shortest = std::min({local_x, local_y, local_z});
    const double aspect_penalty = std::log(longest / shortest);
    return surface_to_volume * (1.0 + 0.05 * aspect_penalty);
}

int linear_rank(int x, int y, int z, const GridShape& grid) {
    return (x * grid.py + y) * grid.pz + z;
}

double topology_score(const GridShape& grid,
                      const DomainShape& domain,
                      const TopologyInfo& topology) {
    if (!topology.available ||
        topology.rank_to_node.size() !=
            static_cast<std::size_t>(grid.px * grid.py * grid.pz)) {
        return balanced_score(grid, domain);
    }

    const double local[3] = {
        domain.lengths[0] / grid.px,
        domain.lengths[1] / grid.py,
        domain.lengths[2] / grid.pz
    };
    const double face_area[3] = {
        local[1] * local[2], local[0] * local[2], local[0] * local[1]
    };
    const int dims[3] = {grid.px, grid.py, grid.pz};
    double communication_cost = 0.0;

    for (int x = 0; x < grid.px; ++x) {
        for (int y = 0; y < grid.py; ++y) {
            for (int z = 0; z < grid.pz; ++z) {
                const int source = linear_rank(x, y, z, grid);
                for (int axis = 0; axis < 3; ++axis) {
                    int neighbor_coordinate[3] = {x, y, z};
                    ++neighbor_coordinate[axis];
                    if (neighbor_coordinate[axis] == dims[axis]) {
                        if (!domain.periodic[axis] || dims[axis] == 1) continue;
                        neighbor_coordinate[axis] = 0;
                    }
                    const int peer = linear_rank(neighbor_coordinate[0],
                                                 neighbor_coordinate[1],
                                                 neighbor_coordinate[2], grid);
                    if (peer == source) continue;
                    const bool same_node = topology.rank_to_node[source] ==
                                           topology.rank_to_node[peer];
                    communication_cost += face_area[axis] *
                        (same_node ? topology.intra_node_weight
                                   : topology.inter_node_weight);
                }
            }
        }
    }
    const double local_volume = local[0] * local[1] * local[2];
    return communication_cost / (local_volume * grid.px * grid.py * grid.pz) +
           0.01 * balanced_score(grid, domain);
}

} // namespace

GridShape choose_process_grid(int nranks,
                              const DomainShape& domain,
                              const TopologyInfo& topology,
                              GridPolicy policy,
                              GridShape manual) {
    if (nranks <= 0) throw std::invalid_argument("dsDecomp: nranks must be positive");
    for (double length : domain.lengths) {
        if (!(length > 0.0)) {
            throw std::invalid_argument("dsDecomp: domain lengths must be positive");
        }
    }
    if (policy == GridPolicy::Manual) {
        if (manual.px <= 0 || manual.py <= 0 || manual.pz <= 0 ||
            manual.px * manual.py * manual.pz != nranks) {
            throw std::invalid_argument("dsDecomp: manual process grid does not match nranks");
        }
        return manual;
    }

    const bool use_topology = policy == GridPolicy::TopologyAware &&
                              topology.available &&
                              topology.rank_to_node.size() ==
                                  static_cast<std::size_t>(nranks);
    double best_score = std::numeric_limits<double>::infinity();
    GridShape best{nranks, 1, 1};
    for (int px = 1; px <= nranks; ++px) {
        if (nranks % px != 0) continue;
        const int remaining = nranks / px;
        for (int py = 1; py <= remaining; ++py) {
            if (remaining % py != 0) continue;
            GridShape candidate{px, py, remaining / py};
            const double score = use_topology
                ? topology_score(candidate, domain, topology)
                : balanced_score(candidate, domain);
            const auto candidate_key = std::make_tuple(score, px, py, candidate.pz);
            const auto best_key = std::make_tuple(best_score, best.px, best.py, best.pz);
            if (candidate_key < best_key) {
                best_score = score;
                best = candidate;
            }
        }
    }
    return best;
}

} // namespace dsdecomp
