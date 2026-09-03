#pragma once

#include "domain.hpp"

#include <array>
#include <vector>

namespace dsdecomp {

struct GridShape {
    int px = 1;
    int py = 1;
    int pz = 1;

    Vec3i as_array() const noexcept { return {px, py, pz}; }
};

struct DomainShape {
    Vec3d lengths{1.0, 1.0, 1.0};
    std::array<bool, 3> periodic{true, true, true};
};

enum class GridPolicy {
    Manual,
    Balanced,
    TopologyAware
};

struct TopologyInfo {
    bool available = false;
    int local_rank = 0;
    int local_size = 1;
    int node_count = 1;
    std::vector<int> rank_to_node;
    double intra_node_weight = 1.0;
    double inter_node_weight = 4.0;
};

GridShape choose_process_grid(int nranks,
                              const DomainShape& domain,
                              const TopologyInfo& topology,
                              GridPolicy policy,
                              GridShape manual = {});

} // namespace dsdecomp
