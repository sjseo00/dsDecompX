#pragma once

#include <array>

namespace dsdecomp {

using Vec3d = std::array<double, 3>;
using Vec3i = std::array<int, 3>;

struct GlobalDomain {
    Vec3d lower{0.0, 0.0, 0.0};
    Vec3d upper{1.0, 1.0, 1.0};
    std::array<bool, 3> periodic{true, true, true};
};

struct LocalDomain {
    Vec3d lower{0.0, 0.0, 0.0};
    Vec3d upper{1.0, 1.0, 1.0};
};

struct ProcessGrid {
    // Zero dimensions are filled by the selected grid planner/backend.
    Vec3i dimensions{0, 0, 0};
    bool allow_rank_reorder = false;
};

struct HaloConfig {
    double width = 0.0;
    bool include_edges_and_corners = true;
};

} // namespace dsdecomp
