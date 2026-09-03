#pragma once

#include "domain.hpp"

#include <vector>

namespace dsdecomp {

namespace detail {
struct TopologyAccess;
}

struct Neighbor {
    int rank = -1;
    Vec3i direction{0, 0, 0};
    Vec3d periodic_shift{0.0, 0.0, 0.0};
    int direction_slot = -1;
    bool is_self = false;
};

class Topology {
public:
    int rank() const noexcept { return rank_; }
    int size() const noexcept { return size_; }
    const Vec3i& dimensions() const noexcept { return dimensions_; }
    const Vec3i& coordinates() const noexcept { return coordinates_; }
    const LocalDomain& local_domain() const noexcept { return local_domain_; }
    const std::vector<Neighbor>& neighbors() const noexcept { return neighbors_; }

private:
    friend struct detail::TopologyAccess;

    int rank_ = 0;
    int size_ = 1;
    Vec3i dimensions_{1, 1, 1};
    Vec3i coordinates_{0, 0, 0};
    LocalDomain local_domain_{};
    std::vector<Neighbor> neighbors_;
};

} // namespace dsdecomp
