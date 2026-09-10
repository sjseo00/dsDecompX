#pragma once

#include "topology.hpp"

namespace dsdecompx::detail {

// Internal mutation gateway. Public consumers receive a read-only Topology.
struct TopologyAccess {
    static int& rank(Topology& topology) { return topology.rank_; }
    static int& size(Topology& topology) { return topology.size_; }
    static Vec3i& dimensions(Topology& topology) { return topology.dimensions_; }
    static Vec3i& coordinates(Topology& topology) { return topology.coordinates_; }
    static LocalDomain& local_domain(Topology& topology) { return topology.local_domain_; }
    static std::vector<Neighbor>& neighbors(Topology& topology) { return topology.neighbors_; }
};

} // namespace dsdecompx::detail
