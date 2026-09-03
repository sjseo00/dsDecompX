#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#ifndef DSDECOMP_ENABLE_PERF_COUNTERS
#define DSDECOMP_ENABLE_PERF_COUNTERS 1
#endif

namespace dsdecomp {

struct SampleAggregate {
    std::uint64_t count = 0;
    double sum = 0.0;
    double min = 0.0;
    double max = 0.0;

    void add(double value) noexcept {
        if (count == 0) min = max = value;
        else { if (value < min) min = value; if (value > max) max = value; }
        ++count;
        sum += value;
    }
};

struct NeighborByteCounter {
    int peer_rank = -1;
    std::uint64_t bytes_sent = 0;
    std::uint64_t bytes_received = 0;
};

struct PerfStats {
    std::uint64_t halo_calls = 0;
    std::uint64_t migration_calls = 0;
    std::uint64_t bytes_sent = 0;
    std::uint64_t bytes_received = 0;
    std::uint64_t buffer_reallocations = 0;
    std::uint64_t peak_communication_buffer_capacity = 0;

    SampleAggregate pack_ms;
    SampleAggregate communication_ms;
    SampleAggregate wait_ms;
    SampleAggregate unpack_ms;
    SampleAggregate migration_classify_ms;
    SampleAggregate migration_compact_ms;

    // Rank-local optional detail; explicit global aggregation omits this vector.
    std::vector<NeighborByteCounter> neighbor_bytes;
};

} // namespace dsdecomp
