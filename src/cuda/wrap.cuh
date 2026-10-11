#pragma once

#include <cmath>

namespace dsdecompx::cuda {

// Maps value into [lower, upper) on a periodic axis. floor()-based wrapping can
// round a tiny negative offset up to exactly `extent`, so fold that case back
// to `lower` to keep the half-open interval invariant.
__device__ inline double wrap_periodic(double value, double lower, double upper,
                                       bool periodic) {
    if (!periodic) return value;
    const double extent = upper - lower;
    double shifted = value - lower;
    shifted -= floor(shifted / extent) * extent;
    if (shifted >= extent) shifted = 0.0;
    return shifted + lower;
}

} // namespace dsdecompx::cuda
