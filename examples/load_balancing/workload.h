// workload.h -- Particle placement and motion, built on random.h.
#pragma once

#include "random.h"
#include <cmath>

// ---- Block placement: one rank holds one cube of edge side, and the particle density is 1.
// The box has open boundaries. A particle that leaves the box is reflected back.
CPU_AND_GPU inline double block_position(long long id, int axis, double lower, double side)
{
    return lower + side * random_start(id, axis);
}

CPU_AND_GPU inline double move_inside(double x, long long id, long long step, int axis, double move, double edge)
{
    x += move * random_move(id, step, axis);
    if (x < 0) x = -x;
    if (x >= edge) x = 2 * edge - x;
    if (x < 0) x = 0;
    if (x >= edge) x = edge * (1.0 - 1e-12);
    return x;
}

// ---- Cluster placement: a part of the particles forms one cluster (Gaussian), and the others are uniformly distributed.
CPU_AND_GPU inline double gaussian(long long id, int axis)
{
    double u1 = random01(6ULL * id + 2 * axis) + 1e-12;
    double u2 = random01(6ULL * id + 2 * axis + 1);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

// A particle with id < blob_count belongs to the cluster (around center), the others have fixed positions. Periodic wrapping puts them in the box.
CPU_AND_GPU inline double blob_position(long long id, int axis, double edge, double center, double sigma, long long blob_count)
{
    double p = id < blob_count ? center + sigma * gaussian(id, axis) : edge * random_start(id, axis);
    return p - edge * floor(p / edge);
}
