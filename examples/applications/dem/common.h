// common.h -- everything the dsDecompX, Cabana, and AMReX programs share: physics, initial state, cell and neighbor lists, force,
// the rank grid used by Cabana and AMReX, and result reporting.
//
// Application: discrete element method (DEM) of equal soft spheres. Two spheres whose distance r is below the diameter 1 push each
// other apart along their line of centers with a spring and a damper: force = k d - g (v_ij . n), where d = 1 - r is the overlap, n the
// unit vector from sphere j to sphere i, v_ij the relative velocity, k = 500, and g = 2 (a negative value is set to 0). The spheres have
// mass 1 and start on a simple cubic lattice with spacing 1 (they only touch) with normally distributed velocities (standard
// deviation 0.3) in a box with reflecting walls. There is no rotation, friction, or gravity.
// Forces come from a Verlet neighbor list with a skin of 0.3 (the neighbor list radius is 1.3). The list holds every pair twice, so no
// reverse reduction is needed. The list and the halo are rebuilt every `rebuild_every` steps.
//
// The force uses the velocity of the neighbor, so the velocities of the ghost particles must be current in every step. The programs
// therefore refresh the whole ghost particle in every step, not only its position.
//
// The three programs solve the same problem. Only the particle storage and the communication (migration and halo exchange) differ.
#pragma once

#include "random.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#define SIM_HD CPU_AND_GPU

// Atomic add used to insert particles into cells. The Kokkos and AMReX programs define it before including this header.
#ifndef SIM_ATOMIC_ADD
#ifdef __CUDA_ARCH__
#define SIM_ATOMIC_ADD(pointer, value) atomicAdd(pointer, value)
#else
inline int sim_host_add(int* pointer, int value)
{
    int old = *pointer;
    *pointer += value;
    return old;
}
#define SIM_ATOMIC_ADD(pointer, value) sim_host_add(pointer, value)
#endif
#endif

typedef std::map<std::string, double> Settings;

// Overrides defaults with key=value arguments. Unknown names stop the run. Arguments that start with -- (Kokkos) are skipped.
inline void parse_settings(int argc, char** argv, Settings& setting, int rank)
{
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') continue;
        char name[64];
        double value;
        if (sscanf(argv[i], "%63[^=]=%lf", name, &value) != 2 || setting.count(name) == 0) {
            if (rank == 0) printf("unknown setting: %s\n", argv[i]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        setting[name] = value;
    }
}

// ---- Physics
const double SIM_DENSITY = 1.0;                            // lattice spacing 1
const double SIM_TEMPERATURE = 0.09;                       // variance of the initial velocity (standard deviation 0.3)
const double SIM_DIAMETER = 1.0;
const double SIM_STIFFNESS = 500.0;
const double SIM_DAMPING = 2.0;
const double SIM_SKIN = 0.3;
const double SIM_LIST_CUTOFF = SIM_DIAMETER + SIM_SKIN;    // neighbor list radius and halo width
const double SIM_DT = 0.005;
enum {
    SIM_MAX_NEIGHBORS = 32,                                // neighbor list row length (a sphere touches 6 neighbors)
    SIM_CELL_CAPACITY = 32                                 // particles per cell (mean is about 2 to 5)
};

// Constants passed to the kernels by value.
struct SimParams {
    double cutoff2, list2, stiffness, damping, dt;
};

inline SimParams sim_params()
{
    SimParams p;
    p.cutoff2 = SIM_DIAMETER * SIM_DIAMETER;
    p.list2 = SIM_LIST_CUTOFF * SIM_LIST_CUTOFF;
    p.stiffness = SIM_STIFFNESS;
    p.damping = SIM_DAMPING;
    p.dt = SIM_DT;
    return p;
}

// ---- Initial state: simple cubic lattice of n x n x n points, shifted by half a spacing away from the walls.
struct Lattice {
    int n;
    double a;                 // lattice constant
    double edge;              // edge of the cubic box
    long long total;
};

inline Lattice make_lattice(long long target)
{
    Lattice lattice;
    lattice.n = std::max(1, (int)std::llround(std::cbrt((double)target)));
    lattice.a = std::cbrt(1.0 / SIM_DENSITY);
    lattice.edge = lattice.n * lattice.a;
    lattice.total = (long long)lattice.n * lattice.n * lattice.n;
    return lattice;
}

// Normally distributed velocity from the particle id (the same on every rank and in every library).
inline double sim_velocity(long long id, int axis)
{
    unsigned long long key = (7ULL << 56) + 6ULL * id + 2 * axis;
    double u1 = random01(key) + 1e-12;
    double u2 = random01(key + 1);
    return std::sqrt(SIM_TEMPERATURE) * std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

struct HostParticles {
    std::vector<double> x, y, z, vx, vy, vz;
    std::vector<int> id;
};

// Creates all lattice points inside [lower, upper). Every rank scans only its own region, so no communication is needed.
inline void generate_block(const Lattice& lattice, const double lower[3], const double upper[3], HostParticles& out)
{
    int first[3], last[3];
    for (int a = 0; a < 3; a++) {
        first[a] = std::max(0, (int)std::floor(lower[a] / lattice.a) - 1);
        last[a] = std::min(lattice.n - 1, (int)std::ceil(upper[a] / lattice.a));
    }
    double volume = (upper[0] - lower[0]) * (upper[1] - lower[1]) * (upper[2] - lower[2]);
    size_t estimate = (size_t)(volume * SIM_DENSITY * 1.05) + 16;
    out.x.reserve(estimate); out.y.reserve(estimate); out.z.reserve(estimate);
    out.vx.reserve(estimate); out.vy.reserve(estimate); out.vz.reserve(estimate); out.id.reserve(estimate);
    for (int i = first[0]; i <= last[0]; i++)
        for (int j = first[1]; j <= last[1]; j++)
            for (int k = first[2]; k <= last[2]; k++) {
                double x = (i + 0.5) * lattice.a;
                double y = (j + 0.5) * lattice.a;
                double z = (k + 0.5) * lattice.a;
                if (x < lower[0] || x >= upper[0] || y < lower[1] || y >= upper[1] || z < lower[2] || z >= upper[2]) continue;
                long long id = ((long long)i * lattice.n + j) * lattice.n + k;
                out.x.push_back(x); out.y.push_back(y); out.z.push_back(z);
                out.vx.push_back(sim_velocity(id, 0)); out.vy.push_back(sim_velocity(id, 1)); out.vz.push_back(sim_velocity(id, 2));
                out.id.push_back((int)id);
            }
}

// Reflects a particle at the box walls.
SIM_HD inline void sim_reflect(double& x, double& v, double edge)
{
    if (x < 0) {
        x = -x;
        v = -v;
    } else if (x >= edge) {
        x = 2 * edge - x;
        v = -v;
    }
}

// ---- Uniform rank grid (used by Cabana and AMReX; dsDecompX has its own grid)
struct SimGrid {
    int dims[3];
    double edge[3];           // whole box
    double halo;
};

SIM_HD inline double sim_cut(const SimGrid& g, int axis, int c)
{
    return c >= g.dims[axis] ? g.edge[axis] : g.edge[axis] * c / g.dims[axis];
}

SIM_HD inline int sim_block_rank(const SimGrid& g, int c0, int c1, int c2)
{
    return (c0 * g.dims[1] + c1) * g.dims[2] + c2;
}

SIM_HD inline void sim_block_coordinates(const SimGrid& g, int rank, int c[3])
{
    c[2] = rank % g.dims[2];
    c[1] = (rank / g.dims[2]) % g.dims[1];
    c[0] = rank / (g.dims[1] * g.dims[2]);
}

// Rank that owns a position. It uses the same expression as the cuts.
SIM_HD inline int sim_owner(const SimGrid& g, double x, double y, double z)
{
    double p[3] = {x, y, z};
    int c[3];
    for (int a = 0; a < 3; a++) {
        int k = (int)floor(p[a] * g.dims[a] / g.edge[a]);
        k = k < 0 ? 0 : (k >= g.dims[a] ? g.dims[a] - 1 : k);
        while (k > 0 && p[a] < sim_cut(g, a, k)) k--;
        while (k < g.dims[a] - 1 && p[a] >= sim_cut(g, a, k + 1)) k++;
        c[a] = k;
    }
    return sim_block_rank(g, c[0], c[1], c[2]);
}

// Writes the neighbor ranks that receive a particle as a ghost into out (at most 26) and returns their number.
// With out == nullptr it only counts.
SIM_HD inline int sim_ghost_targets(const SimGrid& g, int rank, double x, double y, double z, int* out)
{
    int c[3];
    sim_block_coordinates(g, rank, c);
    double p[3] = {x, y, z};
    bool low[3], high[3];
    for (int a = 0; a < 3; a++) {
        low[a] = c[a] > 0 && p[a] < sim_cut(g, a, c[a]) + g.halo;
        high[a] = c[a] < g.dims[a] - 1 && p[a] >= sim_cut(g, a, c[a] + 1) - g.halo;
    }
    int count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        if ((dx < 0 && !low[0]) || (dx > 0 && !high[0])) continue;
        for (int dy = -1; dy <= 1; dy++) {
            if ((dy < 0 && !low[1]) || (dy > 0 && !high[1])) continue;
            for (int dz = -1; dz <= 1; dz++) {
                if ((dz < 0 && !low[2]) || (dz > 0 && !high[2])) continue;
                if (dx == 0 && dy == 0 && dz == 0) continue;
                if (out) out[count] = sim_block_rank(g, c[0] + dx, c[1] + dy, c[2] + dz);
                count++;
            }
        }
    }
    return count;
}

// Neighbor ranks (without the rank itself, no duplicates, sorted).
inline std::vector<int> sim_neighbor_ranks(const SimGrid& g, int rank)
{
    int c[3];
    sim_block_coordinates(g, rank, c);
    std::vector<int> list;
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++) {
                int n0 = c[0] + dx, n1 = c[1] + dy, n2 = c[2] + dz;
                if (n0 < 0 || n1 < 0 || n2 < 0 || n0 >= g.dims[0] || n1 >= g.dims[1] || n2 >= g.dims[2]) continue;
                int r = sim_block_rank(g, n0, n1, n2);
                if (r != rank) list.push_back(r);
            }
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
    return list;
}

// ---- Cell list and neighbor list. Positions are read with pos(i, axis). Owned particles come first, ghosts after them.
struct CellGrid {
    double lower[3];
    double inverse_width[3];
    int n[3];
};

inline CellGrid make_cell_grid(const double lower[3], const double upper[3], double min_width)
{
    CellGrid g;
    for (int a = 0; a < 3; a++) {
        double extent = upper[a] - lower[a];
        g.n[a] = std::max(1, (int)std::floor(extent / min_width));
        g.lower[a] = lower[a];
        g.inverse_width[a] = g.n[a] / extent;
    }
    return g;
}

inline long long cell_count(const CellGrid& g) { return (long long)g.n[0] * g.n[1] * g.n[2]; }

SIM_HD inline int sim_cell_coordinate(const CellGrid& g, double p, int axis)
{
    int c = (int)floor((p - g.lower[axis]) * g.inverse_width[axis]);
    return c < 0 ? 0 : (c >= g.n[axis] ? g.n[axis] - 1 : c);
}

// Puts particle i into its cell. count starts at zero. A full cell sets overflow.
template <class Pos>
SIM_HD inline void sim_insert(const CellGrid& g, Pos pos, int i, int* count, int* members, int* overflow)
{
    int cx = sim_cell_coordinate(g, pos(i, 0), 0);
    int cy = sim_cell_coordinate(g, pos(i, 1), 1);
    int cz = sim_cell_coordinate(g, pos(i, 2), 2);
    long long cell = ((long long)cx * g.n[1] + cy) * g.n[2] + cz;
    int slot = SIM_ATOMIC_ADD(&count[cell], 1);
    if (slot < SIM_CELL_CAPACITY) members[cell * SIM_CELL_CAPACITY + slot] = i;
    else *overflow = 1;
}

// Writes the neighbors (owned and ghost) of owned particle i into the list. A list that is too long sets overflow.
template <class Pos>
SIM_HD inline void sim_find_neighbors(const CellGrid& g, Pos pos, int i, double list2, const int* count, const int* members,
                                      int* neighbors, int* num_neighbors, int* overflow)
{
    double x = pos(i, 0), y = pos(i, 1), z = pos(i, 2);
    int cx = sim_cell_coordinate(g, x, 0), cy = sim_cell_coordinate(g, y, 1), cz = sim_cell_coordinate(g, z, 2);
    int* row = neighbors + (long long)i * SIM_MAX_NEIGHBORS;
    int found = 0;
    for (int dx = -1; dx <= 1; dx++) {
        int nx = cx + dx;
        if (nx < 0 || nx >= g.n[0]) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int ny = cy + dy;
            if (ny < 0 || ny >= g.n[1]) continue;
            for (int dz = -1; dz <= 1; dz++) {
                int nz = cz + dz;
                if (nz < 0 || nz >= g.n[2]) continue;
                long long cell = ((long long)nx * g.n[1] + ny) * g.n[2] + nz;
                int m = count[cell] < SIM_CELL_CAPACITY ? count[cell] : SIM_CELL_CAPACITY;
                const int* member = members + cell * SIM_CELL_CAPACITY;
                for (int k = 0; k < m; k++) {
                    int j = member[k];
                    if (j == i) continue;
                    double ddx = x - pos(j, 0), ddy = y - pos(j, 1), ddz = z - pos(j, 2);
                    if (ddx * ddx + ddy * ddy + ddz * ddz < list2) {
                        if (found < SIM_MAX_NEIGHBORS) row[found] = j;
                        found++;
                    }
                }
            }
        }
    }
    num_neighbors[i] = found;
    if (found > SIM_MAX_NEIGHBORS) *overflow = 1;
}

// Force on owned particle i and its contact energy (half of each pair). vel(j, axis) is the velocity of owned and ghost particles.
template <class Pos, class Vel>
SIM_HD inline void sim_force(const SimParams& p, Pos pos, Vel vel, int i, const int* neighbors, const int* num_neighbors, double f[3],
                             double& energy)
{
    double x = pos(i, 0), y = pos(i, 1), z = pos(i, 2);
    double vx = vel(i, 0), vy = vel(i, 1), vz = vel(i, 2);
    const int* row = neighbors + (long long)i * SIM_MAX_NEIGHBORS;
    int m = num_neighbors[i] < SIM_MAX_NEIGHBORS ? num_neighbors[i] : SIM_MAX_NEIGHBORS;
    double fx = 0, fy = 0, fz = 0, e = 0;
    for (int k = 0; k < m; k++) {
        int j = row[k];
        double dx = x - pos(j, 0), dy = y - pos(j, 1), dz = z - pos(j, 2);
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= p.cutoff2 || r2 < 1e-24) continue;
        double r = sqrt(r2);
        double overlap = SIM_DIAMETER - r;
        double inverse = 1.0 / r;
        double nx = dx * inverse, ny = dy * inverse, nz = dz * inverse;
        double normal_speed = (vx - vel(j, 0)) * nx + (vy - vel(j, 1)) * ny + (vz - vel(j, 2)) * nz;
        double magnitude = p.stiffness * overlap - p.damping * normal_speed;
        if (magnitude < 0) magnitude = 0;
        fx += magnitude * nx;
        fy += magnitude * ny;
        fz += magnitude * nz;
        e += 0.25 * p.stiffness * overlap * overlap;
    }
    f[0] = fx; f[1] = fy; f[2] = fz;
    energy = e;
}

// ---- Problem description that main.cpp hands to the Backend
struct SimCase {
    const Settings* setting;
    Lattice lattice;
    int dims[3];              // rank grid
    SimParams params;
    int rank, ranks;
};

// ---- Result. All programs print the same line.
enum SimSection { SEC_INTEGRATE, SEC_MIGRATE, SEC_HALO_BUILD, SEC_HALO_REFRESH, SEC_NEIGHBOR, SEC_FORCE, SEC_COUNT };

struct SimResult {
    const char* library;
    int gpu, ranks, dims[3];
    long long particles;
    int rebuild_every, warmup, steps, breakdown_steps;
    double step_ms;                       // mean over all steps, timed once for the whole loop (no barriers inside)
    double section_ms[SEC_COUNT];         // mean per step of each section (timed with a barrier around it)
    double ghost_per_owned;
    double neighbor_builds_per_step;
    double potential0, kinetic0, potential1, kinetic1;   // energy per particle at the start and after the timed steps
};

inline void print_result(const SimResult& r)
{
    double e0 = (r.potential0 + r.kinetic0), e1 = (r.potential1 + r.kinetic1);
    printf("RESULT app=dem lib=%s gpu=%d ranks=%d grid=%dx%dx%d particles=%lld particles_per_rank=%.0f rebuild_every=%d warmup=%d steps=%d "
           "breakdown_steps=%d step_ms=%.4f integrate_ms=%.4f migrate_ms=%.4f halo_build_ms=%.4f halo_refresh_ms=%.4f "
           "neighbor_ms=%.4f force_ms=%.4f ghost_per_owned=%.4f neighbor_builds_per_step=%.4f "
           "pe0=%.8f ke0=%.8f e0=%.8f pe1=%.8f ke1=%.8f e1=%.8f\n",
           r.library, r.gpu, r.ranks, r.dims[0], r.dims[1], r.dims[2], r.particles, (double)r.particles / r.ranks,
           r.rebuild_every, r.warmup, r.steps, r.breakdown_steps, r.step_ms, r.section_ms[SEC_INTEGRATE], r.section_ms[SEC_MIGRATE],
           r.section_ms[SEC_HALO_BUILD], r.section_ms[SEC_HALO_REFRESH], r.section_ms[SEC_NEIGHBOR], r.section_ms[SEC_FORCE],
           r.ghost_per_owned, r.neighbor_builds_per_step, r.potential0, r.kinetic0, e0, r.potential1, r.kinetic1, e1);
}

// Collects the time of each section (the slowest rank, averaged over the steps). Each step adds its local time to local_ms,
// and finish() reduces them.
struct SectionTimer {
    double local_ms[SEC_COUNT];
    SectionTimer() { for (int k = 0; k < SEC_COUNT; k++) local_ms[k] = 0; }
    void finish(double out_ms[SEC_COUNT], int steps)
    {
        double slowest[SEC_COUNT];
        MPI_Allreduce(local_ms, slowest, SEC_COUNT, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        for (int k = 0; k < SEC_COUNT; k++) out_ms[k] = steps > 0 ? slowest[k] / steps : -1;
    }
};

// Energy sums. Adds the velocities of the owned particles and the per-particle contact energy (kept in the charge field) over all
// ranks and returns the values per particle.
inline void sim_energy_sum(const double* vx, const double* vy, const double* vz, const double* energy, size_t count, long long total,
                           double& potential, double& kinetic)
{
    double local[2] = {0, 0};
    for (size_t i = 0; i < count; i++) {
        local[0] += energy[i];
        local[1] += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }
    double sum[2];
    MPI_Allreduce(local, sum, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    potential = sum[0] / total;
    kinetic = sum[1] / total;
}
