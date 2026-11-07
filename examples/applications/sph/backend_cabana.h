// backend_cabana.h -- the Cabana part of main.cpp. The particles are stored in a Cabana AoSoA, and migration and halo use Cabana.
//
//   Cabana::Distributor + Cabana::migrate       migrate
//   build Cabana::Halo + Gather::apply          build the halo (sends the whole particle)
//   Gather::apply (one position slice)          refresh the halo (sends only the positions)
//   Gather::apply (whole particle)              second transfer in every step, after the density pass (sends the density, which is in
//                                               the charge field, and the velocity)
// On the GPU the Gather objects are created once and reused. The function Cabana::gather() allocates new send and receive buffers
// in every call. With GPU-aware Cray MPICH, every new device buffer needs a new IPC handle, and then the halo refresh on 4 GPUs takes
// close to 100 ms (the Cabana documentation also recommends a Gather that keeps its buffers).
// On the CPU the function gather is used. A reused Gather on the CPU with 2 or more ranks gave NaN positions after about 50 steps
// (we did not find the cause). The CPU has no IPC, so new buffers cost nothing there.
//
// Cabana sends what it is told to send: it does not cut the domain, find the new owner of a particle, or select the boundary
// particles. Those parts are application code (common.h) and are included in the times.
// A particle has 96 bytes (position, velocity, force, mass, charge, type, id), the same as the dsDecompX particle record.
//
// Kokkos is built with OpenMP for the CPU and with CUDA for the GPU.
// Run: srun -n 16 ./sph_cabana --kokkos-map-device-id-by=mpi_rank particles_total=10000000
#pragma once

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#define SIM_ATOMIC_ADD(pointer, value) Kokkos::atomic_fetch_add(pointer, value)
#include "common.h"

#include <cstdlib>
#include <memory>
#include <type_traits>
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using DataTypes = Cabana::MemberTypes<double[3], double[3], double[3], double, double, int, int>;
using ParticleList = Cabana::AoSoA<DataTypes, MemSpace>;
using IntView = Kokkos::View<int*, MemSpace>;
using HaloType = Cabana::Halo<MemSpace>;
enum { POSITION, VELOCITY, FORCE, MASS, CHARGE, TYPE, ID };

template <class Slice>
struct SlicePos {
    Slice slice;
    KOKKOS_INLINE_FUNCTION double operator()(int i, int axis) const { return slice(i, axis); }
};
template <class Slice>
static SlicePos<Slice> make_pos(Slice slice) { return SlicePos<Slice>{slice}; }

template <class Slice>
struct SliceScalar {
    Slice slice;
    KOKKOS_INLINE_FUNCTION double operator()(int i) const { return slice(i); }
};
template <class Slice>
static SliceScalar<Slice> make_scalar(Slice slice) { return SliceScalar<Slice>{slice}; }

static void place_particles(ParticleList& particles, const HostParticles& host)
{
    Cabana::AoSoA<DataTypes, Kokkos::HostSpace> mirror("mirror", host.id.size());
    auto position = Cabana::slice<POSITION>(mirror);
    auto velocity = Cabana::slice<VELOCITY>(mirror);
    auto force = Cabana::slice<FORCE>(mirror);
    auto mass = Cabana::slice<MASS>(mirror);
    auto charge = Cabana::slice<CHARGE>(mirror);
    auto id = Cabana::slice<ID>(mirror);
    for (size_t i = 0; i < host.id.size(); i++) {
        position(i, 0) = host.x[i]; position(i, 1) = host.y[i]; position(i, 2) = host.z[i];
        velocity(i, 0) = host.vx[i]; velocity(i, 1) = host.vy[i]; velocity(i, 2) = host.vz[i];
        force(i, 0) = force(i, 1) = force(i, 2) = 0.0;
        mass(i) = 1.0;
        charge(i) = 0.0;
        id(i) = host.id[i];
    }
    Cabana::deep_copy(particles, mirror);
}

static void kick_drift_all(ParticleList& particles, int count, const SimParams& params, double edge)
{
    auto position = Cabana::slice<POSITION>(particles);
    auto velocity = Cabana::slice<VELOCITY>(particles);
    auto force = Cabana::slice<FORCE>(particles);
    double dt = params.dt;
    Kokkos::parallel_for("kick_drift", Kokkos::RangePolicy<ExecSpace>(0, count), KOKKOS_LAMBDA(const int i) {
        for (int a = 0; a < 3; a++) {
            velocity(i, a) += 0.5 * dt * force(i, a);
            position(i, a) += dt * velocity(i, a);
            double x = position(i, a), v = velocity(i, a);
            sim_reflect(x, v, edge);
            position(i, a) = x;
            velocity(i, a) = v;
        }
    });
    Kokkos::fence();
}

static void kick_all(ParticleList& particles, int count, const SimParams& params)
{
    auto velocity = Cabana::slice<VELOCITY>(particles);
    auto force = Cabana::slice<FORCE>(particles);
    double dt = params.dt;
    Kokkos::parallel_for("kick", Kokkos::RangePolicy<ExecSpace>(0, count), KOKKOS_LAMBDA(const int i) {
        for (int a = 0; a < 3; a++) velocity(i, a) += 0.5 * dt * force(i, a);
    });
    Kokkos::fence();
}

// migrate: compute the owners, create the Distributor, send
static void migrate_particles(ParticleList& particles, int count, const SimGrid& grid, int rank, const std::vector<int>& neighbors)
{
    auto position = Cabana::slice<POSITION>(particles);
    IntView export_ranks("export_ranks", count);
    Kokkos::parallel_for("owner", Kokkos::RangePolicy<ExecSpace>(0, count), KOKKOS_LAMBDA(const int i) {
        export_ranks(i) = sim_owner(grid, position(i, 0), position(i, 1), position(i, 2));
    });
    std::vector<int> topology = neighbors;                    // Cabana also wants the rank itself in the neighbor list
    topology.push_back(rank);
    std::sort(topology.begin(), topology.end());
    Cabana::Distributor<MemSpace> distributor(MPI_COMM_WORLD, export_ranks, topology);
    Cabana::migrate(distributor, particles);
}

// Build the halo: collect the (particle, neighbor rank) pairs of the ghosts, create the Halo, and make room for the ghosts.
// The ghosts themselves are sent by a Gather.
static std::unique_ptr<HaloType> build_halo(ParticleList& particles, int count, const SimGrid& grid, int rank, const std::vector<int>& neighbors)
{
    auto position = Cabana::slice<POSITION>(particles);
    IntView targets("targets", count), offsets("offsets", count);
    Kokkos::parallel_for("count", Kokkos::RangePolicy<ExecSpace>(0, count), KOKKOS_LAMBDA(const int i) {
        targets(i) = sim_ghost_targets(grid, rank, position(i, 0), position(i, 1), position(i, 2), nullptr);
    });
    int exports = 0;
    Kokkos::parallel_scan("offsets", Kokkos::RangePolicy<ExecSpace>(0, count),
                          KOKKOS_LAMBDA(const int i, int& sum, const bool final_pass) {
                              if (final_pass) offsets(i) = sum;
                              sum += targets(i);
                          }, exports);
    IntView export_ids("export_ids", exports), export_ranks("export_ranks", exports);
    Kokkos::parallel_for("fill", Kokkos::RangePolicy<ExecSpace>(0, count), KOKKOS_LAMBDA(const int i) {
        int out[26];
        int n = sim_ghost_targets(grid, rank, position(i, 0), position(i, 1), position(i, 2), out);
        for (int k = 0; k < n; k++) {
            export_ids(offsets(i) + k) = i;
            export_ranks(offsets(i) + k) = out[k];
        }
    });
    Kokkos::fence();
    std::unique_ptr<HaloType> halo(new HaloType(MPI_COMM_WORLD, count, export_ids, export_ranks, neighbors));
    particles.resize(count + halo->numGhost());
    return halo;
}

struct NeighborData {
    IntView cells, members, neighbors, num_neighbors, flag;
};

static void grow(IntView& view, size_t n, const char* name)
{
    if (view.extent(0) >= n) return;
    view = IntView(std::string(name), std::max(n, view.extent(0) + view.extent(0) / 4));
}

static void build_neighbor_lists(ParticleList& particles, int owned, const double lower[3], const double upper[3], const SimParams& params,
                                 NeighborData& data)
{
    int all = particles.size();
    CellGrid grid = make_cell_grid(lower, upper, SIM_LIST_CUTOFF);
    size_t cell_total = (size_t)cell_count(grid);
    grow(data.cells, cell_total, "cells");
    grow(data.members, cell_total * SIM_CELL_CAPACITY, "members");
    grow(data.neighbors, (size_t)owned * SIM_MAX_NEIGHBORS, "neighbors");
    grow(data.num_neighbors, owned + 1, "num_neighbors");
    grow(data.flag, 1, "flag");
    Kokkos::deep_copy(Kokkos::subview(data.cells, Kokkos::make_pair((size_t)0, cell_total)), 0);
    Kokkos::deep_copy(data.flag, 0);
    auto pos = make_pos(Cabana::slice<POSITION>(particles));
    int *count = data.cells.data(), *members = data.members.data(), *neighbors = data.neighbors.data();
    int *num_neighbors = data.num_neighbors.data(), *overflow = data.flag.data();
    double list2 = params.list2;
    Kokkos::parallel_for("insert", Kokkos::RangePolicy<ExecSpace>(0, all), KOKKOS_LAMBDA(const int i) {
        sim_insert(grid, pos, i, count, members, overflow);
    });
    Kokkos::parallel_for("find", Kokkos::RangePolicy<ExecSpace>(0, owned), KOKKOS_LAMBDA(const int i) {
        sim_find_neighbors(grid, pos, i, list2, count, members, neighbors, num_neighbors, overflow);
    });
    Kokkos::fence();
    auto flag_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.flag);
    if (flag_host(0)) {
        printf("a cell or a neighbor list overflowed (SIM_CELL_CAPACITY, SIM_MAX_NEIGHBORS)\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Density of the owned particles (stored in the charge field)
static void compute_densities(ParticleList& particles, int owned, const SimParams& params, NeighborData& data)
{
    auto pos = make_pos(Cabana::slice<POSITION>(particles));
    auto density = Cabana::slice<CHARGE>(particles);
    const int *neighbors = data.neighbors.data(), *num_neighbors = data.num_neighbors.data();
    Kokkos::parallel_for("density", Kokkos::RangePolicy<ExecSpace>(0, owned), KOKKOS_LAMBDA(const int i) {
        density(i) = sim_density(params, pos, i, neighbors, num_neighbors);
    });
    Kokkos::fence();
}

static void compute_forces_all(ParticleList& particles, int owned, const SimParams& params, NeighborData& data)
{
    auto pos = make_pos(Cabana::slice<POSITION>(particles));
    auto vel = make_pos(Cabana::slice<VELOCITY>(particles));
    auto density = make_scalar(Cabana::slice<CHARGE>(particles));
    auto force = Cabana::slice<FORCE>(particles);
    const int *neighbors = data.neighbors.data(), *num_neighbors = data.num_neighbors.data();
    Kokkos::parallel_for("forces", Kokkos::RangePolicy<ExecSpace>(0, owned), KOKKOS_LAMBDA(const int i) {
        double f[3], e;
        sim_force(params, pos, vel, density, i, neighbors, num_neighbors, f, e);
        force(i, 0) = f[0];
        force(i, 1) = f[1];
        force(i, 2) = f[2];
    });
    Kokkos::fence();
}

// Density sum and kinetic energy sum of the owned particles of this rank
static void sum_energies(ParticleList& particles, int count, double out[2])
{
    auto velocity = Cabana::slice<VELOCITY>(particles);
    auto energy = Cabana::slice<CHARGE>(particles);
    double pe = 0, ke = 0;
    Kokkos::parallel_reduce("pe", Kokkos::RangePolicy<ExecSpace>(0, count),
                            KOKKOS_LAMBDA(const int i, double& sum) { sum += energy(i); }, pe);
    Kokkos::parallel_reduce("ke", Kokkos::RangePolicy<ExecSpace>(0, count), KOKKOS_LAMBDA(const int i, double& sum) {
        sum += 0.5 * (velocity(i, 0) * velocity(i, 0) + velocity(i, 1) * velocity(i, 1) + velocity(i, 2) * velocity(i, 2));
    }, ke);
    out[0] = pe;
    out[1] = ke;
}

using PositionSlice = decltype(Cabana::slice<POSITION>(std::declval<ParticleList&>()));
using GatherAll = Cabana::Gather<HaloType, ParticleList>;
using GatherPosition = Cabana::Gather<HaloType, PositionSlice>;

struct Backend {
    static const char* name() { return "cabana"; }
    static void add_settings(Settings&) {}
    // Selects the GPU before MPI_Init (the GPU that a GPU-aware MPI sees at init must be the GPU that is used; otherwise the direct
    // GPU-to-GPU transfers are switched off).
    static void before_mpi(int, char**)
    {
#if defined(KOKKOS_ENABLE_CUDA)
        if (const char* local = std::getenv("SLURM_LOCALID")) cudaSetDevice(std::atoi(local) % 4);
#endif
    }

    Kokkos::ScopeGuard kokkos;                                // created first and destroyed last
    SimCase problem;
    SimGrid grid;
    std::vector<int> rank_neighbors;
    double lower[3], upper[3], frame_lower[3], frame_upper[3];
    ParticleList particles;
    int owned = 0;                                            // number of owned particles (the ghosts follow them)
    std::unique_ptr<HaloType> halo;
    std::unique_ptr<GatherAll> gather_all;                    // Gathers that keep their buffers (whole particle, positions only)
    std::unique_ptr<GatherPosition> gather_position;
    NeighborData data;

    Backend(const SimCase& p, int argc, char** argv) : kokkos(argc, argv), problem(p), particles("particles", 0)
    {
        for (int a = 0; a < 3; a++) {
            grid.dims[a] = p.dims[a];
            grid.edge[a] = p.lattice.edge;
        }
        grid.halo = SIM_LIST_CUTOFF;
        rank_neighbors = sim_neighbor_ranks(grid, p.rank);
        int c[3];
        sim_block_coordinates(grid, p.rank, c);
        for (int a = 0; a < 3; a++) {
            lower[a] = sim_cut(grid, a, c[a]);
            upper[a] = sim_cut(grid, a, c[a] + 1);
            frame_lower[a] = lower[a] - SIM_LIST_CUTOFF;
            frame_upper[a] = upper[a] + SIM_LIST_CUTOFF;
        }
    }

    bool on_gpu() const { return !std::is_same<MemSpace, Kokkos::HostSpace>::value; }
    void sync() { Kokkos::fence(); }
    unsigned long long ghost_count() const { return halo ? halo->numGhost() : 0; }
    void bounds(double l[3], double u[3]) const
    {
        for (int a = 0; a < 3; a++) {
            l[a] = lower[a];
            u[a] = upper[a];
        }
    }

    void place(const HostParticles& host)
    {
        particles.resize(host.id.size());
        place_particles(particles, host);
        owned = particles.size();
    }

    void kick_drift() { kick_drift_all(particles, owned, problem.params, problem.lattice.edge); }
    void kick() { kick_all(particles, owned, problem.params); }

    void migrate()
    {
        particles.resize(owned);                              // drop the ghosts of the last step
        migrate_particles(particles, owned, grid, problem.rank, rank_neighbors);
        owned = particles.size();
    }

    void halo_build()
    {
        halo = build_halo(particles, owned, grid, problem.rank, rank_neighbors);
        PositionSlice position = Cabana::slice<POSITION>(particles);
        if (!on_gpu()) {                                      // CPU: function gather (see the comment at the top)
            Cabana::gather(*halo, particles);
            return;
        }
        if (!gather_all) {
            gather_all.reset(new GatherAll(*halo, particles, 1.2));
            gather_position.reset(new GatherPosition(*halo, position, 1.2));
        } else {
            gather_all->reserve(*halo, particles, 1.2);       // allocates again only if the buffers are too small
            gather_position->reserve(*halo, position, 1.2);
        }
        gather_all->apply();
    }

    bool halo_refresh()
    {
        if (!on_gpu()) {
            PositionSlice position = Cabana::slice<POSITION>(particles);
            Cabana::gather(*halo, position);
            return false;
        }
        gather_position->apply();
        return false;
    }

    void build_neighbors() { build_neighbor_lists(particles, owned, frame_lower, frame_upper, problem.params, data); }
    void compute_density() { compute_densities(particles, owned, problem.params, data); }

    // Second transfer: the whole particle, with the density
    void density_exchange()
    {
        if (!on_gpu()) Cabana::gather(*halo, particles);
        else gather_all->apply();
    }

    void compute_forces() { compute_forces_all(particles, owned, problem.params, data); }

    void observe(double& potential, double& kinetic)
    {
        double local[2], sum[2];
        sum_energies(particles, owned, local);
        MPI_Allreduce(local, sum, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        potential = sum[0] / problem.lattice.total;
        kinetic = sum[1] / problem.lattice.total;
    }
};
