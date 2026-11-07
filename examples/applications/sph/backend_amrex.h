// backend_amrex.h -- the AMReX part of main.cpp. The particles are stored in an AMReX NeighborParticleContainer, and migration and
// halo use AMReX.
//
//   Redistribute()        migrate
//   fillNeighbors()       build the halo
//   updateNeighbors()     refresh the halo (the whole particle, also for the second transfer in every step, after the density pass)
//
// Each rank owns one Box of cells whose width is at least the halo width (2.8). One layer of neighbor cells (nneighbor = 1) becomes the
// ghosts, so a cell width above the halo width gives more ghosts. A particle has 8 reals (velocity 3, force 3, mass, charge) and
// 2 integers (type, id). AMReX adds the position and its own id and cpu (104 bytes in total). The refresh cannot send only the
// positions, so both transfers of a step send the whole particle.
//
// CPU or GPU is decided when AMReX is built (-DAMReX_GPU_BACKEND=NONE or CUDA).
// Run: srun -n 16 ./sph_amrex particles_total=10000000
#pragma once

#include <AMReX.H>
#include <AMReX_NeighborParticles.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#define SIM_ATOMIC_ADD(pointer, value) amrex::Gpu::Atomic::Add(pointer, value)
#include "common.h"

#include <cstdlib>
#include <memory>
#if defined(AMREX_USE_CUDA)
#include <cuda_runtime.h>
#endif

typedef amrex::NeighborParticleContainer<8, 2> Container;
typedef Container::ParticleType ParticleType;
typedef amrex::ParIter<8, 2> ParticleIterator;       // iterator over the particles
enum { VX, VY, VZ, FX, FY, FZ, MASS, CHARGE };      // real components (rdata)
enum { TYPE, GLOBAL_ID };                           // integer components (idata)

// Reads the owned particles (first) and the ghosts (after them) with one index. In a GPU build the ghosts follow the owned particles
// of the same tile. In a CPU build they are stored separately.
struct AmrexPos {
    const ParticleType* owned_data;
    const ParticleType* ghost_data;
    int owned;
    AMREX_GPU_HOST_DEVICE double operator()(int i, int axis) const
    {
        return i < owned ? owned_data[i].pos(axis) : ghost_data[i - owned].pos(axis);
    }
};

static AmrexPos make_pos(Container& container, ParticleIterator& pti)
{
    AmrexPos p;
    p.owned = pti.numRealParticles();
    p.owned_data = pti.GetArrayOfStructs()().data();
#ifdef AMREX_USE_GPU
    p.ghost_data = p.owned_data + p.owned;
#else
    p.ghost_data = container.GetNeighbors(0, pti.index(), pti.LocalTileIndex()).GetArrayOfStructs()().data();
#endif
    return p;
}

// Reads a real component (rdata) of the owned particles and the ghosts with one index: v(i, axis) = rdata(comp + axis), v(i) = rdata(comp).
struct AmrexReal {
    const ParticleType* owned_data;
    const ParticleType* ghost_data;
    int owned, comp;
    AMREX_GPU_HOST_DEVICE double operator()(int i, int axis) const
    {
        return i < owned ? owned_data[i].rdata(comp + axis) : ghost_data[i - owned].rdata(comp + axis);
    }
    AMREX_GPU_HOST_DEVICE double operator()(int i) const { return (*this)(i, 0); }
};

static AmrexReal make_real(Container& container, ParticleIterator& pti, int comp)
{
    AmrexPos base = make_pos(container, pti);
    AmrexReal r;
    r.owned_data = base.owned_data;
    r.ghost_data = base.ghost_data;
    r.owned = base.owned;
    r.comp = comp;
    return r;
}

static int ghost_count_of(Container& container, ParticleIterator& pti)
{
#ifdef AMREX_USE_GPU
    return pti.numNeighborParticles();
#else
    return container.GetNeighbors(0, pti.index(), pti.LocalTileIndex()).numParticles();
#endif
}

static void place_particles(Container& container, const HostParticles& host)
{
    int count = host.id.size();
    amrex::Gpu::DeviceVector<double> x(count), y(count), z(count), vx(count), vy(count), vz(count);
    amrex::Gpu::DeviceVector<int> id(count);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, host.x.begin(), host.x.end(), x.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, host.y.begin(), host.y.end(), y.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, host.z.begin(), host.z.end(), z.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, host.vx.begin(), host.vx.end(), vx.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, host.vy.begin(), host.vy.end(), vy.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, host.vz.begin(), host.vz.end(), vz.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, host.id.begin(), host.id.end(), id.begin());
    amrex::Gpu::streamSynchronize();
    const double *px = x.data(), *py = y.data(), *pz = z.data(), *pvx = vx.data(), *pvy = vy.data(), *pvz = vz.data();
    const int* pid = id.data();
    for (amrex::MFIter mfi = container.MakeMFIter(0); mfi.isValid(); ++mfi) {
        auto& tile = container.DefineAndReturnParticleTile(0, mfi.index(), mfi.LocalTileIndex());
        tile.resize(count);
        ParticleType* p = tile.GetArrayOfStructs()().data();
        int cpu = amrex::ParallelDescriptor::MyProc();
        amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(int i) {
            p[i].id() = i + 1;
            p[i].cpu() = cpu;
            p[i].pos(0) = px[i];
            p[i].pos(1) = py[i];
            p[i].pos(2) = pz[i];
            for (int k = 0; k < 8; k++) p[i].rdata(k) = 0.0;
            p[i].rdata(VX) = pvx[i];
            p[i].rdata(VY) = pvy[i];
            p[i].rdata(VZ) = pvz[i];
            p[i].rdata(MASS) = 1.0;
            p[i].idata(TYPE) = 0;
            p[i].idata(GLOBAL_ID) = pid[i];
        });
    }
    amrex::Gpu::streamSynchronize();
}

static void kick_drift_all(Container& container, const SimParams& params, double edge)
{
    double dt = params.dt;
    for (ParticleIterator pti(container, 0); pti.isValid(); ++pti) {
        ParticleType* p = pti.GetArrayOfStructs()().data();
        int count = pti.numRealParticles();
        amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(int i) {
            for (int a = 0; a < 3; a++) {
                p[i].rdata(VX + a) += 0.5 * dt * p[i].rdata(FX + a);
                p[i].pos(a) += dt * p[i].rdata(VX + a);
                double x = p[i].pos(a), v = p[i].rdata(VX + a);
                sim_reflect(x, v, edge);
                p[i].pos(a) = x;
                p[i].rdata(VX + a) = v;
            }
        });
    }
    amrex::Gpu::streamSynchronize();
}

static void kick_all(Container& container, const SimParams& params)
{
    double dt = params.dt;
    for (ParticleIterator pti(container, 0); pti.isValid(); ++pti) {
        ParticleType* p = pti.GetArrayOfStructs()().data();
        int count = pti.numRealParticles();
        amrex::ParallelFor(count, [=] AMREX_GPU_DEVICE(int i) {
            for (int a = 0; a < 3; a++) p[i].rdata(VX + a) += 0.5 * dt * p[i].rdata(FX + a);
        });
    }
    amrex::Gpu::streamSynchronize();
}

struct NeighborData {
    amrex::Gpu::DeviceVector<int> cells, members, neighbors, num_neighbors, flag;
};

static void grow(amrex::Gpu::DeviceVector<int>& vector, size_t n)
{
    if (vector.size() < n) vector.resize(std::max(n, vector.size() + vector.size() / 4));
}

static void build_neighbor_lists(Container& container, const double lower[3], const double upper[3], const SimParams& params,
                                 NeighborData& data)
{
    CellGrid grid = make_cell_grid(lower, upper, SIM_LIST_CUTOFF);
    size_t cell_total = (size_t)cell_count(grid);
    for (ParticleIterator pti(container, 0); pti.isValid(); ++pti) {
        AmrexPos pos = make_pos(container, pti);
        int owned = pos.owned, all = owned + ghost_count_of(container, pti);
        grow(data.cells, cell_total);
        grow(data.members, cell_total * SIM_CELL_CAPACITY);
        grow(data.neighbors, (size_t)owned * SIM_MAX_NEIGHBORS);
        grow(data.num_neighbors, owned + 1);
        grow(data.flag, 1);
        int *count = data.cells.data(), *members = data.members.data(), *neighbors = data.neighbors.data();
        int *num_neighbors = data.num_neighbors.data(), *overflow = data.flag.data();
        double list2 = params.list2;
        amrex::ParallelFor((int)cell_total, [=] AMREX_GPU_DEVICE(int c) { count[c] = 0; });
        amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int c) { overflow[0] = 0; });
        amrex::ParallelFor(all, [=] AMREX_GPU_DEVICE(int i) { sim_insert(grid, pos, i, count, members, overflow); });
        amrex::ParallelFor(owned, [=] AMREX_GPU_DEVICE(int i) {
            sim_find_neighbors(grid, pos, i, list2, count, members, neighbors, num_neighbors, overflow);
        });
        amrex::Gpu::streamSynchronize();
        int flag = 0;
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, data.flag.begin(), data.flag.begin() + 1, &flag);
        if (flag) {
            printf("a cell or a neighbor list overflowed (SIM_CELL_CAPACITY, SIM_MAX_NEIGHBORS)\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

// Density of the owned particles (stored in the charge component)
static void compute_densities(Container& container, const SimParams& params, NeighborData& data)
{
    for (ParticleIterator pti(container, 0); pti.isValid(); ++pti) {
        AmrexPos pos = make_pos(container, pti);
        ParticleType* p = pti.GetArrayOfStructs()().data();
        const int *neighbors = data.neighbors.data(), *num_neighbors = data.num_neighbors.data();
        amrex::ParallelFor(pos.owned, [=] AMREX_GPU_DEVICE(int i) {
            p[i].rdata(CHARGE) = sim_density(params, pos, i, neighbors, num_neighbors);
        });
    }
    amrex::Gpu::streamSynchronize();
}

static void compute_forces_all(Container& container, const SimParams& params, NeighborData& data)
{
    for (ParticleIterator pti(container, 0); pti.isValid(); ++pti) {
        AmrexPos pos = make_pos(container, pti);
        AmrexReal vel = make_real(container, pti, VX);
        AmrexReal density = make_real(container, pti, CHARGE);
        ParticleType* p = pti.GetArrayOfStructs()().data();
        const int *neighbors = data.neighbors.data(), *num_neighbors = data.num_neighbors.data();
        amrex::ParallelFor(pos.owned, [=] AMREX_GPU_DEVICE(int i) {
            double f[3], e;
            sim_force(params, pos, vel, density, i, neighbors, num_neighbors, f, e);
            p[i].rdata(FX) = f[0];
            p[i].rdata(FY) = f[1];
            p[i].rdata(FZ) = f[2];
        });
    }
    amrex::Gpu::streamSynchronize();
}

// Density sum and kinetic energy sum of the owned particles of this rank
static void sum_energies(Container& container, double out[2])
{
    out[0] = out[1] = 0;
    for (ParticleIterator pti(container, 0); pti.isValid(); ++pti) {
        const ParticleType* p = pti.GetArrayOfStructs()().data();
        int count = pti.numRealParticles();
        out[0] += amrex::Reduce::Sum<double>(count, [=] AMREX_GPU_DEVICE(int i) -> double { return p[i].rdata(CHARGE); });
        out[1] += amrex::Reduce::Sum<double>(count, [=] AMREX_GPU_DEVICE(int i) -> double {
            return 0.5 * (p[i].rdata(VX) * p[i].rdata(VX) + p[i].rdata(VY) * p[i].rdata(VY) + p[i].rdata(VZ) * p[i].rdata(VZ));
        });
    }
}

// Starts and ends AMReX. It is the first member of Backend, so it is created first and destroyed last.
struct AmrexRuntime {
    AmrexRuntime()
    {
        // AMReX gets no command line. We give it only what it needs. On the GPU it uses GPU-aware MPI (the same condition as for
        // dsDecompX and Cabana).
        static char program[] = "sph_amrex";
        static char gpu_aware[] = "amrex.use_gpu_aware_mpi=1";
        static char quiet[] = "amrex.verbose=0";
        static char* arguments[] = {program, quiet, gpu_aware, nullptr};
#ifdef AMREX_USE_GPU
        int count = 3;
#else
        int count = 2;
#endif
        char** pointer = arguments;
        amrex::Initialize(count, pointer, true, MPI_COMM_WORLD);
    }
    ~AmrexRuntime() { amrex::Finalize(); }
};

struct Backend {
    static const char* name() { return "amrex"; }
    static void add_settings(Settings&) {}
    // Selects the GPU before MPI_Init (the GPU that a GPU-aware MPI sees at init must be the GPU that is used; otherwise the direct
    // GPU-to-GPU transfers are switched off).
    static void before_mpi(int, char**)
    {
#if defined(AMREX_USE_CUDA)
        if (const char* local = std::getenv("SLURM_LOCALID")) cudaSetDevice(std::atoi(local) % 4);
#endif
    }

    AmrexRuntime runtime;
    SimCase problem;
    SimGrid grid;
    int cells[3], ncell[3];
    amrex::Geometry geometry;
    amrex::BoxArray boxes;
    amrex::DistributionMapping mapping;
    std::unique_ptr<Container> container;
    double lower[3], upper[3], frame_lower[3], frame_upper[3];
    NeighborData data;

    Backend(const SimCase& p, int, char**) : problem(p)
    {
        for (int a = 0; a < 3; a++) {
            grid.dims[a] = p.dims[a];
            grid.edge[a] = p.lattice.edge;
        }
        grid.halo = SIM_LIST_CUTOFF;
        // One rank owns one box of cells[0] x cells[1] x cells[2] cells. The number of cells is chosen so that a cell is at least as
        // wide as the halo.
        for (int a = 0; a < 3; a++) {
            cells[a] = std::max(1, (int)std::floor(p.lattice.edge / grid.dims[a] / SIM_LIST_CUTOFF));
            ncell[a] = grid.dims[a] * cells[a];
        }
        amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(ncell[0] - 1, ncell[1] - 1, ncell[2] - 1));
        amrex::RealBox real_box({0.0, 0.0, 0.0}, {p.lattice.edge, p.lattice.edge, p.lattice.edge});
        amrex::Array<int, 3> not_periodic = {0, 0, 0};
        geometry = amrex::Geometry(domain, real_box, 0, not_periodic);
        boxes = amrex::BoxArray(domain);
        boxes.maxSize(amrex::IntVect(cells[0], cells[1], cells[2]));
        amrex::Vector<int> box_to_rank(boxes.size());
        for (int b = 0; b < (int)boxes.size(); b++) {
            amrex::Box box = boxes[b];                        // smallEnd() of a temporary object cannot be used
            amrex::IntVect low = box.smallEnd();
            box_to_rank[b] = sim_block_rank(grid, low[0] / cells[0], low[1] / cells[1], low[2] / cells[2]);
        }
        mapping = amrex::DistributionMapping(box_to_rank);
        container.reset(new Container(geometry, mapping, boxes, 1));

        int c[3];
        sim_block_coordinates(grid, p.rank, c);
        for (int a = 0; a < 3; a++) {
            lower[a] = sim_cut(grid, a, c[a]);
            upper[a] = sim_cut(grid, a, c[a] + 1);
            double width = p.lattice.edge / ncell[a];         // cell width (the ghosts reach this far)
            frame_lower[a] = lower[a] - width;
            frame_upper[a] = upper[a] + width;
        }
    }

    ~Backend() { container.reset(); }                         // destroy the particles before AMReX is finalized

    bool on_gpu() const
    {
#ifdef AMREX_USE_GPU
        return true;
#else
        return false;
#endif
    }
    void sync() { amrex::Gpu::streamSynchronize(); }
    unsigned long long ghost_count() const
    {
        unsigned long long ghosts = 0;
        for (ParticleIterator pti(*container, 0); pti.isValid(); ++pti) ghosts += ghost_count_of(*container, pti);
        return ghosts;
    }
    void bounds(double l[3], double u[3]) const
    {
        for (int a = 0; a < 3; a++) {
            l[a] = lower[a];
            u[a] = upper[a];
        }
    }

    void place(const HostParticles& host)
    {
        place_particles(*container, host);
        container->Redistribute();
    }

    void kick_drift() { kick_drift_all(*container, problem.params, problem.lattice.edge); }
    void kick() { kick_all(*container, problem.params); }
    void migrate() { container->Redistribute(); }
    void halo_build() { container->fillNeighbors(); }
    bool halo_refresh()
    {
        container->updateNeighbors();
        return false;
    }
    void build_neighbors() { build_neighbor_lists(*container, frame_lower, frame_upper, problem.params, data); }
    void compute_density() { compute_densities(*container, problem.params, data); }
    void density_exchange() { container->updateNeighbors(); }
    void compute_forces() { compute_forces_all(*container, problem.params, data); }

    void observe(double& potential, double& kinetic)
    {
        double local[2], sum[2];
        sum_energies(*container, local);
        MPI_Allreduce(local, sum, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        potential = sum[0] / problem.lattice.total;
        kinetic = sum[1] / problem.lattice.total;
    }
};
