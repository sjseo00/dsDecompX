// backend_dsdecompx.h -- the dsDecompX part of main.cpp. The CPU and the GPU use the same code (gpu=1: GPU, gpu=0: CPU).
//
//   dsdecompx::Decomp::migrate()                                migrate
//   dsdecompx::Decomp::exchange_halo(rebuild = true)            build the halo (sends the whole particle)
//   dsdecompx::Decomp::exchange_halo(rebuild = false)           refresh the halo (the force uses the velocity of the ghosts. The GPU sends
//                                                               the whole particle. The CPU refreshes the positions and the velocities of
//                                                               the ghosts in place, which is all the force needs. The neighbor list is
//                                                               rebuilt if the ghosts changed)
// dsDecompX splits the domain (rank grid and cuts) and selects the ghost particles.
#pragma once

#include <dsdecompx.hpp>

#include <cuda_runtime.h>
#include <mpi.h>

#include <cstring>

#include "common.h"

#define CHECK(call)                                                                                   \
    do {                                                                                              \
        cudaError_t status = (call);                                                                  \
        if (status != cudaSuccess) {                                                                  \
            printf("CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(status));         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                             \
        }                                                                                             \
    } while (0)

struct Pos {
    const double *x, *y, *z;
    SIM_HD double operator()(int i, int axis) const { return axis == 0 ? x[i] : (axis == 1 ? y[i] : z[i]); }
};

struct KickDrift {
    double *x, *y, *z, *vx, *vy, *vz;
    const double *fx, *fy, *fz;
    double dt, edge0, edge1, edge2;
    SIM_HD void operator()(int i) const
    {
        vx[i] += 0.5 * dt * fx[i];
        vy[i] += 0.5 * dt * fy[i];
        vz[i] += 0.5 * dt * fz[i];
        x[i] += dt * vx[i];
        y[i] += dt * vy[i];
        z[i] += dt * vz[i];
        sim_reflect(x[i], vx[i], edge0);
        sim_reflect(y[i], vy[i], edge1);
        sim_reflect(z[i], vz[i], edge2);
    }
};

struct Kick {
    double *vx, *vy, *vz;
    const double *fx, *fy, *fz;
    double dt;
    SIM_HD void operator()(int i) const
    {
        vx[i] += 0.5 * dt * fx[i];
        vy[i] += 0.5 * dt * fy[i];
        vz[i] += 0.5 * dt * fz[i];
    }
};

struct Insert {
    CellGrid grid;
    Pos pos;
    int *count, *members, *overflow;
    SIM_HD void operator()(int i) const { sim_insert(grid, pos, i, count, members, overflow); }
};

struct FindNeighbors {
    CellGrid grid;
    Pos pos;
    double list2;
    const int *count, *members;
    int *neighbors, *num_neighbors, *overflow;
    SIM_HD void operator()(int i) const { sim_find_neighbors(grid, pos, i, list2, count, members, neighbors, num_neighbors, overflow); }
};

struct Force {
    SimParams params;
    Pos pos, vel;
    const int *neighbors, *num_neighbors;
    double *fx, *fy, *fz, *energy;
    SIM_HD void operator()(int i) const
    {
        double f[3], e;
        sim_force(params, pos, vel, i, neighbors, num_neighbors, f, e);
        fx[i] = f[0];
        fy[i] = f[1];
        fz[i] = f[2];
        energy[i] = e;
    }
};

template <class F>
__global__ void run_kernel(int n, F f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f(i);
}

// Runs the same work as a kernel on the GPU and as a loop on the CPU.
struct Runner {
    bool gpu;
    cudaStream_t stream;
    template <class F>
    void operator()(int n, F f) const
    {
        if (n <= 0) return;
        if (gpu) run_kernel<<<(n + 255) / 256, 256, 0, stream>>>(n, f);
        else for (int i = 0; i < n; i++) f(i);
    }
};

template <class T>
struct Buffer {
    T* data = nullptr;
    size_t size = 0;
    bool gpu = false;
    ~Buffer() { release(); }
    void release()
    {
        if (!data) return;
        if (gpu) cudaFree(data);
        else std::free(data);
        data = nullptr;
        size = 0;
    }
    void reserve(size_t n)
    {
        if (n <= size) return;
        size_t m = std::max(n, size + size / 4);
        release();
        if (gpu) CHECK(cudaMalloc(&data, m * sizeof(T)));
        else data = (T*)std::malloc(m * sizeof(T));
        size = m;
    }
    void zero(size_t n, cudaStream_t stream)
    {
        if (gpu) CHECK(cudaMemsetAsync(data, 0, n * sizeof(T), stream));
        else std::memset(data, 0, n * sizeof(T));
    }
};

struct Backend {
    static const char* name() { return "dsdecompx"; }
    static void add_settings(Settings& setting)
    {
        setting["gpu"] = 1;                  // 1: GPU, 0: CPU
        setting["backend"] = 1;              // 0 auto, 1 mpi, 2 nccl, 3 nvshmem (GPU only)
    }
    // Selects the GPU before MPI_Init. A GPU-aware MPI initializes for the GPU that is current at that moment. If every rank had
    // GPU 0 then, the direct GPU-to-GPU transfers inside a node (IPC) would be switched off.
    static void before_mpi(int argc, char** argv)
    {
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "gpu=0") return;
        }
        const char* local_id = std::getenv("SLURM_LOCALID");
        cudaSetDevice(local_id ? std::atoi(local_id) : 0);
    }

    SimCase problem;
    bool use_gpu;
    dsdecompx::Decomp decomp;
    dsdecompx::LocalDomain local;
    dsdecompx::ParticleInfo particles;
    cudaStream_t stream = nullptr;
    Runner run;
    Buffer<int> cells, members, neighbors, num_neighbors, flag;

    static dsdecompx::Config make_config(const SimCase& p)
    {
        const Settings& setting = *p.setting;
        dsdecompx::Config config;
        config.domain.upper = {p.lattice.edge, p.lattice.edge, p.lattice.edge};
        config.domain.periodic = {false, false, false};
        config.process_grid.dimensions = {p.dims[0], p.dims[1], p.dims[2]};
        config.grid_policy = dsdecompx::GridPolicy::Manual;
        config.halo.width = SIM_LIST_CUTOFF;
        config.advanced.read_environment = false;
        if (setting.at("gpu") == 1) {
            const char* backend_names[] = {"auto", "mpi", "nccl", "nvshmem"};
            config.execution = dsdecompx::ExecutionMode::Cuda;
            config.advanced.selection = dsdecompx::parse_backend_selection(backend_names[(int)setting.at("backend")]);
            config.advanced.cuda_device = dsdecompx::kCudaDeviceByLocalRank;
        } else {
            config.execution = dsdecompx::ExecutionMode::Host;
            config.advanced.selection = dsdecompx::BackendSelection::Mpi;
        }
        return config;
    }

    Backend(const SimCase& p, int, char**)
        : problem(p), use_gpu(p.setting->at("gpu") == 1), decomp(MPI_COMM_WORLD, make_config(p)), local(decomp.topology().local_domain())
    {
        if (use_gpu) cudaStreamCreate(&stream);
        run = Runner{use_gpu, stream};
        cells.gpu = members.gpu = neighbors.gpu = num_neighbors.gpu = flag.gpu = use_gpu;
        flag.reserve(1);
    }

    bool on_gpu() const { return use_gpu; }
    void sync() { if (use_gpu) CHECK(cudaStreamSynchronize(stream)); }
    unsigned long long ghost_count() const { return particles.ghost_count(); }

    // Region of this rank
    void bounds(double lower[3], double upper[3]) const
    {
        for (int a = 0; a < 3; a++) {
            lower[a] = local.lower[a];
            upper[a] = local.upper[a];
        }
    }

    void place(const HostParticles& host)
    {
        int owned = (int)host.id.size();
        particles = dsdecompx::ParticleInfo::allocate(use_gpu ? dsdecompx::MemorySpace::CudaDevice : dsdecompx::MemorySpace::Host, owned);
        std::vector<double> ones(owned, 1.0), zeros(owned, 0.0);
        auto put = [&](double* destination, const double* source) {
            if (owned == 0) return;
            if (use_gpu) CHECK(cudaMemcpy(destination, source, owned * sizeof(double), cudaMemcpyHostToDevice));
            else std::memcpy(destination, source, owned * sizeof(double));
        };
        put(particles.x_data(), host.x.data()); put(particles.y_data(), host.y.data()); put(particles.z_data(), host.z.data());
        put(particles.vx_data(), host.vx.data()); put(particles.vy_data(), host.vy.data()); put(particles.vz_data(), host.vz.data());
        put(particles.fx_data(), zeros.data()); put(particles.fy_data(), zeros.data()); put(particles.fz_data(), zeros.data());
        put(particles.mass_data(), ones.data()); put(particles.charge_data(), zeros.data());
        if (owned > 0) {
            if (use_gpu) CHECK(cudaMemcpy(particles.global_id_data(), host.id.data(), owned * sizeof(int), cudaMemcpyHostToDevice));
            else std::memcpy(particles.global_id_data(), host.id.data(), owned * sizeof(int));
        }
    }

    void kick_drift()
    {
        double edge = problem.lattice.edge;
        run((int)particles.owned_count(),
            KickDrift{particles.x_data(), particles.y_data(), particles.z_data(), particles.vx_data(), particles.vy_data(),
                      particles.vz_data(), particles.fx_data(), particles.fy_data(), particles.fz_data(), problem.params.dt, edge, edge, edge});
    }

    void kick()
    {
        run((int)particles.owned_count(), Kick{particles.vx_data(), particles.vy_data(), particles.vz_data(), particles.fx_data(),
                                               particles.fy_data(), particles.fz_data(), problem.params.dt});
    }

    void migrate() { decomp.migrate(particles, stream); }

    dsdecompx::HaloResult exchange(bool rebuild, bool positions_only)
    {
        dsdecompx::HaloExchangeOptions options;
        options.rebuild = rebuild;
        options.stream = stream;
        options.positions_only = use_gpu && positions_only;
        return decomp.exchange_halo(particles, options);
    }

    void halo_build() { exchange(true, false); }

    // The whole particle is sent, because the force uses the velocity of the ghosts.
    // True if the ghosts changed, because then the numbers in the neighbor list are wrong (the GPU reuses its plan, so nothing changes).
    bool halo_refresh() { return exchange(false, false).layout_changed; }

    void build_neighbors()
    {
        int owned = (int)particles.owned_count(), all = (int)particles.size();
        double lower[3], upper[3];
        for (int a = 0; a < 3; a++) {
            lower[a] = local.lower[a] - SIM_LIST_CUTOFF;
            upper[a] = local.upper[a] + SIM_LIST_CUTOFF;
        }
        CellGrid grid = make_cell_grid(lower, upper, SIM_LIST_CUTOFF);
        size_t cell_total = (size_t)cell_count(grid);
        cells.reserve(cell_total);
        members.reserve(cell_total * SIM_CELL_CAPACITY);
        neighbors.reserve((size_t)owned * SIM_MAX_NEIGHBORS);
        num_neighbors.reserve(owned + 1);
        cells.zero(cell_total, stream);
        flag.zero(1, stream);
        Pos pos{particles.x_data(), particles.y_data(), particles.z_data()};
        run(all, Insert{grid, pos, cells.data, members.data, flag.data});
        run(owned, FindNeighbors{grid, pos, problem.params.list2, cells.data, members.data, neighbors.data, num_neighbors.data, flag.data});
        int overflow = 0;
        if (use_gpu) CHECK(cudaMemcpyAsync(&overflow, flag.data, sizeof(int), cudaMemcpyDeviceToHost, stream));
        else overflow = *flag.data;
        sync();
        if (overflow) {
            printf("rank %d: a cell or a neighbor list overflowed (SIM_CELL_CAPACITY, SIM_MAX_NEIGHBORS)\n", problem.rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    void compute_forces()
    {
        Pos pos{particles.x_data(), particles.y_data(), particles.z_data()};
        Pos vel{particles.vx_data(), particles.vy_data(), particles.vz_data()};
        run((int)particles.owned_count(), Force{problem.params, pos, vel, neighbors.data, num_neighbors.data, particles.fx_data(),
                                                particles.fy_data(), particles.fz_data(), particles.charge_data()});
    }

    // Energy per particle. The contact energy was stored in the charge field when the force was computed.
    void observe(double& potential, double& kinetic)
    {
        sync();
        size_t n = particles.owned_count();
        std::vector<double> vx(n), vy(n), vz(n), energy(n);
        if (use_gpu && n > 0) {
            CHECK(cudaMemcpy(vx.data(), particles.vx_data(), n * sizeof(double), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(vy.data(), particles.vy_data(), n * sizeof(double), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(vz.data(), particles.vz_data(), n * sizeof(double), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(energy.data(), particles.charge_data(), n * sizeof(double), cudaMemcpyDeviceToHost));
        } else if (n > 0) {
            std::memcpy(vx.data(), particles.vx_data(), n * sizeof(double));
            std::memcpy(vy.data(), particles.vy_data(), n * sizeof(double));
            std::memcpy(vz.data(), particles.vz_data(), n * sizeof(double));
            std::memcpy(energy.data(), particles.charge_data(), n * sizeof(double));
        }
        sim_energy_sum(vx.data(), vy.data(), vz.data(), energy.data(), n, problem.lattice.total, potential, kinetic);
    }
};
