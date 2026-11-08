#include <dsdecompx.hpp>
#include <algorithm>
#include <iostream>

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    {
        dsdecompx::Config config;
        config.execution = dsdecompx::ExecutionMode::Host;
        config.grid_policy = dsdecompx::GridPolicy::Balanced;
        config.domain.upper = {8.0, 8.0, 8.0};
        config.halo.width = 0.25;
        dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);
        const int rank = decomp.topology().rank();
        const auto& local = decomp.topology().local_domain();
        dsdecompx::ParticleInfo particles(1);
        particles.x[0] = local.lower[0] + 0.05;
        particles.y[0] = local.lower[1] + 0.05;
        particles.z[0] = local.lower[2] + 0.05;
        particles.global_id[0] = rank;
        particles.mass[0] = 1.0;
        decomp.migrate(particles);
        decomp.exchange_halo(particles, {true, nullptr});
        std::fill(particles.fx.begin(), particles.fx.end(), 0.0);
        std::fill(particles.fy.begin(), particles.fy.end(), 0.0);
        std::fill(particles.fz.begin(), particles.fz.end(), 0.0);
        for (std::size_t g = particles.owned_count();
             g < particles.size(); ++g) particles.fx[g] = 1.0;
        decomp.reverse_halo(particles);
        std::cout << "rank " << rank << " ghosts=" << particles.ghost_count()
                  << " returned_fx=" << particles.fx[0] << '\n';
    }
    MPI_Finalize();
}
