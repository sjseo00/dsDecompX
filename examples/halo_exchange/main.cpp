#include <dsdecompx.hpp>
#include <iostream>

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    {
        dsdecompx::Config config;
        config.execution = dsdecompx::ExecutionMode::Host;
        config.domain.upper = {10.0, 10.0, 10.0};
        config.halo.width = 0.5;
        dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);
        const auto& local = decomp.topology().local_domain();

        dsdecompx::ParticleInfo particles(1);
        particles.x[0]=local.lower[0]+0.1;
        particles.y[0]=local.lower[1]+0.1;
        particles.z[0]=local.lower[2]+0.1;
        particles.mass[0]=1.0;
        particles.global_id[0]=decomp.topology().rank();

        const auto result = decomp.exchange_halo(particles);
        std::cout << "rank " << decomp.topology().rank() << " received "
                  << result.ghost_count << " ghosts\n";
    }
    MPI_Finalize();
}
