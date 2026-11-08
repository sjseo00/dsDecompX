#include <dsdecompx.hpp>
#include <iostream>

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    {
        dsdecompx::Config config;
        config.execution = dsdecompx::ExecutionMode::Host;
        config.domain.upper = {10.0, 10.0, 10.0};
        dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);
        const int rank = decomp.topology().rank();
        dsdecompx::ParticleInfo particles(1);
        particles.x[0]=rank == 0 ? 10.25 : -0.25;
        particles.y[0]=5.0; particles.z[0]=5.0;
        particles.mass[0]=1.0; particles.global_id[0]=rank;
        const auto result = decomp.migrate(particles);
        std::cout << "rank " << rank << " now owns " << result.new_owned_count
                  << " particles\n";
    }
    MPI_Finalize();
}
