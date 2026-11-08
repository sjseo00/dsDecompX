#include <dsdecompx.hpp>
#include <iostream>

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    {
        dsdecompx::Config config;
        config.execution = dsdecompx::ExecutionMode::Host;
        config.domain.upper = {32.0, 24.0, 16.0};
        config.grid_policy = dsdecompx::GridPolicy::Balanced;
        dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);
        const auto& topology = decomp.topology();
        const auto& d = topology.dimensions();
        const auto& c = topology.coordinates();
        std::cout << "rank " << topology.rank() << " grid "
                  << d[0] << 'x' << d[1] << 'x' << d[2]
                  << " coordinate " << c[0] << ',' << c[1] << ',' << c[2]
                  << " neighbors " << topology.neighbors().size() << '\n';
    }
    MPI_Finalize();
}
