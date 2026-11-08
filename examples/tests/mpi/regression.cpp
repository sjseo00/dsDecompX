#include <dsdecompx.hpp>

#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    {
        int rank = 0, size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        dsdecompx::Config config;
        config.execution = dsdecompx::ExecutionMode::Host;
        config.domain.upper = {8.0, 8.0, 8.0};
        config.halo.width = 0.25;
        dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);
        decomp.set_performance_counters_enabled(true, true);

        const auto& local = decomp.topology().local_domain();
        dsdecompx::ParticleInfo particles(1);
        particles.x[0]=local.lower[0]+0.05;
        particles.y[0]=local.lower[1]+0.05;
        particles.z[0]=local.lower[2]+0.05;
        particles.mass[0]=1.0;
        particles.global_id[0]=rank;
        std::vector<double> temperature(64,0.0);
        std::vector<int> species(64,0);
        std::vector<double> momentum(64*3,0.0);
        std::vector<double> contribution(64,0.0);
        temperature[0]=100.0+rank;
        species[0]=10+rank;
        momentum[0]=rank+0.1;
        momentum[1]=rank+0.2;
        momentum[2]=rank+0.3;
        particles.add_field<double>(
                     "temperature",temperature.data(),64)
                 .add_field<int>("species",species.data(),64)
                 .add_field<double>("momentum",momentum.data(),64,3)
                 .add_field<double>(
                     "contribution",contribution.data(),64);

        const auto first = decomp.exchange_halo(
            particles, dsdecompx::HaloExchangeOptions{true, nullptr});
        std::vector<int> first_ids(
            particles.global_id.begin()+1, particles.global_id.end());
        const auto second = decomp.exchange_halo(particles);
        std::vector<int> second_ids(
            particles.global_id.begin()+1, particles.global_id.end());
        assert(first.ghost_count == second.ghost_count);
        assert(first_ids == second_ids);
        std::vector<double> expected_local(static_cast<std::size_t>(size),0.0);
        for (std::size_t i=1; i<particles.size(); ++i) {
            const int id=particles.global_id[i];
            assert(temperature[i]==100.0+id);
            assert(species[i]==10+id);
            assert(std::abs(momentum[3*i]-(id+0.1))<1.0e-12);
            assert(std::abs(momentum[3*i+1]-(id+0.2))<1.0e-12);
            assert(std::abs(momentum[3*i+2]-(id+0.3))<1.0e-12);
            contribution[i]=1.0;
            expected_local[static_cast<std::size_t>(id)]+=1.0;
        }
        std::vector<double> expected_global(static_cast<std::size_t>(size),0.0);
        MPI_Allreduce(expected_local.data(),expected_global.data(),size,
                      MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        decomp.reverse_exchange(
            particles,{{"contribution",dsdecompx::ReduceOp::Sum}});
        assert(std::abs(contribution[0]-
                        expected_global[static_cast<std::size_t>(rank)])<1.0e-12);
        double global_contribution=0.0;
        MPI_Allreduce(contribution.data(),&global_contribution,1,MPI_DOUBLE,
                      MPI_SUM,MPI_COMM_WORLD);
        unsigned long long local_ghosts=second.ghost_count;
        unsigned long long global_ghosts=0;
        MPI_Allreduce(&local_ghosts,&global_ghosts,1,MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM,MPI_COMM_WORLD);
        assert(std::abs(global_contribution-
                        static_cast<double>(global_ghosts))<1.0e-12);

        const auto& dims = decomp.topology().dimensions();
        if (dims[0] > 1) particles.x[0] = local.upper[0] + 0.05;
        else if (dims[1] > 1) particles.y[0] = local.upper[1] + 0.05;
        else if (dims[2] > 1) particles.z[0] = local.upper[2] + 0.05;
        else particles.x[0] += 8.0;
        const auto migration = decomp.migrate(particles);
        for (std::size_t i=0; i<migration.new_owned_count; ++i) {
            const int id=particles.global_id[i];
            assert(temperature[i]==100.0+id);
            assert(species[i]==10+id);
            assert(std::abs(momentum[3*i]-(id+0.1))<1.0e-12);
            assert(std::abs(momentum[3*i+1]-(id+0.2))<1.0e-12);
            assert(std::abs(momentum[3*i+2]-(id+0.3))<1.0e-12);
        }
        unsigned long long local_count = migration.new_owned_count;
        unsigned long long global_count = 0;
        MPI_Allreduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
        assert(global_count == static_cast<unsigned long long>(size));

        const auto local_stats = decomp.get_stats();
        assert(local_stats.halo_calls == 2);
        assert(local_stats.migration_calls == 1);
        const auto global_stats = decomp.get_global_stats();
        assert(global_stats.halo_calls == static_cast<std::uint64_t>(2 * size));
        assert(global_stats.migration_calls == static_cast<std::uint64_t>(size));
    }
    {
        int rank = 0, size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        dsdecompx::Config config;
        config.execution = dsdecompx::ExecutionMode::Host;
        config.domain.upper = {8.0, 8.0, 8.0};
        config.grid_policy = dsdecompx::GridPolicy::Dynamic;
        config.dynamic.interval = 0;
        config.dynamic.relaxation = 1.0;
        dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);

        const auto before = decomp.topology().axis_cuts();
        const auto local = decomp.topology().local_domain();
        dsdecompx::ParticleInfo particles(1);
        particles.x[0] = 0.5 * (local.lower[0] + local.upper[0]);
        particles.y[0] = 0.5 * (local.lower[1] + local.upper[1]);
        particles.z[0] = 0.5 * (local.lower[2] + local.upper[2]);
        particles.global_id[0] = rank;

        dsdecompx::RebalanceOptions options;
        options.local_work = rank == 0 ? 100.0 : 1.0;
        options.force = true;
        const auto balance = decomp.rebalance(particles, options);
        assert(balance.attempted);
        assert(balance.imbalance_before >= 1.0);
        if (size > 1) assert(balance.geometry_changed);

        const auto& after = decomp.topology().axis_cuts();
        bool cuts_changed = false;
        for (int axis = 0; axis < 3; ++axis) {
            cuts_changed = cuts_changed || before[axis] != after[axis];
        }
        assert(cuts_changed == (size > 1));
        unsigned long long local_count = particles.owned_count();
        unsigned long long global_count = 0;
        MPI_Allreduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
        assert(global_count == static_cast<unsigned long long>(size));
        for (std::size_t i = 0; i < particles.owned_count(); ++i) {
            assert(decomp.owner_rank({particles.x[i], particles.y[i],
                                      particles.z[i]}) == rank);
        }
    }
    {
        int rank = 0, size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        dsdecompx::Config config;
        config.execution = dsdecompx::ExecutionMode::Host;
        config.domain.upper = {8.0, 8.0, 8.0};
        config.grid_policy = dsdecompx::GridPolicy::Dynamic;
        config.dynamic.interval = 1;
        config.dynamic.imbalance_threshold = 1.01;
        config.dynamic.relaxation = 1.0;
        dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);

        const auto before = decomp.topology().axis_cuts();
        const auto local = decomp.topology().local_domain();
        const std::size_t local_count = rank == 0 ? 16u : 1u;
        dsdecompx::ParticleInfo particles(local_count);
        for (std::size_t i = 0; i < local_count; ++i) {
            const double fraction = (static_cast<double>(i) + 0.5) /
                                    static_cast<double>(local_count);
            particles.x[i] = local.lower[0] +
                fraction * (local.upper[0] - local.lower[0]);
            particles.y[i] = local.lower[1] +
                fraction * (local.upper[1] - local.lower[1]);
            particles.z[i] = local.lower[2] +
                fraction * (local.upper[2] - local.lower[2]);
            particles.global_id[i] = rank * 1000 + static_cast<int>(i);
        }
        decomp.migrate(particles);
        bool cuts_changed = false;
        const auto& after = decomp.topology().axis_cuts();
        for (int axis = 0; axis < 3; ++axis) {
            cuts_changed = cuts_changed || before[axis] != after[axis];
        }
        assert(cuts_changed == (size > 1));
        unsigned long long owned = particles.owned_count();
        unsigned long long total = 0;
        MPI_Allreduce(&owned, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        assert(total == static_cast<unsigned long long>(16 + size - 1));
    }
    MPI_Finalize();
}
