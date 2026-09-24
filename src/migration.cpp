#include "decomposition.hpp"

#include "internal/decomp_impl.hpp"
#include "internal/buffer_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <chrono>
#include <numeric>
#include <unordered_set>
#if HAS_CUDA
#include "cuda/execution.hpp"
#include "cuda/kernels.hpp"
#include "cuda/cuda_error.hpp"
#include <cuda_runtime_api.h>
#endif

namespace dsdecompx {

void Decomposition::unpack_migrated_device_particles(
    ParticleView destination, ExecutionStream stream) {
#if HAS_CUDA
    if (!stream) throw std::invalid_argument(
        "dsDecompX: device migration unpack requires an explicit stream");
    if (destination.memory_space!=MemorySpace::CudaDevice &&
        destination.memory_space!=MemorySpace::CudaManaged) {
        throw std::invalid_argument(
            "dsDecompX: device migration unpack needs device output");
    }
    const DeviceParticleBuffer source=impl_->device_migration_result;
    if (destination.count<source.count) throw std::invalid_argument(
        "dsDecompX: migration output capacity is too small");
    cuda::launch_unpack_records(source.data,source.count,destination,0,
                                static_cast<cudaStream_t>(stream));
    impl_->cuda_execution->unpack_ready.record(static_cast<cudaStream_t>(stream));
    impl_->cuda_execution->reuse_event_recorded=true;
#else
    (void)destination; (void)stream;
    throw std::runtime_error("dsDecompX: CUDA execution layer is not compiled");
#endif
}

void Decomposition::unpack_migrated_fields(ParticleView destination) {
    for (const auto& transfer : impl_->migration_fields) {
        ParticleFieldView* begin=destination.fields;
        ParticleFieldView* end=begin+destination.field_count;
        const auto found=std::find_if(
            begin,end,
            [&](const ParticleFieldView& field) {
                return field.name==transfer.name;
            });
        const std::size_t count=transfer.record_bytes
            ? transfer.result.size()/transfer.record_bytes : 0;
        if (found==end || found->type!=transfer.type ||
            found->components!=transfer.components || !found->data ||
            found->capacity<count) {
            throw std::invalid_argument(
                "dsDecompX: missing, incompatible, or undersized migration field '"+
                transfer.name+"'");
        }
        if (!transfer.result.empty()) {
            std::memcpy(found->data,transfer.result.data(),transfer.result.size());
        }
    }
}

int Decomposition::owner_rank(const Vec3d& input_position) const {
    Vec3i coordinates{0, 0, 0};

    for (int axis = 0; axis < 3; ++axis) {
        const double lower = impl_->config.domain.lower[axis];
        const double upper = impl_->config.domain.upper[axis];
        const double extent = upper - lower;
        double position = input_position[axis];

        if (impl_->config.domain.periodic[axis]) {
            position -= lower;
            position -= std::floor(position / extent) * extent;
            position += lower;
        }

        const auto& cuts = impl_->topology.axis_cuts()[axis];
        int coordinate = static_cast<int>(
            std::upper_bound(cuts.begin(), cuts.end(), position) - cuts.begin()) - 1;
        coordinate = std::max(0, std::min(
            coordinate, impl_->topology.dimensions()[axis] - 1));
        coordinates[axis] = coordinate;
    }

    return impl_->backend->cartesian_rank(coordinates);
}

namespace {

ParticleRecord wrapped_record(const ConstParticleView& particles,
                              std::size_t index,
                              const GlobalDomain& domain) {
    ParticleRecord record;
    auto wrap = [&](double value, int axis) {
        if (!domain.periodic[axis]) return value;
        const double lower = domain.lower[axis];
        const double extent = domain.upper[axis] - lower;
        value -= lower;
        value -= std::floor(value / extent) * extent;
        return value + lower;
    };
    record.x = wrap(particles.x[index], 0);
    record.y = wrap(particles.y[index], 1);
    record.z = wrap(particles.z[index], 2);
    if (particles.vx) record.vx = particles.vx[index];
    if (particles.vy) record.vy = particles.vy[index];
    if (particles.vz) record.vz = particles.vz[index];
    if (particles.fx) record.fx = particles.fx[index];
    if (particles.fy) record.fy = particles.fy[index];
    if (particles.fz) record.fz = particles.fz[index];
    if (particles.mass) record.mass = particles.mass[index];
    if (particles.charge) record.charge = particles.charge[index];
    if (particles.type) record.type = particles.type[index];
    record.global_id = particles.global_id[index];
    return record;
}

#ifndef NDEBUG
void ensure_unique(std::vector<int> ids, const char* phase) {
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
        throw std::runtime_error(std::string("dsDecompX: duplicate global particle ID ") + phase);
    }
}
#endif

} // namespace

MigrationResult Decomposition::migrate_particles(
    const ConstParticleView& owned_particles, ExecutionStream stream) {
    (void)stream;
    if (impl_->operation_active) {
        throw std::logic_error("dsDecompX: migration cannot overlap an active halo operation");
    }
    if (owned_particles.memory_space != MemorySpace::Host) {
#if HAS_CUDA
        impl_->migrated_owned.clear();
        if (owned_particles.field_count!=0) {
            throw std::runtime_error(
                "dsDecompX: custom fields currently require host particle memory");
        }
        if (!stream) throw std::invalid_argument(
            "dsDecompX CUDA migration requires an explicit communication stream");
        if (!impl_->backend->supports_device_data_plane()) throw std::runtime_error(
            "dsDecompX: selected backend cannot migrate GPU-resident particles");
        if (owned_particles.count &&
            (!owned_particles.x || !owned_particles.y || !owned_particles.z ||
             !owned_particles.vx || !owned_particles.vy || !owned_particles.vz ||
             !owned_particles.fx || !owned_particles.fy || !owned_particles.fz ||
             !owned_particles.mass || !owned_particles.charge ||
             !owned_particles.type || !owned_particles.global_id)) {
            throw std::invalid_argument("dsDecompX: incomplete CUDA migration view");
        }
        if (!impl_->cuda_execution)
            impl_->cuda_execution=std::make_unique<cuda::ExecutionState>();
        cuda::ExecutionState& state=*impl_->cuda_execution;
        const cudaStream_t cuda_stream=static_cast<cudaStream_t>(stream);
        const std::size_t particle_count=owned_particles.count;
        const std::size_t rank_count=static_cast<std::size_t>(impl_->topology.size());
        const int local_rank=impl_->topology.rank();
        const std::size_t temporary_bytes=
            cuda::migration_layout_temporary_bytes(particle_count,rank_count);
        const bool initial_growth=
            particle_count*sizeof(int)>state.migration_destinations.capacity() ||
            particle_count*sizeof(int)>state.migration_sorted_destinations.capacity() ||
            particle_count*sizeof(unsigned int)>state.migration_indices.capacity() ||
            particle_count*sizeof(unsigned int)>state.migration_sorted_indices.capacity() ||
            rank_count*sizeof(unsigned int)>state.device_counts.capacity() ||
            rank_count*sizeof(unsigned int)>state.device_offsets.capacity() ||
            std::max<std::size_t>(1,particle_count)*sizeof(ParticleRecord)>
                state.device_send.capacity() ||
            temporary_bytes>state.scan_temporary.capacity();
        if (state.reuse_event_recorded) {
            if (initial_growth) state.unpack_ready.synchronize_host();
            else state.unpack_ready.wait(cuda_stream);
        }
        state.migration_destinations.resize(particle_count*sizeof(int));
        state.migration_sorted_destinations.resize(particle_count*sizeof(int));
        state.migration_indices.resize(particle_count*sizeof(unsigned int));
        state.migration_sorted_indices.resize(particle_count*sizeof(unsigned int));
        state.device_counts.resize(rank_count*sizeof(unsigned int));
        state.device_offsets.resize(rank_count*sizeof(unsigned int));
        state.device_send.resize(std::max<std::size_t>(1,particle_count)*
                                 sizeof(ParticleRecord));
        state.host_counts.resize(rank_count*sizeof(unsigned int));
        state.scan_temporary.resize(temporary_bytes);

        const auto& axis_cuts = impl_->topology.axis_cuts();
        Vec3i cut_offsets{0, 0, 0};
        std::vector<double> packed_cuts;
        for (int axis = 0; axis < 3; ++axis) {
            cut_offsets[axis] = static_cast<int>(packed_cuts.size());
            packed_cuts.insert(packed_cuts.end(), axis_cuts[axis].begin(),
                               axis_cuts[axis].end());
        }
        state.domain_cuts.resize(packed_cuts.size()*sizeof(double));
        cuda::detail::check(cudaMemcpyAsync(
            state.domain_cuts.data(), packed_cuts.data(),
            packed_cuts.size()*sizeof(double), cudaMemcpyHostToDevice,
            cuda_stream), "copy dynamic domain cuts");

        cuda::launch_classify_destinations(
            owned_particles,impl_->config.domain,impl_->topology.dimensions(),
            static_cast<const double*>(state.domain_cuts.data()),cut_offsets,
            static_cast<int*>(state.migration_destinations.data()),cuda_stream);
        cuda::launch_build_migration_layout(
            owned_particles,impl_->config.domain,
            static_cast<const int*>(state.migration_destinations.data()),
            static_cast<int*>(state.migration_sorted_destinations.data()),
            static_cast<unsigned int*>(state.migration_indices.data()),
            static_cast<unsigned int*>(state.migration_sorted_indices.data()),
            static_cast<unsigned int*>(state.device_counts.data()),
            static_cast<unsigned int*>(state.device_offsets.data()),
            static_cast<ParticleRecord*>(state.device_send.data()),rank_count,
            state.scan_temporary.data(),state.scan_temporary.size(),cuda_stream);
        cuda::detail::check(cudaMemcpyAsync(
                                state.host_counts.data(),state.device_counts.data(),
                                rank_count*sizeof(unsigned int),cudaMemcpyDeviceToHost,
                                cuda_stream),"copy migration counts");
        state.pack_ready.record(cuda_stream);
        state.pack_ready.synchronize_host();

        const auto* raw_counts=static_cast<const unsigned int*>(state.host_counts.data());
        std::vector<unsigned int> local_counts(raw_counts,raw_counts+rank_count);
        std::vector<std::size_t> local_offsets(rank_count,0);
        for (std::size_t rank=1;rank<rank_count;++rank)
            local_offsets[rank]=local_offsets[rank-1]+local_counts[rank-1];
        const std::size_t retained_count=local_counts[static_cast<std::size_t>(local_rank)];
        local_counts[static_cast<std::size_t>(local_rank)]=0;
        std::vector<unsigned int> receive_counts;
        impl_->backend->exchange_rank_counts(local_counts,receive_counts);
        receive_counts[static_cast<std::size_t>(local_rank)]=0;
        const std::size_t receive_total=std::accumulate(
            receive_counts.begin(),receive_counts.end(),std::size_t{0});

        const bool symmetric=impl_->backend->kind()==BackendKind::Nvshmem;
        ParticleRecord* receive_base=nullptr;
        std::size_t symmetric_stride=0;
        if (symmetric) {
            symmetric_stride=impl_->backend->maximum_size(particle_count);
            if (symmetric_stride>impl_->nvshmem_migration_slot_capacity) {
                if (state.reuse_event_recorded && !initial_growth)
                    state.unpack_ready.synchronize_host();
                if (impl_->nvshmem_migration_receive)
                    impl_->backend->free_symmetric(impl_->nvshmem_migration_receive);
                if (impl_->nvshmem_migration_signals)
                    impl_->backend->free_symmetric(impl_->nvshmem_migration_signals);
                impl_->nvshmem_migration_receive=
                    impl_->backend->allocate_symmetric_zeroed(
                        rank_count*symmetric_stride,sizeof(ParticleRecord));
                impl_->nvshmem_migration_signals=static_cast<std::uint64_t*>(
                    impl_->backend->allocate_symmetric_zeroed(
                        rank_count,sizeof(std::uint64_t)));
                impl_->nvshmem_migration_slot_capacity=symmetric_stride;
#if ENABLE_PERF_COUNTERS
                if (impl_->perf_enabled) ++impl_->perf_stats.buffer_reallocations;
#endif
            }
            receive_base=static_cast<ParticleRecord*>(impl_->nvshmem_migration_receive);
            ++impl_->nvshmem_migration_epoch;
        } else {
            const std::size_t receive_bytes=std::max<std::size_t>(1,receive_total)*
                                            sizeof(ParticleRecord);
            if (state.reuse_event_recorded && !initial_growth &&
                receive_bytes>state.device_receive.capacity())
                state.unpack_ready.synchronize_host();
            state.device_receive.resize(receive_bytes);
            receive_base=static_cast<ParticleRecord*>(state.device_receive.data());
        }

        impl_->device_transfers.clear();
        std::vector<std::size_t> receive_offsets(rank_count,0);
        for (std::size_t rank=1;rank<rank_count;++rank)
            receive_offsets[rank]=receive_offsets[rank-1]+receive_counts[rank-1];
        auto* packed=static_cast<ParticleRecord*>(state.device_send.data());
        for (std::size_t peer=0;peer<rank_count;++peer) {
            if (static_cast<int>(peer)==local_rank) continue;
            detail::DeviceTransfer transfer;
            transfer.peer_rank=static_cast<int>(peer);
            transfer.send_buffer=packed+local_offsets[peer];
            transfer.send_bytes=static_cast<std::size_t>(local_counts[peer])*
                                sizeof(ParticleRecord);
            transfer.receive_bytes=static_cast<std::size_t>(receive_counts[peer])*
                                   sizeof(ParticleRecord);
            transfer.send_tag=transfer.receive_tag=240;
            if (symmetric) {
                transfer.receive_buffer=receive_base+peer*symmetric_stride;
                transfer.remote_receive_buffer=receive_base+
                    static_cast<std::size_t>(local_rank)*symmetric_stride;
                transfer.local_signal=impl_->nvshmem_migration_signals+peer;
                transfer.remote_signal=impl_->nvshmem_migration_signals+local_rank;
                transfer.signal_value=impl_->nvshmem_migration_epoch;
            } else {
                transfer.receive_buffer=receive_base+receive_offsets[peer];
            }
            impl_->device_transfers.push_back(transfer);
        }
        detail::DeviceExchange exchange{
            &impl_->device_transfers,stream,stream,"migration"};
        impl_->backend->begin_device_exchange(exchange);
        impl_->backend->end_device_exchange(exchange);

        const std::size_t new_count=retained_count+receive_total;
        const std::size_t result_bytes=std::max<std::size_t>(1,new_count)*
                                       sizeof(ParticleRecord);
        if (state.reuse_event_recorded && !initial_growth &&
            result_bytes>state.migration_temporary.capacity())
            state.unpack_ready.synchronize_host();
        state.migration_temporary.resize(result_bytes);
        auto* result_records=static_cast<ParticleRecord*>(state.migration_temporary.data());
        if (retained_count) cuda::detail::check(cudaMemcpyAsync(
            result_records,packed+local_offsets[static_cast<std::size_t>(local_rank)],
            retained_count*sizeof(ParticleRecord),cudaMemcpyDeviceToDevice,cuda_stream),
            "copy retained migration particles");
        std::size_t result_offset=retained_count;
        for (std::size_t peer=0;peer<rank_count;++peer) {
            if (!receive_counts[peer]) continue;
            const ParticleRecord* source=symmetric
                ? receive_base+peer*symmetric_stride
                : receive_base+receive_offsets[peer];
            cuda::detail::check(cudaMemcpyAsync(
                result_records+result_offset,source,
                static_cast<std::size_t>(receive_counts[peer])*sizeof(ParticleRecord),
                cudaMemcpyDeviceToDevice,cuda_stream),"compact received migration particles");
            result_offset+=receive_counts[peer];
        }
        state.unpack_ready.record(cuda_stream);
        state.reuse_event_recorded=true;
        impl_->device_migration_result={result_records,new_count};
        invalidate_halo_plan();

        MigrationResult result;
        result.previous_owned_count=particle_count;
        result.retained_count=retained_count;
        result.sent_count=particle_count-retained_count;
        result.received_count=receive_total;
        result.new_owned_count=new_count;
        result.ownership_changed=result.sent_count||result.received_count;
#if ENABLE_PERF_COUNTERS
        if (impl_->perf_enabled) {
            ++impl_->perf_stats.migration_calls;
            impl_->perf_stats.bytes_sent+=result.sent_count*sizeof(ParticleRecord);
            impl_->perf_stats.bytes_received+=result.received_count*sizeof(ParticleRecord);
            std::uint64_t capacity=state.device_send.capacity()+
                                   state.device_receive.capacity()+
                                   state.migration_temporary.capacity();
            if (symmetric) capacity+=rank_count*
                impl_->nvshmem_migration_slot_capacity*sizeof(ParticleRecord)+
                rank_count*sizeof(std::uint64_t);
            impl_->perf_stats.peak_communication_buffer_capacity=std::max(
                impl_->perf_stats.peak_communication_buffer_capacity,capacity);
        }
#endif
        return result;
#else
        (void)stream;
        throw std::invalid_argument(
            "dsDecompX: CUDA migration requires ENABLE_CUDA");
#endif
    }

#if HAS_CUDA
    impl_->device_migration_result={};
#endif

    impl_->validate_owned_view(owned_particles);
    const bool timed = impl_->perf_enabled;
    const auto classify_start = timed ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};

#ifndef NDEBUG
    std::vector<int> ids_before = impl_->backend->allgather_ids(
        owned_particles.global_id, owned_particles.count);
    ensure_unique(ids_before, "before migration");
#endif

    const std::size_t rank_count = static_cast<std::size_t>(impl_->topology.size());
    const int local_rank = impl_->topology.rank();
    impl_->migration_send_buffers.resize(rank_count);
    impl_->migration_receive_buffers.resize(rank_count);
    impl_->migration_counts.assign(rank_count, 0);
    impl_->reserve_buffer(impl_->migration_destination, owned_particles.count);
    impl_->migration_destination.resize(owned_particles.count);

    std::unordered_set<std::string> field_names;
    impl_->migration_fields.clear();
    impl_->migration_fields.reserve(owned_particles.field_count);
    for (std::size_t f=0; f<owned_particles.field_count; ++f) {
        const auto& field=owned_particles.fields[f];
        if (field.name.empty() || !field.data || field.components==0 ||
            field.capacity<owned_particles.count ||
            !field_names.insert(field.name).second) {
            throw std::invalid_argument(
                "dsDecompX: invalid, duplicate, or undersized migration field");
        }
        FieldTransferBuffers transfer;
        transfer.name=field.name;
        transfer.type=field.type;
        transfer.components=field.components;
        transfer.record_bytes=field.record_bytes();
        transfer.send.resize(rank_count);
        transfer.receive.resize(rank_count);
        transfer.result.reserve(owned_particles.count*transfer.record_bytes);
        impl_->migration_fields.push_back(std::move(transfer));
    }

    // classify -> count
    for (std::size_t i = 0; i < owned_particles.count; ++i) {
        const ParticleRecord record = wrapped_record(
            owned_particles, i, impl_->config.domain);
        const int destination = owner_rank(Vec3d{record.x, record.y, record.z});
        impl_->migration_destination[i] = static_cast<std::size_t>(destination);
        if (destination != local_rank) {
            ++impl_->migration_counts[static_cast<std::size_t>(destination)];
        }
    }

    // prefix/capacity generation; vectors retain capacity across migrations.
    for (std::size_t peer = 0; peer < rank_count; ++peer) {
        auto& buffer = impl_->migration_send_buffers[peer];
        buffer.clear();
        impl_->reserve_buffer(buffer, impl_->migration_counts[peer]);
        impl_->migration_receive_buffers[peer].clear();
    }
    impl_->migrated_owned.clear();
    impl_->reserve_buffer(impl_->migrated_owned, owned_particles.count);

    // pack and compact retained particles in their original order.
    for (std::size_t i = 0; i < owned_particles.count; ++i) {
        ParticleRecord record = wrapped_record(
            owned_particles, i, impl_->config.domain);
        const std::size_t destination = impl_->migration_destination[i];
        if (static_cast<int>(destination) == local_rank) {
            impl_->migrated_owned.push_back(record);
        } else {
            impl_->migration_send_buffers[destination].push_back(record);
        }
        for (std::size_t f=0; f<impl_->migration_fields.size(); ++f) {
            auto& transfer=impl_->migration_fields[f];
            auto& output=static_cast<int>(destination)==local_rank
                ? transfer.result : transfer.send[destination];
            const std::size_t offset=output.size();
            output.resize(offset+transfer.record_bytes);
            const auto* source=static_cast<const std::byte*>(
                owned_particles.fields[f].data);
            std::memcpy(output.data()+offset,
                        source+i*transfer.record_bytes,
                        transfer.record_bytes);
        }
    }

    MigrationResult result;
    result.previous_owned_count = owned_particles.count;
    result.retained_count = impl_->migrated_owned.size();
    result.sent_count = owned_particles.count - result.retained_count;

    const double classify_ms = timed
        ? std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - classify_start).count()
        : 0.0;

    // exchange counts -> exchange payload. The MPI backend uses direct
    // destination routing, so particles may safely cross multiple subdomains.
    impl_->backend->exchange_all_to_all_records(
        impl_->migration_send_buffers, impl_->migration_receive_buffers);
    for (auto& field : impl_->migration_fields) {
        impl_->backend->exchange_all_to_all_bytes(field.send,field.receive);
        for (std::size_t peer=0; peer<rank_count; ++peer) {
            const std::size_t expected=impl_->migration_receive_buffers[peer].size()*
                                       field.record_bytes;
            if (field.receive[peer].size()!=expected) {
                throw std::runtime_error(
                    "dsDecompX: custom migration field schema differs across ranks");
            }
            field.result.insert(field.result.end(),field.receive[peer].begin(),
                                field.receive[peer].end());
        }
    }

    const auto compact_start = timed ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
    for (std::size_t peer = 0; peer < rank_count; ++peer) {
        const auto& incoming = impl_->migration_receive_buffers[peer];
        result.received_count += incoming.size();
        impl_->reserve_buffer(impl_->migrated_owned,
                              impl_->migrated_owned.size() + incoming.size());
        impl_->migrated_owned.insert(impl_->migrated_owned.end(),
                                     incoming.begin(), incoming.end());
    }
    result.new_owned_count = impl_->migrated_owned.size();
    result.ownership_changed = result.sent_count != 0 || result.received_count != 0;

#if ENABLE_PERF_COUNTERS
    if (impl_->perf_enabled) {
        ++impl_->perf_stats.migration_calls;
        impl_->perf_stats.bytes_sent += result.sent_count * sizeof(ParticleRecord);
        impl_->perf_stats.bytes_received += result.received_count * sizeof(ParticleRecord);
        for (const auto& field : impl_->migration_fields) {
            for (std::size_t peer=0; peer<field.send.size(); ++peer) {
                impl_->perf_stats.bytes_sent+=field.send[peer].size();
                impl_->perf_stats.bytes_received+=field.receive[peer].size();
            }
        }
        impl_->perf_stats.migration_classify_ms.add(classify_ms);
        impl_->perf_stats.migration_compact_ms.add(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - compact_start).count());
        std::uint64_t capacity = impl_->migrated_owned.capacity() *
                                 sizeof(ParticleRecord);
        for (std::size_t peer = 0; peer < rank_count; ++peer) {
            capacity += (impl_->migration_send_buffers[peer].capacity() +
                         impl_->migration_receive_buffers[peer].capacity()) *
                        sizeof(ParticleRecord);
        }
        for (const auto& field : impl_->migration_fields) {
            capacity+=field.result.capacity();
            for (std::size_t peer=0; peer<field.send.size(); ++peer) {
                capacity+=field.send[peer].capacity()+field.receive[peer].capacity();
            }
        }
        impl_->perf_stats.peak_communication_buffer_capacity = std::max(
            impl_->perf_stats.peak_communication_buffer_capacity, capacity);
    }
#else
    (void)classify_ms;
    (void)compact_start;
#endif

#ifndef NDEBUG
    std::vector<int> local_ids;
    local_ids.reserve(impl_->migrated_owned.size());
    const LocalDomain& local = impl_->topology.local_domain();
    for (const ParticleRecord& record : impl_->migrated_owned) {
        local_ids.push_back(record.global_id);
        if (!(record.x >= local.lower[0] && record.x < local.upper[0] &&
              record.y >= local.lower[1] && record.y < local.upper[1] &&
              record.z >= local.lower[2] && record.z < local.upper[2])) {
            throw std::runtime_error(
                "dsDecompX: migrated particle is outside its owning local domain");
        }
        if (!(record.x >= impl_->config.domain.lower[0] &&
              record.x < impl_->config.domain.upper[0] &&
              record.y >= impl_->config.domain.lower[1] &&
              record.y < impl_->config.domain.upper[1] &&
              record.z >= impl_->config.domain.lower[2] &&
              record.z < impl_->config.domain.upper[2])) {
            throw std::runtime_error("dsDecompX: periodic coordinate wrap is inconsistent");
        }
    }
    std::vector<int> ids_after = impl_->backend->allgather_ids(
        local_ids.data(), local_ids.size());
    ensure_unique(ids_after, "after migration");
    std::sort(ids_before.begin(), ids_before.end());
    std::sort(ids_after.begin(), ids_after.end());
    if (ids_before != ids_after) {
        throw std::runtime_error(
            "dsDecompX: global particle IDs/count changed during migration");
    }
#endif

    invalidate_halo_plan();
    return result;
}

} // namespace dsdecompx
