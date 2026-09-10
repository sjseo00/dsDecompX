#include "decomposition.hpp"

#include "internal/decomp_impl.hpp"
#include "internal/buffer_pool.hpp"

#include <stdexcept>
#include <chrono>
#include <cstring>
#include <numeric>
#include <unordered_set>
#if HAS_CUDA
#include "cuda/execution.hpp"
#include "cuda/kernels.hpp"
#include "cuda/cuda_error.hpp"
#include <cuda_runtime_api.h>
#endif

namespace dsdecompx {
namespace {

constexpr int kForwardCountTagBase = 0;
constexpr int kForwardDataTagBase = 26;
constexpr int kReverseCountTagBase = 52;
constexpr int kReverseDataTagBase = 78;
constexpr int kDeviceCountTagBase = 104;
constexpr int kDeviceDataTagBase = 136;

#if HAS_CUDA
int opposite_direction_slot(const Neighbor& neighbor) {
    const int dx=-neighbor.direction[0], dy=-neighbor.direction[1], dz=-neighbor.direction[2];
    const int raw=(dx+1)*9+(dy+1)*3+(dz+1);
    return raw < 13 ? raw : raw-1;
}
#endif

ParticleRecord make_record(const ConstParticleView& particles,
                           std::size_t index,
                           const Vec3d& shift) {
    ParticleRecord record;
    record.x = particles.x[index] + shift[0];
    record.y = particles.y[index] + shift[1];
    record.z = particles.z[index] + shift[2];
    if (particles.vx) record.vx = particles.vx[index];
    if (particles.vy) record.vy = particles.vy[index];
    if (particles.vz) record.vz = particles.vz[index];
    record.fx = 0.0;
    record.fy = 0.0;
    record.fz = 0.0;
    if (particles.mass) record.mass = particles.mass[index];
    if (particles.charge) record.charge = particles.charge[index];
    if (particles.type) record.type = particles.type[index];
    record.global_id = particles.global_id[index];
    return record;
}

void validate_fields(const ConstParticleView& particles) {
    std::unordered_set<std::string> names;
    for (std::size_t f=0; f<particles.field_count; ++f) {
        const auto& field=particles.fields[f];
        if (field.name.empty() || !field.data || field.components==0 ||
            field.capacity<particles.count) {
            throw std::invalid_argument(
                "dsDecompX: invalid or undersized custom particle field");
        }
        if (!names.insert(field.name).second) {
            throw std::invalid_argument(
                "dsDecompX: duplicate custom particle field '"+field.name+"'");
        }
    }
}

const ParticleFieldView& require_destination_field(
    const ParticleView& destination, const FieldTransferBuffers& transfer,
    std::size_t required) {
    const ParticleFieldView* begin=destination.fields;
    const ParticleFieldView* end=begin+destination.field_count;
    const auto found=std::find_if(
        begin,end,
        [&](const ParticleFieldView& field) { return field.name==transfer.name; });
    if (found==end || found->type!=transfer.type ||
        found->components!=transfer.components || !found->data ||
        found->capacity<required) {
        throw std::invalid_argument(
            "dsDecompX: missing, incompatible, or undersized output field '"+
            transfer.name+"'");
    }
    return *found;
}

template<class ImplType>
void pack_halo_fields(ImplType& impl, const ConstParticleView& particles) {
    validate_fields(particles);
    impl.halo_fields.clear();
    impl.halo_fields.reserve(particles.field_count);
    for (std::size_t f=0; f<particles.field_count; ++f) {
        const auto& field=particles.fields[f];
        FieldTransferBuffers transfer;
        transfer.name=field.name;
        transfer.type=field.type;
        transfer.components=field.components;
        transfer.record_bytes=field.record_bytes();
        transfer.send.resize(impl.halo_plan.size());
        transfer.receive.resize(impl.halo_plan.size());
        const auto* source=static_cast<const std::byte*>(field.data);
        for (std::size_t n=0; n<impl.halo_plan.size(); ++n) {
            const auto& indices=impl.halo_plan[n].source_indices;
            auto& output=transfer.send[n];
            output.resize(indices.size()*transfer.record_bytes);
            for (std::size_t k=0; k<indices.size(); ++k) {
                std::memcpy(output.data()+k*transfer.record_bytes,
                            source+static_cast<std::size_t>(indices[k])*
                                transfer.record_bytes,
                            transfer.record_bytes);
            }
        }
        impl.halo_fields.push_back(std::move(transfer));
    }
}

} // namespace

void Decomposition::Impl::validate_owned_view(
    const ConstParticleView& particles) const {
    if (particles.memory_space != MemorySpace::Host) {
        throw std::invalid_argument(
            "dsDecompX: this host packing operation requires Host particle memory");
    }
    if (particles.count == 0) return;
    if (!particles.x || !particles.y || !particles.z ||
        !particles.global_id) {
        throw std::invalid_argument("dsDecompX: incomplete particle view");
    }
}

void Decomposition::rebuild_halo_plan(
    const ConstParticleView& owned_particles) {
    impl_->validate_owned_view(owned_particles);

    const auto& neighbors = impl_->topology.neighbors();
    const LocalDomain& local = impl_->topology.local_domain();
    const double width = impl_->config.halo.width;

    impl_->halo_plan.resize(neighbors.size());
    impl_->halo_send_buffers.resize(neighbors.size());
    impl_->halo_receive_buffers.resize(neighbors.size());
    std::vector<std::size_t> source_capacities;
    std::vector<std::size_t> send_capacities;
    if (impl_->perf_enabled) {
        source_capacities.resize(neighbors.size());
        send_capacities.resize(neighbors.size());
    }
    for (std::size_t n = 0; n < neighbors.size(); ++n) {
        if (impl_->perf_enabled) {
            source_capacities[n] = impl_->halo_plan[n].source_indices.capacity();
            send_capacities[n] = impl_->halo_send_buffers[n].capacity();
        }
        impl_->halo_plan[n].peer_rank = neighbors[n].rank;
        impl_->halo_plan[n].direction = neighbors[n].direction;
        impl_->halo_plan[n].periodic_shift = neighbors[n].periodic_shift;
        impl_->halo_plan[n].source_indices.clear();
        impl_->halo_plan[n].receive_count = 0;
        impl_->halo_send_buffers[n].clear();
        impl_->halo_receive_buffers[n].clear();
    }

    for (std::size_t i = 0; i < owned_particles.count; ++i) {
        const double px = owned_particles.x[i];
        const double py = owned_particles.y[i];
        const double pz = owned_particles.z[i];

        const bool in_xlo = px < local.lower[0] + width;
        const bool in_xhi = px > local.upper[0] - width;
        const bool in_ylo = py < local.lower[1] + width;
        const bool in_yhi = py > local.upper[1] - width;
        const bool in_zlo = pz < local.lower[2] + width;
        const bool in_zhi = pz > local.upper[2] - width;

        if (!in_xlo && !in_xhi && !in_ylo && !in_yhi &&
            !in_zlo && !in_zhi) {
            continue;
        }

        for (std::size_t n = 0; n < neighbors.size(); ++n) {
            const Neighbor& neighbor = neighbors[n];
            if (neighbor.rank < 0) continue;
            if (neighbor.direction[0] < 0 && !in_xlo) continue;
            if (neighbor.direction[0] > 0 && !in_xhi) continue;
            if (neighbor.direction[1] < 0 && !in_ylo) continue;
            if (neighbor.direction[1] > 0 && !in_yhi) continue;
            if (neighbor.direction[2] < 0 && !in_zlo) continue;
            if (neighbor.direction[2] > 0 && !in_zhi) continue;

            impl_->halo_plan[n].source_indices.push_back(static_cast<int>(i));
            impl_->halo_send_buffers[n].push_back(
                make_record(owned_particles, i, neighbor.periodic_shift));
        }
    }

    if (impl_->perf_enabled) {
        for (std::size_t n = 0; n < neighbors.size(); ++n) {
            if (impl_->halo_plan[n].source_indices.capacity() != source_capacities[n])
                ++impl_->perf_stats.buffer_reallocations;
            if (impl_->halo_send_buffers[n].capacity() != send_capacities[n])
                ++impl_->perf_stats.buffer_reallocations;
        }
    }

    impl_->halo_plan_valid = true;
}

void Decomposition::refresh_halo_buffers(
    const ConstParticleView& owned_particles) {
    impl_->validate_owned_view(owned_particles);
    if (!impl_->halo_plan_valid) {
        rebuild_halo_plan(owned_particles);
        return;
    }

    for (std::size_t n = 0; n < impl_->halo_plan.size(); ++n) {
        const auto& indices = impl_->halo_plan[n].source_indices;
        auto& buffer = impl_->halo_send_buffers[n];
        if (indices.size() != buffer.size()) {
            rebuild_halo_plan(owned_particles);
            return;
        }

        const Vec3d& shift = impl_->halo_plan[n].periodic_shift;
        for (std::size_t k = 0; k < indices.size(); ++k) {
            const int i = indices[k];
            if (i < 0 || static_cast<std::size_t>(i) >= owned_particles.count) {
                rebuild_halo_plan(owned_particles);
                return;
            }
            buffer[k].x = owned_particles.x[i] + shift[0];
            buffer[k].y = owned_particles.y[i] + shift[1];
            buffer[k].z = owned_particles.z[i] + shift[2];
        }
    }
}

void Decomposition::exchange_halo(const ConstParticleView& owned_particles) {
    auto handle = begin_halo_exchange(owned_particles);
    end_halo_exchange(handle);
}

HaloExchangeHandle Decomposition::begin_halo_exchange(
    const ConstParticleView& owned_particles, ExecutionStream stream) {
    (void)stream;
    if (impl_->operation_active) {
        throw std::logic_error("dsDecompX: an operation is already in flight");
    }
    impl_->operation_active = true;
#if HAS_CUDA
    if (owned_particles.memory_space == MemorySpace::Host)
        impl_->device_halo_results.clear();
#endif
    const std::uint64_t generation = ++impl_->operation_generation;
    const bool timed = impl_->timing_enabled || impl_->perf_enabled;
    if (timed) {
        impl_->halo_timings = {};
        impl_->exchange_start = std::chrono::steady_clock::now();
    }

    const auto pack_start = timed ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};

    // A single rank owns the full periodic box. Generating the 26 Cartesian
    // self-images would duplicate interactions; match the established host
    // semantics for both host and CUDA views.
    if (impl_->topology.size() == 1) {
#if HAS_CUDA
        impl_->device_halo_results.clear();
#endif
        for (auto& buffer : impl_->halo_send_buffers) buffer.clear();
        for (auto& buffer : impl_->halo_receive_buffers) buffer.clear();
        for (auto& entry : impl_->halo_plan) {
            entry.source_indices.clear();
            entry.receive_count = 0;
        }
        impl_->halo_plan_valid = false;
        impl_->halo_fields.clear();
        if (timed) {
            impl_->halo_timings.pack_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              pack_start).count();
        }
        return HaloExchangeHandle(this, generation);
    }

    if (owned_particles.memory_space != MemorySpace::Host) {
#if HAS_CUDA
        if (owned_particles.field_count!=0) {
            impl_->operation_active=false;
            throw std::runtime_error(
                "dsDecompX: custom fields currently require host particle memory");
        }
        if (owned_particles.memory_space != MemorySpace::CudaDevice &&
            owned_particles.memory_space != MemorySpace::CudaManaged) {
            impl_->operation_active = false;
            throw std::invalid_argument("dsDecompX: unsupported particle memory space");
        }
        if (!stream) {
            impl_->operation_active = false;
            throw std::invalid_argument(
                "dsDecompX CUDA halo requires an explicit communication stream");
        }
        if (!impl_->backend->supports_device_data_plane()) {
            impl_->operation_active = false;
            throw std::runtime_error(
                "dsDecompX: selected backend cannot exchange GPU-resident buffers");
        }
        if (owned_particles.count &&
            (!owned_particles.x || !owned_particles.y || !owned_particles.z ||
             !owned_particles.vx || !owned_particles.vy || !owned_particles.vz ||
             !owned_particles.mass || !owned_particles.charge ||
             !owned_particles.type || !owned_particles.global_id)) {
            impl_->operation_active = false;
            throw std::invalid_argument("dsDecompX: incomplete CUDA particle view");
        }

        try {
            for (auto& buffer : impl_->halo_receive_buffers) buffer.clear();
            if (!impl_->cuda_execution) {
                impl_->cuda_execution = std::make_unique<cuda::ExecutionState>();
            }
            cuda::ExecutionState& state = *impl_->cuda_execution;
            const auto& neighbors = impl_->topology.neighbors();
            const std::size_t neighbor_count = neighbors.size();
            const std::size_t particle_count = owned_particles.count;
            const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

            // Keep a valid base pointer even for an empty rank. The kernels
            // return immediately for count==0, while transfer descriptors may
            // still form zero-length slice pointers.
            const std::size_t send_bytes=std::max<std::size_t>(
                1,neighbor_count*particle_count)*sizeof(ParticleRecord);
            const std::size_t count_bytes=neighbor_count*sizeof(unsigned int);
            const std::size_t flag_bytes=particle_count*sizeof(unsigned int);
            const std::size_t scan_bytes = particle_count
                ? cuda::prefix_scan_temporary_bytes(particle_count) : 0;
            const bool initial_growth=send_bytes>state.device_send.capacity() ||
                count_bytes>state.device_counts.capacity() ||
                flag_bytes>state.halo_flags.capacity() ||
                flag_bytes>state.halo_offsets.capacity() ||
                scan_bytes>state.scan_temporary.capacity();
            if (state.reuse_event_recorded) {
                if (initial_growth) state.unpack_ready.synchronize_host();
                else state.unpack_ready.wait(cuda_stream);
            }
            state.device_send.resize(send_bytes);
            state.device_counts.resize(count_bytes);
            state.halo_flags.resize(flag_bytes);
            state.halo_offsets.resize(flag_bytes);
            state.scan_temporary.resize(scan_bytes);
            state.host_counts.resize(neighbor_count * sizeof(unsigned int));

            cuda::detail::check(cudaMemsetAsync(
                                    state.device_counts.data(), 0,
                                    state.device_counts.size(), cuda_stream),
                                "clear halo counts");
            auto* send_records = static_cast<ParticleRecord*>(state.device_send.data());
            auto* device_counts = static_cast<unsigned int*>(state.device_counts.data());
            for (std::size_t n=0; n<neighbor_count; ++n) {
                if (neighbors[n].rank < 0) continue;
                cuda::launch_pack_halo_neighbor(
                    owned_particles, impl_->topology.local_domain(),
                    neighbors[n].direction, neighbors[n].periodic_shift,
                    impl_->config.halo.width,
                    static_cast<unsigned int*>(state.halo_flags.data()),
                    static_cast<unsigned int*>(state.halo_offsets.data()),
                    send_records + n*particle_count, device_counts+n,
                    state.scan_temporary.data(), state.scan_temporary.size(),
                    cuda_stream);
            }
            cuda::detail::check(cudaMemcpyAsync(
                                    state.host_counts.data(), state.device_counts.data(),
                                    neighbor_count*sizeof(unsigned int),
                                    cudaMemcpyDeviceToHost, cuda_stream),
                                "copy halo counts to control plane");
            state.pack_ready.record(cuda_stream);
            // Count metadata is needed to size persistent receive storage and
            // post sparse transfers. This waits only for pack/count readiness,
            // never for the communication payload or the whole device.
            state.pack_ready.synchronize_host();

            auto* host_send_counts =
                static_cast<unsigned int*>(state.host_counts.data());
            impl_->device_halo_send_counts.assign(
                host_send_counts, host_send_counts+neighbor_count);
            impl_->backend->exchange_neighbor_counts(
                neighbors, impl_->device_halo_send_counts,
                impl_->device_halo_receive_counts, kDeviceCountTagBase);

            const bool symmetric = impl_->backend->kind()==BackendKind::Nvshmem;
            std::size_t receive_total = 0;
            for (unsigned int count : impl_->device_halo_receive_counts)
                receive_total += count;

            ParticleRecord* receive_base = nullptr;
            std::size_t symmetric_stride = 0;
            if (symmetric) {
                symmetric_stride = impl_->backend->maximum_size(particle_count);
                if (symmetric_stride > impl_->nvshmem_halo_slot_capacity) {
                    if (state.reuse_event_recorded && !initial_growth)
                        state.unpack_ready.synchronize_host();
                    if (impl_->nvshmem_halo_receive)
                        impl_->backend->free_symmetric(impl_->nvshmem_halo_receive);
                    if (impl_->nvshmem_halo_signals)
                        impl_->backend->free_symmetric(impl_->nvshmem_halo_signals);
                    impl_->nvshmem_halo_receive =
                        impl_->backend->allocate_symmetric_zeroed(
                            26*symmetric_stride, sizeof(ParticleRecord));
                    impl_->nvshmem_halo_signals =
                        static_cast<std::uint64_t*>(
                            impl_->backend->allocate_symmetric_zeroed(
                                26, sizeof(std::uint64_t)));
                    impl_->nvshmem_halo_slot_capacity = symmetric_stride;
#if ENABLE_PERF_COUNTERS
                    if (impl_->perf_enabled) ++impl_->perf_stats.buffer_reallocations;
#endif
                }
                receive_base = static_cast<ParticleRecord*>(
                    impl_->nvshmem_halo_receive);
                ++impl_->nvshmem_halo_epoch;
            } else {
                const std::size_t receive_bytes=std::max<std::size_t>(1,receive_total)*
                                                sizeof(ParticleRecord);
                if (state.reuse_event_recorded && !initial_growth &&
                    receive_bytes>state.device_receive.capacity())
                    state.unpack_ready.synchronize_host();
                state.device_receive.resize(receive_bytes);
                receive_base = static_cast<ParticleRecord*>(state.device_receive.data());
            }

            impl_->device_transfers.clear();
            impl_->device_halo_results.clear();
            impl_->device_transfers.reserve(neighbor_count);
            impl_->device_halo_results.reserve(neighbor_count);
            std::size_t receive_offset = 0;
            for (std::size_t n=0; n<neighbor_count; ++n) {
                const Neighbor& neighbor = neighbors[n];
                detail::DeviceTransfer transfer;
                transfer.peer_rank = neighbor.rank;
                transfer.send_buffer = send_records + n*particle_count;
                transfer.send_bytes =
                    static_cast<std::size_t>(impl_->device_halo_send_counts[n]) *
                    sizeof(ParticleRecord);
                transfer.receive_bytes =
                    static_cast<std::size_t>(impl_->device_halo_receive_counts[n]) *
                    sizeof(ParticleRecord);
                transfer.send_tag = kDeviceDataTagBase + neighbor.direction_slot;
                transfer.receive_tag =
                    kDeviceDataTagBase + opposite_direction_slot(neighbor);
                if (symmetric) {
                    const std::size_t local_slot =
                        static_cast<std::size_t>(neighbor.direction_slot);
                    const std::size_t remote_slot =
                        static_cast<std::size_t>(opposite_direction_slot(neighbor));
                    transfer.receive_buffer = receive_base + local_slot*symmetric_stride;
                    transfer.remote_receive_buffer = receive_base + remote_slot*symmetric_stride;
                    transfer.local_signal = impl_->nvshmem_halo_signals + local_slot;
                    transfer.remote_signal = impl_->nvshmem_halo_signals + remote_slot;
                    transfer.signal_value = impl_->nvshmem_halo_epoch;
                } else {
                    transfer.receive_buffer = receive_base + receive_offset;
                }
                impl_->device_transfers.push_back(transfer);
                impl_->device_halo_results.push_back(DeviceHaloBuffer{
                    neighbor.rank,
                    static_cast<const ParticleRecord*>(transfer.receive_buffer),
                    impl_->device_halo_receive_counts[n]});
                receive_offset += impl_->device_halo_receive_counts[n];
            }

            detail::DeviceExchange exchange{
                &impl_->device_transfers, stream, stream, "halo"};
            impl_->backend->begin_device_exchange(exchange);
            impl_->device_operation_active = true;
            impl_->device_communication_stream = stream;
            if (timed) {
                impl_->halo_timings.pack_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now()-
                                                  pack_start).count();
            }
            return HaloExchangeHandle(this, generation);
        } catch (...) {
            impl_->operation_active = false;
            impl_->device_operation_active = false;
            impl_->device_communication_stream = nullptr;
            throw;
        }
#else
        impl_->operation_active = false;
        (void)stream;
        throw std::invalid_argument(
            "dsDecompX: CUDA particle view requires ENABLE_CUDA");
#endif
    }
    if (impl_->halo_plan_valid) {
        refresh_halo_buffers(owned_particles);
    } else {
        rebuild_halo_plan(owned_particles);
    }

    pack_halo_fields(*impl_,owned_particles);

    if (timed) {
        impl_->halo_timings.pack_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          pack_start).count();
    }

    const auto communication_start = timed ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};
    try {
        impl_->backend->begin_exchange_records(
            impl_->topology.neighbors(),
            impl_->halo_send_buffers,
            impl_->halo_receive_buffers,
            kForwardCountTagBase,
            kForwardDataTagBase,
            true);
    } catch (...) {
        impl_->operation_active = false;
        throw;
    }
    if (timed) {
        impl_->halo_timings.communication_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          communication_start).count();
    }
    return HaloExchangeHandle(this, generation);
}

void Decomposition::end_halo_exchange(HaloExchangeHandle& handle,
                                      ExecutionStream stream) {
#if HAS_CUDA
    if (!stream && impl_->device_operation_active)
        stream = impl_->device_communication_stream;
#endif
    (void)stream;
    if (!handle.active_ || handle.owner_ != this ||
        handle.generation_ != impl_->operation_generation ||
        !impl_->operation_active) {
        throw std::logic_error("dsDecompX: invalid or stale halo exchange handle");
    }
    const bool timed = impl_->timing_enabled || impl_->perf_enabled;
    const auto wait_start = timed ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
#if HAS_CUDA
    if (impl_->device_operation_active) {
        detail::DeviceExchange exchange{
            &impl_->device_transfers, stream, stream, "halo"};
        impl_->backend->end_device_exchange(exchange);
        impl_->cuda_execution->receive_ready.record(
            static_cast<cudaStream_t>(stream));
        impl_->cuda_execution->unpack_ready.record(
            static_cast<cudaStream_t>(stream));
        impl_->cuda_execution->reuse_event_recorded = true;
        impl_->device_operation_active = false;
    } else if (impl_->topology.size() != 1) {
        impl_->backend->end_exchange_records();
    }
#else
    if (impl_->topology.size() != 1) {
        impl_->backend->end_exchange_records();
    }
#endif
    if (impl_->topology.size()!=1) {
        for (auto& field : impl_->halo_fields) {
            impl_->backend->exchange_neighbor_bytes(
                impl_->topology.neighbors(),field.send,field.receive,
                kForwardCountTagBase,kForwardDataTagBase,true);
            for (std::size_t n=0; n<field.receive.size(); ++n) {
                const std::size_t expected=impl_->halo_receive_buffers[n].size()*
                                           field.record_bytes;
                if (field.receive[n].size()!=expected) {
                    throw std::runtime_error(
                        "dsDecompX: custom halo field schema differs across ranks");
                }
            }
        }
    }
    if (timed) {
        impl_->halo_timings.wait_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          wait_start).count();
    }

    const auto unpack_start = timed ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
#if HAS_CUDA
    if (impl_->device_halo_results.empty())
#endif
    {
        for (std::size_t n = 0; n < impl_->halo_plan.size(); ++n) {
            impl_->halo_plan[n].receive_count = impl_->halo_receive_buffers[n].size();
        }
    }
    if (timed) {
        impl_->halo_timings.unpack_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          unpack_start).count();
        impl_->halo_timings.total_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          impl_->exchange_start).count();
    }
    impl_->operation_active = false;
#if HAS_CUDA
    impl_->device_communication_stream = nullptr;
#endif
    handle.active_ = false;
    handle.owner_ = nullptr;

#if ENABLE_PERF_COUNTERS
    if (impl_->perf_enabled) {
        ++impl_->perf_stats.halo_calls;
        std::uint64_t sent = 0;
        std::uint64_t received = 0;
        std::uint64_t capacity = 0;
        const auto& neighbors = impl_->topology.neighbors();
        if (impl_->neighbor_detail_enabled &&
            impl_->perf_stats.neighbor_bytes.size() != neighbors.size()) {
            impl_->perf_stats.neighbor_bytes.resize(neighbors.size());
            for (std::size_t n = 0; n < neighbors.size(); ++n)
                impl_->perf_stats.neighbor_bytes[n].peer_rank = neighbors[n].rank;
        }
#if HAS_CUDA
        if (!impl_->device_halo_results.empty()) {
            for (std::size_t n=0; n<impl_->device_halo_results.size(); ++n) {
                const std::uint64_t send_bytes =
                    static_cast<std::uint64_t>(impl_->device_halo_send_counts[n]) *
                    sizeof(ParticleRecord);
                const std::uint64_t receive_bytes =
                    static_cast<std::uint64_t>(impl_->device_halo_receive_counts[n]) *
                    sizeof(ParticleRecord);
                sent += send_bytes;
                received += receive_bytes;
                if (impl_->neighbor_detail_enabled) {
                    impl_->perf_stats.neighbor_bytes[n].bytes_sent += send_bytes;
                    impl_->perf_stats.neighbor_bytes[n].bytes_received += receive_bytes;
                }
            }
            capacity = impl_->cuda_execution->device_send.capacity() +
                       impl_->cuda_execution->device_receive.capacity();
            if (impl_->backend->kind()==BackendKind::Nvshmem) {
                capacity += 26*impl_->nvshmem_halo_slot_capacity*
                            sizeof(ParticleRecord)+26*sizeof(std::uint64_t);
            }
        } else
#endif
        for (std::size_t n = 0; n < impl_->halo_send_buffers.size(); ++n) {
            const std::uint64_t send_bytes = impl_->halo_send_buffers[n].size() *
                                              sizeof(ParticleRecord);
            const std::uint64_t receive_bytes = impl_->halo_receive_buffers[n].size() *
                                                 sizeof(ParticleRecord);
            sent += send_bytes;
            received += receive_bytes;
            capacity += (impl_->halo_send_buffers[n].capacity() +
                         impl_->halo_receive_buffers[n].capacity()) *
                        sizeof(ParticleRecord);
            if (impl_->neighbor_detail_enabled) {
                impl_->perf_stats.neighbor_bytes[n].bytes_sent += send_bytes;
                impl_->perf_stats.neighbor_bytes[n].bytes_received += receive_bytes;
            }
        }
        for (const auto& field : impl_->halo_fields) {
            for (std::size_t n=0; n<field.send.size(); ++n) {
                sent+=field.send[n].size();
                received+=field.receive[n].size();
                capacity+=field.send[n].capacity()+field.receive[n].capacity();
                if (impl_->neighbor_detail_enabled) {
                    impl_->perf_stats.neighbor_bytes[n].bytes_sent+=
                        field.send[n].size();
                    impl_->perf_stats.neighbor_bytes[n].bytes_received+=
                        field.receive[n].size();
                }
            }
        }
        impl_->perf_stats.bytes_sent += sent;
        impl_->perf_stats.bytes_received += received;
        impl_->perf_stats.peak_communication_buffer_capacity = std::max(
            impl_->perf_stats.peak_communication_buffer_capacity, capacity);
        impl_->perf_stats.pack_ms.add(impl_->halo_timings.pack_seconds * 1000.0);
        impl_->perf_stats.communication_ms.add(
            impl_->halo_timings.communication_seconds * 1000.0);
        impl_->perf_stats.wait_ms.add(impl_->halo_timings.wait_seconds * 1000.0);
        impl_->perf_stats.unpack_ms.add(impl_->halo_timings.unpack_seconds * 1000.0);
    }
#endif
}

void Decomposition::unpack_halo_fields(ParticleView destination,
                                       std::size_t destination_offset) {
    std::size_t total=0;
    for (const auto& buffer : impl_->halo_receive_buffers) total+=buffer.size();
    const std::size_t required=destination_offset+total;
    for (const auto& transfer : impl_->halo_fields) {
        const auto& field=require_destination_field(destination,transfer,required);
        auto* output=static_cast<std::byte*>(field.data)+
                     destination_offset*transfer.record_bytes;
        for (const auto& buffer : transfer.receive) {
            if (!buffer.empty()) {
                std::memcpy(output,buffer.data(),buffer.size());
                output+=buffer.size();
            }
        }
    }
}

const std::vector<DeviceHaloBuffer>&
Decomposition::received_device_halos() const noexcept {
#if HAS_CUDA
    return impl_->device_halo_results;
#else
    static const std::vector<DeviceHaloBuffer> empty;
    return empty;
#endif
}

void Decomposition::unpack_device_halos(ParticleView destination,
                                        std::size_t destination_offset,
                                        ExecutionStream stream) {
#if HAS_CUDA
    if (!stream) throw std::invalid_argument(
        "dsDecompX: device halo unpack requires an explicit stream");
    if (destination.memory_space != MemorySpace::CudaDevice &&
        destination.memory_space != MemorySpace::CudaManaged) {
        throw std::invalid_argument("dsDecompX: device halo unpack needs device output");
    }
    std::size_t total=0;
    for (const DeviceHaloBuffer& buffer : impl_->device_halo_results) total+=buffer.count;
    if (destination_offset+total > destination.count) {
        throw std::invalid_argument("dsDecompX: device halo output capacity is too small");
    }
    const cudaStream_t cuda_stream=static_cast<cudaStream_t>(stream);
    std::size_t offset=destination_offset;
    for (const DeviceHaloBuffer& buffer : impl_->device_halo_results) {
        cuda::launch_unpack_records(buffer.data,buffer.count,destination,offset,cuda_stream);
        offset+=buffer.count;
    }
    impl_->cuda_execution->unpack_ready.record(cuda_stream);
    impl_->cuda_execution->reuse_event_recorded = true;
#else
    (void)destination; (void)destination_offset; (void)stream;
    throw std::runtime_error("dsDecompX: CUDA execution layer is not compiled");
#endif
}

void Decomposition::reverse_halo(ParticleView particles,
                                 std::size_t owned_count,
                                 ExecutionStream stream) {
    if (impl_->operation_active) {
        throw std::logic_error("dsDecompX: reverse halo cannot overlap an active operation");
    }
    if (particles.memory_space == MemorySpace::CudaDevice ||
        particles.memory_space == MemorySpace::CudaManaged) {
#if HAS_CUDA
        if (!stream) {
            throw std::invalid_argument(
                "dsDecompX: device reverse halo requires an explicit stream");
        }
        if (impl_->backend->kind()==BackendKind::Nvshmem) {
            throw std::runtime_error(
                "dsDecompX: GPU reverse halo is not yet available for NVSHMEM");
        }
        const std::size_t ghost_count=total_received_halo_particles();
        if (particles.count<owned_count+ghost_count || !particles.fx ||
            !particles.fy || !particles.fz || !particles.global_id) {
            throw std::invalid_argument("dsDecompX: invalid device reverse-halo view");
        }
        auto& state=*impl_->cuda_execution;
        const auto& neighbors=impl_->topology.neighbors();
        const std::size_t send_total=std::accumulate(
            impl_->device_halo_receive_counts.begin(),
            impl_->device_halo_receive_counts.end(),std::size_t{0});
        const std::size_t receive_total=std::accumulate(
            impl_->device_halo_send_counts.begin(),
            impl_->device_halo_send_counts.end(),std::size_t{0});
        state.device_send.resize(std::max<std::size_t>(1,send_total)*
                                 sizeof(ParticleRecord));
        state.device_receive.resize(std::max<std::size_t>(1,receive_total)*
                                    sizeof(ParticleRecord));
        auto* send_base=static_cast<ParticleRecord*>(state.device_send.data());
        auto* receive_base=static_cast<ParticleRecord*>(state.device_receive.data());
        const cudaStream_t cuda_stream=static_cast<cudaStream_t>(stream);

        impl_->device_transfers.clear();
        impl_->device_transfers.reserve(neighbors.size());
        std::size_t ghost_offset=owned_count;
        std::size_t send_offset=0,receive_offset=0;
        for (std::size_t n=0;n<neighbors.size();++n) {
            const std::size_t send_count=impl_->device_halo_receive_counts[n];
            const std::size_t receive_count=impl_->device_halo_send_counts[n];
            cuda::launch_pack_reverse_records(
                particles,ghost_offset,send_count,send_base+send_offset,cuda_stream);
            detail::DeviceTransfer transfer;
            transfer.peer_rank=neighbors[n].rank;
            transfer.send_buffer=send_base+send_offset;
            transfer.send_bytes=send_count*sizeof(ParticleRecord);
            transfer.receive_buffer=receive_base+receive_offset;
            transfer.receive_bytes=receive_count*sizeof(ParticleRecord);
            transfer.send_tag=kReverseDataTagBase+neighbors[n].direction_slot;
            transfer.receive_tag=kReverseDataTagBase+
                                 opposite_direction_slot(neighbors[n]);
            impl_->device_transfers.push_back(transfer);
            ghost_offset+=send_count;
            send_offset+=send_count;
            receive_offset+=receive_count;
        }
        detail::DeviceExchange exchange{
            &impl_->device_transfers,stream,stream,"reverse-halo"};
        impl_->backend->begin_device_exchange(exchange);
        impl_->backend->end_device_exchange(exchange);
        cuda::launch_accumulate_reverse_records(
            receive_base,receive_total,particles,owned_count,cuda_stream);
        return;
#else
        (void)stream;
        throw std::runtime_error("dsDecompX: CUDA execution layer is not compiled");
#endif
    }
    if (particles.memory_space != MemorySpace::Host) {
        throw std::invalid_argument(
            "dsDecompX: reverse halo currently requires Host particle memory");
    }
    const std::size_t ghost_count = total_received_halo_particles();
    if (particles.count < owned_count + ghost_count ||
        !particles.fx || !particles.fy || !particles.fz) {
        throw std::invalid_argument("dsDecompX: invalid reverse-halo particle view");
    }
    ParticleFieldView force_fields[] = {
        {"__force_x",FieldType::Float64,1,particles.count,particles.fx},
        {"__force_y",FieldType::Float64,1,particles.count,particles.fy},
        {"__force_z",FieldType::Float64,1,particles.count,particles.fz}};
    particles.fields=force_fields;
    particles.field_count=3;
    reverse_exchange(
        std::move(particles),owned_count,
        {{"__force_x",ReduceOp::Sum},
         {"__force_y",ReduceOp::Sum},
         {"__force_z",ReduceOp::Sum}},stream);
}

namespace {

template<class T>
void reduce_field_values(const std::vector<std::byte>& input,
                         const std::vector<int>& indices,
                         const ParticleFieldView& field,
                         ReduceOp operation) {
    auto* output=static_cast<T*>(field.data);
    for (std::size_t k=0; k<indices.size(); ++k) {
        for (std::size_t component=0; component<field.components; ++component) {
            T value{};
            std::memcpy(&value,input.data()+
                        (k*field.components+component)*sizeof(T),sizeof(T));
            T& destination=output[
                static_cast<std::size_t>(indices[k])*field.components+component];
            switch (operation) {
            case ReduceOp::Sum: destination+=value; break;
            case ReduceOp::Min: destination=std::min(destination,value); break;
            case ReduceOp::Max: destination=std::max(destination,value); break;
            case ReduceOp::Replace: destination=value; break;
            }
        }
    }
}

void reduce_field(const std::vector<std::byte>& input,
                  const std::vector<int>& indices,
                  const ParticleFieldView& field,
                  ReduceOp operation) {
    switch (field.type) {
    case FieldType::Int32:
        reduce_field_values<std::int32_t>(input,indices,field,operation); break;
    case FieldType::Int64:
        reduce_field_values<std::int64_t>(input,indices,field,operation); break;
    case FieldType::UInt32:
        reduce_field_values<std::uint32_t>(input,indices,field,operation); break;
    case FieldType::UInt64:
        reduce_field_values<std::uint64_t>(input,indices,field,operation); break;
    case FieldType::Float32:
        reduce_field_values<float>(input,indices,field,operation); break;
    case FieldType::Float64:
        reduce_field_values<double>(input,indices,field,operation); break;
    }
}

} // namespace

void Decomposition::reverse_exchange(
    ParticleView particles, std::size_t owned_count,
    const std::vector<ReverseExchangeField>& fields,
    ExecutionStream stream) {
    (void)stream;
    if (impl_->operation_active) {
        throw std::logic_error(
            "dsDecompX: reverse exchange cannot overlap an active operation");
    }
    if (particles.memory_space!=MemorySpace::Host) {
        throw std::runtime_error(
            "dsDecompX: generic reverse exchange currently requires host memory");
    }
    const std::size_t ghost_count=total_received_halo_particles();
    if (particles.count<owned_count+ghost_count) {
        throw std::invalid_argument(
            "dsDecompX: reverse exchange particle view is too small");
    }
    std::unordered_set<std::string> selected;
    for (const auto& specification : fields) {
        if (!selected.insert(specification.name).second) {
            throw std::invalid_argument(
                "dsDecompX: duplicate reverse exchange field '"+
                specification.name+"'");
        }
        ParticleFieldView* begin=particles.fields;
        ParticleFieldView* end=begin+particles.field_count;
        const auto found=std::find_if(
            begin,end,
            [&](const ParticleFieldView& field) {
                return field.name==specification.name;
            });
        if (found==end || !found->data ||
            found->components==0 ||
            found->capacity<owned_count+ghost_count) {
            throw std::invalid_argument(
                "dsDecompX: missing or undersized reverse exchange field '"+
                specification.name+"'");
        }
        const std::size_t record_bytes=found->record_bytes();
        std::vector<std::vector<std::byte>> send(impl_->halo_plan.size());
        std::vector<std::vector<std::byte>> receive;
        const auto* source=static_cast<const std::byte*>(found->data);
        std::size_t ghost_offset=owned_count;
        for (std::size_t n=0; n<impl_->halo_plan.size(); ++n) {
            const std::size_t count=impl_->halo_plan[n].receive_count;
            send[n].resize(count*record_bytes);
            if (!send[n].empty()) {
                std::memcpy(send[n].data(),source+ghost_offset*record_bytes,
                            send[n].size());
            }
            ghost_offset+=count;
        }
        impl_->backend->exchange_neighbor_bytes(
            impl_->topology.neighbors(),send,receive,
            kReverseCountTagBase,kReverseDataTagBase,false);
        for (std::size_t n=0; n<receive.size(); ++n) {
            const auto& indices=impl_->halo_plan[n].source_indices;
            if (receive[n].size()!=indices.size()*record_bytes) {
                throw std::runtime_error(
                    "dsDecompX: reverse field schema differs across ranks");
            }
            reduce_field(receive[n],indices,*found,specification.operation);
        }
    }
}

} // namespace dsdecompx
