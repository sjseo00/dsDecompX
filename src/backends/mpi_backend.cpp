#include "backends/mpi.hpp"

#include "internal/backend.hpp"
#include "internal/buffer_pool.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <limits>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>
#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
// Open MPI (and HPC-X): MPIX_CUDA_AWARE_SUPPORT and MPIX_Query_cuda_support().
#include <mpi-ext.h>
#endif
#endif
#if HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace dsdecompx::detail {
namespace {

int opposite_direction_slot(const Neighbor& neighbor) {
    const int dx = -neighbor.direction[0];
    const int dy = -neighbor.direction[1];
    const int dz = -neighbor.direction[2];
    const int raw = (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1);
    return raw < 13 ? raw : raw - 1;
}

class MpiBackend final : public Backend {
public:
    MpiBackend(MPI_Comm communicator, const mpi::BackendOptions& options)
        : options_(options) {
        if (communicator == MPI_COMM_NULL) {
            throw std::invalid_argument("dsDecompX: MPI communicator is null");
        }

        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            throw std::runtime_error("dsDecompX: MPI must be initialized before backend creation");
        }

        int provided = MPI_THREAD_SINGLE;
        MPI_Query_thread(&provided);
        if (provided < options.required_thread_level) {
            throw std::runtime_error("dsDecompX: MPI thread support is below the requested level");
        }

        if (options.duplicate_communicator) {
            MPI_Comm_dup(communicator, &world_);
            owns_world_ = true;
        } else {
            world_ = communicator;
        }

        MPI_Comm_rank(world_, &rank_);
        MPI_Comm_size(world_, &size_);

        const bool capability = detect_cuda_aware_mpi();
        if (options.cuda_aware == mpi::CudaAwareMode::Required && !capability) {
            cleanup();
            throw std::runtime_error(
                "dsDecompX: CUDA-aware MPI was required but could not be confirmed");
        }
        cuda_aware_enabled_ =
            options.cuda_aware != mpi::CudaAwareMode::Disabled && capability;

        create_particle_record_type();
#if HAS_CUDA
        if (!cuda_aware_enabled_ && options_.allow_host_staging) {
            cuda_check(cudaEventCreateWithFlags(&device_copy_event_,
                                                cudaEventDisableTiming),
                       "cudaEventCreate staging event");
        }
#endif
    }

    ~MpiBackend() override { cleanup(); }

    BackendKind kind() const noexcept override { return BackendKind::Mpi; }
    int rank() const noexcept override { return rank_; }
    int size() const noexcept override { return size_; }

    void create_cartesian(const ProcessGrid& grid,
                          const std::array<bool, 3>& periodic) override {
        if (cart_ != MPI_COMM_NULL) {
            MPI_Comm_free(&cart_);
            cart_ = MPI_COMM_NULL;
        }

        int dims[3] = {
            grid.dimensions[0],
            grid.dimensions[1],
            grid.dimensions[2]
        };

        if (dims[0] < 0 || dims[1] < 0 || dims[2] < 0) {
            throw std::invalid_argument("dsDecompX: process-grid dimensions cannot be negative");
        }

        MPI_Dims_create(size_, 3, dims);
        if (dims[0] * dims[1] * dims[2] != size_) {
            throw std::invalid_argument(
                "dsDecompX: process-grid dimensions do not match communicator size");
        }

        int periods[3] = {
            periodic[0] ? 1 : 0,
            periodic[1] ? 1 : 0,
            periodic[2] ? 1 : 0
        };

        MPI_Cart_create(world_, 3, dims, periods,
                        grid.allow_rank_reorder ? 1 : 0, &cart_);
        if (cart_ == MPI_COMM_NULL) {
            throw std::runtime_error("dsDecompX: MPI_Cart_create failed");
        }

        MPI_Comm_rank(cart_, &rank_);
        MPI_Comm_size(cart_, &size_);
        MPI_Cart_coords(cart_, rank_, 3, coordinates_.data());
        dimensions_ = {dims[0], dims[1], dims[2]};
    }

    Vec3i cartesian_dimensions() const noexcept override { return dimensions_; }
    Vec3i cartesian_coordinates() const noexcept override { return coordinates_; }

    int cartesian_rank(const Vec3i& coordinates) const override {
        if (cart_ == MPI_COMM_NULL) {
            throw std::runtime_error("dsDecompX: Cartesian topology is not initialized");
        }
        int coords[3] = {coordinates[0], coordinates[1], coordinates[2]};
        int result = MPI_PROC_NULL;
        MPI_Cart_rank(cart_, coords, &result);
        return result;
    }

    void exchange_neighbor_counts(
        const std::vector<Neighbor>& neighbors,
        const std::vector<unsigned int>& send_counts,
        std::vector<unsigned int>& receive_counts,
        int tag_base) override {
        if (neighbors.size() != send_counts.size()) {
            throw std::invalid_argument("dsDecompX MPI: neighbor/count size mismatch");
        }
        receive_counts.assign(neighbors.size(), 0u);
        std::vector<MPI_Request> count_requests;
        reserve_backend(count_requests, neighbors.size() * 2);
        for (std::size_t i = 0; i < neighbors.size(); ++i) {
            const Neighbor& neighbor = neighbors[i];
            if (neighbor.rank < 0 || neighbor.rank == MPI_PROC_NULL) continue;
            if (neighbor.rank == rank_) {
                const int source_slot=opposite_direction_slot(neighbor);
                const auto source=std::find_if(neighbors.begin(),neighbors.end(),
                    [&](const Neighbor& candidate) {
                        return candidate.rank==rank_ &&
                               candidate.direction_slot==source_slot;
                    });
                if (source!=neighbors.end())
                    receive_counts[i]=send_counts[static_cast<std::size_t>(
                        std::distance(neighbors.begin(),source))];
                continue;
            }
            MPI_Request receive = MPI_REQUEST_NULL, send = MPI_REQUEST_NULL;
            MPI_Irecv(&receive_counts[i], 1, MPI_UNSIGNED, neighbor.rank,
                      tag_base + opposite_direction_slot(neighbor),
                      data_comm(), &receive);
            MPI_Isend(const_cast<unsigned int*>(&send_counts[i]), 1, MPI_UNSIGNED,
                      neighbor.rank, tag_base + neighbor.direction_slot,
                      data_comm(), &send);
            count_requests.push_back(receive);
            count_requests.push_back(send);
        }
        if (!count_requests.empty()) {
            MPI_Waitall(static_cast<int>(count_requests.size()),
                        count_requests.data(), MPI_STATUSES_IGNORE);
        }
    }

    void exchange_rank_counts(const std::vector<unsigned int>& send,
                              std::vector<unsigned int>& receive) override {
        if (send.size()!=static_cast<std::size_t>(size_))
            throw std::invalid_argument("dsDecompX MPI: migration count table size mismatch");
        receive.resize(static_cast<std::size_t>(size_));
        MPI_Alltoall(send.data(),1,MPI_UNSIGNED,receive.data(),1,MPI_UNSIGNED,
                     data_comm());
    }

    bool supports_device_data_plane() const noexcept override {
        return cuda_aware_enabled_
#if HAS_CUDA
            || options_.allow_host_staging
#endif
            ;
    }

    void begin_device_exchange(const DeviceExchange& exchange) override {
        if (!cuda_aware_enabled_ && !options_.allow_host_staging) {
            throw std::runtime_error(
                "dsDecompX MPI: CUDA-aware MPI unavailable and host staging disabled");
        }
        if (exchange_active_) {
            throw std::logic_error("dsDecompX MPI: another exchange is active");
        }
        if (!exchange.transfers) {
            throw std::invalid_argument("dsDecompX MPI: missing device transfers");
        }
        for (const DeviceTransfer& transfer : *exchange.transfers) {
            if (transfer.send_bytes>static_cast<std::size_t>(INT_MAX) ||
                transfer.receive_bytes>static_cast<std::size_t>(INT_MAX)) {
                throw std::overflow_error(
                    "dsDecompX MPI: device transfer exceeds MPI int count");
            }
            if ((transfer.send_bytes && !transfer.send_buffer) ||
                (transfer.receive_bytes && !transfer.receive_buffer)) {
                throw std::invalid_argument(
                    "dsDecompX MPI: nonzero device transfer has a null buffer");
            }
        }
        requests_.clear();
        reserve_backend(requests_, exchange.transfers->size() * 2);
        active_device_transfers_ = *exchange.transfers;
        active_device_stream_ = exchange.communication_stream;
        active_consumer_stream_ = exchange.consumer_stream;
        active_device_staging_ = !cuda_aware_enabled_;

#if HAS_CUDA
        if (active_device_staging_) {
            prepare_staging(active_device_transfers_);
            const cudaStream_t stream =
                static_cast<cudaStream_t>(exchange.communication_stream);
            for (std::size_t i = 0; i < active_device_transfers_.size(); ++i) {
                const DeviceTransfer& transfer = active_device_transfers_[i];
                if (transfer.send_bytes) {
                    cuda_check(cudaMemcpyAsync(staging_[i].send,
                                               transfer.send_buffer,
                                               transfer.send_bytes,
                                               cudaMemcpyDeviceToHost, stream),
                               "cudaMemcpyAsync staged send");
                }
            }
            cuda_check(cudaEventRecord(device_copy_event_, stream),
                       "record staged pack copy");
            // MPI cannot consume a CUDA event. This is the single documented
            // host wait in the staging path; CUDA-aware MPI avoids it.
            cuda_check(cudaEventSynchronize(device_copy_event_),
                       "wait staged pack copy");
        }
#else
        if (active_device_staging_) {
            throw std::runtime_error(
                "dsDecompX MPI: host staging requires ENABLE_CUDA");
        }
#endif
        exchange_active_ = true;
        for (std::size_t i = 0; i < active_device_transfers_.size(); ++i) {
            const DeviceTransfer& transfer = active_device_transfers_[i];
            if (transfer.peer_rank < 0 || transfer.peer_rank == MPI_PROC_NULL) continue;
            if (transfer.receive_bytes) {
                MPI_Request request = MPI_REQUEST_NULL;
                void* receive_pointer = transfer.receive_buffer;
#if HAS_CUDA
                if (active_device_staging_) receive_pointer = staging_[i].receive;
#endif
                MPI_Irecv(receive_pointer,
                          static_cast<int>(transfer.receive_bytes), MPI_BYTE,
                          transfer.peer_rank, transfer.receive_tag,
                          data_comm(), &request);
                requests_.push_back(request);
            }
            if (transfer.send_bytes) {
                MPI_Request request = MPI_REQUEST_NULL;
                void* send_pointer = const_cast<void*>(transfer.send_buffer);
#if HAS_CUDA
                if (active_device_staging_) send_pointer = staging_[i].send;
#endif
                MPI_Isend(send_pointer,
                          static_cast<int>(transfer.send_bytes), MPI_BYTE,
                          transfer.peer_rank, transfer.send_tag,
                          data_comm(), &request);
                requests_.push_back(request);
            }
        }
    }

    void end_device_exchange(const DeviceExchange&) override {
        if (!exchange_active_) {
            throw std::logic_error("dsDecompX MPI: no device exchange is active");
        }
        if (!requests_.empty()) {
            MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(),
                        MPI_STATUSES_IGNORE);
        }
        requests_.clear();
#if HAS_CUDA
        if (active_device_staging_) {
            const cudaStream_t communication =
                static_cast<cudaStream_t>(active_device_stream_);
            for (std::size_t i = 0; i < active_device_transfers_.size(); ++i) {
                const DeviceTransfer& transfer = active_device_transfers_[i];
                if (transfer.receive_bytes) {
                    cuda_check(cudaMemcpyAsync(transfer.receive_buffer,
                                               staging_[i].receive,
                                               transfer.receive_bytes,
                                               cudaMemcpyHostToDevice,
                                               communication),
                               "cudaMemcpyAsync staged receive");
                }
            }
            cuda_check(cudaEventRecord(device_copy_event_, communication),
                       "record staged receive copy");
            if (active_consumer_stream_ &&
                active_consumer_stream_ != active_device_stream_) {
                cuda_check(cudaStreamWaitEvent(
                               static_cast<cudaStream_t>(active_consumer_stream_),
                               device_copy_event_, 0),
                           "consumer waits for staged receive");
            }
        }
#endif
        active_device_transfers_.clear();
        active_device_stream_ = nullptr;
        active_consumer_stream_ = nullptr;
        active_device_staging_ = false;
        exchange_active_ = false;
    }

    std::size_t maximum_size(std::size_t local) const override {
        unsigned long long input = static_cast<unsigned long long>(local);
        unsigned long long output = 0;
        MPI_Allreduce(&input, &output, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX, data_comm());
        return static_cast<std::size_t>(output);
    }

    void begin_exchange_records(
        const std::vector<Neighbor>& neighbors,
        const std::vector<std::vector<ParticleRecord>>& send_buffers,
        std::vector<std::vector<ParticleRecord>>& receive_buffers,
        int count_tag_base,
        int data_tag_base,
        bool direct_self_copy) override {
        if (exchange_active_) {
            throw std::logic_error("dsDecompX: an MPI exchange is already active");
        }
        if (neighbors.size() != send_buffers.size()) {
            throw std::invalid_argument("dsDecompX: neighbor/send-buffer size mismatch");
        }

        receive_buffers.resize(neighbors.size());
        for (auto& buffer : receive_buffers) buffer.clear();

        active_indices_.clear();
        reserve_backend(active_indices_, neighbors.size());
        for (std::size_t n = 0; n < neighbors.size(); ++n) {
            const Neighbor& neighbor = neighbors[n];
            if (neighbor.rank < 0 || neighbor.rank == MPI_PROC_NULL) continue;
            if (direct_self_copy && neighbor.rank == rank_) {
                const int source_slot=opposite_direction_slot(neighbor);
                const auto source=std::find_if(neighbors.begin(),neighbors.end(),
                    [&](const Neighbor& candidate) {
                        return candidate.rank==rank_ &&
                               candidate.direction_slot==source_slot;
                    });
                if (source!=neighbors.end())
                    receive_buffers[n]=send_buffers[static_cast<std::size_t>(
                        std::distance(neighbors.begin(),source))];
            } else {
                active_indices_.push_back(static_cast<int>(n));
            }
        }

        exchange_active_ = true;
        if (active_indices_.empty()) return;

        reserve_backend(send_counts_, active_indices_.size());
        reserve_backend(receive_counts_, active_indices_.size());
        send_counts_.assign(active_indices_.size(), 0);
        receive_counts_.assign(active_indices_.size(), 0);
        requests_.clear();
        reserve_backend(requests_, active_indices_.size() * 2);

        for (std::size_t i = 0; i < active_indices_.size(); ++i) {
            const int n = active_indices_[i];
            const Neighbor& neighbor = neighbors[n];
            send_counts_[i] = static_cast<int>(send_buffers[n].size());

            MPI_Request receive_request = MPI_REQUEST_NULL;
            MPI_Request send_request = MPI_REQUEST_NULL;
            MPI_Irecv(&receive_counts_[i], 1, MPI_INT, neighbor.rank,
                      count_tag_base + opposite_direction_slot(neighbor),
                      data_comm(), &receive_request);
            MPI_Isend(&send_counts_[i], 1, MPI_INT, neighbor.rank,
                      count_tag_base + neighbor.direction_slot,
                      data_comm(), &send_request);
            requests_.push_back(receive_request);
            requests_.push_back(send_request);
        }
        MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(),
                    MPI_STATUSES_IGNORE);

        requests_.clear();
        for (std::size_t i = 0; i < active_indices_.size(); ++i) {
            const int n = active_indices_[i];
            const Neighbor& neighbor = neighbors[n];
            const int send_count = send_counts_[i];
            const int receive_count = receive_counts_[i];
            reserve_backend(receive_buffers[n],
                            static_cast<std::size_t>(receive_count));
            receive_buffers[n].resize(static_cast<std::size_t>(receive_count));

            if (receive_count > 0) {
                MPI_Request request = MPI_REQUEST_NULL;
                MPI_Irecv(receive_buffers[n].data(), receive_count, record_type_,
                          neighbor.rank,
                          data_tag_base + opposite_direction_slot(neighbor),
                          data_comm(), &request);
                requests_.push_back(request);
            }
            if (send_count > 0) {
                MPI_Request request = MPI_REQUEST_NULL;
                MPI_Isend(send_buffers[n].data(), send_count, record_type_,
                          neighbor.rank,
                          data_tag_base + neighbor.direction_slot,
                          data_comm(), &request);
                requests_.push_back(request);
            }
        }
    }

    void end_exchange_records() override {
        if (!exchange_active_) {
            throw std::logic_error("dsDecompX: no MPI exchange is active");
        }
        if (!requests_.empty()) {
            MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(),
                        MPI_STATUSES_IGNORE);
        }
        requests_.clear();
        exchange_active_ = false;
    }

    void exchange_all_to_all_records(
        const std::vector<std::vector<ParticleRecord>>& send_by_rank,
        std::vector<std::vector<ParticleRecord>>& receive_by_rank) override {
        if (exchange_active_) {
            throw std::logic_error("dsDecompX: halo and migration cannot overlap on one backend");
        }
        if (send_by_rank.size() != static_cast<std::size_t>(size_)) {
            throw std::invalid_argument("dsDecompX: migration send table does not match MPI size");
        }

        migration_send_counts_.assign(size_, 0);
        migration_receive_counts_.assign(size_, 0);
        migration_send_displacements_.assign(size_, 0);
        migration_receive_displacements_.assign(size_, 0);
        std::size_t total_send = 0;
        for (int peer = 0; peer < size_; ++peer) {
            migration_send_counts_[peer] =
                static_cast<int>(send_by_rank[peer].size());
            migration_send_displacements_[peer] = static_cast<int>(total_send);
            total_send += send_by_rank[peer].size();
        }
        reserve_backend(migration_send_flat_, total_send);
        migration_send_flat_.clear();
        for (const auto& buffer : send_by_rank) {
            migration_send_flat_.insert(migration_send_flat_.end(),
                                        buffer.begin(), buffer.end());
        }

        MPI_Alltoall(migration_send_counts_.data(), 1, MPI_INT,
                     migration_receive_counts_.data(), 1, MPI_INT, data_comm());

        std::size_t total_receive = 0;
        for (int peer = 0; peer < size_; ++peer) {
            migration_receive_displacements_[peer] =
                static_cast<int>(total_receive);
            total_receive += static_cast<std::size_t>(migration_receive_counts_[peer]);
        }
        reserve_backend(migration_receive_flat_, total_receive);
        migration_receive_flat_.resize(total_receive);

        MPI_Alltoallv(migration_send_flat_.empty() ? nullptr : migration_send_flat_.data(),
                      migration_send_counts_.data(), migration_send_displacements_.data(),
                      record_type_,
                      migration_receive_flat_.empty() ? nullptr : migration_receive_flat_.data(),
                      migration_receive_counts_.data(), migration_receive_displacements_.data(),
                      record_type_, data_comm());

        receive_by_rank.resize(static_cast<std::size_t>(size_));
        for (int peer = 0; peer < size_; ++peer) {
            auto& destination = receive_by_rank[peer];
            const std::size_t count =
                static_cast<std::size_t>(migration_receive_counts_[peer]);
            reserve_backend(destination, count);
            destination.assign(
                migration_receive_flat_.begin() + migration_receive_displacements_[peer],
                migration_receive_flat_.begin() + migration_receive_displacements_[peer] + count);
        }
    }

    void exchange_neighbor_bytes(
        const std::vector<Neighbor>& neighbors,
        const std::vector<std::vector<std::byte>>& send_buffers,
        std::vector<std::vector<std::byte>>& receive_buffers,
        int count_tag_base, int data_tag_base,
        bool direct_self_copy) override {
        if (exchange_active_) {
            throw std::logic_error(
                "dsDecompX: byte exchange cannot overlap another exchange");
        }
        if (neighbors.size() != send_buffers.size()) {
            throw std::invalid_argument(
                "dsDecompX: neighbor/field-buffer size mismatch");
        }
        receive_buffers.assign(neighbors.size(), {});
        std::vector<int> active;
        for (std::size_t n=0; n<neighbors.size(); ++n) {
            const Neighbor& neighbor=neighbors[n];
            if (neighbor.rank < 0 || neighbor.rank == MPI_PROC_NULL) continue;
            if (direct_self_copy && neighbor.rank == rank_) {
                const int source_slot=opposite_direction_slot(neighbor);
                const auto source=std::find_if(
                    neighbors.begin(),neighbors.end(),
                    [&](const Neighbor& candidate) {
                        return candidate.rank==rank_ &&
                               candidate.direction_slot==source_slot;
                    });
                if (source!=neighbors.end()) {
                    receive_buffers[n]=send_buffers[static_cast<std::size_t>(
                        std::distance(neighbors.begin(),source))];
                }
            } else {
                active.push_back(static_cast<int>(n));
            }
        }
        std::vector<int> send_counts(active.size(),0);
        std::vector<int> receive_counts(active.size(),0);
        std::vector<MPI_Request> requests;
        requests.reserve(active.size()*2);
        for (std::size_t i=0; i<active.size(); ++i) {
            const int n=active[i];
            if (send_buffers[n].size() > static_cast<std::size_t>(INT_MAX)) {
                throw std::overflow_error("dsDecompX: custom field message is too large");
            }
            send_counts[i]=static_cast<int>(send_buffers[n].size());
            MPI_Request receive_request=MPI_REQUEST_NULL;
            MPI_Request send_request=MPI_REQUEST_NULL;
            MPI_Irecv(&receive_counts[i],1,MPI_INT,neighbors[n].rank,
                      count_tag_base+opposite_direction_slot(neighbors[n]),
                      data_comm(),&receive_request);
            MPI_Isend(&send_counts[i],1,MPI_INT,neighbors[n].rank,
                      count_tag_base+neighbors[n].direction_slot,
                      data_comm(),&send_request);
            requests.push_back(receive_request);
            requests.push_back(send_request);
        }
        if (!requests.empty()) {
            MPI_Waitall(static_cast<int>(requests.size()),requests.data(),
                        MPI_STATUSES_IGNORE);
        }
        requests.clear();
        for (std::size_t i=0; i<active.size(); ++i) {
            const int n=active[i];
            receive_buffers[n].resize(static_cast<std::size_t>(receive_counts[i]));
            if (receive_counts[i]) {
                MPI_Request request=MPI_REQUEST_NULL;
                MPI_Irecv(receive_buffers[n].data(),receive_counts[i],MPI_BYTE,
                          neighbors[n].rank,
                          data_tag_base+opposite_direction_slot(neighbors[n]),
                          data_comm(),&request);
                requests.push_back(request);
            }
            if (send_counts[i]) {
                MPI_Request request=MPI_REQUEST_NULL;
                MPI_Isend(send_buffers[n].data(),send_counts[i],MPI_BYTE,
                          neighbors[n].rank,
                          data_tag_base+neighbors[n].direction_slot,
                          data_comm(),&request);
                requests.push_back(request);
            }
        }
        if (!requests.empty()) {
            MPI_Waitall(static_cast<int>(requests.size()),requests.data(),
                        MPI_STATUSES_IGNORE);
        }
    }

    void exchange_all_to_all_bytes(
        const std::vector<std::vector<std::byte>>& send_by_rank,
        std::vector<std::vector<std::byte>>& receive_by_rank) override {
        if (send_by_rank.size()!=static_cast<std::size_t>(size_)) {
            throw std::invalid_argument(
                "dsDecompX: custom field migration table does not match MPI size");
        }
        std::vector<int> send_counts(size_,0),receive_counts(size_,0);
        std::vector<int> send_offsets(size_,0),receive_offsets(size_,0);
        std::size_t send_total=0;
        for (int peer=0; peer<size_; ++peer) {
            if (send_by_rank[peer].size()>static_cast<std::size_t>(INT_MAX) ||
                send_total>static_cast<std::size_t>(INT_MAX)) {
                throw std::overflow_error("dsDecompX: custom field migration is too large");
            }
            send_counts[peer]=static_cast<int>(send_by_rank[peer].size());
            send_offsets[peer]=static_cast<int>(send_total);
            send_total+=send_by_rank[peer].size();
        }
        std::vector<std::byte> send_flat;
        send_flat.reserve(send_total);
        for (const auto& buffer : send_by_rank) {
            send_flat.insert(send_flat.end(),buffer.begin(),buffer.end());
        }
        MPI_Alltoall(send_counts.data(),1,MPI_INT,
                     receive_counts.data(),1,MPI_INT,data_comm());
        std::size_t receive_total=0;
        for (int peer=0; peer<size_; ++peer) {
            if (receive_total>static_cast<std::size_t>(INT_MAX)) {
                throw std::overflow_error("dsDecompX: custom field migration is too large");
            }
            receive_offsets[peer]=static_cast<int>(receive_total);
            receive_total+=static_cast<std::size_t>(receive_counts[peer]);
        }
        std::vector<std::byte> receive_flat(receive_total);
        MPI_Alltoallv(send_flat.empty()?nullptr:send_flat.data(),
                      send_counts.data(),send_offsets.data(),MPI_BYTE,
                      receive_flat.empty()?nullptr:receive_flat.data(),
                      receive_counts.data(),receive_offsets.data(),MPI_BYTE,
                      data_comm());
        receive_by_rank.resize(static_cast<std::size_t>(size_));
        for (int peer=0; peer<size_; ++peer) {
            const auto first=receive_flat.begin()+receive_offsets[peer];
            receive_by_rank[peer].assign(
                first,first+receive_counts[peer]);
        }
    }

#ifndef NDEBUG
    std::vector<int> allgather_ids(const int* local_ids,
                                   std::size_t count) override {
        validation_counts_.assign(size_, 0);
        const int local_count = static_cast<int>(count);
        MPI_Allgather(&local_count, 1, MPI_INT,
                      validation_counts_.data(), 1, MPI_INT, data_comm());
        validation_displacements_.assign(size_, 0);
        int total = 0;
        for (int peer = 0; peer < size_; ++peer) {
            validation_displacements_[peer] = total;
            total += validation_counts_[peer];
        }
        std::vector<int> result(static_cast<std::size_t>(total));
        MPI_Allgatherv(local_ids, local_count, MPI_INT,
                       result.data(), validation_counts_.data(),
                       validation_displacements_.data(), MPI_INT, data_comm());
        return result;
    }
#endif

    bool cuda_aware_enabled() const noexcept override {
        return cuda_aware_enabled_;
    }

    PerfStats aggregate_stats(const PerfStats& local) const override {
        PerfStats global;
        auto sum_u64 = [&](std::uint64_t value) {
            unsigned long long input = static_cast<unsigned long long>(value);
            unsigned long long output = 0;
            MPI_Allreduce(&input, &output, 1, MPI_UNSIGNED_LONG_LONG,
                          MPI_SUM, data_comm());
            return static_cast<std::uint64_t>(output);
        };
        auto max_u64 = [&](std::uint64_t value) {
            unsigned long long input = static_cast<unsigned long long>(value);
            unsigned long long output = 0;
            MPI_Allreduce(&input, &output, 1, MPI_UNSIGNED_LONG_LONG,
                          MPI_MAX, data_comm());
            return static_cast<std::uint64_t>(output);
        };
        auto aggregate_metric = [&](const SampleAggregate& input) {
            SampleAggregate output;
            output.count = sum_u64(input.count);
            MPI_Allreduce(&input.sum, &output.sum, 1, MPI_DOUBLE, MPI_SUM, data_comm());
            const double local_min = input.count ? input.min
                : std::numeric_limits<double>::infinity();
            const double local_max = input.count ? input.max
                : -std::numeric_limits<double>::infinity();
            MPI_Allreduce(&local_min, &output.min, 1, MPI_DOUBLE, MPI_MIN, data_comm());
            MPI_Allreduce(&local_max, &output.max, 1, MPI_DOUBLE, MPI_MAX, data_comm());
            if (output.count == 0) output.min = output.max = 0.0;
            return output;
        };

        global.halo_calls = sum_u64(local.halo_calls);
        global.migration_calls = sum_u64(local.migration_calls);
        global.bytes_sent = sum_u64(local.bytes_sent);
        global.bytes_received = sum_u64(local.bytes_received);
        global.buffer_reallocations = sum_u64(local.buffer_reallocations);
        global.peak_communication_buffer_capacity =
            max_u64(local.peak_communication_buffer_capacity);
        global.pack_ms = aggregate_metric(local.pack_ms);
        global.communication_ms = aggregate_metric(local.communication_ms);
        global.wait_ms = aggregate_metric(local.wait_ms);
        global.unpack_ms = aggregate_metric(local.unpack_ms);
        global.migration_classify_ms = aggregate_metric(local.migration_classify_ms);
        global.migration_compact_ms = aggregate_metric(local.migration_compact_ms);
        return global;
    }

    std::uint64_t buffer_reallocations() const noexcept override {
        return buffer_reallocations_;
    }

    std::uint64_t peak_buffer_capacity() const noexcept override {
        return peak_buffer_capacity_;
    }

private:
    MPI_Comm data_comm() const noexcept {
        return cart_ != MPI_COMM_NULL ? cart_ : world_;
    }
#if HAS_CUDA
    void cuda_check(cudaError_t status, const char* operation) const {
        if (status == cudaSuccess) return;
        throw std::runtime_error(
            std::string("dsDecompX MPI rank=") + std::to_string(rank_) +
            " operation=" + operation + ": " + cudaGetErrorString(status));
    }

    void grow_pinned(void*& pointer, std::size_t& capacity,
                     std::size_t required, const char* operation) {
        if (required <= capacity) return;
        std::size_t next = capacity ? capacity : 4096;
        while (next < required) next += next / 2;
        void* replacement = nullptr;
        cuda_check(cudaMallocHost(&replacement, next), operation);
        if (pointer) cuda_check(cudaFreeHost(pointer), "cudaFreeHost old staging");
        pointer = replacement;
        capacity = next;
        ++buffer_reallocations_;
        peak_buffer_capacity_ = std::max(
            peak_buffer_capacity_, static_cast<std::uint64_t>(next));
    }

    void prepare_staging(const std::vector<DeviceTransfer>& transfers) {
        staging_.resize(transfers.size());
        for (std::size_t i = 0; i < transfers.size(); ++i) {
            grow_pinned(staging_[i].send, staging_[i].send_capacity,
                        transfers[i].send_bytes, "cudaMallocHost send staging");
            grow_pinned(staging_[i].receive, staging_[i].receive_capacity,
                        transfers[i].receive_bytes, "cudaMallocHost receive staging");
        }
    }
#endif

    template<class T>
    void reserve_backend(std::vector<T>& buffer, std::size_t required) {
        const std::size_t before = buffer.capacity();
        detail::reserve_geometric(buffer, required);
        if (buffer.capacity() != before) ++buffer_reallocations_;
        const std::uint64_t bytes = static_cast<std::uint64_t>(buffer.capacity()) *
                                    static_cast<std::uint64_t>(sizeof(T));
        peak_buffer_capacity_ = std::max(peak_buffer_capacity_, bytes);
    }

    bool detect_cuda_aware_mpi() const noexcept {
        if (const char* override_value=std::getenv("CUDA_AWARE_MPI")) {
            const std::string value(override_value);
            if (value=="1" || value=="true" || value=="TRUE") return true;
            if (value=="0" || value=="false" || value=="FALSE") return false;
        }
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        // Open MPI built with CUDA support: ask the library itself.
        if (MPIX_Query_cuda_support() == 1) return true;
#endif
        const char* value = std::getenv("OMPI_MCA_opal_cuda_support");
        return value && (std::string(value) == "true" || std::string(value) == "1");
    }

    void create_particle_record_type() {
        ParticleRecord record{};
        MPI_Aint base = 0;
        MPI_Aint displacements[13]{};
        MPI_Get_address(&record, &base);
        MPI_Get_address(&record.x, &displacements[0]);
        MPI_Get_address(&record.y, &displacements[1]);
        MPI_Get_address(&record.z, &displacements[2]);
        MPI_Get_address(&record.vx, &displacements[3]);
        MPI_Get_address(&record.vy, &displacements[4]);
        MPI_Get_address(&record.vz, &displacements[5]);
        MPI_Get_address(&record.fx, &displacements[6]);
        MPI_Get_address(&record.fy, &displacements[7]);
        MPI_Get_address(&record.fz, &displacements[8]);
        MPI_Get_address(&record.mass, &displacements[9]);
        MPI_Get_address(&record.charge, &displacements[10]);
        MPI_Get_address(&record.type, &displacements[11]);
        MPI_Get_address(&record.global_id, &displacements[12]);
        for (MPI_Aint& displacement : displacements) displacement -= base;

        int block_lengths[13] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        MPI_Datatype types[13] = {
            MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
            MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
            MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
            MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT
        };
        MPI_Type_create_struct(13, block_lengths, displacements, types,
                               &record_type_);
        MPI_Type_commit(&record_type_);
    }

    void cleanup() noexcept {
#if HAS_CUDA
        for (StagingSlot& slot : staging_) {
            if (slot.send) cudaFreeHost(slot.send);
            if (slot.receive) cudaFreeHost(slot.receive);
        }
        staging_.clear();
        if (device_copy_event_) cudaEventDestroy(device_copy_event_);
        device_copy_event_ = nullptr;
#endif
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (finalized) return;

        if (record_type_ != MPI_DATATYPE_NULL) {
            MPI_Type_free(&record_type_);
            record_type_ = MPI_DATATYPE_NULL;
        }
        if (cart_ != MPI_COMM_NULL) {
            MPI_Comm_free(&cart_);
            cart_ = MPI_COMM_NULL;
        }
        if (owns_world_ && world_ != MPI_COMM_NULL) {
            MPI_Comm_free(&world_);
            world_ = MPI_COMM_NULL;
        }
    }

    mpi::BackendOptions options_{};
    MPI_Comm world_ = MPI_COMM_NULL;
    MPI_Comm cart_ = MPI_COMM_NULL;
    MPI_Datatype record_type_ = MPI_DATATYPE_NULL;
    bool owns_world_ = false;
    bool cuda_aware_enabled_ = false;
    int rank_ = 0;
    int size_ = 1;
    Vec3i dimensions_{1, 1, 1};
    Vec3i coordinates_{0, 0, 0};
    bool exchange_active_ = false;
    std::vector<DeviceTransfer> active_device_transfers_;
    ExecutionStream active_device_stream_ = nullptr;
    ExecutionStream active_consumer_stream_ = nullptr;
    bool active_device_staging_ = false;
#if HAS_CUDA
    struct StagingSlot {
        void* send = nullptr;
        void* receive = nullptr;
        std::size_t send_capacity = 0;
        std::size_t receive_capacity = 0;
    };
    std::vector<StagingSlot> staging_;
    cudaEvent_t device_copy_event_ = nullptr;
#endif
    std::vector<int> active_indices_;
    std::vector<int> send_counts_;
    std::vector<int> receive_counts_;
    std::vector<MPI_Request> requests_;
    std::vector<int> migration_send_counts_;
    std::vector<int> migration_receive_counts_;
    std::vector<int> migration_send_displacements_;
    std::vector<int> migration_receive_displacements_;
    std::vector<ParticleRecord> migration_send_flat_;
    std::vector<ParticleRecord> migration_receive_flat_;
    std::uint64_t buffer_reallocations_ = 0;
    std::uint64_t peak_buffer_capacity_ = 0;
#ifndef NDEBUG
    std::vector<int> validation_counts_;
    std::vector<int> validation_displacements_;
#endif
};

} // namespace
} // namespace dsdecompx::detail

namespace dsdecompx::mpi {

Backend make_backend(MPI_Comm communicator, const BackendOptions& options) {
    return Backend(std::make_unique<detail::MpiBackend>(communicator, options));
}

TopologyInfo discover_topology(MPI_Comm communicator) noexcept {
    TopologyInfo info;
    try {
    if (communicator == MPI_COMM_NULL) return info;
    int rank = 0;
    int size = 1;
    if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
        MPI_Comm_size(communicator, &size) != MPI_SUCCESS || size <= 0) {
        return info;
    }

    char local_name[MPI_MAX_PROCESSOR_NAME]{};
    int name_length = 0;
    if (MPI_Get_processor_name(local_name, &name_length) != MPI_SUCCESS) {
        return info;
    }
    std::vector<char> names(
        static_cast<std::size_t>(size) * MPI_MAX_PROCESSOR_NAME, '\0');
    if (MPI_Allgather(local_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                      names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                      communicator) != MPI_SUCCESS) {
        return info;
    }

    std::unordered_map<std::string, int> node_ids;
    info.rank_to_node.resize(static_cast<std::size_t>(size));
    for (int peer = 0; peer < size; ++peer) {
        const char* begin = names.data() +
            static_cast<std::size_t>(peer) * MPI_MAX_PROCESSOR_NAME;
        std::size_t length = 0;
        while (length < MPI_MAX_PROCESSOR_NAME && begin[length] != '\0') ++length;
        const std::string name(begin, length);
        auto inserted = node_ids.emplace(name, static_cast<int>(node_ids.size()));
        info.rank_to_node[peer] = inserted.first->second;
    }
    const int local_node = info.rank_to_node[rank];
    info.local_rank = 0;
    info.local_size = 0;
    for (int peer = 0; peer < size; ++peer) {
        if (info.rank_to_node[peer] != local_node) continue;
        if (peer < rank) ++info.local_rank;
        ++info.local_size;
    }
    info.node_count = static_cast<int>(node_ids.size());
    info.available = true;
    return info;
    } catch (...) {
        return TopologyInfo{};
    }
}

} // namespace dsdecompx::mpi
