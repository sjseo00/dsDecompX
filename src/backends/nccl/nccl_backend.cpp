#include "backends/nccl.hpp"

#include "internal/backend.hpp"

#include <cuda_runtime_api.h>
#include <nccl.h>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsdecompx::detail {
namespace {

int opposite_slot(const Neighbor& neighbor) {
    const int dx=-neighbor.direction[0], dy=-neighbor.direction[1], dz=-neighbor.direction[2];
    const int raw=(dx+1)*9+(dy+1)*3+(dz+1);
    return raw < 13 ? raw : raw-1;
}

class NcclBackend final : public Backend {
public:
    explicit NcclBackend(MPI_Comm bootstrap) {
      try {
        if (bootstrap == MPI_COMM_NULL) {
            throw std::invalid_argument("dsDecompX NCCL: bootstrap communicator is null");
        }
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            throw std::runtime_error("dsDecompX NCCL: MPI must be initialized first");
        }
        mpi_check(MPI_Comm_dup(bootstrap, &control_), "MPI_Comm_dup bootstrap");
        mpi_check(MPI_Comm_rank(control_, &bootstrap_rank_), "MPI_Comm_rank");
        mpi_check(MPI_Comm_size(control_, &size_), "MPI_Comm_size");

        ncclUniqueId id{};
        if (bootstrap_rank_ == 0) nccl_check(ncclGetUniqueId(&id), "ncclGetUniqueId");
        mpi_check(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, control_),
                  "MPI_Bcast ncclUniqueId");
        nccl_check(ncclCommInitRank(&nccl_, size_, id, bootstrap_rank_),
                   "ncclCommInitRank");
        cuda_check(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming),
                   "cudaEventCreate completion");
        rank_ = bootstrap_rank_;
      } catch (...) {
        cleanup_resources();
        throw;
      }
    }

    ~NcclBackend() override {
        if (operation_active_ && nccl_) {
            ncclCommAbort(nccl_);
            nccl_=nullptr;
        }
        cleanup_resources();
    }

private:
    void cleanup_resources() noexcept {
        if (completion_) cudaEventDestroy(completion_);
        completion_ = nullptr;
        if (cart_ != MPI_COMM_NULL) MPI_Comm_free(&cart_);
        if (nccl_) ncclCommDestroy(nccl_);
        nccl_ = nullptr;
        if (control_ != MPI_COMM_NULL) MPI_Comm_free(&control_);
        control_ = MPI_COMM_NULL;
    }

public:

    BackendKind kind() const noexcept override { return BackendKind::Nccl; }
    int rank() const noexcept override { return rank_; }
    int size() const noexcept override { return size_; }

    void create_cartesian(const ProcessGrid& grid,
                          const std::array<bool, 3>& periodic) override {
        int dims[3] = {grid.dimensions[0], grid.dimensions[1], grid.dimensions[2]};
        if (dims[0] < 0 || dims[1] < 0 || dims[2] < 0) {
            throw std::invalid_argument("dsDecompX NCCL: negative process-grid dimension");
        }
        mpi_check(MPI_Dims_create(size_, 3, dims), "MPI_Dims_create");
        if (dims[0] * dims[1] * dims[2] != size_) {
            throw std::invalid_argument("dsDecompX NCCL: process grid does not match rank count");
        }
        int periods[3] = {periodic[0] ? 1 : 0,
                          periodic[1] ? 1 : 0,
                          periodic[2] ? 1 : 0};
        mpi_check(MPI_Cart_create(control_, 3, dims, periods,
                                  grid.allow_rank_reorder ? 1 : 0, &cart_),
                  "MPI_Cart_create");
        if (cart_ == MPI_COMM_NULL) {
            throw std::runtime_error("dsDecompX NCCL: MPI_Cart_create returned null");
        }
        mpi_check(MPI_Comm_rank(cart_, &rank_), "MPI cart rank");
        mpi_check(MPI_Cart_coords(cart_, rank_, 3, coordinates_.data()),
                  "MPI_Cart_coords");
        dimensions_ = {dims[0], dims[1], dims[2]};

        // NCCL ranks use bootstrap ordering. Translate Cartesian peers once.
        cart_to_bootstrap_.resize(static_cast<std::size_t>(size_));
        mpi_check(MPI_Allgather(&bootstrap_rank_, 1, MPI_INT,
                                cart_to_bootstrap_.data(), 1, MPI_INT, cart_),
                  "MPI_Allgather Cartesian-to-NCCL rank map");
    }

    Vec3i cartesian_dimensions() const noexcept override { return dimensions_; }
    Vec3i cartesian_coordinates() const noexcept override { return coordinates_; }

    int cartesian_rank(const Vec3i& coordinates) const override {
        int input[3] = {coordinates[0], coordinates[1], coordinates[2]};
        int result = MPI_PROC_NULL;
        mpi_check(MPI_Cart_rank(cart_, input, &result), "MPI_Cart_rank");
        return result;
    }

    void exchange_neighbor_counts(const std::vector<Neighbor>& neighbors,
                                  const std::vector<unsigned int>& send,
                                  std::vector<unsigned int>& receive,
                                  int tag_base) override {
        if (neighbors.size()!=send.size()) throw std::invalid_argument("NCCL count size mismatch");
        receive.assign(neighbors.size(),0u);
        std::vector<MPI_Request> requests;
        requests.reserve(neighbors.size()*2);
        for (std::size_t i=0;i<neighbors.size();++i) {
            const Neighbor& n=neighbors[i];
            if (n.rank<0) continue;
            if (n.rank==rank_) {
                const int source_slot=opposite_slot(n);
                const auto source=std::find_if(neighbors.begin(),neighbors.end(),
                    [&](const Neighbor& candidate) {
                        return candidate.rank==rank_ &&
                               candidate.direction_slot==source_slot;
                    });
                if (source!=neighbors.end())
                    receive[i]=send[static_cast<std::size_t>(
                        std::distance(neighbors.begin(),source))];
                continue;
            }
            MPI_Request r=MPI_REQUEST_NULL,s=MPI_REQUEST_NULL;
            mpi_check(MPI_Irecv(&receive[i],1,MPI_UNSIGNED,n.rank,
                                tag_base+opposite_slot(n),cart_,&r),"count Irecv");
            mpi_check(MPI_Isend(const_cast<unsigned int*>(&send[i]),1,MPI_UNSIGNED,n.rank,
                                tag_base+n.direction_slot,cart_,&s),"count Isend");
            requests.push_back(r); requests.push_back(s);
        }
        if (!requests.empty()) mpi_check(MPI_Waitall(static_cast<int>(requests.size()),
                                                      requests.data(),MPI_STATUSES_IGNORE),
                                         "count Waitall");
    }

    void exchange_rank_counts(const std::vector<unsigned int>& send,
                              std::vector<unsigned int>& receive) override {
        if (send.size()!=static_cast<std::size_t>(size_))
            throw std::invalid_argument("NCCL migration count table size mismatch");
        receive.resize(static_cast<std::size_t>(size_));
        mpi_check(MPI_Alltoall(send.data(),1,MPI_UNSIGNED,receive.data(),1,
                               MPI_UNSIGNED,cart_),"migration count Alltoall");
    }

    bool supports_device_data_plane() const noexcept override { return true; }

    void begin_device_exchange(const DeviceExchange& exchange) override {
        if (!nccl_) {
            fail(exchange.operation,
                 "communicator was aborted by an earlier failure; backend is unusable");
        }
        if (operation_active_) {
            fail(exchange.operation, "another operation is already active");
        }
        if (!exchange.transfers || !exchange.communication_stream) {
            fail(exchange.operation, "missing transfers or CUDA communication stream");
        }
        const cudaStream_t stream =
            static_cast<cudaStream_t>(exchange.communication_stream);
        operation_active_ = true;
        try {
            std::vector<const DeviceTransfer*> receives;
            std::vector<const DeviceTransfer*> sends;
            for (const DeviceTransfer& transfer : *exchange.transfers) {
                if (transfer.peer_rank < 0) continue;
                const int peer = nccl_rank(transfer.peer_rank);
                if (peer == bootstrap_rank_) {
                    if (!transfer.receive_bytes) continue;
                    const DeviceTransfer* source = nullptr;
                    for (const DeviceTransfer& candidate : *exchange.transfers) {
                        if (candidate.peer_rank == transfer.peer_rank &&
                            candidate.send_tag == transfer.receive_tag) {
                            source = &candidate; break;
                        }
                    }
                    if (!source || source->send_bytes != transfer.receive_bytes) {
                        fail(exchange.operation, "unmatched self transfer");
                    }
                    cuda_check(cudaMemcpyAsync(
                                   transfer.receive_buffer, source->send_buffer,
                                   transfer.receive_bytes, cudaMemcpyDeviceToDevice,
                                   stream),
                               context(exchange.operation, "self cudaMemcpyAsync"));
                    continue;
                }
                if (transfer.receive_bytes) receives.push_back(&transfer);
                if (transfer.send_bytes) sends.push_back(&transfer);
            }
            auto receive_order = [&](const DeviceTransfer* a,
                                     const DeviceTransfer* b) {
                const int peer_a=nccl_rank(a->peer_rank), peer_b=nccl_rank(b->peer_rank);
                return peer_a!=peer_b ? peer_a<peer_b : a->receive_tag<b->receive_tag;
            };
            auto send_order = [&](const DeviceTransfer* a,
                                  const DeviceTransfer* b) {
                const int peer_a=nccl_rank(a->peer_rank), peer_b=nccl_rank(b->peer_rank);
                return peer_a!=peer_b ? peer_a<peer_b : a->send_tag<b->send_tag;
            };
            std::sort(receives.begin(),receives.end(),receive_order);
            std::sort(sends.begin(),sends.end(),send_order);

            nccl_check(ncclGroupStart(), context(exchange.operation, "ncclGroupStart"));
            for (const DeviceTransfer* transfer : receives) {
                    nccl_check(ncclRecv(transfer->receive_buffer,
                                        transfer->receive_bytes, ncclUint8,
                                        nccl_rank(transfer->peer_rank), nccl_, stream),
                               context(exchange.operation, "ncclRecv"));
            }
            for (const DeviceTransfer* transfer : sends) {
                    nccl_check(ncclSend(transfer->send_buffer,
                                        transfer->send_bytes, ncclUint8,
                                        nccl_rank(transfer->peer_rank), nccl_, stream),
                               context(exchange.operation, "ncclSend"));
            }
            nccl_check(ncclGroupEnd(), context(exchange.operation, "ncclGroupEnd"));
        } catch (...) {
            operation_active_ = false;
            ncclCommAbort(nccl_);
            nccl_ = nullptr;
            throw;
        }
    }

    void end_device_exchange(const DeviceExchange& exchange) override {
        if (!operation_active_) fail(exchange.operation, "no operation is active");
        const cudaStream_t communication =
            static_cast<cudaStream_t>(exchange.communication_stream);
        cuda_check(cudaEventRecord(completion_, communication),
                   context(exchange.operation, "record NCCL completion"));
        if (exchange.consumer_stream &&
            exchange.consumer_stream != exchange.communication_stream) {
            cuda_check(cudaStreamWaitEvent(
                           static_cast<cudaStream_t>(exchange.consumer_stream),
                           completion_, 0),
                       context(exchange.operation, "consumer stream wait"));
        }
        operation_active_ = false;
    }

    std::size_t maximum_size(std::size_t local) const override {
        return static_cast<std::size_t>(maximum(local));
    }

    void begin_exchange_records(const std::vector<Neighbor>&,
                                const std::vector<std::vector<ParticleRecord>>&,
                                std::vector<std::vector<ParticleRecord>>&,
                                int, int, bool) override {
        throw std::runtime_error(
            "dsDecompX NCCL: host ParticleView is unsupported; use CUDA execution plans");
    }
    void end_exchange_records() override {
        throw std::runtime_error("dsDecompX NCCL: no host exchange exists");
    }
    void exchange_all_to_all_records(
        const std::vector<std::vector<ParticleRecord>>&,
        std::vector<std::vector<ParticleRecord>>&) override {
        throw std::runtime_error(
            "dsDecompX NCCL: host migration is unsupported; use CUDA execution plans");
    }

#ifndef NDEBUG
    std::vector<int> allgather_ids(const int* local_ids,
                                   std::size_t count) override {
        const int local_count = static_cast<int>(count);
        std::vector<int> counts(static_cast<std::size_t>(size_));
        mpi_check(MPI_Allgather(&local_count, 1, MPI_INT,
                                counts.data(), 1, MPI_INT, cart_),
                  "MPI_Allgather validation counts");
        std::vector<int> offsets(static_cast<std::size_t>(size_));
        int total = 0;
        for (int i = 0; i < size_; ++i) { offsets[i] = total; total += counts[i]; }
        std::vector<int> result(static_cast<std::size_t>(total));
        mpi_check(MPI_Allgatherv(local_ids, local_count, MPI_INT,
                                 result.data(), counts.data(), offsets.data(),
                                 MPI_INT, cart_),
                  "MPI_Allgatherv validation IDs");
        return result;
    }
#endif

    bool cuda_aware_enabled() const noexcept override { return true; }

    PerfStats aggregate_stats(const PerfStats& local) const override {
        PerfStats global;
        global.halo_calls = sum(local.halo_calls);
        global.migration_calls = sum(local.migration_calls);
        global.bytes_sent = sum(local.bytes_sent);
        global.bytes_received = sum(local.bytes_received);
        global.buffer_reallocations = sum(local.buffer_reallocations);
        global.peak_communication_buffer_capacity =
            maximum(local.peak_communication_buffer_capacity);
        global.pack_ms = aggregate(local.pack_ms);
        global.communication_ms = aggregate(local.communication_ms);
        global.wait_ms = aggregate(local.wait_ms);
        global.unpack_ms = aggregate(local.unpack_ms);
        global.migration_classify_ms = aggregate(local.migration_classify_ms);
        global.migration_compact_ms = aggregate(local.migration_compact_ms);
        return global;
    }

    std::uint64_t buffer_reallocations() const noexcept override { return 0; }
    std::uint64_t peak_buffer_capacity() const noexcept override { return 0; }

private:
    std::string context(const char* operation, const char* action) const {
        std::ostringstream out;
        out << "dsDecompX NCCL rank=" << rank_
            << " operation=" << (operation ? operation : "unknown")
            << " action=" << action;
        return out.str();
    }
    [[noreturn]] void fail(const char* operation, const char* message) const {
        throw std::runtime_error(context(operation, message));
    }
    void nccl_check(ncclResult_t status, const std::string& action) const {
        if (status != ncclSuccess) {
            throw std::runtime_error(action + ": " + ncclGetErrorString(status));
        }
    }
    void mpi_check(int status, const char* action) const {
        if (status == MPI_SUCCESS) return;
        char text[MPI_MAX_ERROR_STRING]{}; int length = 0;
        MPI_Error_string(status, text, &length);
        throw std::runtime_error(context("control-plane", action) + ": " +
                                 std::string(text, static_cast<std::size_t>(length)));
    }
    void cuda_check(cudaError_t status, const std::string& action) const {
        if (status != cudaSuccess) {
            throw std::runtime_error(action + ": " + cudaGetErrorString(status));
        }
    }
    int nccl_rank(int cartesian_peer) const {
        if (cartesian_peer < 0 || cartesian_peer >= size_) {
            throw std::out_of_range("dsDecompX NCCL: invalid Cartesian peer");
        }
        return cart_to_bootstrap_.empty()
            ? cartesian_peer
            : cart_to_bootstrap_[static_cast<std::size_t>(cartesian_peer)];
    }
    std::uint64_t sum(std::uint64_t value) const {
        unsigned long long in = value, out = 0;
        mpi_check(MPI_Allreduce(&in, &out, 1, MPI_UNSIGNED_LONG_LONG,
                                MPI_SUM, cart_), "MPI_Allreduce sum");
        return out;
    }
    std::uint64_t maximum(std::uint64_t value) const {
        unsigned long long in = value, out = 0;
        mpi_check(MPI_Allreduce(&in, &out, 1, MPI_UNSIGNED_LONG_LONG,
                                MPI_MAX, cart_), "MPI_Allreduce max");
        return out;
    }
    SampleAggregate aggregate(const SampleAggregate& in) const {
        SampleAggregate out;
        out.count = sum(in.count);
        mpi_check(MPI_Allreduce(&in.sum, &out.sum, 1, MPI_DOUBLE,
                                MPI_SUM, cart_), "MPI_Allreduce metric sum");
        const double local_min = in.count ? in.min : std::numeric_limits<double>::infinity();
        const double local_max = in.count ? in.max : -std::numeric_limits<double>::infinity();
        mpi_check(MPI_Allreduce(&local_min, &out.min, 1, MPI_DOUBLE,
                                MPI_MIN, cart_), "MPI_Allreduce metric min");
        mpi_check(MPI_Allreduce(&local_max, &out.max, 1, MPI_DOUBLE,
                                MPI_MAX, cart_), "MPI_Allreduce metric max");
        if (!out.count) out.min = out.max = 0.0;
        return out;
    }

    MPI_Comm control_ = MPI_COMM_NULL;
    MPI_Comm cart_ = MPI_COMM_NULL;
    ncclComm_t nccl_ = nullptr;
    cudaEvent_t completion_ = nullptr;
    int bootstrap_rank_ = 0;
    int rank_ = 0;
    int size_ = 1;
    Vec3i dimensions_{1,1,1};
    Vec3i coordinates_{0,0,0};
    std::vector<int> cart_to_bootstrap_;
    bool operation_active_ = false;
};

} // namespace
} // namespace dsdecompx::detail

namespace dsdecompx::nccl {

Backend make_backend(MPI_Comm bootstrap_communicator) {
    return Backend(std::make_unique<detail::NcclBackend>(bootstrap_communicator));
}

} // namespace dsdecompx::nccl
