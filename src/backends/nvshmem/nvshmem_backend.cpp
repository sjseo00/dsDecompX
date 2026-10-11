#include "backends/nvshmem.hpp"

#include "internal/backend.hpp"

#include <cuda_runtime_api.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <array>
#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsdecompx::detail {
namespace {

// NVSHMEM has no nvshmem_initialized(); the documented query is
// nvshmemx_init_status(). A merely bootstrapped library is not yet usable.
bool nvshmem_is_initialized() {
    const int status = nvshmemx_init_status();
    return status == NVSHMEM_STATUS_IS_INITIALIZED ||
           status == NVSHMEM_STATUS_LIMITED_MPG ||
           status == NVSHMEM_STATUS_FULL_MPG;
}

int opposite_slot(const Neighbor& neighbor) {
    const int dx=-neighbor.direction[0], dy=-neighbor.direction[1], dz=-neighbor.direction[2];
    const int raw=(dx+1)*9+(dy+1)*3+(dz+1);
    return raw < 13 ? raw : raw-1;
}

class NvshmemBackend final : public Backend {
public:
    explicit NvshmemBackend(MPI_Comm bootstrap) {
      try {
        if (bootstrap == MPI_COMM_NULL) {
            throw std::invalid_argument("dsDecompX NVSHMEM: bootstrap communicator is null");
        }
        int initialized=0;
        MPI_Initialized(&initialized);
        if (!initialized) throw std::runtime_error(
            "dsDecompX NVSHMEM: MPI must be initialized first");
        mpi_check(MPI_Comm_dup(bootstrap, &control_), "MPI_Comm_dup bootstrap");
        mpi_check(MPI_Comm_rank(control_, &bootstrap_rank_), "MPI_Comm_rank");
        mpi_check(MPI_Comm_size(control_, &size_), "MPI_Comm_size");

        if (!nvshmem_is_initialized()) {
            nvshmemx_init_attr_t attributes{};
            attributes.mpi_comm = &control_;
            nvshmem_check(
                nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attributes),
                "nvshmemx_init_attr");
            owns_nvshmem_ = true;
        }
        if (nvshmem_n_pes() != size_ || nvshmem_my_pe() != bootstrap_rank_) {
            throw std::runtime_error(
                "dsDecompX NVSHMEM: PE ordering does not match MPI bootstrap ordering");
        }
        cuda_check(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming),
                   "cudaEventCreate completion");
        rank_ = bootstrap_rank_;
      } catch (...) {
        cleanup_resources();
        throw;
      }
    }

    ~NvshmemBackend() override {
        if (operation_active_) {
            // Do not add a hidden hot-path barrier. Destruction with an active
            // operation is a lifecycle error; abort the local communicator.
            MPI_Abort(control_, 1);
        }
        cleanup_resources();
    }

private:
    void cleanup_resources() noexcept {
        for (void* allocation : symmetric_allocations_) nvshmem_free(allocation);
        symmetric_allocations_.clear();
        if (completion_) cudaEventDestroy(completion_);
        completion_=nullptr;
        if (cart_ != MPI_COMM_NULL) MPI_Comm_free(&cart_);
        if (owns_nvshmem_ && nvshmem_is_initialized()) nvshmem_finalize();
        owns_nvshmem_=false;
        if (control_ != MPI_COMM_NULL) MPI_Comm_free(&control_);
        control_=MPI_COMM_NULL;
    }

public:

    BackendKind kind() const noexcept override { return BackendKind::Nvshmem; }
    int rank() const noexcept override { return rank_; }
    int size() const noexcept override { return size_; }

    void create_cartesian(const ProcessGrid& grid,
                          const std::array<bool, 3>& periodic) override {
        int dims[3] = {grid.dimensions[0], grid.dimensions[1], grid.dimensions[2]};
        if (dims[0] < 0 || dims[1] < 0 || dims[2] < 0) {
            throw std::invalid_argument("dsDecompX NVSHMEM: negative process-grid dimension");
        }
        mpi_check(MPI_Dims_create(size_, 3, dims), "MPI_Dims_create");
        if (dims[0] * dims[1] * dims[2] != size_) {
            throw std::invalid_argument("dsDecompX NVSHMEM: process grid does not match rank count");
        }
        int periods[3] = {periodic[0] ? 1 : 0,
                          periodic[1] ? 1 : 0,
                          periodic[2] ? 1 : 0};
        mpi_check(MPI_Cart_create(control_, 3, dims, periods,
                                  grid.allow_rank_reorder ? 1 : 0, &cart_),
                  "MPI_Cart_create");
        if (cart_ == MPI_COMM_NULL) {
            throw std::runtime_error("dsDecompX NVSHMEM: MPI_Cart_create returned null");
        }
        mpi_check(MPI_Comm_rank(cart_, &rank_), "MPI cart rank");
        mpi_check(MPI_Cart_coords(cart_, rank_, 3, coordinates_.data()),
                  "MPI_Cart_coords");
        dimensions_ = {dims[0], dims[1], dims[2]};
        cart_to_pe_.resize(static_cast<std::size_t>(size_));
        mpi_check(MPI_Allgather(&bootstrap_rank_, 1, MPI_INT,
                                cart_to_pe_.data(), 1, MPI_INT, cart_),
                  "MPI_Allgather Cartesian-to-PE map");
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
        if (neighbors.size()!=send.size()) throw std::invalid_argument("NVSHMEM count size mismatch");
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
            throw std::invalid_argument("NVSHMEM migration count table size mismatch");
        receive.resize(static_cast<std::size_t>(size_));
        mpi_check(MPI_Alltoall(send.data(),1,MPI_UNSIGNED,receive.data(),1,
                               MPI_UNSIGNED,cart_),"migration count Alltoall");
    }

    bool supports_device_data_plane() const noexcept override { return true; }

    void* allocate_symmetric(std::size_t bytes) override {
        // Symmetric allocation is collective and therefore belongs to plan
        // creation/capacity growth, never the normal timestep hot path.
        void* result = nvshmem_malloc(bytes ? bytes : 1);
        if (!result) throw std::bad_alloc();
        symmetric_allocations_.push_back(result);
        return result;
    }

    void* allocate_symmetric_zeroed(std::size_t count,
                                     std::size_t element_size) override {
        void* result = nvshmem_calloc(count ? count : 1,
                                      element_size ? element_size : 1);
        if (!result) throw std::bad_alloc();
        symmetric_allocations_.push_back(result);
        return result;
    }

    void free_symmetric(void* pointer) override {
        if (!pointer) return;
        const auto found = std::find(symmetric_allocations_.begin(),
                                     symmetric_allocations_.end(), pointer);
        if (found == symmetric_allocations_.end()) {
            throw std::invalid_argument("dsDecompX NVSHMEM: unknown symmetric allocation");
        }
        nvshmem_free(pointer);
        symmetric_allocations_.erase(found);
    }

    void begin_device_exchange(const DeviceExchange& exchange) override {
        if (operation_active_) fail(exchange.operation, "another operation is active");
        if (!exchange.transfers || !exchange.communication_stream) {
            fail(exchange.operation, "missing transfers or communication stream");
        }
        const cudaStream_t stream =
            static_cast<cudaStream_t>(exchange.communication_stream);
        operation_active_ = true;
        try {
            // Payload writes are one-sided. remote_receive_buffer and both
            // signal addresses must come from symmetric plan allocations.
            for (const DeviceTransfer& transfer : *exchange.transfers) {
                if (transfer.peer_rank < 0 || !transfer.send_bytes) continue;
                const int pe = peer(transfer.peer_rank);
                if (pe == bootstrap_rank_) {
                    cuda_check(cudaMemcpyAsync(
                                   transfer.remote_receive_buffer,
                                   transfer.send_buffer,
                                   transfer.send_bytes, cudaMemcpyDeviceToDevice,
                                   stream),
                               context(exchange.operation, "self copy"));
                } else {
                    if (!transfer.remote_receive_buffer || !transfer.remote_signal) {
                        fail(exchange.operation, "missing symmetric remote buffer or signal");
                    }
                    nvshmemx_putmem_on_stream(
                        transfer.remote_receive_buffer,
                        transfer.send_buffer,
                        transfer.send_bytes, pe, stream);
                }
            }

            // quiet provides remote completion for payload puts. The following
            // signal is therefore the visibility/completion notification.
            nvshmemx_quiet_on_stream(stream);
            for (const DeviceTransfer& transfer : *exchange.transfers) {
                if (transfer.peer_rank < 0 || !transfer.send_bytes) continue;
                const int pe = peer(transfer.peer_rank);
                if (pe == bootstrap_rank_) continue;
                nvshmemx_signal_op_on_stream(
                    transfer.remote_signal,
                    transfer.signal_value,
                    NVSHMEM_SIGNAL_SET, pe, stream);
            }

            // Wait only for incoming neighbors, on the communication stream.
            // There is no global barrier and the CPU does not poll each peer.
            for (const DeviceTransfer& transfer : *exchange.transfers) {
                if (transfer.peer_rank < 0 || !transfer.receive_bytes) continue;
                const int pe = peer(transfer.peer_rank);
                if (pe == bootstrap_rank_) continue;
                if (!transfer.local_signal) {
                    fail(exchange.operation, "missing local symmetric signal");
                }
                nvshmemx_signal_wait_until_on_stream(
                    transfer.local_signal, NVSHMEM_CMP_GE,
                    transfer.signal_value, stream);
            }
        } catch (...) {
            operation_active_ = false;
            throw;
        }
    }

    void end_device_exchange(const DeviceExchange& exchange) override {
        if (!operation_active_) fail(exchange.operation, "no operation is active");
        const cudaStream_t communication =
            static_cast<cudaStream_t>(exchange.communication_stream);
        cuda_check(cudaEventRecord(completion_, communication),
                   context(exchange.operation, "record completion"));
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
            "dsDecompX NVSHMEM: host ParticleView is unsupported; use CUDA plans");
    }
    void end_exchange_records() override {
        throw std::runtime_error("dsDecompX NVSHMEM: no host exchange exists");
    }
    void exchange_all_to_all_records(
        const std::vector<std::vector<ParticleRecord>>&,
        std::vector<std::vector<ParticleRecord>>&) override {
        throw std::runtime_error(
            "dsDecompX NVSHMEM: host migration is unsupported; use CUDA plans");
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
        for (int i=0; i<size_; ++i) { offsets[i]=total; total+=counts[i]; }
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
        global.halo_calls=sum(local.halo_calls); global.migration_calls=sum(local.migration_calls);
        global.bytes_sent=sum(local.bytes_sent); global.bytes_received=sum(local.bytes_received);
        global.buffer_reallocations=sum(local.buffer_reallocations);
        global.peak_communication_buffer_capacity=maximum(local.peak_communication_buffer_capacity);
        global.pack_ms=aggregate(local.pack_ms); global.communication_ms=aggregate(local.communication_ms);
        global.wait_ms=aggregate(local.wait_ms); global.unpack_ms=aggregate(local.unpack_ms);
        global.migration_classify_ms=aggregate(local.migration_classify_ms);
        global.migration_compact_ms=aggregate(local.migration_compact_ms);
        return global;
    }
    std::uint64_t buffer_reallocations() const noexcept override { return 0; }
    std::uint64_t peak_buffer_capacity() const noexcept override { return 0; }

private:
    std::string context(const char* operation, const char* action) const {
        std::ostringstream out;
        out << "dsDecompX NVSHMEM rank=" << rank_
            << " operation=" << (operation ? operation : "unknown")
            << " action=" << action;
        return out.str();
    }
    [[noreturn]] void fail(const char* operation, const char* message) const {
        throw std::runtime_error(context(operation, message));
    }
    void mpi_check(int status, const char* action) const {
        if (status == MPI_SUCCESS) return;
        char text[MPI_MAX_ERROR_STRING]{}; int length=0;
        MPI_Error_string(status, text, &length);
        throw std::runtime_error(context("control-plane", action) + ": " +
                                 std::string(text, static_cast<std::size_t>(length)));
    }
    void nvshmem_check(int status, const std::string& action) const {
        if (status != 0) throw std::runtime_error(action + ": status=" + std::to_string(status));
    }
    void cuda_check(cudaError_t status, const std::string& action) const {
        if (status != cudaSuccess) throw std::runtime_error(action + ": " + cudaGetErrorString(status));
    }
    int peer(int cartesian_peer) const {
        if (cartesian_peer < 0 || cartesian_peer >= size_) throw std::out_of_range("invalid peer");
        return cart_to_pe_.empty() ? cartesian_peer : cart_to_pe_[static_cast<std::size_t>(cartesian_peer)];
    }
    std::uint64_t reduce(std::uint64_t value, MPI_Op op) const {
        unsigned long long in=value, out=0;
        mpi_check(MPI_Allreduce(&in,&out,1,MPI_UNSIGNED_LONG_LONG,op,cart_), "MPI_Allreduce");
        return out;
    }
    std::uint64_t sum(std::uint64_t value) const { return reduce(value, MPI_SUM); }
    std::uint64_t maximum(std::uint64_t value) const { return reduce(value, MPI_MAX); }
    SampleAggregate aggregate(const SampleAggregate& in) const {
        SampleAggregate out; out.count=sum(in.count);
        mpi_check(MPI_Allreduce(&in.sum,&out.sum,1,MPI_DOUBLE,MPI_SUM,cart_), "metric sum");
        const double min_value=in.count?in.min:std::numeric_limits<double>::infinity();
        const double max_value=in.count?in.max:-std::numeric_limits<double>::infinity();
        mpi_check(MPI_Allreduce(&min_value,&out.min,1,MPI_DOUBLE,MPI_MIN,cart_), "metric min");
        mpi_check(MPI_Allreduce(&max_value,&out.max,1,MPI_DOUBLE,MPI_MAX,cart_), "metric max");
        if (!out.count) out.min=out.max=0.0; return out;
    }

    MPI_Comm control_=MPI_COMM_NULL;
    MPI_Comm cart_=MPI_COMM_NULL;
    cudaEvent_t completion_=nullptr;
    int bootstrap_rank_=0, rank_=0, size_=1;
    Vec3i dimensions_{1,1,1};
    Vec3i coordinates_{0,0,0};
    std::vector<int> cart_to_pe_;
    std::vector<void*> symmetric_allocations_;
    bool owns_nvshmem_=false;
    bool operation_active_=false;
};

} // namespace
} // namespace dsdecompx::detail

namespace dsdecompx::nvshmem {

Backend make_backend(MPI_Comm bootstrap_communicator) {
    return Backend(std::make_unique<detail::NvshmemBackend>(bootstrap_communicator));
}

} // namespace dsdecompx::nvshmem
