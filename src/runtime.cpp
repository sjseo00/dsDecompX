#include "dsdecompx.hpp"

#include "decomposition.hpp"
#include "backends/mpi.hpp"
#if HAS_NCCL
#include "backends/nccl.hpp"
#endif
#if HAS_NVSHMEM
#include "backends/nvshmem.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace dsdecompx {
namespace {

BackendSelection requested_backend(const Config& config) {
    if (config.advanced.selection != BackendSelection::Auto) {
        return config.advanced.selection;
    }
    if (!config.advanced.read_environment) return BackendSelection::Auto;
    const char* value = std::getenv("BACKEND");
    return value && *value ? parse_backend_selection(value)
                           : BackendSelection::Auto;
}

Backend create_selected_backend(MPI_Comm communicator, const Config& config) {
    const BackendSelection requested = requested_backend(config);
    if (requested == BackendSelection::Nccl) {
        if (config.execution != ExecutionMode::Cuda) {
            throw std::invalid_argument(
                "dsDecompX: NCCL backend requires ExecutionMode::Cuda");
        }
#if HAS_NCCL
        return nccl::make_backend(communicator);
#else
        throw std::runtime_error(
            "dsDecompX: NCCL backend was explicitly requested but not compiled");
#endif
    }
    if (requested == BackendSelection::Nvshmem) {
        if (config.execution != ExecutionMode::Cuda) {
            throw std::invalid_argument(
                "dsDecompX: NVSHMEM backend requires ExecutionMode::Cuda");
        }
#if HAS_NVSHMEM
        return nvshmem::make_backend(communicator);
#else
        throw std::runtime_error(
            "dsDecompX: NVSHMEM backend was explicitly requested but not compiled");
#endif
    }
    if (requested == BackendSelection::Mpi ||
        config.execution == ExecutionMode::Host) {
        mpi::BackendOptions options;
        options.allow_host_staging = config.advanced.allow_host_staging;
        return mpi::make_backend(communicator, options);
    }
#if HAS_NVSHMEM
    try { return nvshmem::make_backend(communicator); } catch (...) {}
#endif
#if HAS_NCCL
    try { return nccl::make_backend(communicator); } catch (...) {}
#endif
    mpi::BackendOptions options;
    options.allow_host_staging = config.advanced.allow_host_staging;
    return mpi::make_backend(communicator, options);
}

void copy_record(ParticleView& view, std::size_t index,
                 const ParticleRecord& record) {
    if (view.x) view.x[index]=record.x;
    if (view.y) view.y[index]=record.y;
    if (view.z) view.z[index]=record.z;
    if (view.vx) view.vx[index]=record.vx;
    if (view.vy) view.vy[index]=record.vy;
    if (view.vz) view.vz[index]=record.vz;
    if (view.fx) view.fx[index]=record.fx;
    if (view.fy) view.fy[index]=record.fy;
    if (view.fz) view.fz[index]=record.fz;
    if (view.mass) view.mass[index]=record.mass;
    if (view.charge) view.charge[index]=record.charge;
    if (view.type) view.type[index]=record.type;
    if (view.global_id) view.global_id[index]=record.global_id;
}

void require_output_capacity(const ParticleView& view, std::size_t required) {
    if (view.count < required) {
        throw std::invalid_argument(
            "dsDecompX: particle storage returned insufficient output capacity");
    }
}

std::uint64_t field_schema_hash(const ConstParticleView& view) {
    std::uint64_t hash=1469598103934665603ull;
    const auto mix=[&](std::uint64_t value) {
        hash^=value;
        hash*=1099511628211ull;
    };
    mix(view.field_count);
    for (std::size_t f=0; f<view.field_count; ++f) {
        const auto& field=view.fields[f];
        for (const unsigned char character : field.name) mix(character);
        mix(0xffu);
        mix(static_cast<std::uint8_t>(field.type));
        mix(field.components);
    }
    return hash;
}

void validate_collective_field_schema(MPI_Comm communicator,
                                      std::uint64_t hash) {
    // One reduction instead of a MIN and a MAX: the minimum of the hash and of
    // its complement gives both extremes (max == ~min(~hash)).
    unsigned long long local[2]={hash,~hash},extreme[2]={0,0};
    MPI_Allreduce(local,extreme,2,MPI_UNSIGNED_LONG_LONG,MPI_MIN,communicator);
    const unsigned long long minimum=extreme[0],maximum=~extreme[1];
    if (minimum!=maximum) {
        throw std::invalid_argument(
            "dsDecompX: custom field names, types, components, and order "
            "must match on every rank");
    }
}

#if HAS_CUDA
void cuda_require(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("dsDecompX: ") + operation +
                                 ": " + cudaGetErrorString(status));
    }
}

// Collective. Selects this rank's GPU per Config::advanced.cuda_device and, for
// backends that need one GPU per rank (NCCL, NVSHMEM), verifies that ranks on
// the same node did not end up sharing a GPU while enough GPUs were visible.
void select_cuda_device(MPI_Comm communicator, const Config& config) {
    constexpr int kBusIdBytes = 32;
    const int requested = config.advanced.cuda_device;
    const BackendSelection backend = requested_backend(config);
    const bool exclusive_required = backend == BackendSelection::Nccl ||
                                    backend == BackendSelection::Nvshmem;

    int device_count = 0;
    int local_error = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
        local_error = 1;

    MPI_Comm node = MPI_COMM_NULL;
    MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &node);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_size(node, &local_size);

    int device = -1;
    if (!local_error) {
        if (requested == kCudaDeviceByLocalRank) {
            device = local_rank % device_count;
        } else if (requested >= 0) {
            if (requested < device_count) device = requested;
            else local_error = 2;
        } else if (cudaGetDevice(&device) != cudaSuccess) {
            local_error = 3;
        }
        if (!local_error && requested != kCudaDeviceCurrent &&
            cudaSetDevice(device) != cudaSuccess) {
            local_error = 3;
        }
    }

    // Physical identity (PCI bus id) so CUDA_VISIBLE_DEVICES remapping cannot
    // make two different GPUs look equal or one GPU look different.
    char bus_id[kBusIdBytes] = {};
    if (!local_error &&
        cudaDeviceGetPCIBusId(bus_id, kBusIdBytes, device) != cudaSuccess) {
        local_error = 3;
    }
    std::vector<char> node_bus_ids(static_cast<std::size_t>(kBusIdBytes) *
                                   static_cast<std::size_t>(local_size));
    MPI_Allgather(bus_id, kBusIdBytes, MPI_CHAR, node_bus_ids.data(),
                  kBusIdBytes, MPI_CHAR, node);
    int shared = 0;
    if (exclusive_required && !local_error && local_size <= device_count) {
        for (int i = 0; i < local_size && !shared; ++i) {
            const char* first = node_bus_ids.data() +
                                static_cast<std::size_t>(i) * kBusIdBytes;
            if (!*first) continue;
            for (int j = i + 1; j < local_size; ++j) {
                if (std::strncmp(first,
                        node_bus_ids.data() +
                            static_cast<std::size_t>(j) * kBusIdBytes,
                        kBusIdBytes) == 0) {
                    shared = 1;
                    break;
                }
            }
        }
    }
    MPI_Comm_free(&node);

    // Reduce so every rank fails together instead of some hanging in the next
    // collective.
    int flags[2] = {local_error, shared};
    int worst[2] = {0, 0};
    MPI_Allreduce(flags, worst, 2, MPI_INT, MPI_MAX, communicator);
    if (worst[0] != 0) {
        throw std::runtime_error(
            "dsDecompX: could not select a CUDA device (no visible device, "
            "Config::advanced.cuda_device out of range, or cudaSetDevice failed)");
    }
    if (worst[1] != 0) {
        throw std::runtime_error(
            "dsDecompX: the NCCL/NVSHMEM backend needs one GPU per rank, but ranks "
            "on the same node share a GPU although enough GPUs are visible; set "
            "Config::advanced.cuda_device = kCudaDeviceByLocalRank or call "
            "cudaSetDevice per rank before creating Decomp");
    }
}

ParticleView allocate_cuda_particles(MemorySpace space, std::size_t capacity) {
    ParticleView view;
    view.memory_space = space;
    view.count = capacity;
    if (capacity == 0) return view;
    auto allocate = [space](auto** pointer, std::size_t bytes) {
        if (space == MemorySpace::CudaManaged) {
            cuda_require(cudaMallocManaged(reinterpret_cast<void**>(pointer), bytes),
                         "cudaMallocManaged failed");
        } else {
            cuda_require(cudaMalloc(reinterpret_cast<void**>(pointer), bytes),
                         "cudaMalloc failed");
        }
    };
    try {
        allocate(&view.x,capacity*sizeof(double)); allocate(&view.y,capacity*sizeof(double));
        allocate(&view.z,capacity*sizeof(double)); allocate(&view.vx,capacity*sizeof(double));
        allocate(&view.vy,capacity*sizeof(double)); allocate(&view.vz,capacity*sizeof(double));
        allocate(&view.fx,capacity*sizeof(double)); allocate(&view.fy,capacity*sizeof(double));
        allocate(&view.fz,capacity*sizeof(double)); allocate(&view.mass,capacity*sizeof(double));
        allocate(&view.charge,capacity*sizeof(double)); allocate(&view.type,capacity*sizeof(int));
        allocate(&view.global_id,capacity*sizeof(int));
    } catch (...) {
        if (view.x) cudaFree(view.x); if (view.y) cudaFree(view.y);
        if (view.z) cudaFree(view.z); if (view.vx) cudaFree(view.vx);
        if (view.vy) cudaFree(view.vy); if (view.vz) cudaFree(view.vz);
        if (view.fx) cudaFree(view.fx); if (view.fy) cudaFree(view.fy);
        if (view.fz) cudaFree(view.fz); if (view.mass) cudaFree(view.mass);
        if (view.charge) cudaFree(view.charge); if (view.type) cudaFree(view.type);
        if (view.global_id) cudaFree(view.global_id);
        throw;
    }
    return view;
}

void free_cuda_particles(ParticleView& view) noexcept {
    if (view.x) cudaFree(view.x); if (view.y) cudaFree(view.y);
    if (view.z) cudaFree(view.z); if (view.vx) cudaFree(view.vx);
    if (view.vy) cudaFree(view.vy); if (view.vz) cudaFree(view.vz);
    if (view.fx) cudaFree(view.fx); if (view.fy) cudaFree(view.fy);
    if (view.fz) cudaFree(view.fz); if (view.mass) cudaFree(view.mass);
    if (view.charge) cudaFree(view.charge); if (view.type) cudaFree(view.type);
    if (view.global_id) cudaFree(view.global_id);
    view = {};
}

void copy_cuda_particles(const ParticleView& source, ParticleView& destination,
                         std::size_t count) {
    if (!count) return;
    auto copy = [count](auto* destination_pointer, const auto* source_pointer) {
        using Value = std::remove_pointer_t<decltype(destination_pointer)>;
        cuda_require(cudaMemcpy(destination_pointer, source_pointer,
                                count*sizeof(Value), cudaMemcpyDefault),
                     "cudaMemcpy while growing ParticleInfo failed");
    };
    copy(destination.x,source.x); copy(destination.y,source.y); copy(destination.z,source.z);
    copy(destination.vx,source.vx); copy(destination.vy,source.vy); copy(destination.vz,source.vz);
    copy(destination.fx,source.fx); copy(destination.fy,source.fy); copy(destination.fz,source.fz);
    copy(destination.mass,source.mass); copy(destination.charge,source.charge);
    copy(destination.type,source.type); copy(destination.global_id,source.global_id);
}
#endif

} // namespace

struct Decomp::Impl {
    MPI_Comm collective_comm = MPI_COMM_NULL;
    std::unique_ptr<Decomposition> decomposition;
    double halo_width = 0.0;
    DynamicLoadBalanceConfig dynamic;
    std::size_t migration_calls = 0;
    bool rebalance_in_progress = false;
    // Schema hash last agreed on by every rank (device halo only, see
    // exchange_halo): while a halo plan is reused the schema cannot change, so
    // the collective check is skipped.
    bool schema_validated = false;
    std::uint64_t validated_schema_hash = 0;

    // A split-phase halo exchange between begin_halo_exchange() and
    // end_halo_exchange().
    struct PendingHalo {
        HaloExchangeHandle handle;
        HaloExchangeOptions options;
        std::size_t owned = 0;
        MemorySpace memory_space = MemorySpace::Host;
        std::vector<int> previous_ghost_ids;
        bool active = false;
    } pending_halo;

    ~Impl() {
        decomposition.reset();
        if (collective_comm != MPI_COMM_NULL) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) MPI_Comm_free(&collective_comm);
        }
    }
};

namespace {

struct RankWorkSample {
    double work = 0.0;
    int coordinates[3]{0, 0, 0};
};

struct GeometryBalanceResult {
    bool attempted = false;
    bool changed = false;
    double imbalance = 1.0;
};

template<class ImplType>
GeometryBalanceResult rebalance_geometry(ImplType& impl,
                                         double local_work,
                                         bool force) {
    if (!std::isfinite(local_work) || local_work < 0.0) {
        throw std::invalid_argument(
            "dsDecompX: rebalance local_work must be finite and non-negative");
    }

    const Topology& topology = impl.decomposition->topology();
    const int rank_count = topology.size();
    RankWorkSample local;
    local.work = local_work;
    for (int axis = 0; axis < 3; ++axis) {
        local.coordinates[axis] = topology.coordinates()[axis];
    }
    std::vector<RankWorkSample> samples(static_cast<std::size_t>(rank_count));
    MPI_Allgather(&local, static_cast<int>(sizeof(local)), MPI_BYTE,
                  samples.data(), static_cast<int>(sizeof(local)), MPI_BYTE,
                  impl.collective_comm);

    double total_work = 0.0;
    double maximum_work = 0.0;
    for (const auto& sample : samples) {
        total_work += sample.work;
        maximum_work = std::max(maximum_work, sample.work);
    }
    GeometryBalanceResult result;
    result.attempted = true;
    if (total_work <= 0.0 || rank_count <= 1) return result;
    result.imbalance = maximum_work /
        (total_work / static_cast<double>(rank_count));
    if (!force && result.imbalance < impl.dynamic.imbalance_threshold) {
        return result;
    }

    auto cuts = topology.axis_cuts();
    const Vec3i& dimensions = topology.dimensions();
    const GlobalDomain& domain = impl.decomposition->global_domain();
    for (int axis = 0; axis < 3; ++axis) {
        const int slab_count = dimensions[axis];
        if (slab_count <= 1) continue;
        std::vector<double> slab_work(static_cast<std::size_t>(slab_count), 0.0);
        for (const auto& sample : samples) {
            slab_work[static_cast<std::size_t>(sample.coordinates[axis])] +=
                sample.work;
        }

        const std::vector<double> old_cuts = cuts[axis];
        const double target = total_work / static_cast<double>(slab_count);
        const double extent = domain.upper[axis] - domain.lower[axis];
        const double uniform_width = extent / static_cast<double>(slab_count);
        // A 26-neighbor exchange reaches only the immediately adjacent cell.
        // Do not let dynamic balancing make a cell narrower than the halo.
        const double minimum_width = std::min(
            uniform_width,
            std::max(uniform_width * impl.dynamic.minimum_cell_fraction,
                     impl.halo_width));
        cuts[axis].front() = domain.lower[axis];
        cuts[axis].back() = domain.upper[axis];
        for (int boundary = 1; boundary < slab_count; ++boundary) {
            const double desired = target * static_cast<double>(boundary);
            double cumulative = 0.0;
            int slab = 0;
            while (slab < slab_count - 1 &&
                   cumulative + slab_work[static_cast<std::size_t>(slab)] < desired) {
                cumulative += slab_work[static_cast<std::size_t>(slab++)];
            }
            const double weight = slab_work[static_cast<std::size_t>(slab)];
            const double fraction = weight > 0.0
                ? std::clamp((desired - cumulative) / weight, 0.0, 1.0)
                : 0.5;
            const double proposed = old_cuts[static_cast<std::size_t>(slab)] +
                fraction * (old_cuts[static_cast<std::size_t>(slab + 1)] -
                            old_cuts[static_cast<std::size_t>(slab)]);
            const double old = old_cuts[static_cast<std::size_t>(boundary)];
            cuts[axis][static_cast<std::size_t>(boundary)] = old +
                impl.dynamic.relaxation * (proposed - old);
        }
        for (int i = 1; i < slab_count; ++i) {
            cuts[axis][static_cast<std::size_t>(i)] = std::max(
                cuts[axis][static_cast<std::size_t>(i)],
                cuts[axis][static_cast<std::size_t>(i - 1)] + minimum_width);
        }
        for (int i = slab_count - 1; i > 0; --i) {
            cuts[axis][static_cast<std::size_t>(i)] = std::min(
                cuts[axis][static_cast<std::size_t>(i)],
                cuts[axis][static_cast<std::size_t>(i + 1)] - minimum_width);
        }
    }

    const auto& old_cuts = topology.axis_cuts();
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < cuts[axis].size(); ++i) {
            if (std::abs(cuts[axis][i] - old_cuts[axis][i]) > 1.0e-12) {
                result.changed = true;
                break;
            }
        }
    }
    if (result.changed) impl.decomposition->set_axis_cuts(cuts);
    return result;
}

} // namespace

ParticleInfo::~ParticleInfo() { release_device_memory(); }

ParticleInfo::ParticleInfo(ParticleInfo&& other) noexcept {
    *this = std::move(other);
}

ParticleInfo& ParticleInfo::operator=(ParticleInfo&& other) noexcept {
    if (this == &other) return *this;
    release_device_memory();
    x=std::move(other.x); y=std::move(other.y); z=std::move(other.z);
    vx=std::move(other.vx); vy=std::move(other.vy); vz=std::move(other.vz);
    fx=std::move(other.fx); fy=std::move(other.fy); fz=std::move(other.fz);
    mass=std::move(other.mass); charge=std::move(other.charge);
    type=std::move(other.type); global_id=std::move(other.global_id);
    num_particles=other.num_particles;
    external_=other.external_; owns_device_memory_=other.owns_device_memory_;
    owned_count_=other.owned_count_; ghost_count_=other.ghost_count_;
    fields_=std::move(other.fields_);
    storage_=other.storage_; provider_=other.provider_;
    storage_.fields=fields_.empty()?nullptr:fields_.data();
    storage_.field_count=fields_.size();
    other.num_particles=0; other.external_=false;
    other.owns_device_memory_=false; other.owned_count_=0; other.ghost_count_=0;
    other.fields_.clear(); other.storage_={}; other.provider_=nullptr;
    return *this;
}

void ParticleInfo::release_device_memory() noexcept {
    if (!owns_device_memory_) return;
#if HAS_CUDA
    free_cuda_particles(storage_);
#endif
    owns_device_memory_ = false;
}

ParticleInfo ParticleInfo::allocate(MemorySpace memory_space,
                                    std::size_t owned) {
    if (memory_space == MemorySpace::Host) return ParticleInfo(owned);
#if HAS_CUDA
    ParticleInfo result;
    result.external_ = true;
    result.owns_device_memory_ = true;
    result.owned_count_ = owned;
    result.storage_ = allocate_cuda_particles(memory_space, owned);
    return result;
#else
    (void)owned;
    throw std::runtime_error(
        "dsDecompX: CUDA ParticleInfo allocation requires a CUDA build");
#endif
}

std::size_t ParticleInfo::capacity() const noexcept {
    return external_ ? storage_.count : x.size();
}

MemorySpace ParticleInfo::memory_space() const noexcept {
    return external_ ? storage_.memory_space : MemorySpace::Host;
}

double* ParticleInfo::x_data() noexcept { return external_ ? storage_.x : x.data(); }
double* ParticleInfo::y_data() noexcept { return external_ ? storage_.y : y.data(); }
double* ParticleInfo::z_data() noexcept { return external_ ? storage_.z : z.data(); }
double* ParticleInfo::vx_data() noexcept { return external_ ? storage_.vx : vx.data(); }
double* ParticleInfo::vy_data() noexcept { return external_ ? storage_.vy : vy.data(); }
double* ParticleInfo::vz_data() noexcept { return external_ ? storage_.vz : vz.data(); }
double* ParticleInfo::fx_data() noexcept { return external_ ? storage_.fx : fx.data(); }
double* ParticleInfo::fy_data() noexcept { return external_ ? storage_.fy : fy.data(); }
double* ParticleInfo::fz_data() noexcept { return external_ ? storage_.fz : fz.data(); }
double* ParticleInfo::mass_data() noexcept { return external_ ? storage_.mass : mass.data(); }
double* ParticleInfo::charge_data() noexcept { return external_ ? storage_.charge : charge.data(); }
int* ParticleInfo::type_data() noexcept { return external_ ? storage_.type : type.data(); }
int* ParticleInfo::global_id_data() noexcept {
    return external_ ? storage_.global_id : global_id.data();
}

void ParticleInfo::commit_counts(std::size_t owned,
                                 std::size_t ghosts) noexcept {
    owned_count_ = owned;
    ghost_count_ = ghosts;
}

void ParticleInfo::reset_counts(std::size_t owned, std::size_t ghosts) {
    const std::size_t total = owned + ghosts;
    if (read_view(total).count < total) {
        throw std::invalid_argument(
            "dsDecompX: particle storage is smaller than the supplied counts");
    }
    commit_counts(owned, ghosts);
}

void ParticleInfo::resize(std::size_t count) {
    x.resize(count); y.resize(count); z.resize(count);
    vx.resize(count); vy.resize(count); vz.resize(count);
    fx.resize(count); fy.resize(count); fz.resize(count);
    mass.resize(count); charge.resize(count);
    type.resize(count); global_id.resize(count);
    num_particles = count;
}

ConstParticleView ParticleInfo::host_const_view(std::size_t count) const {
    ConstParticleView view;
    view.memory_space=MemorySpace::Host; view.count=count;
    view.x=x.data(); view.y=y.data(); view.z=z.data();
    view.vx=vx.data(); view.vy=vy.data(); view.vz=vz.data();
    view.fx=fx.data(); view.fy=fy.data(); view.fz=fz.data();
    view.mass=mass.data(); view.charge=charge.data();
    view.type=type.data(); view.global_id=global_id.data();
    view.fields=fields_.empty()?nullptr:fields_.data();
    view.field_count=fields_.size();
    return view;
}

ParticleView ParticleInfo::host_view(std::size_t count) {
    ParticleView view;
    view.memory_space=MemorySpace::Host; view.count=count;
    view.x=x.data(); view.y=y.data(); view.z=z.data();
    view.vx=vx.data(); view.vy=vy.data(); view.vz=vz.data();
    view.fx=fx.data(); view.fy=fy.data(); view.fz=fz.data();
    view.mass=mass.data(); view.charge=charge.data();
    view.type=type.data(); view.global_id=global_id.data();
    view.fields=fields_.empty()?nullptr:fields_.data();
    view.field_count=fields_.size();
    return view;
}

void ParticleInfo::resize_owned(std::size_t count) {
    if (external_) {
        if (!owns_device_memory_) {
            throw std::logic_error(
                "dsDecompX: resize_owned cannot resize borrowed particle memory");
        }
        resize_storage(count, std::min(count, owned_count_));
        commit_counts(count, 0);
        return;
    }
    for (const auto& field : fields_) {
        if (field.capacity<count) {
            throw std::length_error(
                "dsDecompX: custom particle field '"+field.name+
                "' has insufficient capacity");
        }
    }
    resize(count);
    commit_counts(count, 0);
}

ConstParticleView ParticleInfo::read_view(std::size_t count) const {
    if (external_) {
        if (storage_.count < count) {
            throw std::invalid_argument(
                "dsDecompX: external particle memory has insufficient capacity");
        }
        return as_const_view(storage_);
    }
    return host_const_view(count);
}

ParticleInfo::ParticleInfo(
    std::size_t owned, std::size_t ghosts, ParticleView storage)
    : external_(true), owned_count_(owned), ghost_count_(ghosts),
      storage_(storage) {
    if (storage_.count < owned + ghosts) {
        throw std::invalid_argument(
            "dsDecompX: external particle view is smaller than its particle counts");
    }
    if (storage.fields && storage.field_count) {
        fields_.assign(storage.fields,storage.fields+storage.field_count);
        storage_.fields=fields_.data();
        storage_.field_count=fields_.size();
    }
}

ParticleInfo::ParticleInfo(
    std::size_t owned, std::size_t ghosts, Provider& provider)
    : ParticleInfo(owned, ghosts, provider.particle_info_view(owned + ghosts)) {
    provider_ = &provider;
}

ParticleInfo ParticleInfo::external(MemorySpace memory_space,
                                    std::size_t capacity,
                                    std::size_t owned,
                                    std::size_t ghosts) {
    ParticleView view;
    view.memory_space = memory_space;
    view.count = capacity;
    return ParticleInfo(owned, ghosts, view);
}

ParticleInfo& ParticleInfo::positions(double* x_value, double* y_value,
                                      double* z_value) noexcept {
    storage_.x=x_value; storage_.y=y_value; storage_.z=z_value;
    return *this;
}

ParticleInfo& ParticleInfo::velocities(double* vx_value, double* vy_value,
                                       double* vz_value) noexcept {
    storage_.vx=vx_value; storage_.vy=vy_value; storage_.vz=vz_value;
    return *this;
}

ParticleInfo& ParticleInfo::forces(double* fx_value, double* fy_value,
                                   double* fz_value) noexcept {
    storage_.fx=fx_value; storage_.fy=fy_value; storage_.fz=fz_value;
    return *this;
}

ParticleInfo& ParticleInfo::properties(double* mass_value,
                                       double* charge_value,
                                       int* type_value,
                                       int* global_id_value) noexcept {
    storage_.mass=mass_value; storage_.charge=charge_value;
    storage_.type=type_value; storage_.global_id=global_id_value;
    return *this;
}

ParticleInfo& ParticleInfo::add_field_view(ParticleFieldView field) {
    if (field.name.empty() || !field.data || field.components==0 ||
        field.capacity<size()) {
        throw std::invalid_argument(
            "dsDecompX: invalid or undersized custom particle field");
    }
    const auto duplicate=std::find_if(
        fields_.begin(),fields_.end(),
        [&](const ParticleFieldView& existing) {
            return existing.name==field.name;
        });
    if (duplicate!=fields_.end()) {
        throw std::invalid_argument(
            "dsDecompX: duplicate custom particle field '"+field.name+"'");
    }
    fields_.push_back(std::move(field));
    storage_.fields=fields_.data();
    storage_.field_count=fields_.size();
    return *this;
}

StorageResizeResult ParticleInfo::resize_storage(
    std::size_t required, std::size_t preserve) {
    for (const auto& field : fields_) {
        if (field.capacity<required) {
            throw std::length_error(
                "dsDecompX: custom particle field '"+field.name+
                "' has insufficient capacity");
        }
    }
    if (!external_) {
        const double* old_x = x.data();
        resize(required);
        return {host_view(required), old_x != x.data()};
    }
    if (required <= storage_.count) return {storage_, false};
    if (owns_device_memory_) {
#if HAS_CUDA
        const std::size_t grown = std::max(required,
            std::max<std::size_t>(64, storage_.count + storage_.count/2));
        ParticleView replacement =
            allocate_cuda_particles(storage_.memory_space, grown);
        try {
            copy_cuda_particles(storage_, replacement,
                                std::min(preserve, storage_.count));
        } catch (...) {
            free_cuda_particles(replacement);
            throw;
        }
        free_cuda_particles(storage_);
        storage_ = replacement;
        storage_.fields=fields_.empty()?nullptr:fields_.data();
        storage_.field_count=fields_.size();
        return {storage_, true};
#else
        throw std::runtime_error(
            "dsDecompX: CUDA ParticleInfo resize requires a CUDA build");
#endif
    }
    if (!provider_) {
        throw std::length_error(
            "dsDecompX: fixed external particle storage is too small");
    }
    StorageResizeResult result =
        provider_->resize_particle_info(required, preserve);
    storage_ = result.view;
    storage_.fields=fields_.empty()?nullptr:fields_.data();
    storage_.field_count=fields_.size();
    result.view=storage_;
    return result;
}

void ParticleInfo::reset_view(ParticleView storage) {
    if (storage.count < size()) {
        throw std::invalid_argument(
            "dsDecompX: replacement particle view is smaller than current particles");
    }
    release_device_memory();
    external_ = true;
    owns_device_memory_ = false;
    provider_ = nullptr;
    storage_ = storage;
    fields_.clear();
    if (storage.fields && storage.field_count) {
        fields_.assign(storage.fields,storage.fields+storage.field_count);
        storage_.fields=fields_.data();
        storage_.field_count=fields_.size();
    }
}

Decomp::Decomp(MPI_Comm communicator, Config config)
    : impl_(std::make_unique<Impl>()) {
    if (communicator == MPI_COMM_NULL) {
        throw std::invalid_argument("dsDecompX: MPI communicator is null");
    }
#if !HAS_CUDA
    if (config.execution == ExecutionMode::Cuda) {
        throw std::runtime_error(
            "dsDecompX: CUDA execution requested but ENABLE_CUDA is OFF");
    }
#endif
    if (config.grid_policy == GridPolicy::TopologyAware &&
        !config.topology_info.available) {
        const double intra_weight=config.topology_info.intra_node_weight;
        const double inter_weight=config.topology_info.inter_node_weight;
        config.topology_info = mpi::discover_topology(communicator);
        config.topology_info.intra_node_weight=intra_weight;
        config.topology_info.inter_node_weight=inter_weight;
    }
    if (!(config.dynamic.imbalance_threshold >= 1.0) ||
        !(config.dynamic.relaxation > 0.0 && config.dynamic.relaxation <= 1.0) ||
        !(config.dynamic.minimum_cell_fraction > 0.0 &&
          config.dynamic.minimum_cell_fraction <= 1.0)) {
        throw std::invalid_argument("dsDecompX: invalid dynamic load-balance configuration");
    }
#if HAS_CUDA
    if (config.execution == ExecutionMode::Cuda) {
        select_cuda_device(communicator, config);
    }
#endif
    MPI_Comm_dup(communicator, &impl_->collective_comm);
    impl_->halo_width = config.halo.width;
    impl_->dynamic = config.dynamic;
    Backend backend = create_selected_backend(communicator, config);
    impl_->decomposition = std::make_unique<Decomposition>(
        std::move(config), std::move(backend));
}

// HaloExchangeOptions is passed by value, so its layout is part of the ABI. A client compares this
// with its own sizeof to catch headers and library from different builds (see MDSystem).
std::size_t halo_exchange_options_size() noexcept;
std::size_t halo_exchange_options_size() noexcept { return sizeof(HaloExchangeOptions); }

Decomp::~Decomp() = default;
Decomp::Decomp(Decomp&&) noexcept = default;
Decomp& Decomp::operator=(Decomp&&) noexcept = default;

const Topology& Decomp::topology() const noexcept {
    return impl_->decomposition->topology();
}

BackendKind Decomp::backend_kind() const noexcept {
    return impl_->decomposition->backend_kind();
}

bool Decomp::mpi_takes_device_pointers() const noexcept {
    return impl_->decomposition->backend_kind() == BackendKind::Mpi &&
           impl_->decomposition->cuda_aware_mpi_enabled();
}

int Decomp::owner_rank(const Vec3d& position) const {
    return impl_->decomposition->owner_rank(position);
}

bool Decomp::any_rank_true(bool local_flag) const {
    int local = local_flag ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX,
                  impl_->collective_comm);
    return global != 0;
}

bool Decomp::has_out_of_domain_particles(
    const ParticleInfo& particles) const {
    const ConstParticleView view = particles.read_view(particles.owned_count());
    bool local_out = false;
    for (std::size_t i=0; i<particles.owned_count(); ++i) {
        if (owner_rank(Vec3d{view.x[i], view.y[i], view.z[i]}) !=
            topology().rank()) {
            local_out = true;
            break;
        }
    }
    return any_rank_true(local_out);
}

void Decomp::update_domain(const GlobalDomain& domain) {
    impl_->decomposition->update_domain(domain);
}

void Decomp::set_halo_width(double width) {
    impl_->decomposition->set_halo_width(width);
    impl_->halo_width = width;
}

double Decomp::halo_width() const noexcept { return impl_->halo_width; }

void Decomp::invalidate_halo() noexcept {
    impl_->decomposition->invalidate_halo_plan();
}

HaloResult Decomp::exchange_halo(ParticleInfo& particles,
                                 HaloExchangeOptions options) {
    begin_halo_exchange(particles, options);
    return end_halo_exchange(particles);
}

bool Decomp::can_refresh_halo(const ParticleInfo& particles) const {
    const std::size_t owned = particles.owned_count();
    ConstParticleView input = particles.read_view(owned);
    input.count = owned;
    return impl_->decomposition->can_refresh_halo_plan(input);
}

void Decomp::begin_halo_exchange(ParticleInfo& particles,
                                 HaloExchangeOptions options) {
    if (impl_->pending_halo.active) {
        throw std::logic_error(
            "dsDecompX: a split-phase halo exchange is already in flight");
    }
    if (options.rebuild) invalidate_halo();
    const std::size_t owned = particles.owned_count();
    ConstParticleView input = particles.read_view(owned);
    input.count = owned;
    {
        const std::uint64_t schema_hash = field_schema_hash(input);
        // A reused device plan (options.rebuild == false) is only taken when
        // every rank refreshes it together, so a schema that was already agreed
        // on needs no new collective on those calls. Host views always check.
        const bool skip_validation =
            input.memory_space != MemorySpace::Host && !options.rebuild &&
            impl_->schema_validated && impl_->validated_schema_hash == schema_hash;
        if (!skip_validation) {
            validate_collective_field_schema(impl_->collective_comm, schema_hash);
            impl_->schema_validated = true;
            impl_->validated_schema_hash = schema_hash;
        }
    }

    std::vector<int> previous_ghost_ids;
    if (input.memory_space == MemorySpace::Host && particles.ghost_count() &&
        input.global_id) {
        ConstParticleView all = particles.read_view(particles.size());
        previous_ghost_ids.assign(all.global_id + owned,
                                  all.global_id + particles.size());
    }

    auto& pending = impl_->pending_halo;
    pending.handle = impl_->decomposition->begin_halo_exchange(
        input, options.stream, options.positions_only);
    pending.options = options;
    pending.owned = owned;
    pending.memory_space = input.memory_space;
    pending.previous_ghost_ids = std::move(previous_ghost_ids);
    pending.active = true;
}

HaloResult Decomp::end_halo_exchange(ParticleInfo& particles) {
    auto& pending = impl_->pending_halo;
    if (!pending.active) {
        throw std::logic_error(
            "dsDecompX: end_halo_exchange without a matching begin_halo_exchange");
    }
    pending.active = false;
    HaloExchangeHandle handle = std::move(pending.handle);
    const HaloExchangeOptions options = pending.options;
    const std::size_t owned = pending.owned;
    const MemorySpace input_space = pending.memory_space;
    const std::vector<int> previous_ghost_ids = std::move(pending.previous_ghost_ids);

    impl_->decomposition->end_halo_exchange(handle, options.stream);

    const std::size_t ghosts =
        impl_->decomposition->total_received_halo_particles();
    const std::size_t required = owned + ghosts;
    StorageResizeResult output = particles.resize_storage(required, owned);
    require_output_capacity(output.view, required);
    if (output.reallocated && impl_->decomposition->halo_positions_only()) {
        // Only the owned particles survive a reallocation, and a positions-only refresh sends nothing
        // but x, y, z: the ghosts would lose their velocity, mass, charge, type and id.
        throw std::logic_error(
            "dsDecompX: a positions-only halo refresh needed a larger particle storage; "
            "the ghosts' other fields are gone (use a full exchange)");
    }

    if (input_space == MemorySpace::Host) {
        if (output.view.memory_space != MemorySpace::Host) {
            throw std::invalid_argument(
                "dsDecompX: host halo input requires host output storage");
        }
        std::size_t index = owned;
        for (const auto& buffer : impl_->decomposition->received_halos()) {
            for (const ParticleRecord& record : buffer) {
                copy_record(output.view, index++, record);
            }
        }
        impl_->decomposition->unpack_halo_fields(output.view,owned);
    } else if (ghosts != 0) {
        impl_->decomposition->unpack_device_halos(output.view, owned,
                                                   options.stream);
    }

    // A device halo that did not refresh an existing plan in place built a new
    // one, whose ghost order may differ from the previous call's even when the
    // count matches (the host path compares global IDs below instead).
    bool layout_changed = options.rebuild ||
                          particles.ghost_count() != ghosts ||
                          (input_space != MemorySpace::Host && ghosts != 0 &&
                           !impl_->decomposition->halo_plan_reused());
    if (!layout_changed && input_space == MemorySpace::Host) {
        for (std::size_t i=0; i<ghosts; ++i) {
            if (previous_ghost_ids[i] != output.view.global_id[owned+i]) {
                layout_changed = true;
                break;
            }
        }
    }
    particles.commit_counts(owned, ghosts);
    return {ghosts, layout_changed, output.reallocated,
            impl_->decomposition->halo_plan_reused()};
}

MigrationResult Decomp::migrate(ParticleInfo& particles,
                                ExecutionStream stream) {
    ++impl_->migration_calls;
    if (!impl_->rebalance_in_progress &&
        impl_->decomposition->grid_policy() == GridPolicy::Dynamic &&
        impl_->dynamic.interval != 0 &&
        impl_->migration_calls % impl_->dynamic.interval == 0) {
        rebalance_geometry(*impl_, static_cast<double>(particles.owned_count()),
                           false);
    }
    const std::size_t owned = particles.owned_count();
    ConstParticleView input = particles.read_view(owned);
    input.count = owned;
    validate_collective_field_schema(impl_->collective_comm,
                                     field_schema_hash(input));
    MigrationResult result = impl_->decomposition->migrate_particles(input,
                                                                      stream);
    StorageResizeResult output =
        particles.resize_storage(result.new_owned_count, 0);
    require_output_capacity(output.view, result.new_owned_count);
    if (input.memory_space == MemorySpace::Host) {
        if (output.view.memory_space != MemorySpace::Host) {
            throw std::invalid_argument(
                "dsDecompX: host migration input requires host output storage");
        }
        const auto& records = impl_->decomposition->migrated_owned_particles();
        for (std::size_t i=0; i<records.size(); ++i) {
            copy_record(output.view, i, records[i]);
        }
        impl_->decomposition->unpack_migrated_fields(output.view);
    } else {
        impl_->decomposition->unpack_migrated_device_particles(output.view,
                                                                stream);
    }
    particles.commit_counts(result.new_owned_count, 0);
    return result;
}

RebalanceResult Decomp::rebalance(ParticleInfo& particles,
                                  RebalanceOptions options,
                                  ExecutionStream stream) {
    const double work = options.local_work >= 0.0
        ? options.local_work
        : static_cast<double>(particles.owned_count());
    const GeometryBalanceResult geometry =
        rebalance_geometry(*impl_, work, options.force);
    RebalanceResult result;
    result.attempted = geometry.attempted;
    result.geometry_changed = geometry.changed;
    result.imbalance_before = geometry.imbalance;

    struct Guard {
        bool& flag;
        explicit Guard(bool& value) : flag(value) { flag = true; }
        ~Guard() { flag = false; }
    } guard(impl_->rebalance_in_progress);
    result.migration = migrate(particles, stream);
    return result;
}

void Decomp::reverse_halo(ParticleInfo& particles,
                          ExecutionStream stream) {
    StorageResizeResult storage =
        particles.resize_storage(particles.size(), particles.size());
    require_output_capacity(storage.view, particles.size());
    impl_->decomposition->reverse_halo(storage.view,
                                       particles.owned_count(), stream);
}

void Decomp::reverse_exchange(
    ParticleInfo& particles,
    const std::vector<ReverseExchangeField>& fields,
    ExecutionStream stream) {
    StorageResizeResult storage=
        particles.resize_storage(particles.size(),particles.size());
    require_output_capacity(storage.view,particles.size());
    impl_->decomposition->reverse_exchange(
        storage.view,particles.owned_count(),fields,stream);
}

void Decomp::set_performance_counters_enabled(bool enabled,
                                              bool neighbor_detail) noexcept {
    impl_->decomposition->set_performance_counters_enabled(enabled,
                                                            neighbor_detail);
}

PerfStats Decomp::get_stats() const { return impl_->decomposition->get_stats(); }
PerfStats Decomp::get_global_stats() const {
    return impl_->decomposition->get_global_stats();
}
void Decomp::reset_stats() noexcept { impl_->decomposition->reset_stats(); }

} // namespace dsdecompx
