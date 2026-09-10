#include "runtime.hpp"

#include "dsdecompx.hpp"
#include "backends/mpi.hpp"
#if HAS_NCCL
#include "backends/nccl.hpp"
#endif
#if HAS_NVSHMEM
#include "backends/nvshmem.hpp"
#endif

#include <algorithm>
#include <cstdlib>
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

void validate_collective_field_schema(MPI_Comm communicator,
                                      const ConstParticleView& view) {
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
    unsigned long long local=hash,minimum=0,maximum=0;
    MPI_Allreduce(&local,&minimum,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,communicator);
    MPI_Allreduce(&local,&maximum,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,communicator);
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

    ~Impl() {
        decomposition.reset();
        if (collective_comm != MPI_COMM_NULL) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) MPI_Comm_free(&collective_comm);
        }
    }
};

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
    MPI_Comm_dup(communicator, &impl_->collective_comm);
    impl_->halo_width = config.halo.width;
    Backend backend = create_selected_backend(communicator, config);
    impl_->decomposition = std::make_unique<Decomposition>(
        std::move(config), std::move(backend));
}

Decomp::~Decomp() = default;
Decomp::Decomp(Decomp&&) noexcept = default;
Decomp& Decomp::operator=(Decomp&&) noexcept = default;

const Topology& Decomp::topology() const noexcept {
    return impl_->decomposition->topology();
}

BackendKind Decomp::backend_kind() const noexcept {
    return impl_->decomposition->backend_kind();
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
    if (options.rebuild) invalidate_halo();
    const std::size_t owned = particles.owned_count();
    ConstParticleView input = particles.read_view(owned);
    input.count = owned;
    validate_collective_field_schema(impl_->collective_comm,input);

    std::vector<int> previous_ghost_ids;
    if (input.memory_space == MemorySpace::Host && particles.ghost_count() &&
        input.global_id) {
        ConstParticleView all = particles.read_view(particles.size());
        previous_ghost_ids.assign(all.global_id + owned,
                                  all.global_id + particles.size());
    }

    auto handle = impl_->decomposition->begin_halo_exchange(input,
                                                             options.stream);
    impl_->decomposition->end_halo_exchange(handle, options.stream);

    const std::size_t ghosts =
        impl_->decomposition->total_received_halo_particles();
    const std::size_t required = owned + ghosts;
    StorageResizeResult output = particles.resize_storage(required, owned);
    require_output_capacity(output.view, required);

    if (input.memory_space == MemorySpace::Host) {
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

    bool layout_changed = options.rebuild ||
                          particles.ghost_count() != ghosts;
    if (!layout_changed && input.memory_space == MemorySpace::Host) {
        for (std::size_t i=0; i<ghosts; ++i) {
            if (previous_ghost_ids[i] != output.view.global_id[owned+i]) {
                layout_changed = true;
                break;
            }
        }
    }
    particles.commit_counts(owned, ghosts);
    return {ghosts, layout_changed, output.reallocated};
}

MigrationResult Decomp::migrate(ParticleInfo& particles,
                                ExecutionStream stream) {
    const std::size_t owned = particles.owned_count();
    ConstParticleView input = particles.read_view(owned);
    input.count = owned;
    validate_collective_field_schema(impl_->collective_comm,input);
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
