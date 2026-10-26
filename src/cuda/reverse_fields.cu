#include "kernels.hpp"
#include "cuda_error.hpp"

namespace dsdecompx::cuda {
namespace {

struct DeviceReverseFields {
    int count;
    std::size_t stride;
    unsigned char type[kMaxDeviceFields];
    unsigned char op[kMaxDeviceFields];
    unsigned int components[kMaxDeviceFields];
    unsigned int words[kMaxDeviceFields];
    std::size_t offset[kMaxDeviceFields];
    void* data[kMaxDeviceFields];
};

DeviceReverseFields to_device(const ReverseFields& fields) {
    DeviceReverseFields device{};
    device.count = static_cast<int>(fields.count);
    device.stride = fields.stride;
    for (std::size_t f = 0; f < fields.count; ++f) {
        const std::size_t element = field_type_size(fields.type[f]);
        device.type[f] = static_cast<unsigned char>(fields.type[f]);
        device.op[f] = static_cast<unsigned char>(fields.op[f]);
        device.components[f] = static_cast<unsigned int>(fields.components[f]);
        device.words[f] = static_cast<unsigned int>(fields.components[f] * element / 4);
        device.offset[f] = fields.offset[f];
        device.data[f] = fields.data[f];
    }
    return device;
}

// ---- Atomic reductions ------------------------------------------------------
// int64 uses `long long` and uint64 `unsigned long long`, the widths CUDA's
// atomics are defined for; the field's own std::int64_t is the same size.

__device__ inline void atomic_add_value(int* a, int v) { atomicAdd(a, v); }
__device__ inline void atomic_add_value(unsigned int* a, unsigned int v) { atomicAdd(a, v); }
__device__ inline void atomic_add_value(unsigned long long* a, unsigned long long v) {
    atomicAdd(a, v);
}
__device__ inline void atomic_add_value(long long* a, long long v) {
    // Two's-complement addition is the same bit operation as unsigned addition.
    atomicAdd(reinterpret_cast<unsigned long long*>(a),
              static_cast<unsigned long long>(v));
}
__device__ inline void atomic_add_value(float* a, float v) { atomicAdd(a, v); }
__device__ inline void atomic_add_value(double* a, double v) { atomicAdd(a, v); }

__device__ inline void atomic_min_value(int* a, int v) { atomicMin(a, v); }
__device__ inline void atomic_min_value(unsigned int* a, unsigned int v) { atomicMin(a, v); }
__device__ inline void atomic_min_value(long long* a, long long v) { atomicMin(a, v); }
__device__ inline void atomic_min_value(unsigned long long* a, unsigned long long v) {
    atomicMin(a, v);
}
__device__ inline void atomic_max_value(int* a, int v) { atomicMax(a, v); }
__device__ inline void atomic_max_value(unsigned int* a, unsigned int v) { atomicMax(a, v); }
__device__ inline void atomic_max_value(long long* a, long long v) { atomicMax(a, v); }
__device__ inline void atomic_max_value(unsigned long long* a, unsigned long long v) {
    atomicMax(a, v);
}

// Floating-point min/max have no native atomic: compare-and-swap on the bits.
// The result matches std::min / std::max applied to (destination, value).
__device__ inline void atomic_min_value(float* a, float v) {
    auto* bits = reinterpret_cast<unsigned int*>(a);
    unsigned int old = *bits, assumed;
    do {
        assumed = old;
        if (!(v < __uint_as_float(assumed))) return;
        old = atomicCAS(bits, assumed, __float_as_uint(v));
    } while (assumed != old);
}
__device__ inline void atomic_max_value(float* a, float v) {
    auto* bits = reinterpret_cast<unsigned int*>(a);
    unsigned int old = *bits, assumed;
    do {
        assumed = old;
        if (!(v > __uint_as_float(assumed))) return;
        old = atomicCAS(bits, assumed, __float_as_uint(v));
    } while (assumed != old);
}
__device__ inline void atomic_min_value(double* a, double v) {
    auto* bits = reinterpret_cast<unsigned long long*>(a);
    unsigned long long old = *bits, assumed;
    do {
        assumed = old;
        if (!(v < __longlong_as_double(static_cast<long long>(assumed)))) return;
        old = atomicCAS(bits, assumed,
                        static_cast<unsigned long long>(__double_as_longlong(v)));
    } while (assumed != old);
}
__device__ inline void atomic_max_value(double* a, double v) {
    auto* bits = reinterpret_cast<unsigned long long*>(a);
    unsigned long long old = *bits, assumed;
    do {
        assumed = old;
        if (!(v > __longlong_as_double(static_cast<long long>(assumed)))) return;
        old = atomicCAS(bits, assumed,
                        static_cast<unsigned long long>(__double_as_longlong(v)));
    } while (assumed != old);
}

template<class T>
__device__ inline void reduce_value(T* address, T value, int op) {
    switch (op) {
    case 0: atomic_add_value(address, value); break;   // Sum
    case 1: atomic_min_value(address, value); break;   // Min
    case 2: atomic_max_value(address, value); break;   // Max
    default: *address = value; break;                  // Replace
    }
}

template<class T>
__device__ inline void reduce_components(void* data, const unsigned char* field_bytes,
                                         std::size_t owner, unsigned int components,
                                         int op) {
    T* out = static_cast<T*>(data) + owner * components;
    const T* in = reinterpret_cast<const T*>(field_bytes);
    for (unsigned int c = 0; c < components; ++c) reduce_value(out + c, in[c], op);
}

__global__ void pack_reverse_fields_kernel(DeviceReverseFields fields,
                                           std::size_t ghost_offset,
                                           std::size_t count,
                                           unsigned char* output) {
    const std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const std::size_t ghost = ghost_offset + k;
    unsigned char* record = output + k * fields.stride;
    for (int f = 0; f < fields.count; ++f) {
        const unsigned int* source =
            static_cast<const unsigned int*>(fields.data[f]) + ghost * fields.words[f];
        unsigned int* target = reinterpret_cast<unsigned int*>(record + fields.offset[f]);
        for (unsigned int w = 0; w < fields.words[f]; ++w) target[w] = source[w];
    }
}

// Record k is the value a neighbor computed for the ghost that this rank sent
// it, in send order, so sources[k] is the owned particle it reduces into.
__global__ void reduce_reverse_fields_kernel(DeviceReverseFields fields,
                                             const unsigned char* records,
                                             std::size_t count,
                                             const unsigned int* sources) {
    const std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const unsigned char* record = records + k * fields.stride;
    const std::size_t owner = sources[k];
    for (int f = 0; f < fields.count; ++f) {
        const unsigned char* value = record + fields.offset[f];
        switch (static_cast<FieldType>(fields.type[f])) {
        case FieldType::Int32:
            reduce_components<int>(fields.data[f], value, owner,
                                   fields.components[f], fields.op[f]); break;
        case FieldType::Int64:
            reduce_components<long long>(fields.data[f], value, owner,
                                         fields.components[f], fields.op[f]); break;
        case FieldType::UInt32:
            reduce_components<unsigned int>(fields.data[f], value, owner,
                                            fields.components[f], fields.op[f]); break;
        case FieldType::UInt64:
            reduce_components<unsigned long long>(fields.data[f], value, owner,
                                                  fields.components[f], fields.op[f]);
            break;
        case FieldType::Float32:
            reduce_components<float>(fields.data[f], value, owner,
                                     fields.components[f], fields.op[f]); break;
        case FieldType::Float64:
            reduce_components<double>(fields.data[f], value, owner,
                                      fields.components[f], fields.op[f]); break;
        }
    }
}

// One block per record: the threads walk over the 4-byte words of every field, so a wide field (a node feature vector of
// hundreds of values) is read and written coalesced.
__global__ void gather_fields_kernel(DeviceReverseFields fields, const unsigned int* sources,
                                     std::size_t count, unsigned char* output) {
    const std::size_t k = blockIdx.x;
    if (k >= count) return;
    const std::size_t owner = sources[k];
    unsigned char* record = output + k * fields.stride;
    for (int f = 0; f < fields.count; ++f) {
        const unsigned int* source =
            static_cast<const unsigned int*>(fields.data[f]) + owner * fields.words[f];
        unsigned int* target = reinterpret_cast<unsigned int*>(record + fields.offset[f]);
        for (unsigned int w = threadIdx.x; w < fields.words[f]; w += blockDim.x) target[w] = source[w];
    }
}

__global__ void scatter_fields_kernel(DeviceReverseFields fields, const unsigned char* records,
                                      std::size_t count, std::size_t ghost_offset) {
    const std::size_t k = blockIdx.x;
    if (k >= count) return;
    const std::size_t ghost = ghost_offset + k;
    const unsigned char* record = records + k * fields.stride;
    for (int f = 0; f < fields.count; ++f) {
        const unsigned int* source = reinterpret_cast<const unsigned int*>(record + fields.offset[f]);
        unsigned int* target = static_cast<unsigned int*>(fields.data[f]) + ghost * fields.words[f];
        for (unsigned int w = threadIdx.x; w < fields.words[f]; w += blockDim.x) target[w] = source[w];
    }
}

} // namespace

void launch_gather_fields(const ReverseFields& fields, const unsigned int* sources,
                          std::size_t count, unsigned char* output, cudaStream_t stream) {
    if (!count || !fields.count) return;
    gather_fields_kernel<<<static_cast<unsigned>(count), 128, 0, stream>>>(
        to_device(fields), sources, count, output);
    detail::check(cudaGetLastError(), "launch forward field gather kernel");
}

void launch_scatter_fields(const ReverseFields& fields, const unsigned char* records,
                           std::size_t count, std::size_t ghost_offset, cudaStream_t stream) {
    if (!count || !fields.count) return;
    scatter_fields_kernel<<<static_cast<unsigned>(count), 128, 0, stream>>>(
        to_device(fields), records, count, ghost_offset);
    detail::check(cudaGetLastError(), "launch forward field scatter kernel");
}

void launch_pack_reverse_fields(const ReverseFields& fields,
                                std::size_t ghost_offset, std::size_t count,
                                unsigned char* output, cudaStream_t stream) {
    if (!count || !fields.count) return;
    constexpr int threads = 256;
    pack_reverse_fields_kernel<<<static_cast<unsigned>((count + threads - 1) / threads),
                                 threads, 0, stream>>>(
        to_device(fields), ghost_offset, count, output);
    detail::check(cudaGetLastError(), "launch reverse field pack kernel");
}

void launch_reduce_reverse_fields(const ReverseFields& fields,
                                  const unsigned char* records, std::size_t count,
                                  const unsigned int* sources, cudaStream_t stream) {
    if (!count || !fields.count) return;
    constexpr int threads = 256;
    reduce_reverse_fields_kernel<<<static_cast<unsigned>((count + threads - 1) / threads),
                                   threads, 0, stream>>>(
        to_device(fields), records, count, sources);
    detail::check(cudaGetLastError(), "launch reverse field reduce kernel");
}

} // namespace dsdecompx::cuda
