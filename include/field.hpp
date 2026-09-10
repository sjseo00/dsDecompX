#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace dsdecompx {

enum class MemorySpace : std::uint8_t {
    Host,
    CudaDevice,
    CudaManaged
};

using ExecutionStream = void*;

// Runtime type information for application-defined per-particle fields.
// Fields are non-owning, contiguous arrays. A field with components > 1 is
// laid out as AoS: [particle0_component0, particle0_component1, ...].
enum class FieldType : std::uint8_t {
    Int32,
    Int64,
    UInt32,
    UInt64,
    Float32,
    Float64
};

enum class ReduceOp : std::uint8_t {
    Sum,
    Min,
    Max,
    Replace
};

inline std::size_t field_type_size(FieldType type) {
    switch (type) {
    case FieldType::Int32: return sizeof(std::int32_t);
    case FieldType::Int64: return sizeof(std::int64_t);
    case FieldType::UInt32: return sizeof(std::uint32_t);
    case FieldType::UInt64: return sizeof(std::uint64_t);
    case FieldType::Float32: return sizeof(float);
    case FieldType::Float64: return sizeof(double);
    }
    throw std::invalid_argument("dsDecompX: unsupported field type");
}

template<class T> struct FieldTypeOf;
template<> struct FieldTypeOf<std::int32_t> {
    static constexpr FieldType value = FieldType::Int32;
};
template<> struct FieldTypeOf<std::int64_t> {
    static constexpr FieldType value = FieldType::Int64;
};
template<> struct FieldTypeOf<std::uint32_t> {
    static constexpr FieldType value = FieldType::UInt32;
};
template<> struct FieldTypeOf<std::uint64_t> {
    static constexpr FieldType value = FieldType::UInt64;
};
template<> struct FieldTypeOf<float> {
    static constexpr FieldType value = FieldType::Float32;
};
template<> struct FieldTypeOf<double> {
    static constexpr FieldType value = FieldType::Float64;
};

template<class T>
inline constexpr FieldType field_type_v =
    FieldTypeOf<std::remove_cv_t<T>>::value;

struct ParticleFieldView {
    std::string name;
    FieldType type = FieldType::Float64;
    std::size_t components = 1;
    std::size_t capacity = 0;
    void* data = nullptr;

    std::size_t record_bytes() const {
        return components * field_type_size(type);
    }
};

struct ReverseExchangeField {
    std::string name;
    ReduceOp operation = ReduceOp::Sum;
};

// Non-owning SoA view backed by host, CUDA device, or managed pointers.
struct ConstParticleView {
    MemorySpace memory_space = MemorySpace::Host;
    std::size_t count = 0;

    const double* x = nullptr;
    const double* y = nullptr;
    const double* z = nullptr;
    const double* vx = nullptr;
    const double* vy = nullptr;
    const double* vz = nullptr;
    const double* fx = nullptr;
    const double* fy = nullptr;
    const double* fz = nullptr;
    const double* mass = nullptr;
    const double* charge = nullptr;
    const int* type = nullptr;
    const int* global_id = nullptr;
    const ParticleFieldView* fields = nullptr;
    std::size_t field_count = 0;
};

struct ParticleView {
    MemorySpace memory_space = MemorySpace::Host;
    std::size_t count = 0;

    double* x = nullptr;
    double* y = nullptr;
    double* z = nullptr;
    double* vx = nullptr;
    double* vy = nullptr;
    double* vz = nullptr;
    double* fx = nullptr;
    double* fy = nullptr;
    double* fz = nullptr;
    double* mass = nullptr;
    double* charge = nullptr;
    int* type = nullptr;
    int* global_id = nullptr;
    ParticleFieldView* fields = nullptr;
    std::size_t field_count = 0;
};

inline ConstParticleView as_const_view(const ParticleView& source) noexcept {
    ConstParticleView view;
    view.memory_space = source.memory_space;
    view.count = source.count;
    view.x=source.x; view.y=source.y; view.z=source.z;
    view.vx=source.vx; view.vy=source.vy; view.vz=source.vz;
    view.fx=source.fx; view.fy=source.fy; view.fz=source.fz;
    view.mass=source.mass; view.charge=source.charge;
    view.type=source.type; view.global_id=source.global_id;
    view.fields=source.fields;
    view.field_count=source.field_count;
    return view;
}

// Stable wire record. Application particle storage remains non-owning;
// dsDecompX owns only communication buffers containing these records.
struct ParticleRecord {
    double x = 0.0, y = 0.0, z = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    double mass = 0.0, charge = 0.0;
    int type = 0;
    int global_id = 0;
};

} // namespace dsdecompx
