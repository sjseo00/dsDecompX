#include "internal/device_fields.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace dsdecompx::detail {
namespace {

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace

DeviceWireLayout make_device_wire_layout(const ConstParticleView& particles) {
    DeviceWireLayout layout;
    if (particles.field_count == 0) return layout;
    if (!particles.fields) {
        throw std::invalid_argument("dsDecompX: custom field array is null");
    }
    if (particles.field_count > cuda::kMaxDeviceFields) {
        throw std::invalid_argument(
            "dsDecompX: at most " + std::to_string(cuda::kMaxDeviceFields) +
            " custom fields are supported on CUDA particle memory");
    }
    std::unordered_set<std::string> names;
    std::size_t offset = sizeof(ParticleRecord);
    for (std::size_t f = 0; f < particles.field_count; ++f) {
        const ParticleFieldView& field = particles.fields[f];
        if (field.name.empty() || !field.data || field.components == 0 ||
            field.capacity < particles.count) {
            throw std::invalid_argument(
                "dsDecompX: invalid or undersized custom particle field");
        }
        if (!names.insert(field.name).second) {
            throw std::invalid_argument(
                "dsDecompX: duplicate custom particle field '" + field.name + "'");
        }
        DeviceWireLayout::Field entry;
        entry.name = field.name;
        entry.type = field.type;
        entry.components = field.components;
        entry.bytes = field.record_bytes();
        entry.offset = offset;
        offset += entry.bytes;
        layout.fields.push_back(std::move(entry));
    }
    // Keep every record 8-byte aligned so the ParticleRecord header can be
    // accessed directly in the packed buffer.
    layout.stride = align_up(offset, 8);
    return layout;
}

cuda::WireFields bind_source_fields(const DeviceWireLayout& layout,
                                    const ConstParticleView& particles) {
    cuda::WireFields wire;
    wire.stride = layout.stride;
    wire.count = layout.fields.size();
    for (std::size_t f = 0; f < layout.fields.size(); ++f) {
        wire.bytes[f] = layout.fields[f].bytes;
        wire.offset[f] = layout.fields[f].offset;
        // The kernels only read a source field; the pointer is shared with the
        // unpack path, which writes through it.
        wire.data[f] = const_cast<void*>(particles.fields[f].data);
    }
    return wire;
}

cuda::WireFields bind_destination_fields(const DeviceWireLayout& layout,
                                         const ParticleView& destination,
                                         std::size_t required_capacity) {
    cuda::WireFields wire;
    wire.stride = layout.stride;
    wire.count = layout.fields.size();
    for (std::size_t f = 0; f < layout.fields.size(); ++f) {
        const DeviceWireLayout::Field& expected = layout.fields[f];
        const ParticleFieldView* begin = destination.fields;
        const ParticleFieldView* end = begin ? begin + destination.field_count : begin;
        const ParticleFieldView* found = std::find_if(
            begin, end, [&](const ParticleFieldView& field) {
                return field.name == expected.name;
            });
        if (found == end || found->type != expected.type ||
            found->components != expected.components || !found->data ||
            found->capacity < required_capacity) {
            throw std::invalid_argument(
                "dsDecompX: missing, incompatible, or undersized output field '" +
                expected.name + "'");
        }
        wire.bytes[f] = expected.bytes;
        wire.offset[f] = expected.offset;
        wire.data[f] = found->data;
    }
    return wire;
}

} // namespace dsdecompx::detail
