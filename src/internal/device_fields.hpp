#pragma once

#include "cuda/kernels.hpp"
#include "field.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace dsdecompx::detail {

// Wire format of one device exchange: a ParticleRecord followed by the bytes of
// every custom field, in the order the fields appear in the sending view. The
// layout is kept after the exchange so the receiving side can match fields by
// name when it unpacks.
struct DeviceWireLayout {
    struct Field {
        std::string name;
        FieldType type = FieldType::Float64;
        std::size_t components = 1;
        std::size_t bytes = 0;   // per particle
        std::size_t offset = 0;  // inside the wire record
    };
    std::size_t stride = sizeof(ParticleRecord);
    std::vector<Field> fields;
};

// Validates the custom fields of a device particle view and derives the wire
// layout. Throws std::invalid_argument for unusable fields.
DeviceWireLayout make_device_wire_layout(const ConstParticleView& particles);

// Source pointers of `particles`' fields for packing (same order as layout).
cuda::WireFields bind_source_fields(const DeviceWireLayout& layout,
                                    const ConstParticleView& particles);

// Destination pointers for unpacking, matched by name, checked for type,
// component count and room for `required_capacity` particles.
cuda::WireFields bind_destination_fields(const DeviceWireLayout& layout,
                                         const ParticleView& destination,
                                         std::size_t required_capacity);

} // namespace dsdecompx::detail
