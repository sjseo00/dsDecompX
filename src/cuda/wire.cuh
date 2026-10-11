#pragma once

#include "kernels.hpp"

namespace dsdecompx::cuda {

// Device-side copy of WireFields. Every field is moved as 32-bit words, which
// is valid for all supported field types (4- or 8-byte elements) and keeps the
// kernels independent of the field's element type and component count.
struct DeviceWireFields {
    int count;
    std::size_t stride;
    unsigned int words[kMaxDeviceFields];
    std::size_t offset[kMaxDeviceFields];
    void* data[kMaxDeviceFields];
};

inline DeviceWireFields to_device_fields(const WireFields& fields) {
    DeviceWireFields device{};
    device.count = static_cast<int>(fields.count);
    device.stride = fields.stride;
    for (std::size_t f = 0; f < fields.count; ++f) {
        device.words[f] = static_cast<unsigned int>(fields.bytes[f] / 4);
        device.offset[f] = fields.offset[f];
        device.data[f] = fields.data[f];
    }
    return device;
}

// Copies particle `index` of every custom field into its wire record.
__device__ inline void store_wire_fields(const DeviceWireFields& fields,
                                         unsigned char* record,
                                         std::size_t index) {
    for (int f = 0; f < fields.count; ++f) {
        const unsigned int* source =
            static_cast<const unsigned int*>(fields.data[f]) + index * fields.words[f];
        unsigned int* target = reinterpret_cast<unsigned int*>(record + fields.offset[f]);
        for (unsigned int w = 0; w < fields.words[f]; ++w) target[w] = source[w];
    }
}

// Copies a wire record's custom-field bytes into particle `index`.
__device__ inline void load_wire_fields(const DeviceWireFields& fields,
                                        const unsigned char* record,
                                        std::size_t index) {
    for (int f = 0; f < fields.count; ++f) {
        const unsigned int* source =
            reinterpret_cast<const unsigned int*>(record + fields.offset[f]);
        unsigned int* target =
            static_cast<unsigned int*>(fields.data[f]) + index * fields.words[f];
        for (unsigned int w = 0; w < fields.words[f]; ++w) target[w] = source[w];
    }
}

} // namespace dsdecompx::cuda
