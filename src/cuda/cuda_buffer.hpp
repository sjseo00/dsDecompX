#pragma once

#include <cstddef>
#include <cstdint>

namespace dsdecomp::cuda {

class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;
    ~DeviceBuffer();
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void reserve(std::size_t required_bytes);
    void clear() noexcept { logical_bytes_ = 0; }
    void resize(std::size_t logical_bytes);
    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return logical_bytes_; }
    std::size_t capacity() const noexcept { return capacity_bytes_; }
    std::uint64_t reallocations() const noexcept { return reallocations_; }

private:
    void release() noexcept;
    void* data_ = nullptr;
    std::size_t logical_bytes_ = 0;
    std::size_t capacity_bytes_ = 0;
    std::uint64_t reallocations_ = 0;
};

class PinnedBuffer {
public:
    PinnedBuffer() noexcept = default;
    ~PinnedBuffer();
    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void reserve(std::size_t required_bytes);
    void resize(std::size_t logical_bytes);
    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return logical_bytes_; }
    std::size_t capacity() const noexcept { return capacity_bytes_; }
    std::uint64_t reallocations() const noexcept { return reallocations_; }

private:
    void release() noexcept;
    void* data_ = nullptr;
    std::size_t logical_bytes_ = 0;
    std::size_t capacity_bytes_ = 0;
    std::uint64_t reallocations_ = 0;
};

} // namespace dsdecomp::cuda
