#include "cuda_buffer.hpp"
#include "cuda_error.hpp"

#include <algorithm>
#include <utility>

namespace dsdecomp::cuda {
namespace {

std::size_t grown_capacity(std::size_t current, std::size_t required) {
    std::size_t result = current ? current : 4096;
    while (result < required) {
        const std::size_t next = result + result / 2;
        if (next <= result) return required;
        result = next;
    }
    return result;
}

} // namespace

DeviceBuffer::~DeviceBuffer() { release(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      logical_bytes_(std::exchange(other.logical_bytes_, 0)),
      capacity_bytes_(std::exchange(other.capacity_bytes_, 0)),
      reallocations_(std::exchange(other.reallocations_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) return *this;
    release();
    data_ = std::exchange(other.data_, nullptr);
    logical_bytes_ = std::exchange(other.logical_bytes_, 0);
    capacity_bytes_ = std::exchange(other.capacity_bytes_, 0);
    reallocations_ = std::exchange(other.reallocations_, 0);
    return *this;
}

void DeviceBuffer::reserve(std::size_t required_bytes) {
    if (required_bytes <= capacity_bytes_) return;
    const std::size_t next_capacity = grown_capacity(capacity_bytes_, required_bytes);
    void* replacement = nullptr;
    detail::check(cudaMalloc(&replacement, next_capacity), "cudaMalloc buffer growth");
    if (data_ && logical_bytes_) {
        detail::check(cudaMemcpy(replacement, data_, logical_bytes_,
                                 cudaMemcpyDeviceToDevice),
                      "preserve device buffer during growth");
    }
    if (data_) detail::check(cudaFree(data_), "cudaFree replaced buffer");
    data_ = replacement;
    capacity_bytes_ = next_capacity;
    ++reallocations_;
}

void DeviceBuffer::resize(std::size_t logical_bytes) {
    reserve(logical_bytes);
    logical_bytes_ = logical_bytes;
}

void DeviceBuffer::release() noexcept {
    if (data_) cudaFree(data_);
    data_ = nullptr;
    logical_bytes_ = 0;
    capacity_bytes_ = 0;
}

PinnedBuffer::~PinnedBuffer() { release(); }

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      logical_bytes_(std::exchange(other.logical_bytes_, 0)),
      capacity_bytes_(std::exchange(other.capacity_bytes_, 0)),
      reallocations_(std::exchange(other.reallocations_, 0)) {}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
    if (this == &other) return *this;
    release();
    data_ = std::exchange(other.data_, nullptr);
    logical_bytes_ = std::exchange(other.logical_bytes_, 0);
    capacity_bytes_ = std::exchange(other.capacity_bytes_, 0);
    reallocations_ = std::exchange(other.reallocations_, 0);
    return *this;
}

void PinnedBuffer::reserve(std::size_t required_bytes) {
    if (required_bytes <= capacity_bytes_) return;
    const std::size_t next_capacity = grown_capacity(capacity_bytes_, required_bytes);
    void* replacement = nullptr;
    detail::check(cudaMallocHost(&replacement, next_capacity),
                  "cudaMallocHost staging growth");
    if (data_ && logical_bytes_) {
        std::copy_n(static_cast<const unsigned char*>(data_), logical_bytes_,
                    static_cast<unsigned char*>(replacement));
    }
    if (data_) detail::check(cudaFreeHost(data_), "cudaFreeHost replaced staging");
    data_ = replacement;
    capacity_bytes_ = next_capacity;
    ++reallocations_;
}

void PinnedBuffer::resize(std::size_t logical_bytes) {
    reserve(logical_bytes);
    logical_bytes_ = logical_bytes;
}

void PinnedBuffer::release() noexcept {
    if (data_) cudaFreeHost(data_);
    data_ = nullptr;
    logical_bytes_ = 0;
    capacity_bytes_ = 0;
}

} // namespace dsdecomp::cuda
