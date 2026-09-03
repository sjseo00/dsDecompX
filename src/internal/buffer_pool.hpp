#pragma once

#include "field.hpp"

#include <cstddef>
#include <algorithm>
#include <vector>

namespace dsdecomp::detail {

// Host MPI buffers use std::vector capacity as the reuse mechanism.
template<class T>
class ReusableBuffer {
public:
    void reserve(std::size_t required) {
        if (required <= storage_.capacity()) return;
        const std::size_t grown = storage_.capacity() + storage_.capacity() / 2 + 1;
        storage_.reserve(std::max(required, grown));
    }
    void resize(std::size_t size) {
        reserve(size);
        storage_.resize(size);
    }
    void clear() noexcept { storage_.clear(); }

    T* data() noexcept { return storage_.data(); }
    const T* data() const noexcept { return storage_.data(); }
    std::size_t size() const noexcept { return storage_.size(); }
    std::size_t capacity() const noexcept { return storage_.capacity(); }

    std::vector<T>& vector() noexcept { return storage_; }
    const std::vector<T>& vector() const noexcept { return storage_; }

private:
    std::vector<T> storage_;
};

template<class T>
void reserve_geometric(std::vector<T>& storage, std::size_t required) {
    if (required <= storage.capacity()) return;
    const std::size_t grown = storage.capacity() + storage.capacity() / 2 + 1;
    storage.reserve(std::max(required, grown));
}

} // namespace dsdecomp::detail
