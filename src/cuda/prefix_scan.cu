#include "kernels.hpp"
#include "cuda_error.hpp"

#include <cub/device/device_scan.cuh>

namespace dsdecomp::cuda {

std::size_t prefix_scan_temporary_bytes(std::size_t count) {
    std::size_t bytes = 0;
    detail::check(cub::DeviceScan::ExclusiveSum(
                      nullptr, bytes,
                      static_cast<const unsigned int*>(nullptr),
                      static_cast<unsigned int*>(nullptr), count),
                  "query CUB prefix scan storage");
    return bytes;
}

void launch_exclusive_scan(const unsigned int* counts, unsigned int* offsets,
                           std::size_t count, void* temporary,
                           std::size_t temporary_bytes, cudaStream_t stream) {
    if (!count) return;
    detail::check(cub::DeviceScan::ExclusiveSum(
                      temporary, temporary_bytes, counts, offsets, count, stream),
                  "launch CUB prefix scan");
}

} // namespace dsdecomp::cuda
