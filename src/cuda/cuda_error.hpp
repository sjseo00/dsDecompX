#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace dsdecomp::cuda::detail {

inline void check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(
        std::string("dsDecomp CUDA: ") + operation + ": " +
        cudaGetErrorString(status));
}

} // namespace dsdecomp::cuda::detail
