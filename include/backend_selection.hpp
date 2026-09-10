#pragma once

#include <string_view>

namespace dsdecompx {

enum class BackendKind {
    Mpi,
    Nccl,
    Nvshmem
};

enum class BackendSelection {
    Auto,
    Mpi,
    Nccl,
    Nvshmem
};

enum class ExecutionMode {
    Host,
    Cuda
};

const char* backend_selection_name(BackendSelection selection) noexcept;
BackendSelection parse_backend_selection(std::string_view value);
bool backend_compiled(BackendSelection selection) noexcept;

} // namespace dsdecompx
