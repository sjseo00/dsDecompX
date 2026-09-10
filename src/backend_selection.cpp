#include "backend_selection.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace dsdecompx {

const char* backend_selection_name(BackendSelection selection) noexcept {
    switch (selection) {
    case BackendSelection::Auto: return "auto";
    case BackendSelection::Mpi: return "mpi";
    case BackendSelection::Nccl: return "nccl";
    case BackendSelection::Nvshmem: return "nvshmem";
    }
    return "unknown";
}

BackendSelection parse_backend_selection(std::string_view value) {
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (normalized == "auto") return BackendSelection::Auto;
    if (normalized == "mpi") return BackendSelection::Mpi;
    if (normalized == "nccl") return BackendSelection::Nccl;
    if (normalized == "nvshmem") return BackendSelection::Nvshmem;
    throw std::invalid_argument(
        "dsDecompX: BACKEND must be auto, mpi, nccl, or nvshmem");
}

bool backend_compiled(BackendSelection selection) noexcept {
    switch (selection) {
    case BackendSelection::Auto:
    case BackendSelection::Mpi:
        return true;
    case BackendSelection::Nccl:
#if HAS_NCCL
        return true;
#else
        return false;
#endif
    case BackendSelection::Nvshmem:
#if HAS_NVSHMEM
        return true;
#else
        return false;
#endif
    }
    return false;
}

} // namespace dsdecompx
