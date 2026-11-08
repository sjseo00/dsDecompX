#include <config.hpp>

#include <cassert>
#include <stdexcept>

int main() {
    using dsdecompx::BackendSelection;
    assert(dsdecompx::parse_backend_selection("AUTO") == BackendSelection::Auto);
    assert(dsdecompx::parse_backend_selection("Mpi") == BackendSelection::Mpi);
    assert(dsdecompx::parse_backend_selection("NCCL") == BackendSelection::Nccl);
    assert(dsdecompx::parse_backend_selection("nvshmem") == BackendSelection::Nvshmem);
    bool rejected = false;
    try { (void)dsdecompx::parse_backend_selection("invalid"); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);

    const dsdecompx::Config defaults;
    assert(defaults.advanced.selection == BackendSelection::Auto);
#if HAS_CUDA
    assert(defaults.execution == dsdecompx::ExecutionMode::Cuda);
#else
    assert(defaults.execution == dsdecompx::ExecutionMode::Host);
#endif
}
