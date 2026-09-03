#pragma once

#include "backend.hpp"
#include "grid.hpp"

#include <mpi.h>

namespace dsdecomp::mpi {

enum class CudaAwareMode {
    Auto,
    Disabled,
    Required
};

struct BackendOptions {
    CudaAwareMode cuda_aware = CudaAwareMode::Auto;
    bool allow_host_staging = true;
    bool duplicate_communicator = true;
    int required_thread_level = MPI_THREAD_FUNNELED;
};

Backend make_backend(MPI_Comm communicator,
                     const BackendOptions& options = {});

// Best-effort discovery. Failure returns available=false and never throws.
TopologyInfo discover_topology(MPI_Comm communicator) noexcept;

} // namespace dsdecomp::mpi
