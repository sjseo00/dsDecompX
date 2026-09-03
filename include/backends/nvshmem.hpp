#pragma once

#include "backend.hpp"

#include <mpi.h>

namespace dsdecomp::nvshmem {

// MPI remains available as the topology/control plane. Timestep payloads use
// symmetric NVSHMEM allocations and one-sided operations.
Backend make_backend(MPI_Comm bootstrap_communicator);

} // namespace dsdecomp::nvshmem
