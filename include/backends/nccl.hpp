#pragma once

#include "backend.hpp"

#include <mpi.h>

namespace dsdecomp::nccl {

// MPI is the control plane used to broadcast ncclUniqueId. Halo and migration
// payloads use NCCL after initialization.
Backend make_backend(MPI_Comm bootstrap_communicator);

} // namespace dsdecomp::nccl
