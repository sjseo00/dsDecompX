#pragma once

#include "backend_selection.hpp"
#include "domain.hpp"
#include "grid.hpp"

#include <cstddef>

namespace dsdecomp {

// Expert-only backend controls. Normal applications only select
// Config::execution and leave these automatic defaults unchanged.
struct AdvancedBackendConfig {
    BackendSelection selection = BackendSelection::Auto;
    bool read_environment = true;
    bool allow_host_staging = true;
};

struct Config {
    GlobalDomain domain;
    ProcessGrid process_grid;
    HaloConfig halo;
    GridPolicy grid_policy = GridPolicy::Manual;
    TopologyInfo topology_info;
#if DSDECOMP_HAS_CUDA
    ExecutionMode execution = ExecutionMode::Cuda;
#else
    ExecutionMode execution = ExecutionMode::Host;
#endif
    AdvancedBackendConfig advanced;
};

struct MigrationResult {
    bool ownership_changed = false;
    std::size_t previous_owned_count = 0;
    std::size_t retained_count = 0;
    std::size_t sent_count = 0;
    std::size_t received_count = 0;
    std::size_t new_owned_count = 0;
};

} // namespace dsdecomp
