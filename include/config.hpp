#pragma once

#include "backend_selection.hpp"
#include "domain.hpp"
#include "grid.hpp"

#include <cstddef>

namespace dsdecompx {

// Values for AdvancedBackendConfig::cuda_device besides an explicit ordinal.
// kCudaDeviceCurrent keeps whatever device the caller already selected with
// cudaSetDevice; kCudaDeviceByLocalRank gives each rank on a node its own GPU
// (node-local rank modulo the visible device count).
constexpr int kCudaDeviceCurrent = -1;
constexpr int kCudaDeviceByLocalRank = -2;

// Expert-only backend controls. Normal applications only select
// Config::execution and leave these automatic defaults unchanged.
struct AdvancedBackendConfig {
    BackendSelection selection = BackendSelection::Auto;
    bool read_environment = true;
    bool allow_host_staging = true;
    // GPU used by this rank in ExecutionMode::Cuda: an ordinal, or one of the
    // constants above. Applied before the NCCL/NVSHMEM backends initialize.
    int cuda_device = kCudaDeviceCurrent;
};

struct DynamicLoadBalanceConfig {
    // Rebalance automatically every N migrate() calls. Zero disables the
    // automatic path while leaving the explicit rebalance() API available.
    std::size_t interval = 1;
    // max(rank work) / average(rank work) required before moving boundaries.
    double imbalance_threshold = 1.10;
    // Fraction of the newly estimated cut position applied per rebalance.
    double relaxation = 0.5;
    // Smallest cell width as a fraction of a uniform cell width.
    double minimum_cell_fraction = 0.10;
};

struct Config {
    GlobalDomain domain;
    ProcessGrid process_grid;
    HaloConfig halo;
    GridPolicy grid_policy = GridPolicy::Manual;
    TopologyInfo topology_info;
    DynamicLoadBalanceConfig dynamic;
#if HAS_CUDA
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

} // namespace dsdecompx
