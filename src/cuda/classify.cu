#include "kernels.hpp"
#include "cuda_error.hpp"
#include "wrap.cuh"

namespace dsdecompx::cuda {
namespace {

struct DeviceDomain {
    double lower[3];
    double upper[3];
    bool periodic[3];
};
struct DeviceGrid { int value[3]; };

__global__ void classify_kernel(ConstParticleView p, DeviceDomain domain,
                                DeviceGrid grid, const double* axis_cuts,
                                DeviceGrid cut_offsets, int* destinations) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.count) return;
    const double position[3] = {p.x[i], p.y[i], p.z[i]};
    int coordinate[3];
    for (int axis = 0; axis < 3; ++axis) {
        const double value = wrap_periodic(position[axis], domain.lower[axis],
                                           domain.upper[axis],
                                           domain.periodic[axis]);
        const double* cuts = axis_cuts + cut_offsets.value[axis];
        int lo = 0;
        int hi = grid.value[axis];
        while (lo + 1 < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (value < cuts[mid]) hi = mid;
            else lo = mid;
        }
        int c = lo;
        c = c < 0 ? 0 : (c >= grid.value[axis] ? grid.value[axis] - 1 : c);
        coordinate[axis] = c;
    }
    // Cartesian rank for reorder=false. CUDA execution rejects reordered
    // topologies during initialization.
    destinations[i] = (coordinate[0] * grid.value[1] + coordinate[1]) * grid.value[2] +
                      coordinate[2];
}

} // namespace

void launch_classify_destinations(const ConstParticleView& particles,
                                  GlobalDomain domain, Vec3i process_grid,
                                  const double* axis_cuts, Vec3i cut_offsets,
                                  int* destinations, cudaStream_t stream) {
    if (!particles.count) return;
    const DeviceDomain device_domain{
        {domain.lower[0], domain.lower[1], domain.lower[2]},
        {domain.upper[0], domain.upper[1], domain.upper[2]},
        {domain.periodic[0], domain.periodic[1], domain.periodic[2]}};
    const DeviceGrid grid{{process_grid[0], process_grid[1], process_grid[2]}};
    const DeviceGrid offsets{{cut_offsets[0], cut_offsets[1], cut_offsets[2]}};
    constexpr int threads = 256;
    classify_kernel<<<static_cast<unsigned>((particles.count + threads - 1) / threads),
                      threads, 0, stream>>>(particles, device_domain, grid,
                                            axis_cuts, offsets,
                                            destinations);
    detail::check(cudaGetLastError(), "launch classify kernel");
}

} // namespace dsdecompx::cuda
