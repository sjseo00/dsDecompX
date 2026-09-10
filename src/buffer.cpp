#include "internal/buffer_pool.hpp"

#include "field.hpp"

namespace dsdecompx::detail {

// Host communication buffers use geometrically-grown reusable vectors. CUDA
// device and pinned buffers are implemented by the separate execution layer.
static_assert(sizeof(ParticleRecord) > 0, "ParticleRecord must be complete");

} // namespace dsdecompx::detail
