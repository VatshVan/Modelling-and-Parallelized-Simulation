#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bh_spatial {

// ParticleSystem — particle data in strict Structure-of-Arrays layout.
//
// SoA is mandatory for:
//   * CPU auto-vectorisation: iterating one physical quantity at a time maps
//     cleanly to SIMD register widths.
//   * GPU coalesced reads: consecutive threads access consecutive elements of
//     the same array, maximising memory-bus utilisation.
//
// Device pointers are forward-compatible placeholders for a future CUDA upload
// step (cudaMalloc + cudaMemcpy). They are declared here so that CUDA
// translation units can use this struct directly. They are NEVER dereferenced
// anywhere in this spatial-indexing module.
struct ParticleSystem {
    std::size_t count{0};

    // ---- Host (CPU) SoA arrays — one element per particle ----
    std::vector<float> host_pos_x;
    std::vector<float> host_pos_y;
    std::vector<float> host_mass;

    // ---- Device (GPU) placeholder pointers — null until GPU upload ----
    float* device_pos_x{nullptr};
    float* device_pos_y{nullptr};
    float* device_mass{nullptr};
};

} // namespace bh_spatial
