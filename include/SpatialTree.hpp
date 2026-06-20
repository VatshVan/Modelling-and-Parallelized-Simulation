#pragma once

#include "Storage.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace bh_spatial {

// Sentinel value for absent/null array indices throughout the tree.
// Chosen as the maximum uint32_t value so it can never collide with a valid
// zero-based array index in any realistically-sized tree.
inline constexpr uint32_t INVALID_INDEX = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// CompactQuadtreeNode
//
// Occupies EXACTLY one 64-byte cache line (verified by static_assert below).
// All inter-node relationships are expressed as uint32_t indices into the flat
// LinearQuadtree::host_nodes array — zero raw pointers inside the struct.
//
// Field semantics
// ---------------
//   center_of_mass_x/y   Mass-weighted centre of mass for this subtree.
//                         Set during Phase D (bottom-up mass reduction);
//                         meaningless before Phase D completes.
//
//   total_mass           Sum of all particle masses in this subtree.
//
//   box_center_x/y       Geometric centre of the node's square bounding region.
//                         Set during construction; never changes afterwards.
//
//   box_half_width       Half the side length of the square region.
//                         Children have half_width = parent.half_width * 0.5.
//
//   first_child_index    Index of the FIRST of four contiguous child nodes in
//                         host_nodes[first_child_index .. first_child_index+3].
//                         INVALID_INDEX for a leaf or empty-quadrant node.
//
//   first_particle_index  Index into LinearQuadtree::host_leaf_particle_indices
//                          for the first particle held at this leaf.
//                          INVALID_INDEX for internal nodes and empty quadrants.
//
//   particle_count        Number of particles held at this leaf.
//                          0 for internal nodes and empty-quadrant nodes.
//
//   reserved_padding      Explicit padding to reach exactly 64 bytes.
//                          Required by static_assert; also ensures the struct
//                          is trivially copyable (no vtable, no heap members).
//
// Layout (64 bytes total):
//   [0..3]   center_of_mass_x     float       4
//   [4..7]   center_of_mass_y     float       4
//   [8..11]  total_mass           float       4
//   [12..15] box_center_x         float       4
//   [16..19] box_center_y         float       4
//   [20..23] box_half_width       float       4
//   [24..27] first_child_index    uint32_t    4
//   [28..31] first_particle_index uint32_t    4
//   [32..35] particle_count       uint32_t    4
//   [36..63] reserved_padding     std::byte  28
// ---------------------------------------------------------------------------
struct alignas(64) CompactQuadtreeNode {
    float    center_of_mass_x;
    float    center_of_mass_y;
    float    total_mass;
    float    box_center_x;
    float    box_center_y;
    float    box_half_width;
    uint32_t first_child_index;
    uint32_t first_particle_index;
    uint32_t particle_count;
    std::byte reserved_padding[28];
};

static_assert(sizeof(CompactQuadtreeNode) == 64u,
    "CompactQuadtreeNode must occupy exactly one 64-byte cache line; "
    "adjust reserved_padding if field sizes change.");

static_assert(std::is_trivially_copyable_v<CompactQuadtreeNode>,
    "CompactQuadtreeNode must be trivially copyable for future cudaMemcpy "
    "GPU transfer; remove any non-trivial constructors, destructors, or "
    "heap-allocated members.");

// ---------------------------------------------------------------------------
// LinearQuadtree
//
// Flat, pointer-free quadtree container. All nodes live in one contiguous
// host_nodes vector; all inter-node links use uint32_t indices.
//
// After build_linear_quadtree completes:
//   host_nodes[0 .. node_count-1]         — all allocated nodes in
//                                           construction order (root is [0]).
//   host_leaf_particle_indices[...]        — particle indices grouped by leaf;
//                                           leaf node L's particles are at
//                                           [L.first_particle_index ..
//                                            L.first_particle_index + L.particle_count).
//   node_count                             — number of valid nodes.
//
// Device pointers are null placeholders for a future CUDA upload step.
// ---------------------------------------------------------------------------
struct LinearQuadtree {
    std::vector<CompactQuadtreeNode> host_nodes;
    std::vector<uint32_t>            host_leaf_particle_indices;
    uint32_t                         node_count{0u};

    CompactQuadtreeNode* device_nodes{nullptr};
    uint32_t*            device_leaf_particle_indices{nullptr};
};

// ---------------------------------------------------------------------------
// build_linear_quadtree
//
// Constructs the linearised quadtree from particles in four explicit phases:
//
//   Phase A  Compute axis-aligned bounding box; derive a square root cell.
//   Phase B  Pre-allocate all storage (the ONLY heap-allocation phase).
//            Reserves host_nodes to 8*N+64, host_leaf_particle_indices to N.
//            No push_back / resize / new / malloc occurs after this phase.
//   Phase C  Iterative top-down subdivision using an explicit pre-reserved
//            work stack. Particles are partitioned in-place via counting sort.
//            Subdivision stops when any of:
//              • node.particle_count <= max_particles_per_leaf
//              • node depth >= max_tree_depth
//              • node.box_half_width <= root_half_width * 2^{-20}  (epsilon)
//            Nodes that hit the latter two limits become leaves holding ALL
//            their particles — this is the mechanism that gracefully handles
//            near-coincident or exactly-coincident particle coordinates.
//   Phase D  Bottom-up mass reduction. host_nodes is iterated in reverse
//            index order; because children always receive higher indices than
//            their parent during Phase C, all children are processed before
//            their parent without any separate tree walk.
//
// Sets tree.node_count to the total number of allocated nodes on return.
//
// Throws:
//   std::invalid_argument  if max_particles_per_leaf == 0 or max_tree_depth == 0.
//   std::runtime_error     if the 8*N+64 node capacity is exceeded during
//                          Phase C subdivision (should never occur for any
//                          physically realisable particle distribution).
// ---------------------------------------------------------------------------
void build_linear_quadtree(
    const ParticleSystem& particles,
    LinearQuadtree&       tree,
    uint32_t              max_particles_per_leaf = 1u,
    uint32_t              max_tree_depth         = 48u);

} // namespace bh_spatial
