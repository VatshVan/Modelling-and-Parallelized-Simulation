#include "SpatialTree.hpp"

#include <algorithm>  // std::max
#include <array>
#include <cmath>      // std::ldexp
#include <cstdint>
#include <limits>     // std::numeric_limits (not used after Phase A refactor, kept for clarity)
#include <stdexcept>

namespace bh_spatial {

// ---------------------------------------------------------------------------
// File-scope helpers (anonymous namespace — not visible outside this TU)
// ---------------------------------------------------------------------------
namespace {

// compute_quadrant_index
//
// Returns which of the four quadrants a point falls into relative to a node's
// geometric centre, using a 2-bit encoding:
//
//   bit 0 — east  half: particle_x >= node_center_x
//   bit 1 — north half: particle_y >= node_center_y
//
//   0 = SW  (x <  cx, y <  cy)
//   1 = SE  (x >= cx, y <  cy)
//   2 = NW  (x <  cx, y >= cy)
//   3 = NE  (x >= cx, y >= cy)
//
// Particles on the boundary (x == cx or y == cy) are deterministically
// assigned to the east / north half respectively — this prevents any particle
// from ever being "between" two quadrants and ensures identical inputs always
// produce identical trees (determinism requirement).
[[nodiscard]] inline uint32_t compute_quadrant_index(
    const float particle_x,
    const float particle_y,
    const float node_center_x,
    const float node_center_y) noexcept
{
    const uint32_t east_bit  = (particle_x >= node_center_x) ? 1u : 0u;
    const uint32_t north_bit = (particle_y >= node_center_y) ? 2u : 0u;
    return east_bit | north_bit;
}

// compute_child_box_geometry
//
// Computes the geometric centre and half-width of child quadrant `quadrant`
// given the parent's geometry. The child's square region has half the side
// length of the parent and is centred at the midpoint of the corresponding
// quadrant.
//
// child_quadrant encoding is identical to compute_quadrant_index (0=SW,
// 1=SE, 2=NW, 3=NE).
inline void compute_child_box_geometry(
    const float    parent_center_x,
    const float    parent_center_y,
    const float    parent_half_width,
    const uint32_t child_quadrant,
    float&         out_center_x,
    float&         out_center_y,
    float&         out_half_width) noexcept
{
    out_half_width = parent_half_width * 0.5f;
    // Offset from parent centre: positive for east/north, negative for west/south.
    const float east_offset  = (child_quadrant & 1u) ?  out_half_width : -out_half_width;
    const float north_offset = (child_quadrant & 2u) ?  out_half_width : -out_half_width;
    out_center_x = parent_center_x + east_offset;
    out_center_y = parent_center_y + north_offset;
}

// initialize_node_defaults
//
// Resets a node's fields to known-good initial values.  Must be called for
// every node immediately after it is allocated (before any other code reads
// its fields), because vector::resize value-initialises POD members to zero,
// meaning first_child_index and first_particle_index would be 0 — not
// INVALID_INDEX — without this call.
inline void initialize_node_defaults(
    CompactQuadtreeNode& node,
    const float          box_center_x,
    const float          box_center_y,
    const float          box_half_width) noexcept
{
    node.center_of_mass_x     = 0.0f;
    node.center_of_mass_y     = 0.0f;
    node.total_mass           = 0.0f;
    node.box_center_x         = box_center_x;
    node.box_center_y         = box_center_y;
    node.box_half_width       = box_half_width;
    node.first_child_index    = INVALID_INDEX;
    node.first_particle_index = INVALID_INDEX;
    node.particle_count       = 0u;
    // reserved_padding is zero-filled by vector::resize; no explicit write needed.
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// build_linear_quadtree — full implementation
// ---------------------------------------------------------------------------
void build_linear_quadtree(
    const ParticleSystem& particles,
    LinearQuadtree&       tree,
    const uint32_t        max_particles_per_leaf,
    const uint32_t        max_tree_depth)
{
    // =========================================================================
    // Parameter validation — the ONLY block that throws std::invalid_argument.
    // =========================================================================
    if (max_particles_per_leaf == 0u) {
        throw std::invalid_argument(
            "bh_spatial::build_linear_quadtree: "
            "max_particles_per_leaf must be >= 1");
    }
    if (max_tree_depth == 0u) {
        throw std::invalid_argument(
            "bh_spatial::build_linear_quadtree: "
            "max_tree_depth must be >= 1");
    }

    const uint32_t particle_count = static_cast<uint32_t>(particles.count);

    // Trivial empty-system: produce an empty tree rather than crashing.
    if (particle_count == 0u) {
        tree.host_nodes.clear();
        tree.host_leaf_particle_indices.clear();
        tree.node_count = 0u;
        return;
    }

    // =========================================================================
    // Phase A — Compute axis-aligned bounding box (single linear pass)
    // =========================================================================

    // Bootstrap from the first element so we avoid numeric_limits::max/lowest
    // which could interact oddly with -ffast-math NaN propagation.
    float global_min_x = particles.host_pos_x[0];
    float global_max_x = particles.host_pos_x[0];
    float global_min_y = particles.host_pos_y[0];
    float global_max_y = particles.host_pos_y[0];

    for (uint32_t particle_index = 1u;
         particle_index < particle_count;
         ++particle_index)
    {
        const float px = particles.host_pos_x[particle_index];
        const float py = particles.host_pos_y[particle_index];
        if (px < global_min_x) { global_min_x = px; }
        if (px > global_max_x) { global_max_x = px; }
        if (py < global_min_y) { global_min_y = py; }
        if (py > global_max_y) { global_max_y = py; }
    }

    // Derive a SQUARE root bounding box centred on the coordinate midpoint.
    // We take the larger of the two axis spans so that both axes are identical
    // — quadtree subdivision assumes square children, so a non-square root
    // cell would break the geometry.
    //
    // A small absolute epsilon (1e-4 metres) is added so that particles
    // sitting exactly on the computed boundary are strictly inside the root
    // box, preventing them from being mis-classified to a neighbouring
    // quadrant on the first split.
    const float box_midpoint_x       = 0.5f * (global_min_x + global_max_x);
    const float box_midpoint_y       = 0.5f * (global_min_y + global_max_y);
    const float larger_span          = std::max(global_max_x - global_min_x,
                                                global_max_y - global_min_y);
    const float root_box_half_width  = 0.5f * larger_span + 1e-4f;

    // Geometric epsilon: subdivision is forbidden once a node's half-width
    // falls at or below this threshold.  Defined as root_half_width * 2^{-20}
    // so that ~20 binary subdivisions are possible before floating-point
    // arithmetic can no longer distinguish child box boundaries.
    const float geometric_epsilon =
        root_box_half_width * std::ldexp(1.0f, -20);

    // =========================================================================
    // Phase B — Pre-allocate ALL storage
    //
    // After the resize/clear calls in this section, NO push_back, resize,
    // new, or malloc is permitted anywhere in Phases C or D.
    // =========================================================================

    // Node capacity: 8*N+64 is an empirically safe upper bound for the total
    // number of quadtree nodes produced by any physically plausible particle
    // distribution, including heavily clustered inputs.
    const uint32_t node_capacity = 8u * particle_count + 64u;

    // Resize host_nodes to full capacity for O(1) direct index access in
    // Phase C.  vector::resize value-initialises all fields to zero, which is
    // a safe default — initialize_node_defaults must still be called for each
    // node before any code reads its sentinel fields (first_child_index etc.).
    tree.host_nodes.clear();
    tree.host_nodes.resize(node_capacity);

    // host_leaf_particle_indices serves DUAL PURPOSE:
    //   During Phase C — scratch array holding the working permutation of
    //                    particle indices; partitioned in-place as nodes split.
    //   After Phase D   — final leaf-bucket array; leaf node L's particles are
    //                    at indices [L.first_particle_index,
    //                                L.first_particle_index + L.particle_count).
    tree.host_leaf_particle_indices.clear();
    tree.host_leaf_particle_indices.resize(particle_count);

    // Initialise scratch as the identity permutation [0, 1, ..., N-1].
    for (uint32_t i = 0u; i < particle_count; ++i) {
        tree.host_leaf_particle_indices[i] = i;
    }

    // NodeBuildState tracks the particle range and tree depth for each node
    // during Phase C.  Declared locally to avoid leaking build internals.
    struct NodeBuildState {
        uint32_t particle_range_start;   // offset into host_leaf_particle_indices
        uint32_t particle_range_count;   // number of particles in this node's range
        uint32_t depth;                  // depth of this node (root = 0)
    };

    // Parallel to host_nodes; one entry per allocated node.
    std::vector<NodeBuildState> node_build_states;
    node_build_states.resize(node_capacity);

    // Pre-reserved work stack: node indices awaiting subdivision evaluation.
    // Accessed via a manual top-pointer; no push_back or pop_back are used,
    // satisfying the zero-dynamic-allocation requirement for Phase C.
    std::vector<uint32_t> work_stack;
    work_stack.resize(node_capacity);

    // Temporary buffer for the counting-sort scatter step in particle partitioning.
    std::vector<uint32_t> partition_scratch;
    partition_scratch.resize(particle_count);

    // ---- Initialise root node (index 0) ----
    uint32_t active_node_count = 1u;

    initialize_node_defaults(
        tree.host_nodes[0],
        box_midpoint_x,
        box_midpoint_y,
        root_box_half_width);

    // Root begins as a leaf owning the entire particle set.
    tree.host_nodes[0].particle_count       = particle_count;
    tree.host_nodes[0].first_particle_index = 0u;

    node_build_states[0] = {0u, particle_count, 0u};

    // Push root onto work stack.
    uint32_t work_stack_top = 0u;
    work_stack[work_stack_top++] = 0u;

    // =========================================================================
    // Phase C — Iterative top-down subdivision
    //
    // INVARIANT: after Phase C completes, every child node has a strictly
    // higher index than its parent.  Phase D exploits this to perform the
    // bottom-up mass reduction by iterating host_nodes in reverse index
    // order — no separate post-order traversal is needed.
    // =========================================================================

    // Cache raw data pointers to avoid repeated vector::operator[] overhead.
    const float* const pos_x_data = particles.host_pos_x.data();
    const float* const pos_y_data = particles.host_pos_y.data();

    while (work_stack_top > 0u) {
        const uint32_t current_node_index = work_stack[--work_stack_top];

        CompactQuadtreeNode& current_node    = tree.host_nodes[current_node_index];
        const NodeBuildState& current_state  = node_build_states[current_node_index];

        const uint32_t range_start   = current_state.particle_range_start;
        const uint32_t range_count   = current_state.particle_range_count;
        const uint32_t current_depth = current_state.depth;
        const uint32_t range_end     = range_start + range_count;

        // ---- Termination conditions ----
        // If ANY of these holds, the node becomes a leaf with all its particles.
        // This is the mechanism that gracefully handles near-coincident and
        // exactly-coincident particle coordinates: they accumulate in one leaf
        // without causing infinite recursion.
        const bool within_leaf_capacity  = (range_count <= max_particles_per_leaf);
        const bool max_depth_reached     = (current_depth >= max_tree_depth);
        const bool box_too_small         =
            (current_node.box_half_width <= geometric_epsilon);

        if (within_leaf_capacity || max_depth_reached || box_too_small) {
            // Node stays a leaf.  particle_count and first_particle_index were
            // set during node initialisation — nothing more to do here.
            continue;
        }

        // ---- Partition particles into 4 quadrants (counting sort) ----
        // Pass 1: count how many particles fall into each quadrant.
        const float node_center_x = current_node.box_center_x;
        const float node_center_y = current_node.box_center_y;

        std::array<uint32_t, 4u> quadrant_particle_count = {0u, 0u, 0u, 0u};

        for (uint32_t i = range_start; i < range_end; ++i) {
            const uint32_t pidx     = tree.host_leaf_particle_indices[i];
            const uint32_t quadrant =
                compute_quadrant_index(
                    pos_x_data[pidx], pos_y_data[pidx],
                    node_center_x, node_center_y);
            ++quadrant_particle_count[quadrant];
        }

        // Prefix-sum: quadrant_scatter_offset[q] is the starting position in
        // partition_scratch[0..range_count) where quadrant q's particles land.
        std::array<uint32_t, 4u> quadrant_scatter_offset;
        quadrant_scatter_offset[0] = 0u;
        quadrant_scatter_offset[1] = quadrant_particle_count[0];
        quadrant_scatter_offset[2] = quadrant_scatter_offset[1]
                                     + quadrant_particle_count[1];
        quadrant_scatter_offset[3] = quadrant_scatter_offset[2]
                                     + quadrant_particle_count[2];

        // Pass 2: scatter particle indices into partition_scratch.
        std::array<uint32_t, 4u> write_cursor = quadrant_scatter_offset;

        for (uint32_t i = range_start; i < range_end; ++i) {
            const uint32_t pidx     = tree.host_leaf_particle_indices[i];
            const uint32_t quadrant =
                compute_quadrant_index(
                    pos_x_data[pidx], pos_y_data[pidx],
                    node_center_x, node_center_y);
            partition_scratch[write_cursor[quadrant]++] = pidx;
        }

        // Pass 3: copy rearranged indices back into host_leaf_particle_indices.
        for (uint32_t i = 0u; i < range_count; ++i) {
            tree.host_leaf_particle_indices[range_start + i] = partition_scratch[i];
        }

        // ---- Allocate four child nodes ----
        // This is the only place active_node_count advances; the capacity guard
        // produces a std::runtime_error rather than allowing a silent buffer
        // overrun or an unexpected reallocation.
        if (active_node_count + 4u > node_capacity) {
            throw std::runtime_error(
                "bh_spatial::build_linear_quadtree: node capacity exceeded. "
                "The 8*N+64 reservation was insufficient for this particle "
                "distribution. This should not occur for any physically "
                "realisable input.");
        }

        const uint32_t first_child_node_index = active_node_count;
        active_node_count += 4u;

        // ---- Promote current node from leaf to internal ----
        current_node.first_child_index    = first_child_node_index;
        current_node.first_particle_index = INVALID_INDEX;
        current_node.particle_count       = 0u;

        const uint32_t child_depth = current_depth + 1u;

        // ---- Initialise and (conditionally) enqueue each child ----
        for (uint32_t quadrant = 0u; quadrant < 4u; ++quadrant) {
            const uint32_t child_node_index = first_child_node_index + quadrant;

            float child_center_x  = 0.0f;
            float child_center_y  = 0.0f;
            float child_half_width = 0.0f;
            compute_child_box_geometry(
                node_center_x, node_center_y,
                current_node.box_half_width,
                quadrant,
                child_center_x, child_center_y, child_half_width);

            initialize_node_defaults(
                tree.host_nodes[child_node_index],
                child_center_x, child_center_y, child_half_width);

            const uint32_t child_range_start =
                range_start + quadrant_scatter_offset[quadrant];
            const uint32_t child_range_count =
                quadrant_particle_count[quadrant];

            // Set particle-range fields on the child node.
            tree.host_nodes[child_node_index].particle_count = child_range_count;
            tree.host_nodes[child_node_index].first_particle_index =
                (child_range_count > 0u) ? child_range_start : INVALID_INDEX;

            node_build_states[child_node_index] = {
                child_range_start,
                child_range_count,
                child_depth
            };

            // Only push children that still need potential further subdivision.
            // Empty quadrant nodes (count == 0) and already-leaf-sized nodes
            // are left on the stack-not-pushed path; they are correct leaves
            // from the moment of initialisation.
            if (child_range_count > max_particles_per_leaf) {
                if (work_stack_top >= node_capacity) {
                    // Unreachable under normal conditions; included as a guard.
                    throw std::runtime_error(
                        "bh_spatial::build_linear_quadtree: work stack "
                        "capacity exceeded.");
                }
                work_stack[work_stack_top++] = child_node_index;
            }
        }
    }

    // =========================================================================
    // Phase D — Bottom-up mass reduction
    //
    // Iterate host_nodes in REVERSE INDEX ORDER.
    //
    // Correctness argument: by construction in Phase C, every child node
    // receives an index strictly greater than its parent's index.  Therefore,
    // when we visit node i in reverse order, all nodes j > i (including all
    // of i's children) have already been processed and their total_mass /
    // center_of_mass fields are finalised.  No separate post-order traversal
    // or visited-flag is needed.
    // =========================================================================
    for (int64_t node_idx = static_cast<int64_t>(active_node_count) - 1;
         node_idx >= 0;
         --node_idx)
    {
        CompactQuadtreeNode& node =
            tree.host_nodes[static_cast<uint32_t>(node_idx)];

        if (node.first_child_index == INVALID_INDEX) {
            // ---- Leaf or empty-quadrant node ----
            if (node.particle_count == 0u) {
                // Empty quadrant: zero mass; place CoM at box centre as a
                // well-defined fallback (unused by Barnes-Hut traversal).
                node.total_mass        = 0.0f;
                node.center_of_mass_x  = node.box_center_x;
                node.center_of_mass_y  = node.box_center_y;
            } else {
                // Populated leaf: compute total mass and mass-weighted CoM
                // directly from the particle data referenced by this leaf.
                float accumulated_mass   = 0.0f;
                float accumulated_mass_x = 0.0f;
                float accumulated_mass_y = 0.0f;

                const uint32_t leaf_range_start = node.first_particle_index;
                const uint32_t leaf_range_count = node.particle_count;

                for (uint32_t i = 0u; i < leaf_range_count; ++i) {
                    const uint32_t particle_index =
                        tree.host_leaf_particle_indices[leaf_range_start + i];
                    const float particle_mass = particles.host_mass[particle_index];
                    accumulated_mass   += particle_mass;
                    accumulated_mass_x +=
                        particle_mass * particles.host_pos_x[particle_index];
                    accumulated_mass_y +=
                        particle_mass * particles.host_pos_y[particle_index];
                }

                node.total_mass = accumulated_mass;
                if (accumulated_mass > 0.0f) {
                    const float inv_total_mass = 1.0f / accumulated_mass;
                    node.center_of_mass_x = accumulated_mass_x * inv_total_mass;
                    node.center_of_mass_y = accumulated_mass_y * inv_total_mass;
                } else {
                    // All particles have zero mass — place CoM at box centre.
                    node.center_of_mass_x = node.box_center_x;
                    node.center_of_mass_y = node.box_center_y;
                }
            }
        } else {
            // ---- Internal node — accumulate from four children ----
            // Per spec: skip child slots where particle_count == 0 AND
            // first_child_index == INVALID_INDEX (empty, unoccupied quadrants).
            const uint32_t first_child_index = node.first_child_index;

            float accumulated_mass   = 0.0f;
            float accumulated_mass_x = 0.0f;
            float accumulated_mass_y = 0.0f;

            for (uint32_t child_offset = 0u; child_offset < 4u; ++child_offset) {
                const CompactQuadtreeNode& child_node =
                    tree.host_nodes[first_child_index + child_offset];

                if (child_node.particle_count == 0u &&
                    child_node.first_child_index == INVALID_INDEX)
                {
                    continue; // empty quadrant — no mass contribution
                }

                accumulated_mass   += child_node.total_mass;
                accumulated_mass_x +=
                    child_node.total_mass * child_node.center_of_mass_x;
                accumulated_mass_y +=
                    child_node.total_mass * child_node.center_of_mass_y;
            }

            node.total_mass = accumulated_mass;
            if (accumulated_mass > 0.0f) {
                const float inv_total_mass = 1.0f / accumulated_mass;
                node.center_of_mass_x = accumulated_mass_x * inv_total_mass;
                node.center_of_mass_y = accumulated_mass_y * inv_total_mass;
            } else {
                node.center_of_mass_x = node.box_center_x;
                node.center_of_mass_y = node.box_center_y;
            }
        }
    }

    tree.node_count = active_node_count;
}

} // namespace bh_spatial
