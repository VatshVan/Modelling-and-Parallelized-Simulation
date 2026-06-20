// ============================================================
// FILE: src/TreeOperations.cpp
// PURPOSE: Complete implementation of gravitational force traversal,
//          OpenMP-parallelised all-particle force computation,
//          Velocity Verlet integration, and tree diagnostic functions.
// DEPENDENCIES: TreeOperations.hpp, SpatialTree.hpp, Storage.hpp, Config.hpp
// DOES NOT: Implement tree construction, CUDA operations, or visualisation.
// ============================================================

#include "TreeOperations.hpp"
#include "Config.hpp"
#include "SpatialTree.hpp"
#include "Storage.hpp"

#include <array>
#include <cmath>       // std::sqrt
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ===========================================================================
// compute_gravitational_acceleration_for_particle
//
// Uses an explicit 64-element stack (on the thread's call stack — zero heap
// allocation) to perform a depth-first traversal of the quadtree, applying
// the Barnes-Hut multipole-acceptance criterion to decide whether a node is
// treated as a point mass or opened further.
// ===========================================================================
void compute_gravitational_acceleration_for_particle(
    const uint32_t                                query_particle_index,
    const ParticleSystem&                         system,
    const std::vector<CompactQuadtreeNode>&       tree_nodes,
    const float                                   theta_squared,
    const float                                   softening_length_squared,
    float&                                        out_acceleration_x,
    float&                                        out_acceleration_y)
{
    out_acceleration_x = 0.0f;
    out_acceleration_y = 0.0f;

    if (tree_nodes.empty()) {
        return;
    }

    const float query_position_x =
        system.host_position_x[query_particle_index];
    const float query_position_y =
        system.host_position_y[query_particle_index];

    // Fixed-size explicit stack: avoids std::stack's internal deque/heap.
    // Each internal node push adds 4 entries; with a maximum realistic tree
    // depth of ~50 levels that requires at most ~200 simultaneous stack entries.
    // 512 entries provides a safe upper bound for any physically plausible
    // particle distribution without any heap allocation.
    std::array<uint32_t, 512> traversal_node_stack {};
    int32_t stack_top_index = 0;
    traversal_node_stack[0] = 0u;  // root node index

    while (stack_top_index >= 0) {
        const uint32_t current_node_index =
            traversal_node_stack[static_cast<uint32_t>(stack_top_index)];
        --stack_top_index;

        const CompactQuadtreeNode& current_node = tree_nodes[current_node_index];

        // Skip empty nodes (no mass contribution).
        if (current_node.node_total_mass == 0.0f) {
            continue;
        }

        const float displacement_x =
            current_node.node_center_of_mass_x - query_position_x;
        const float displacement_y =
            current_node.node_center_of_mass_y - query_position_y;
        const float distance_squared =
            displacement_x * displacement_x + displacement_y * displacement_y;

        // ------------------------------------------------------------------
        // Barnes-Hut criterion (squared — no sqrt):
        //   node_width^2 < theta^2 * distance^2  => treat as point mass
        // ------------------------------------------------------------------
        const float node_width         = current_node.bounding_box_half_width * 2.0f;
        const float node_width_squared = node_width * node_width;

        const bool is_leaf =
            (current_node.first_child_node_index == EMPTY_NODE_SENTINEL);

        // Accept as point mass if: leaf node, OR criterion is satisfied.
        const bool accept_as_point_mass =
            is_leaf || (node_width_squared < theta_squared * distance_squared);

        if (accept_as_point_mass) {
            // Skip self-interaction for leaf nodes.
            if (is_leaf &&
                current_node.contained_particle_index == query_particle_index)
            {
                continue;
            }

            // Gravitational acceleration: a = G * M / (r^2 + eps^2)^(3/2) * r_hat
            //
            // inv_r3 = 1 / (softened_distance)^3
            //        = 1 / sqrt(softened_dist_sq) * 1 / softened_dist_sq
            // This form requires exactly one std::sqrt per interaction.
            const float softened_distance_squared =
                distance_squared + softening_length_squared;
            const float inv_r       = 1.0f / std::sqrt(softened_distance_squared);
            const float inv_r_cubed = inv_r / softened_distance_squared;
            const float acceleration_magnitude_factor =
                GRAVITATIONAL_CONSTANT * current_node.node_total_mass * inv_r_cubed;

            out_acceleration_x += acceleration_magnitude_factor * displacement_x;
            out_acceleration_y += acceleration_magnitude_factor * displacement_y;
        } else {
            // Open this internal node: push all four children onto the stack.
            // Ascending order push ensures NE is processed last (arbitrary;
            // traversal order does not affect the final result).
            const uint32_t first_child_index =
                current_node.first_child_node_index;

            // Each push adds 4 entries; one was already popped, so the net growth
            // per step is +3.  At tree depth D, the maximum simultaneous stack
            // occupancy is D*3+1.  With 512 entries this covers trees up to
            // depth 170, which is physically unreachable for any real particle count.
            for (uint32_t child_offset = 0u; child_offset < 4u; ++child_offset) {
                ++stack_top_index;
                traversal_node_stack[static_cast<uint32_t>(stack_top_index)] =
                    first_child_index + child_offset;
            }
        }
    }
}

// ===========================================================================
// compute_all_gravitational_forces_cpu
//
// Evaluates compute_gravitational_acceleration_for_particle for every particle
// using OpenMP guided scheduling.  Each loop iteration is fully independent:
// the query particle reads from shared (const) tree and particle position
// arrays, and writes only to its own slot in host_acceleration_x/y.
// No reduction or critical section is required.
// ===========================================================================
void compute_all_gravitational_forces_cpu(
    ParticleSystem&       system,
    const LinearQuadtree& tree,
    const float           theta_squared,
    const float           softening_length_squared)
{
    // Zero the output arrays before accumulation.
    const uint32_t particle_count = system.particle_count;
    for (uint32_t particle_index = 0u;
         particle_index < particle_count;
         ++particle_index)
    {
        system.host_acceleration_x[particle_index] = 0.0f;
        system.host_acceleration_y[particle_index] = 0.0f;
    }

    // Take a const reference to the node vector so the parallel region
    // does not need a mutable handle to the tree.
    const std::vector<CompactQuadtreeNode>& node_array = tree.host_nodes;

#ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
#endif
    // Use int64_t loop variable: OpenMP requires a signed integer type
    // for the loop control variable in a parallel for construct.
    for (int64_t signed_particle_index = 0;
         signed_particle_index < static_cast<int64_t>(particle_count);
         ++signed_particle_index)
    {
        const uint32_t particle_index =
            static_cast<uint32_t>(signed_particle_index);

        // Each thread accumulates into local variables and writes once.
        float local_acceleration_x = 0.0f;
        float local_acceleration_y = 0.0f;

        compute_gravitational_acceleration_for_particle(
            particle_index,
            system,
            node_array,
            theta_squared,
            softening_length_squared,
            local_acceleration_x,
            local_acceleration_y);

        system.host_acceleration_x[particle_index] = local_acceleration_x;
        system.host_acceleration_y[particle_index] = local_acceleration_y;
    }
}

// ===========================================================================
// integrate_velocity_verlet
//
// Velocity Verlet (symplectic Euler) step:
//   v_new = v_old + a * dt
//   x_new = x_old + v_new * dt
//
// Both loops are trivially parallel with no data dependency between particles.
// ===========================================================================
void integrate_velocity_verlet(
    ParticleSystem& system,
    const float     delta_time_seconds)
{
    const uint32_t particle_count = system.particle_count;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t signed_particle_index = 0;
         signed_particle_index < static_cast<int64_t>(particle_count);
         ++signed_particle_index)
    {
        const uint32_t particle_index =
            static_cast<uint32_t>(signed_particle_index);

        // Velocity update: v += a * dt
        system.host_velocity_x[particle_index] +=
            system.host_acceleration_x[particle_index] * delta_time_seconds;
        system.host_velocity_y[particle_index] +=
            system.host_acceleration_y[particle_index] * delta_time_seconds;

        // Position update: x += v_new * dt
        system.host_position_x[particle_index] +=
            system.host_velocity_x[particle_index] * delta_time_seconds;
        system.host_position_y[particle_index] +=
            system.host_velocity_y[particle_index] * delta_time_seconds;
    }
}

// ===========================================================================
// compute_tree_maximum_depth
//
// Iterative DFS using a stack of (node_index, current_depth) pairs.
// Returns the maximum depth encountered at any leaf.
// ===========================================================================
[[nodiscard]] uint32_t compute_tree_maximum_depth(const LinearQuadtree& tree)
{
    if (tree.active_node_count == 0u) {
        return 0u;
    }

    uint32_t maximum_depth_found = 0u;

    // Stack of (node_index, depth_of_node) pairs.
    std::vector<std::pair<uint32_t, uint32_t>> traversal_stack;
    traversal_stack.reserve(64u);
    traversal_stack.emplace_back(0u, 0u);  // root at depth 0

    while (!traversal_stack.empty()) {
        const auto [current_node_index, current_depth] = traversal_stack.back();
        traversal_stack.pop_back();

        const CompactQuadtreeNode& current_node =
            tree.host_nodes[current_node_index];

        if (current_node.first_child_node_index == EMPTY_NODE_SENTINEL) {
            // Leaf node — compare depth.
            if (current_depth > maximum_depth_found) {
                maximum_depth_found = current_depth;
            }
        } else {
            // Internal node — push children with incremented depth.
            const uint32_t first_child_index =
                current_node.first_child_node_index;
            const uint32_t child_depth = current_depth + 1u;

            for (uint32_t child_offset = 0u; child_offset < 4u; ++child_offset) {
                traversal_stack.emplace_back(
                    first_child_index + child_offset, child_depth);
            }
        }
    }

    return maximum_depth_found;
}

// ===========================================================================
// count_internal_nodes
//
// Linear scan over the active node region; a node is internal when
// first_child_node_index != EMPTY_NODE_SENTINEL.
// ===========================================================================
[[nodiscard]] uint32_t count_internal_nodes(const LinearQuadtree& tree)
{
    uint32_t internal_node_count = 0u;

    for (uint32_t node_index = 0u;
         node_index < tree.active_node_count;
         ++node_index)
    {
        if (tree.host_nodes[node_index].first_child_node_index
            != EMPTY_NODE_SENTINEL)
        {
            ++internal_node_count;
        }
    }

    return internal_node_count;
}
