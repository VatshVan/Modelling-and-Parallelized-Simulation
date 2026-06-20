// ============================================================
// FILE: include/TreeOperations.hpp
// PURPOSE: Declares query and diagnostic operations that use the built
//          LinearQuadtree without modifying it.  These are the functions
//          that will eventually be parallelised for CPU (OpenMP) and GPU (CUDA).
// DEPENDENCIES: SpatialTree.hpp, Storage.hpp, <vector>, <cstdint>
// DOES NOT: Implement any function bodies, perform CUDA operations, or define
//           tree construction logic.
// ============================================================

#pragma once

#include "SpatialTree.hpp"
#include "Storage.hpp"

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// compute_gravitational_acceleration_for_particle
//
// Traverses the quadtree for a single query particle using the Barnes-Hut
// multipole-acceptance criterion and accumulates the net gravitational
// acceleration from all other particles / clusters.
//
// The traversal uses an explicit fixed-size stack (array on the thread stack,
// not heap) to avoid std::stack's internal heap allocation, which would
// otherwise serialise multi-threaded calls through the allocator.
//
// Barnes-Hut opening criterion (squared form — no sqrt in the condition):
//   node_width = bounding_box_half_width * 2
//   if (node_width^2 < theta^2 * distance^2)  =>  treat node as point mass
//
// Gravitational acceleration kernel (per interaction):
//   dx = node_com_x - query_x
//   dy = node_com_y - query_y
//   softened_dist_sq = dx^2 + dy^2 + epsilon^2
//   inv_r  = 1 / sqrt(softened_dist_sq)
//   inv_r3 = inv_r / softened_dist_sq          (= inv_r^3)
//   acc_x += G * M * inv_r3 * dx
//   acc_y += G * M * inv_r3 * dy
//
// Special cases:
//   - The query particle itself (leaf with contained_particle_index == query_particle_index)
//     is skipped to avoid self-interaction.
//   - Empty leaf nodes (contained_particle_index == EMPTY_NODE_SENTINEL,
//     first_child_node_index == EMPTY_NODE_SENTINEL) are skipped.
//
// Parameters:
//   query_particle_index      — index into system's SoA arrays for this particle.
//   system                    — particle data (positions, masses; const).
//   tree_nodes                — flat node array from the built LinearQuadtree.
//   theta_squared             — Barnes-Hut theta^2 (= BARNES_HUT_THETA_THRESHOLD^2).
//   softening_length_squared  — epsilon^2 for force softening.
//   out_acceleration_x        — accumulated x-acceleration output (written, not read).
//   out_acceleration_y        — accumulated y-acceleration output (written, not read).
// ---------------------------------------------------------------------------
void compute_gravitational_acceleration_for_particle(
    uint32_t                                query_particle_index,
    const ParticleSystem&                   system,
    const std::vector<CompactQuadtreeNode>& tree_nodes,
    float                                   theta_squared,
    float                                   softening_length_squared,
    float&                                  out_acceleration_x,
    float&                                  out_acceleration_y
);

// ---------------------------------------------------------------------------
// compute_all_gravitational_forces_cpu
//
// Evaluates compute_gravitational_acceleration_for_particle for every particle
// in the system using OpenMP task-parallel scheduling.
//
// Parallelisation strategy:
//   - Zeroes host_acceleration_x/y before the parallel region.
//   - Uses #pragma omp parallel for schedule(guided) so that threads with
//     heavier traversal workloads (particles near dense regions) do not
//     become bottlenecks.
//   - Each iteration is fully independent; no reduction or critical section
//     is needed because each thread writes only to its own particle's slot.
//
// If OpenMP is not available (compiled without -fopenmp / /openmp), the loop
// runs sequentially via the #ifdef _OPENMP guard.
//
// Parameters:
//   system                   — particle data; acceleration arrays are overwritten.
//   tree                     — fully built LinearQuadtree (not modified).
//   theta_squared            — Barnes-Hut opening criterion (theta^2).
//   softening_length_squared — epsilon^2 for force softening.
// ---------------------------------------------------------------------------
void compute_all_gravitational_forces_cpu(
    ParticleSystem&       system,
    const LinearQuadtree& tree,
    float                 theta_squared,
    float                 softening_length_squared
);

// ---------------------------------------------------------------------------
// integrate_velocity_verlet
//
// Advances each particle's velocity and position by one timestep using the
// Velocity Verlet (leapfrog) integration scheme:
//
//   v_new = v_old + a * dt
//   x_new = x_old + v_new * dt
//
// Note: This is the simplified (Euler-corrected) form.  A full second-order
// Verlet would require the previous acceleration, which is not stored in this
// deliverable.  The scheme here is explicit and first-order accurate in
// velocity, second-order in position given constant acceleration — sufficient
// for benchmarking purposes.
//
// Parallelised with #pragma omp parallel for (trivially data-parallel; no
// inter-particle dependencies).
//
// Parameters:
//   system              — particle data; velocity and position arrays updated.
//   delta_time_seconds  — integration timestep (seconds).
// ---------------------------------------------------------------------------
void integrate_velocity_verlet(
    ParticleSystem& system,
    float           delta_time_seconds
);

// ---------------------------------------------------------------------------
// compute_tree_maximum_depth
//
// Traverses the tree depth-first and returns the depth of the deepest leaf.
// The root has depth 0.  Used for diagnostics and benchmark reports.
//
// Parameters:
//   tree — fully built LinearQuadtree (not modified).
//
// Returns: maximum leaf depth (0 for a single-node tree).
// ---------------------------------------------------------------------------
[[nodiscard]] uint32_t compute_tree_maximum_depth(const LinearQuadtree& tree);

// ---------------------------------------------------------------------------
// count_internal_nodes
//
// Returns the number of internal (non-leaf) nodes in the tree.
// A node is internal if first_child_node_index != EMPTY_NODE_SENTINEL.
//
// Parameters:
//   tree — fully built LinearQuadtree (not modified).
//
// Returns: count of internal nodes.
// ---------------------------------------------------------------------------
[[nodiscard]] uint32_t count_internal_nodes(const LinearQuadtree& tree);
