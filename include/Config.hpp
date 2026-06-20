// ============================================================
// FILE: include/Config.hpp
// PURPOSE: Compile-time simulation constants for the Barnes-Hut N-body quadtree.
//          All physical, numerical, and structural parameters live here.
//          Every other translation unit must reference these names — no magic
//          numbers are permitted anywhere else in the codebase.
// DEPENDENCIES: <cstdint> (for uint32_t)
// DOES NOT: Define any types, functions, or mutable state.
// ============================================================

#pragma once

#include <cstdint>

// ---- Physical constants --------------------------------------------------------

/// Newton's gravitational constant (SI units: m^3 kg^-1 s^-2).
/// Used in the acceleration formula: a = G * M / r^2 (direction implied by dx, dy).
constexpr float GRAVITATIONAL_CONSTANT = 6.674e-11f;

/// Barnes-Hut multipole-acceptance criterion.
/// A node of width s at distance d is treated as a point mass when s/d < theta,
/// equivalently s^2 < theta^2 * d^2.  Smaller theta => more accurate but slower.
constexpr float BARNES_HUT_THETA_THRESHOLD = 0.5f;

/// Gravitational softening length squared (epsilon^2, SI: m^2).
/// Prevents the force from diverging when two particles are very close.
/// The softened distance is sqrt(r^2 + epsilon^2) in all force calculations.
constexpr float SOFTENING_LENGTH_SQUARED = 1e-4f;

// ---- Simulation domain ---------------------------------------------------------

/// Half-width of the square simulation domain (SI: metres).
/// The domain spans [-SIMULATION_DOMAIN_HALF_WIDTH, +SIMULATION_DOMAIN_HALF_WIDTH]
/// in both x and y.  The root quadtree node's bounding box uses this value.
constexpr float SIMULATION_DOMAIN_HALF_WIDTH = 1000.0f;

// ---- Tree structural constants -------------------------------------------------

/// Maximum number of particles that may occupy a single leaf node before
/// it is subdivided.  Set to 1 for the classic Barnes-Hut tree where every
/// leaf holds exactly one particle.
constexpr uint32_t MAX_PARTICLES_PER_LEAF_NODE = 1u;

/// Sentinel value stored in node index fields (first_child_node_index,
/// contained_particle_index) to indicate "empty / no child / no particle".
/// Chosen to be the maximum uint32_t value so it is unambiguous and can never
/// collide with a valid zero-based array index in any realistically sized tree.
constexpr uint32_t EMPTY_NODE_SENTINEL = 0xFFFFFFFFu;

/// Pre-allocation multiplier for the flat node array.
/// host_nodes.capacity() is reserved to (particle_count * this value) before
/// the build loop begins, guaranteeing zero reallocations during insertion.
/// For a balanced quadtree with MAX_PARTICLES_PER_LEAF_NODE == 1, the worst-case
/// node count is bounded well below 4*N; a multiplier of 8 provides a safe margin
/// for pathological spatial distributions.
constexpr uint32_t TREE_NODE_CAPACITY_MULTIPLIER = 8u;

// ---- Numerical guard -----------------------------------------------------------

/// Minimum bounding box half-width for any tree node (metres).
/// When two particles occupy the same position, subdivision would otherwise loop
/// forever.  If parent_half_width < this value the insertion stops subdividing
/// and both particles are placed in the same leaf node.
constexpr float MINIMUM_BOUNDING_BOX_HALF_WIDTH = 1e-6f;
