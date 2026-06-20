#include "SpatialTree.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------------------
// Compile-time structural assertions (run as part of translation)
// ---------------------------------------------------------------------------

// Redundant with the ones in SpatialTree.hpp but intentional: a test TU that
// includes only the header should also catch these at test-build time.
static_assert(sizeof(bh_spatial::CompactQuadtreeNode) == 64u,
    "TEST FAIL: CompactQuadtreeNode must be exactly 64 bytes (one cache line).");

static_assert(alignof(bh_spatial::CompactQuadtreeNode) == 64u,
    "TEST FAIL: CompactQuadtreeNode must be 64-byte aligned.");

static_assert(std::is_trivially_copyable_v<bh_spatial::CompactQuadtreeNode>,
    "TEST FAIL: CompactQuadtreeNode must be trivially copyable.");

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

namespace {

// Counts total particles recorded across all leaf nodes of a built tree.
// A correctly built tree must return particle_count here.
[[nodiscard]] uint32_t count_all_leaf_particles(
    const bh_spatial::LinearQuadtree& tree) noexcept
{
    uint32_t total = 0u;
    for (uint32_t i = 0u; i < tree.node_count; ++i) {
        const bh_spatial::CompactQuadtreeNode& node = tree.host_nodes[i];
        if (node.first_child_index == bh_spatial::INVALID_INDEX) {
            total += node.particle_count;
        }
    }
    return total;
}

// Verifies that every particle index appears exactly once across all leaf
// buckets of a built tree (partition correctness).
[[nodiscard]] bool particles_are_partitioned_correctly(
    const bh_spatial::LinearQuadtree& tree,
    const uint32_t                    expected_particle_count)
{
    std::vector<uint32_t> seen(expected_particle_count, 0u);

    for (uint32_t i = 0u; i < tree.node_count; ++i) {
        const bh_spatial::CompactQuadtreeNode& node = tree.host_nodes[i];
        if (node.first_child_index != bh_spatial::INVALID_INDEX) {
            continue; // internal node — skip
        }
        if (node.particle_count == 0u) {
            continue; // empty quadrant — skip
        }
        for (uint32_t j = 0u; j < node.particle_count; ++j) {
            const uint32_t pidx =
                tree.host_leaf_particle_indices[node.first_particle_index + j];
            if (pidx >= expected_particle_count) { return false; }
            if (seen[pidx] != 0u) { return false; } // duplicate
            seen[pidx] = 1u;
        }
    }
    for (uint32_t i = 0u; i < expected_particle_count; ++i) {
        if (seen[i] != 1u) { return false; } // missing
    }
    return true;
}

// Builds a ParticleSystem from raw vectors.
[[nodiscard]] bh_spatial::ParticleSystem make_particle_system(
    std::vector<float> xs,
    std::vector<float> ys,
    std::vector<float> masses)
{
    bh_spatial::ParticleSystem ps;
    ps.count     = xs.size();
    ps.host_pos_x = std::move(xs);
    ps.host_pos_y = std::move(ys);
    ps.host_mass  = std::move(masses);
    return ps;
}

// PASS/FAIL reporting.
void pass(const char* test_name)
{
    std::printf("PASS  %s\n", test_name);
}

void fail(const char* test_name, const char* reason)
{
    std::fprintf(stderr, "FAIL  %s — %s\n", test_name, reason);
    std::abort();
}

// ---------------------------------------------------------------------------
// Test 1 — Single particle
//
// The tree must consist of exactly one node (the root), which is a leaf
// holding the single particle.  Its total_mass and center_of_mass must
// equal the particle's mass and position exactly.
// ---------------------------------------------------------------------------
void test_single_particle()
{
    static constexpr const char* TEST_NAME = "single_particle";

    bh_spatial::ParticleSystem ps =
        make_particle_system({3.0f}, {7.0f}, {5.0f});

    bh_spatial::LinearQuadtree tree;
    bh_spatial::build_linear_quadtree(ps, tree, 1u, 48u);

    if (tree.node_count != 1u) {
        fail(TEST_NAME, "expected node_count == 1");
    }
    const bh_spatial::CompactQuadtreeNode& root = tree.host_nodes[0];
    if (root.first_child_index != bh_spatial::INVALID_INDEX) {
        fail(TEST_NAME, "root must be a leaf (first_child_index == INVALID_INDEX)");
    }
    if (root.particle_count != 1u) {
        fail(TEST_NAME, "root.particle_count must be 1");
    }
    if (root.total_mass != 5.0f) {
        fail(TEST_NAME, "root.total_mass must equal the particle mass 5.0");
    }
    if (root.center_of_mass_x != 3.0f || root.center_of_mass_y != 7.0f) {
        fail(TEST_NAME, "root.center_of_mass must equal the particle position (3,7)");
    }
    if (count_all_leaf_particles(tree) != 1u) {
        fail(TEST_NAME, "count_all_leaf_particles must return 1");
    }
    pass(TEST_NAME);
}

// ---------------------------------------------------------------------------
// Test 2 — Two particles in opposite quadrants
//
// Particles at NE and SW corners must force the root to become an internal
// node with at least two non-empty leaf children.  Total particle count
// across leaves must equal 2.  Each particle must appear exactly once.
// ---------------------------------------------------------------------------
void test_two_particles_opposite_quadrants()
{
    static constexpr const char* TEST_NAME = "two_particles_opposite_quadrants";

    // Place one particle firmly in NE (positive x, positive y) and one in SW.
    bh_spatial::ParticleSystem ps =
        make_particle_system({1.0f, -1.0f}, {1.0f, -1.0f}, {2.0f, 3.0f});

    bh_spatial::LinearQuadtree tree;
    bh_spatial::build_linear_quadtree(ps, tree, 1u, 48u);

    const bh_spatial::CompactQuadtreeNode& root = tree.host_nodes[0];
    if (root.first_child_index == bh_spatial::INVALID_INDEX) {
        fail(TEST_NAME, "root must be an internal node after splitting two particles");
    }
    if (root.particle_count != 0u) {
        fail(TEST_NAME, "internal root.particle_count must be 0");
    }
    if (count_all_leaf_particles(tree) != 2u) {
        fail(TEST_NAME, "total leaf particles must be 2");
    }
    if (!particles_are_partitioned_correctly(tree, 2u)) {
        fail(TEST_NAME, "partition correctness check failed — duplicate or missing index");
    }
    // Mass and CoM of root: (2*1 + 3*(-1))/(2+3) = -0.2 for x, same for y
    const float expected_total_mass = 5.0f;
    const float expected_com_x      = (2.0f * 1.0f + 3.0f * -1.0f) / 5.0f; // = -0.2
    const float expected_com_y      = (2.0f * 1.0f + 3.0f * -1.0f) / 5.0f; // = -0.2
    if (std::fabs(root.total_mass - expected_total_mass) > 1e-5f) {
        fail(TEST_NAME, "root.total_mass mismatch");
    }
    if (std::fabs(root.center_of_mass_x - expected_com_x) > 1e-5f) {
        fail(TEST_NAME, "root.center_of_mass_x mismatch");
    }
    if (std::fabs(root.center_of_mass_y - expected_com_y) > 1e-5f) {
        fail(TEST_NAME, "root.center_of_mass_y mismatch");
    }
    pass(TEST_NAME);
}

// ---------------------------------------------------------------------------
// Test 3 — N particles uniformly distributed: total mass conservation
//
// The root's total_mass must equal the arithmetic sum of all particle masses
// to within floating-point rounding (tolerance: 1e-2 * N).
// Every particle must appear in exactly one leaf (partition correctness).
// ---------------------------------------------------------------------------
void test_mass_conservation_uniform_distribution()
{
    static constexpr const char* TEST_NAME =
        "mass_conservation_uniform_distribution";
    static constexpr uint32_t    N          = 200u;
    static constexpr float       UNIT_MASS  = 1.0f;

    std::vector<float> xs(N), ys(N), masses(N, UNIT_MASS);
    for (uint32_t i = 0u; i < N; ++i) {
        // 10x20 uniform grid
        xs[i] = static_cast<float>(i % 10u) - 4.5f;
        ys[i] = static_cast<float>(i / 10u) - 9.5f;
    }

    bh_spatial::ParticleSystem ps;
    ps.count      = N;
    ps.host_pos_x = xs;
    ps.host_pos_y = ys;
    ps.host_mass  = masses;

    bh_spatial::LinearQuadtree tree;
    bh_spatial::build_linear_quadtree(ps, tree, 1u, 48u);

    if (count_all_leaf_particles(tree) != N) {
        fail(TEST_NAME, "count_all_leaf_particles != N");
    }
    if (!particles_are_partitioned_correctly(tree, N)) {
        fail(TEST_NAME, "partition correctness check failed");
    }

    const float expected_total_mass = static_cast<float>(N) * UNIT_MASS;
    if (std::fabs(tree.host_nodes[0].total_mass - expected_total_mass) > 1e-2f) {
        fail(TEST_NAME, "root total_mass deviates from sum of particle masses");
    }
    pass(TEST_NAME);
}

// ---------------------------------------------------------------------------
// Test 4 — All particles at one exact coincident point (leaf termination)
//
// The geometric epsilon termination rule must prevent infinite subdivision.
// All N particles must be accounted for in the tree.
// Total mass must be conserved.
// ---------------------------------------------------------------------------
void test_coincident_particles_leaf_termination()
{
    static constexpr const char* TEST_NAME =
        "coincident_particles_leaf_termination";
    static constexpr uint32_t N = 64u;

    std::vector<float> xs(N, 0.0f);
    std::vector<float> ys(N, 0.0f);
    std::vector<float> masses(N, 1.0f);

    bh_spatial::ParticleSystem ps;
    ps.count      = N;
    ps.host_pos_x = xs;
    ps.host_pos_y = ys;
    ps.host_mass  = masses;

    bh_spatial::LinearQuadtree tree;
    // Should complete without crash, infinite recursion, or throwing.
    bh_spatial::build_linear_quadtree(ps, tree, 1u, 48u);

    if (count_all_leaf_particles(tree) != N) {
        fail(TEST_NAME, "count_all_leaf_particles != N after coincident-point build");
    }
    const float expected_mass = static_cast<float>(N);
    if (std::fabs(tree.host_nodes[0].total_mass - expected_mass) > 1e-2f) {
        fail(TEST_NAME, "total_mass not conserved for coincident particles");
    }
    if (!particles_are_partitioned_correctly(tree, N)) {
        fail(TEST_NAME, "partition correctness check failed for coincident particles");
    }
    pass(TEST_NAME);
}

// ---------------------------------------------------------------------------
// Test 5 — Two-rebuilds determinism
//
// Calling build_linear_quadtree twice on the same ParticleSystem must produce
// bit-identical host_nodes contents (same node_count, same field values).
// ---------------------------------------------------------------------------
void test_rebuild_determinism()
{
    static constexpr const char* TEST_NAME = "rebuild_determinism";
    static constexpr uint32_t    N          = 50u;

    std::vector<float> xs(N), ys(N), masses(N);
    for (uint32_t i = 0u; i < N; ++i) {
        xs[i]     = static_cast<float>(i) * 0.17f - 4.0f;
        ys[i]     = static_cast<float>(i) * 0.23f - 5.5f;
        masses[i] = static_cast<float>(i + 1u);
    }

    bh_spatial::ParticleSystem ps;
    ps.count      = N;
    ps.host_pos_x = xs;
    ps.host_pos_y = ys;
    ps.host_mass  = masses;

    bh_spatial::LinearQuadtree tree_a;
    bh_spatial::LinearQuadtree tree_b;

    bh_spatial::build_linear_quadtree(ps, tree_a, 1u, 48u);
    bh_spatial::build_linear_quadtree(ps, tree_b, 1u, 48u);

    if (tree_a.node_count != tree_b.node_count) {
        fail(TEST_NAME, "node_count differs between two builds of the same input");
    }
    for (uint32_t i = 0u; i < tree_a.node_count; ++i) {
        const auto& na = tree_a.host_nodes[i];
        const auto& nb = tree_b.host_nodes[i];
        // Bit-identical comparison on every float field.
        if (na.total_mass           != nb.total_mass           ||
            na.center_of_mass_x     != nb.center_of_mass_x     ||
            na.center_of_mass_y     != nb.center_of_mass_y     ||
            na.box_center_x         != nb.box_center_x         ||
            na.box_center_y         != nb.box_center_y         ||
            na.box_half_width       != nb.box_half_width       ||
            na.first_child_index    != nb.first_child_index    ||
            na.first_particle_index != nb.first_particle_index ||
            na.particle_count       != nb.particle_count)
        {
            fail(TEST_NAME, "host_nodes content differs between two builds of identical input");
        }
    }
    pass(TEST_NAME);
}

// ---------------------------------------------------------------------------
// Test 6 — invalid_argument for bad parameters
//
// build_linear_quadtree must throw std::invalid_argument for
// max_particles_per_leaf == 0 and for max_tree_depth == 0.
// ---------------------------------------------------------------------------
void test_invalid_argument_exceptions()
{
    static constexpr const char* TEST_NAME = "invalid_argument_exceptions";

    bh_spatial::ParticleSystem ps =
        make_particle_system({0.0f}, {0.0f}, {1.0f});
    bh_spatial::LinearQuadtree tree;

    bool caught_zero_leaf = false;
    try {
        bh_spatial::build_linear_quadtree(ps, tree, 0u, 48u);
    } catch (const std::invalid_argument&) {
        caught_zero_leaf = true;
    }
    if (!caught_zero_leaf) {
        fail(TEST_NAME, "must throw std::invalid_argument for max_particles_per_leaf == 0");
    }

    bool caught_zero_depth = false;
    try {
        bh_spatial::build_linear_quadtree(ps, tree, 1u, 0u);
    } catch (const std::invalid_argument&) {
        caught_zero_depth = true;
    }
    if (!caught_zero_depth) {
        fail(TEST_NAME, "must throw std::invalid_argument for max_tree_depth == 0");
    }

    pass(TEST_NAME);
}

// ---------------------------------------------------------------------------
// Test 7 — Cache-line size static assertion (verified at compile time above;
// this function is a runtime echo for the pass log)
// ---------------------------------------------------------------------------
void test_cache_line_size_static_assertion()
{
    static constexpr const char* TEST_NAME = "cache_line_size_static_assertion";
    // Compile-time asserts at the top of this file already validate these.
    // If we reach here, they passed.
    static_assert(sizeof(bh_spatial::CompactQuadtreeNode) == 64u);
    static_assert(std::is_trivially_copyable_v<bh_spatial::CompactQuadtreeNode>);
    pass(TEST_NAME);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main()
{
    std::printf("=== bh_spatial::LinearQuadtree correctness tests ===\n\n");

    test_cache_line_size_static_assertion();
    test_single_particle();
    test_two_particles_opposite_quadrants();
    test_mass_conservation_uniform_distribution();
    test_coincident_particles_leaf_termination();
    test_rebuild_determinism();
    test_invalid_argument_exceptions();

    std::printf("\nAll tests PASSED.\n");
    return 0;
}
