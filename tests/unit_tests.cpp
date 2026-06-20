// ============================================================
// FILE: tests/unit_tests.cpp
// PURPOSE: Single-threaded correctness validation for the quadtree library.
//          Tests struct layout, particle count preservation, mass conservation,
//          centre-of-mass accuracy, force symmetry, and rebuild stability.
//          No OpenMP parallelism is used; all tests are sequential.
// DEPENDENCIES: SpatialTree.hpp, Storage.hpp, TreeOperations.hpp, Config.hpp
// DOES NOT: Benchmark performance, test GPU paths, or validate integration
//           accuracy over many timesteps.
// ============================================================

#include "Config.hpp"
#include "SpatialTree.hpp"
#include "Storage.hpp"
#include "TreeOperations.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>    // std::abort
#include <cstring>    // std::memcpy
#include <iostream>
#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------
// test_report_pass / test_abort
//
// Helpers to print pass messages or abort with a descriptive failure message.
// ---------------------------------------------------------------------------
static void test_report_pass(const std::string& test_name)
{
    std::cout << "PASS: " << test_name << '\n';
}

static void test_abort_with_message(
    const std::string& test_name,
    const std::string& failure_reason)
{
    std::cerr << "FAIL: " << test_name << "\n  Reason: " << failure_reason << '\n';
    std::abort();
}

// ---------------------------------------------------------------------------
// approximately_equal
//
// Returns true if |a - b| <= tolerance.
// ---------------------------------------------------------------------------
static bool approximately_equal(
    const float value_a,
    const float value_b,
    const float tolerance)
{
    const float absolute_difference =
        (value_a >= value_b) ? (value_a - value_b) : (value_b - value_a);
    return absolute_difference <= tolerance;
}

// ===========================================================================
// Individual test functions
// ===========================================================================

// ---------------------------------------------------------------------------
// test_node_struct_size
//
// Verifies at runtime (in addition to the compile-time static_assert) that
// the node struct is exactly 64 bytes.
// ---------------------------------------------------------------------------
static void test_node_struct_size()
{
    static_assert(sizeof(CompactQuadtreeNode) == 64,
        "CompactQuadtreeNode must be exactly 64 bytes");

    if (sizeof(CompactQuadtreeNode) != 64u) {
        test_abort_with_message(
            "node_struct_size",
            "sizeof(CompactQuadtreeNode) != 64");
    }
    test_report_pass("node_struct_size");
}

// ---------------------------------------------------------------------------
// test_trivial_copyability
//
// Verifies at runtime (in addition to the compile-time static_assert) that
// CompactQuadtreeNode satisfies std::is_trivially_copyable.
// ---------------------------------------------------------------------------
static void test_trivial_copyability()
{
    static_assert(std::is_trivially_copyable_v<CompactQuadtreeNode>,
        "CompactQuadtreeNode must be trivially copyable for GPU transfer");

    if (!std::is_trivially_copyable_v<CompactQuadtreeNode>) {
        test_abort_with_message(
            "trivial_copyability",
            "CompactQuadtreeNode is not trivially copyable");
    }
    test_report_pass("trivial_copyability");
}

// ---------------------------------------------------------------------------
// test_particle_count_preservation
//
// Builds a tree with N = 1, 2, 4, 1000 particles and asserts that
// count_particles_in_tree returns N in each case.
// ---------------------------------------------------------------------------
static void test_particle_count_preservation()
{
    const std::array<uint32_t, 4> test_particle_counts = {1u, 2u, 4u, 1000u};

    for (const uint32_t expected_count : test_particle_counts) {
        ParticleSystem test_system = create_particle_system(expected_count);
        initialize_uniform_disk(test_system, 500.0f, 1.0e30f, 12345u);

        LinearQuadtree test_tree;
        build_quadtree(test_tree, test_system);

        const uint32_t counted_particles = count_particles_in_tree(test_tree);

        if (counted_particles != expected_count) {
            test_abort_with_message(
                "particle_count_preservation",
                "count_particles_in_tree returned "
                + std::to_string(counted_particles)
                + " but expected "
                + std::to_string(expected_count));
        }
    }

    test_report_pass("particle_count_preservation");
}

// ---------------------------------------------------------------------------
// test_single_particle_tree
//
// Builds a tree with exactly one particle and asserts that the root is a leaf
// with contained_particle_index == 0 and first_child_node_index == SENTINEL.
// ---------------------------------------------------------------------------
static void test_single_particle_tree()
{
    ParticleSystem single_particle_system = create_particle_system(1u);
    single_particle_system.host_position_x[0] = 10.0f;
    single_particle_system.host_position_y[0] = 20.0f;
    single_particle_system.host_mass[0]        = 5.0e30f;

    LinearQuadtree single_particle_tree;
    build_quadtree(single_particle_tree, single_particle_system);

    const CompactQuadtreeNode& root_node = single_particle_tree.host_nodes[0];

    if (root_node.first_child_node_index != EMPTY_NODE_SENTINEL) {
        test_abort_with_message(
            "single_particle_tree",
            "root node should be a leaf "
            "(first_child_node_index should be EMPTY_NODE_SENTINEL)");
    }

    if (root_node.contained_particle_index != 0u) {
        test_abort_with_message(
            "single_particle_tree",
            "root node's contained_particle_index should be 0, got "
            + std::to_string(root_node.contained_particle_index));
    }

    test_report_pass("single_particle_tree");
}

// ---------------------------------------------------------------------------
// test_two_particles_same_quadrant
//
// Inserts two particles that are initially in the same quadrant relative to
// the root and asserts that count_particles_in_tree returns 2 (no infinite
// recursion, no crash, all particles accounted for).
// ---------------------------------------------------------------------------
static void test_two_particles_same_quadrant()
{
    ParticleSystem two_particle_system = create_particle_system(2u);

    // Both particles in the NE quadrant (x > 0, y > 0) but at different
    // sub-positions so they require at least one level of subdivision.
    two_particle_system.host_position_x[0] = 100.0f;
    two_particle_system.host_position_y[0] = 100.0f;
    two_particle_system.host_position_x[1] = 200.0f;
    two_particle_system.host_position_y[1] = 200.0f;
    two_particle_system.host_mass[0]        = 1.0e30f;
    two_particle_system.host_mass[1]        = 1.0e30f;

    LinearQuadtree two_particle_tree;
    build_quadtree(two_particle_tree, two_particle_system);

    const uint32_t counted_particles = count_particles_in_tree(two_particle_tree);

    if (counted_particles != 2u) {
        test_abort_with_message(
            "two_particles_same_quadrant",
            "expected count 2, got " + std::to_string(counted_particles));
    }

    test_report_pass("two_particles_same_quadrant");
}

// ---------------------------------------------------------------------------
// test_mass_conservation
//
// For a 2-particle system, the root's node_total_mass must equal m1 + m2.
// ---------------------------------------------------------------------------
static void test_mass_conservation()
{
    constexpr float mass_particle_0 = 3.0e30f;
    constexpr float mass_particle_1 = 7.0e30f;
    constexpr float expected_total_mass = mass_particle_0 + mass_particle_1;
    constexpr float mass_tolerance = 1.0f;  // absolute tolerance in kg

    ParticleSystem two_mass_system = create_particle_system(2u);
    two_mass_system.host_position_x[0] = -300.0f;
    two_mass_system.host_position_y[0] =    0.0f;
    two_mass_system.host_position_x[1] =  300.0f;
    two_mass_system.host_position_y[1] =    0.0f;
    two_mass_system.host_mass[0]        = mass_particle_0;
    two_mass_system.host_mass[1]        = mass_particle_1;

    LinearQuadtree mass_conservation_tree;
    build_quadtree(mass_conservation_tree, two_mass_system);

    const float root_total_mass =
        mass_conservation_tree.host_nodes[0].node_total_mass;

    if (!approximately_equal(root_total_mass, expected_total_mass, mass_tolerance)) {
        test_abort_with_message(
            "mass_conservation",
            "root total mass " + std::to_string(root_total_mass)
            + " does not equal m0 + m1 = "
            + std::to_string(expected_total_mass));
    }

    test_report_pass("mass_conservation");
}

// ---------------------------------------------------------------------------
// test_center_of_mass_correctness
//
// For two equal-mass particles at (-1, 0) and (+1, 0), the root centre of
// mass must be (0, 0) ± 1e-5.
// ---------------------------------------------------------------------------
static void test_center_of_mass_correctness()
{
    constexpr float equal_particle_mass = 1.0e30f;
    constexpr float com_tolerance = 1e-5f;

    ParticleSystem symmetric_system = create_particle_system(2u);
    symmetric_system.host_position_x[0] = -1.0f;
    symmetric_system.host_position_y[0] =  0.0f;
    symmetric_system.host_position_x[1] =  1.0f;
    symmetric_system.host_position_y[1] =  0.0f;
    symmetric_system.host_mass[0]        = equal_particle_mass;
    symmetric_system.host_mass[1]        = equal_particle_mass;

    LinearQuadtree com_test_tree;
    build_quadtree(com_test_tree, symmetric_system);

    const float root_com_x =
        com_test_tree.host_nodes[0].node_center_of_mass_x;
    const float root_com_y =
        com_test_tree.host_nodes[0].node_center_of_mass_y;

    if (!approximately_equal(root_com_x, 0.0f, com_tolerance)) {
        test_abort_with_message(
            "center_of_mass_correctness",
            "centre-of-mass x = " + std::to_string(root_com_x)
            + ", expected 0.0 ± " + std::to_string(com_tolerance));
    }

    if (!approximately_equal(root_com_y, 0.0f, com_tolerance)) {
        test_abort_with_message(
            "center_of_mass_correctness",
            "centre-of-mass y = " + std::to_string(root_com_y)
            + ", expected 0.0 ± " + std::to_string(com_tolerance));
    }

    test_report_pass("center_of_mass_correctness");
}

// ---------------------------------------------------------------------------
// test_bounding_box_computation
//
// Calls compute_particle_bounding_box on a known 4-particle configuration
// and verifies exact expected bounds.
// ---------------------------------------------------------------------------
static void test_bounding_box_computation()
{
    // Particles at corners of a known rectangle.
    constexpr float expected_min_x = -50.0f;
    constexpr float expected_min_y = -75.0f;
    constexpr float expected_max_x =  40.0f;
    constexpr float expected_max_y =  60.0f;
    constexpr float bounds_tolerance = 1e-5f;

    ParticleSystem four_particle_system = create_particle_system(4u);
    four_particle_system.host_position_x[0] = expected_min_x;
    four_particle_system.host_position_y[0] = expected_min_y;
    four_particle_system.host_position_x[1] = expected_max_x;
    four_particle_system.host_position_y[1] = expected_min_y;
    four_particle_system.host_position_x[2] = expected_min_x;
    four_particle_system.host_position_y[2] = expected_max_y;
    four_particle_system.host_position_x[3] = expected_max_x;
    four_particle_system.host_position_y[3] = expected_max_y;
    four_particle_system.host_mass[0] = 1.0f;
    four_particle_system.host_mass[1] = 1.0f;
    four_particle_system.host_mass[2] = 1.0f;
    four_particle_system.host_mass[3] = 1.0f;

    const std::array<float, 4> computed_bounds =
        compute_particle_bounding_box(four_particle_system);

    if (!approximately_equal(computed_bounds[0], expected_min_x, bounds_tolerance)) {
        test_abort_with_message("bounding_box_computation",
            "min_x = " + std::to_string(computed_bounds[0])
            + ", expected " + std::to_string(expected_min_x));
    }
    if (!approximately_equal(computed_bounds[1], expected_min_y, bounds_tolerance)) {
        test_abort_with_message("bounding_box_computation",
            "min_y = " + std::to_string(computed_bounds[1])
            + ", expected " + std::to_string(expected_min_y));
    }
    if (!approximately_equal(computed_bounds[2], expected_max_x, bounds_tolerance)) {
        test_abort_with_message("bounding_box_computation",
            "max_x = " + std::to_string(computed_bounds[2])
            + ", expected " + std::to_string(expected_max_x));
    }
    if (!approximately_equal(computed_bounds[3], expected_max_y, bounds_tolerance)) {
        test_abort_with_message("bounding_box_computation",
            "max_y = " + std::to_string(computed_bounds[3])
            + ", expected " + std::to_string(expected_max_y));
    }

    test_report_pass("bounding_box_computation");
}

// ---------------------------------------------------------------------------
// test_force_symmetry
//
// Newton's third law: for two equal-mass particles, the gravitational
// acceleration on particle 0 must equal the negation of acceleration on
// particle 1, within a tolerance of 1e-5 m/s^2.
// ---------------------------------------------------------------------------
static void test_force_symmetry()
{
    constexpr float force_symmetry_tolerance = 1e-5f;

    ParticleSystem two_body_system = create_particle_system(2u);
    two_body_system.host_position_x[0] = -100.0f;
    two_body_system.host_position_y[0] =    0.0f;
    two_body_system.host_position_x[1] =  100.0f;
    two_body_system.host_position_y[1] =    0.0f;
    two_body_system.host_mass[0]        = 1.0e30f;
    two_body_system.host_mass[1]        = 1.0e30f;

    LinearQuadtree two_body_tree;
    build_quadtree(two_body_tree, two_body_system);

    // Use theta^2 = 0 to force exact direct summation (all leaves visited).
    const float zero_theta_squared = 0.0f;

    float acceleration_x_particle_0 = 0.0f;
    float acceleration_y_particle_0 = 0.0f;
    compute_gravitational_acceleration_for_particle(
        0u,
        two_body_system,
        two_body_tree.host_nodes,
        zero_theta_squared,
        SOFTENING_LENGTH_SQUARED,
        acceleration_x_particle_0,
        acceleration_y_particle_0);

    float acceleration_x_particle_1 = 0.0f;
    float acceleration_y_particle_1 = 0.0f;
    compute_gravitational_acceleration_for_particle(
        1u,
        two_body_system,
        two_body_tree.host_nodes,
        zero_theta_squared,
        SOFTENING_LENGTH_SQUARED,
        acceleration_x_particle_1,
        acceleration_y_particle_1);

    // Newton's third law: a0 == -a1
    if (!approximately_equal(
            acceleration_x_particle_0,
            -acceleration_x_particle_1,
            force_symmetry_tolerance))
    {
        test_abort_with_message(
            "force_symmetry",
            "acc_x[0] = " + std::to_string(acceleration_x_particle_0)
            + " is not the negation of acc_x[1] = "
            + std::to_string(acceleration_x_particle_1));
    }

    if (!approximately_equal(
            acceleration_y_particle_0,
            -acceleration_y_particle_1,
            force_symmetry_tolerance))
    {
        test_abort_with_message(
            "force_symmetry",
            "acc_y[0] = " + std::to_string(acceleration_y_particle_0)
            + " is not the negation of acc_y[1] = "
            + std::to_string(acceleration_y_particle_1));
    }

    test_report_pass("force_symmetry");
}

// ---------------------------------------------------------------------------
// test_tree_rebuild_stability
//
// Builds the tree 3 times on the same data and asserts that
// count_particles_in_tree returns N each time.
// ---------------------------------------------------------------------------
static void test_tree_rebuild_stability()
{
    constexpr uint32_t rebuild_particle_count = 500u;
    constexpr uint32_t expected_count         = rebuild_particle_count;

    ParticleSystem stable_system = create_particle_system(rebuild_particle_count);
    initialize_uniform_disk(stable_system, 400.0f, 1.0e30f, 99999u);

    LinearQuadtree stable_tree;

    for (uint32_t rebuild_iteration = 0u;
         rebuild_iteration < 3u;
         ++rebuild_iteration)
    {
        build_quadtree(stable_tree, stable_system);

        const uint32_t counted = count_particles_in_tree(stable_tree);
        if (counted != expected_count) {
            test_abort_with_message(
                "tree_rebuild_stability",
                "iteration " + std::to_string(rebuild_iteration)
                + ": count_particles_in_tree returned "
                + std::to_string(counted)
                + " but expected " + std::to_string(expected_count));
        }
    }

    test_report_pass("tree_rebuild_stability");
}

// ---------------------------------------------------------------------------
// test_empty_system
//
// Builds a tree with particle_count == 0 and asserts no crash and
// active_node_count == 1 (empty root only).
// ---------------------------------------------------------------------------
static void test_empty_system()
{
    ParticleSystem empty_system = create_particle_system(0u);
    LinearQuadtree empty_tree;
    build_quadtree(empty_tree, empty_system);

    if (empty_tree.active_node_count != 1u) {
        test_abort_with_message(
            "empty_system",
            "expected active_node_count == 1 for empty system, got "
            + std::to_string(empty_tree.active_node_count));
    }

    const CompactQuadtreeNode& root = empty_tree.host_nodes[0];
    if (root.first_child_node_index != EMPTY_NODE_SENTINEL) {
        test_abort_with_message(
            "empty_system",
            "root of empty tree should have no children");
    }
    if (root.contained_particle_index != EMPTY_NODE_SENTINEL) {
        test_abort_with_message(
            "empty_system",
            "root of empty tree should contain no particle");
    }

    test_report_pass("empty_system");
}

// ---------------------------------------------------------------------------
// main — runs all tests sequentially (single-threaded, no OpenMP).
// ---------------------------------------------------------------------------
int main()
{
    std::cout << "=== Barnes-Hut Quadtree Unit Tests ===\n\n";

    test_node_struct_size();
    test_trivial_copyability();
    test_particle_count_preservation();
    test_single_particle_tree();
    test_two_particles_same_quadrant();
    test_mass_conservation();
    test_center_of_mass_correctness();
    test_bounding_box_computation();
    test_force_symmetry();
    test_tree_rebuild_stability();
    test_empty_system();

    std::cout << "\nAll tests passed.\n";
    return 0;
}
