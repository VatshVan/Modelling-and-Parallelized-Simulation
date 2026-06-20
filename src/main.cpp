// ============================================================
// FILE: src/main.cpp
// PURPOSE: Benchmark orchestrator for the Barnes-Hut quadtree CPU pipeline.
//          Parses CLI arguments, drives the build → forces → integrate loop,
//          and reports per-step and summary performance metrics.
// DEPENDENCIES: Config.hpp, Storage.hpp, SpatialTree.hpp, TreeOperations.hpp
// DOES NOT: Implement tree construction, force traversal, CUDA operations,
//           or output file writing.
// ============================================================

#include "Config.hpp"
#include "SpatialTree.hpp"
#include "Storage.hpp"
#include "TreeOperations.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>   // std::atoi, std::atof, std::strtoul
#include <cstring>   // std::strcmp
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// parse_cli_arguments
//
// Parses --particles N, --steps S, --dt D from argc/argv.
// Uses no third-party libraries.  Unknown flags are silently ignored to allow
// future extension without breaking existing scripts.
//
// Parameters:
//   argc              — argument count from main().
//   argv              — argument vector from main().
//   out_particle_count — parsed particle count (output).
//   out_step_count     — parsed step count (output).
//   out_delta_time     — parsed timestep in seconds (output).
// ---------------------------------------------------------------------------
static void parse_cli_arguments(
    const int          argc,
    const char* const* argv,
    uint32_t&          out_particle_count,
    uint32_t&          out_step_count,
    float&             out_delta_time)
{
    // Defaults from Config.hpp-consistent choices.
    out_particle_count = 10000u;
    out_step_count     = 10u;
    out_delta_time     = 0.01f;

    for (int argument_index = 1; argument_index + 1 < argc; ++argument_index) {
        const char* const flag_string  = argv[argument_index];
        const char* const value_string = argv[argument_index + 1];

        if (std::strcmp(flag_string, "--particles") == 0) {
            const long parsed_value = std::atol(value_string);
            if (parsed_value > 0) {
                out_particle_count = static_cast<uint32_t>(parsed_value);
            }
            ++argument_index;  // consume the value token
        } else if (std::strcmp(flag_string, "--steps") == 0) {
            const long parsed_value = std::atol(value_string);
            if (parsed_value > 0) {
                out_step_count = static_cast<uint32_t>(parsed_value);
            }
            ++argument_index;
        } else if (std::strcmp(flag_string, "--dt") == 0) {
            const double parsed_value = std::atof(value_string);
            if (parsed_value > 0.0) {
                out_delta_time = static_cast<float>(parsed_value);
            }
            ++argument_index;
        }
    }
}

// ---------------------------------------------------------------------------
// format_milliseconds
//
// Formats a duration in milliseconds with two decimal places.
// ---------------------------------------------------------------------------
static std::string format_milliseconds(const double milliseconds)
{
    std::ostringstream formatted_stream;
    formatted_stream << std::fixed << std::setprecision(2) << milliseconds;
    return formatted_stream.str();
}

// ---------------------------------------------------------------------------
// main
//
// Benchmark loop:
//   For each step:
//     1. Build quadtree (timed).
//     2. Compute all gravitational forces (timed).
//     3. Integrate velocity and position.
//     4. Print per-step stats.
//   After all steps: print summary with throughput.
// ---------------------------------------------------------------------------
int main(const int argc, const char* const* const argv)
{
    // ---- Parse CLI -----------------------------------------------------------
    uint32_t benchmark_particle_count = 0u;
    uint32_t benchmark_step_count     = 0u;
    float    benchmark_delta_time     = 0.0f;

    parse_cli_arguments(
        argc, argv,
        benchmark_particle_count,
        benchmark_step_count,
        benchmark_delta_time);

    std::cout << "Barnes-Hut Quadtree CPU Benchmark\n"
              << "  Particles : " << benchmark_particle_count << '\n'
              << "  Steps     : " << benchmark_step_count << '\n'
              << "  dt        : " << benchmark_delta_time << " s\n\n";

    // ---- Particle system initialisation ------------------------------------
    ParticleSystem particle_system =
        create_particle_system(benchmark_particle_count);

    // Disk radius = half the simulation domain width for a typical N-body setup.
    constexpr float disk_radius_metres = SIMULATION_DOMAIN_HALF_WIDTH * 0.5f;
    constexpr float uniform_particle_mass_kg = 1.989e30f;  // 1 solar mass
    constexpr uint32_t initialisation_seed = 42u;

    initialize_uniform_disk(
        particle_system,
        disk_radius_metres,
        uniform_particle_mass_kg,
        initialisation_seed);

    // ---- Tree storage -------------------------------------------------------
    LinearQuadtree quadtree;
    // Pre-reserve once; build_quadtree calls initialize_tree_storage internally,
    // but we can give the vector an initial hint to avoid any early reallocation.
    quadtree.host_nodes.reserve(
        static_cast<size_t>(benchmark_particle_count)
        * TREE_NODE_CAPACITY_MULTIPLIER + 1u);

    // ---- Per-step accumulators for summary ----------------------------------
    double accumulated_build_time_ms    = 0.0;
    double accumulated_force_time_ms    = 0.0;
    uint32_t peak_tree_depth            = 0u;
    uint64_t accumulated_internal_nodes = 0u;

    const float theta_squared_value =
        BARNES_HUT_THETA_THRESHOLD * BARNES_HUT_THETA_THRESHOLD;

    // ---- Main benchmark loop ------------------------------------------------
    for (uint32_t step_index = 0u;
         step_index < benchmark_step_count;
         ++step_index)
    {
        // --- Build phase -------------------------------------------------------
        const auto build_start_time =
            std::chrono::high_resolution_clock::now();

        build_quadtree(quadtree, particle_system);

        const auto build_end_time =
            std::chrono::high_resolution_clock::now();

        const double step_build_time_ms =
            std::chrono::duration<double, std::milli>(
                build_end_time - build_start_time).count();

        accumulated_build_time_ms += step_build_time_ms;

        // --- Force computation phase ------------------------------------------
        const auto force_start_time =
            std::chrono::high_resolution_clock::now();

        compute_all_gravitational_forces_cpu(
            particle_system,
            quadtree,
            theta_squared_value,
            SOFTENING_LENGTH_SQUARED);

        const auto force_end_time =
            std::chrono::high_resolution_clock::now();

        const double step_force_time_ms =
            std::chrono::duration<double, std::milli>(
                force_end_time - force_start_time).count();

        accumulated_force_time_ms += step_force_time_ms;

        // --- Integration phase ------------------------------------------------
        integrate_velocity_verlet(particle_system, benchmark_delta_time);

        // --- Diagnostics ------------------------------------------------------
        const uint32_t step_tree_depth   = compute_tree_maximum_depth(quadtree);
        const uint32_t step_internal_nodes = count_internal_nodes(quadtree);

        if (step_tree_depth > peak_tree_depth) {
            peak_tree_depth = step_tree_depth;
        }
        accumulated_internal_nodes += step_internal_nodes;

        // --- Per-step output --------------------------------------------------
        std::cout << "Step "
                  << std::setw(3) << std::setfill('0') << (step_index + 1u)
                  << std::setfill(' ')
                  << " | Build: "
                  << std::setw(8) << format_milliseconds(step_build_time_ms)
                  << " ms | Forces: "
                  << std::setw(8) << format_milliseconds(step_force_time_ms)
                  << " ms | Depth: "
                  << std::setw(4) << step_tree_depth
                  << " | Internal nodes: "
                  << step_internal_nodes
                  << '\n';
    }

    // ---- Summary -------------------------------------------------------------
    const double average_build_time_ms =
        accumulated_build_time_ms / static_cast<double>(benchmark_step_count);
    const double average_force_time_ms =
        accumulated_force_time_ms / static_cast<double>(benchmark_step_count);
    const double average_internal_node_count =
        static_cast<double>(accumulated_internal_nodes)
        / static_cast<double>(benchmark_step_count);

    // Throughput: particle-node interactions per second.
    // Defined as: (particles * avg_internal_nodes) / avg_force_time_seconds.
    const double average_force_time_seconds = average_force_time_ms * 1e-3;
    const double cpu_throughput_interactions_per_second =
        (average_force_time_seconds > 0.0)
        ? (static_cast<double>(benchmark_particle_count)
           * average_internal_node_count
           / average_force_time_seconds)
        : 0.0;

    std::cout << "\n=== Benchmark Summary ===\n"
              << "Particles         : " << benchmark_particle_count << '\n'
              << "Steps completed   : " << benchmark_step_count << '\n'
              << "Avg build time    : "
              << format_milliseconds(average_build_time_ms) << " ms\n"
              << "Avg force time    : "
              << format_milliseconds(average_force_time_ms) << " ms\n"
              << "Peak tree depth   : " << peak_tree_depth << '\n'
              << "CPU throughput    : "
              << std::scientific << std::setprecision(2)
              << cpu_throughput_interactions_per_second
              << " particle-node interactions/sec\n";

    return 0;
}
