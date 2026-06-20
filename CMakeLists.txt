cmake_minimum_required(VERSION 3.22)
project(BarnesHutSpatialIndex LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Global C++20 configuration
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)   # -std=c++20, not -std=gnu++20

# ---------------------------------------------------------------------------
# Compiler warning flags
# The spec requires zero warnings under -Wall -Wextra -Wpedantic -Werror
# for both GCC 13+ and Clang 16+.
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -Wno-unused-result   # allow discarding [[nodiscard]] in test helpers
    )
endif()

if(MSVC)
    add_compile_options(/W4 /WX /permissive-)
endif()

# ---------------------------------------------------------------------------
# Release optimisation flags
# ---------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-O3 -march=native -fno-math-errno)
    endif()
    if(MSVC)
        add_compile_options(/O2 /GL)
    endif()
endif()

# ---------------------------------------------------------------------------
# bh_spatial_lib — the spatial-index library
#
# Contains:
#   src/SpatialTree.cpp   — build_linear_quadtree (Phases A-D)
#
# Out of scope (not compiled here):
#   Force calculation, time integration, CUDA kernels — per spec.
# ---------------------------------------------------------------------------
add_library(bh_spatial_lib STATIC
    src/SpatialTree.cpp
)

target_include_directories(bh_spatial_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---------------------------------------------------------------------------
# bh_spatial_tests — correctness test executable
# ---------------------------------------------------------------------------
add_executable(bh_spatial_tests
    tests/test_spatial_tree.cpp
)

target_link_libraries(bh_spatial_tests PRIVATE bh_spatial_lib)

target_include_directories(bh_spatial_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---------------------------------------------------------------------------
# CTest integration
# ---------------------------------------------------------------------------
enable_testing()
add_test(NAME SpatialTreeCorrectness COMMAND bh_spatial_tests)

# ---------------------------------------------------------------------------
# Optional: print a helpful summary at the end of cmake configuration
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "  Barnes-Hut Spatial Index — build summary")
message(STATUS "  ----------------------------------------")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler     : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "  Targets")
message(STATUS "    bh_spatial_lib    — spatial index static library")
message(STATUS "    bh_spatial_tests  — correctness test suite")
message(STATUS "")
message(STATUS "  Build:  cmake --build <build_dir>")
message(STATUS "  Test:   ctest --test-dir <build_dir> --output-on-failure")
message(STATUS "")
