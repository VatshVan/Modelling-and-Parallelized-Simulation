# Modelling and Parallelized Simulation of Natural Phenomenon: Barnes-Hut Algorithm

## Executive Summary
This repository contains a high-performance numerical computing framework for simulating large-scale natural phenomena driven by N-body interactions (e.g., gravitational or electrostatic forces). By implementing the Barnes-Hut spatial partitioning algorithm, the system reduces the classical $\mathcal{O}(N^2)$ brute-force computational bound to a scalable $\mathcal{O}(N \log N)$ paradigm. The architecture is explicitly designed for parallel execution, maximizing CPU throughput and minimizing latency bottlenecks associated with deep tree traversals.

## Architectural Overview

The operational pipeline consists of three primary, parallelized phases executed per simulation timestep:

1. **Domain Decomposition & Spatial Partitioning:**
   The simulation space is recursively subdivided utilizing a Quadtree (2D) or Octree (3D) structure. To maintain optimal cache locality and mitigate dynamic allocation overhead, nodes are allocated within pre-defined, contiguous memory pools. Particle insertion leverages spatial hashing (Morton encoding/Z-order curves) to ensure adjacent spatial regions map to adjacent memory blocks.

2. **Mass & Center of Mass Aggregation:**
   A bottom-up, post-order tree traversal computes the aggregate mass and the center of mass for all internal nodes. This hierarchical data structure forms the foundation for the far-field force approximations.

3. **Force Computation & Integration:**
   The framework computes net forces acting on each particle by traversing the tree structure. Nodes are evaluated against the Multipole Acceptance Criterion (MAC) to determine if their internal particle distribution can be approximated as a single point mass. The evaluation is defined as:

   $$\frac{s}{d} < \theta$$

   Where:
   * $s$ = The width of the spatial region represented by the node.
   * $d$ = The distance between the target particle and the node's center of mass.
   * $\theta$ = A configurable threshold parameter balancing accuracy and computational expense.

## Parallelization Strategy

To achieve maximum resource utilization, the framework employs a parallelized execution model addressing the specific compute bottlenecks of the algorithm:

* **Concurrent Force Evaluation:** The force computation phase is inherently embarrassingly parallel. Workloads are distributed across available threads using dynamic scheduling to ensure load balancing, as spatial particle density dictates tree depth and traversal time.
* **Lock-Free Tree Construction:** Depending on the configuration, tree building utilizes localized thread structures or lock-free atomic operations to prevent serialization during the domain decomposition phase.
* **Vectorization (SIMD):** Leaf node interactions (particles within the same terminal spatial bounds) are structured to leverage SIMD instructions for direct, vectorized force calculations.

## System Requirements and Compilation

The framework requires a modern C++ compiler supporting C++17 or higher, alongside standard parallelization libraries (e.g., OpenMP or Intel TBB, depending on the specific branch configuration). 

Compilation is handled via standard CMake build pipelines. Users must configure the build environment to target their specific hardware architecture to ensure maximum vectorization and optimization flags are applied during compilation. 

## Configuration Parameters

Simulation parameters are managed via a centralized configuration matrix, allowing adjustments to:
* `N_PARTICLES`: Total entity count.
* `THETA_MAC`: Threshold for the Multipole Acceptance Criterion.
* `DT`: Timestep resolution for the numerical integrator (e.g., Runge-Kutta or Leapfrog).
* `THREAD_COUNT`: Explicit definition of the worker pool size.
* `LEAF_CAPACITY`: Maximum particle density per terminal tree node before subdivision.

## Licensing
This project is distributed under the GNU GENERAL PUBLIC LICENSE Version 3. Reference the LICENSE file for explicit terms and conditions regarding modification and redistribution.
