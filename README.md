# Barnes-Hut Spatial Index — Linear Quadtree

A production-grade, hardware-agnostic **2D linearised quadtree** for the
spatial-indexing front-end of a hybrid CPU/GPU Barnes-Hut N-body simulation.
Written in **C++20**, targeting zero warnings under GCC 13+ and Clang 16+
with `-Wall -Wextra -Wpedantic -Werror`.

> **Scope of this module:** quadtree construction and mass aggregation only.
> Force calculation (Barnes-Hut θ traversal), time integration, and CUDA
> kernels are explicitly *out of scope* and are not implemented here.

---

## Table of Contents

1. [Algorithm Overview](#algorithm-overview)
2. [Architecture & Design Constraints](#architecture--design-constraints)
3. [Data Structures](#data-structures)
4. [Build Phases](#build-phases)
5. [File Layout](#file-layout)
6. [Building](#building)
7. [Running the Tests](#running-the-tests)
8. [Design Decisions & Trade-offs](#design-decisions--trade-offs)
9. [GPU Readiness Roadmap](#gpu-readiness-roadmap)
10. [License](#license)

---

## Algorithm Overview

The Barnes-Hut algorithm reduces the N-body force-calculation complexity from
$\mathcal{O}(N^2)$ to $\mathcal{O}(N \log N)$ by grouping distant particles
into a spatial hierarchy.  Each internal node stores the aggregate mass and
centre-of-mass of its entire subtree; a traversal step then applies the
multipole-acceptance criterion

$$\frac{s}{d} < \theta$$

to decide whether the node's aggregate can replace individual particle
interactions (not implemented here — this module builds the tree only).

The quadtree recursively subdivides a square region into four equal children
(SW, SE, NW, NE).  This implementation stores the entire tree in a single
flat `std::vector`, with all inter-node links expressed as `uint32_t` array
indices — no raw pointers inside the tree.

---

## Architecture & Design Constraints

All constraints below are *hard requirements*, verified by inspection and
enforced by `static_assert` where possible.

| Constraint | How it is enforced |
|---|---|
| **Zero global variables** | All state passed as function arguments; no `static` mutable state |
| **No pointer-chasing in nodes** | Inter-node links are `uint32_t` indices into `host_nodes[]` |
| **Exactly one 64-byte cache line per node** | `alignas(64)` + `static_assert(sizeof == 64)` |
| **Trivially copyable nodes** | `static_assert(std::is_trivially_copyable_v<...>)` — ensures future `cudaMemcpy` compatibility |
| **Zero heap allocation after Phase B** | All storage reserved before the build loop; no `push_back`, `resize`, `new`, or `malloc` in Phases C/D |
| **Iterative construction (no recursion)** | Explicit pre-reserved work stack; depth-first order |
| **Determinism** | Counting-sort partitioning is stable given identical input order |
| **Graceful coincident-coordinate handling** | Geometric epsilon termination: subdivision stops when `box_half_width ≤ root_half_width × 2⁻²⁰` |
| **SoA particle layout** | `ParticleSystem` holds separate `host_pos_x`, `host_pos_y`, `host_mass` vectors |
| **`const`-correct API** | `build_linear_quadtree` takes `const ParticleSystem&`; all helpers are `noexcept` |

Everything lives in namespace **`bh_spatial`** to avoid global-namespace pollution.

---

## Data Structures

### `ParticleSystem` (`include/Storage.hpp`)

Strict Structure-of-Arrays layout:

```
ParticleSystem
├── count            : size_t
├── host_pos_x[]     : float   ─┐
├── host_pos_y[]     : float    ├─ one element per particle
├── host_mass[]      : float   ─┘
├── device_pos_x*    : float*  ─┐
├── device_pos_y*    : float*   ├─ null placeholders for future GPU upload
└── device_mass*     : float*  ─┘
```

SoA improves CPU auto-vectorisation (iterating one physical quantity at a time
maps cleanly to SIMD registers) and GPU global-memory coalescing (consecutive
threads read consecutive floats from the same array).

### `CompactQuadtreeNode` (`include/SpatialTree.hpp`)

Exactly **64 bytes**, `alignas(64)`:

```
Offset  Field                    Type       Bytes  Description
──────  ───────────────────────  ─────────  ─────  ──────────────────────────────────
 0      center_of_mass_x         float       4     Mass-weighted CoM (set in Phase D)
 4      center_of_mass_y         float       4
 8      total_mass               float       4     Sum of subtree particle masses
12      box_center_x             float       4     Geometric centre of bounding square
16      box_center_y             float       4
20      box_half_width           float       4     Half the side length
24      first_child_index        uint32_t    4     Index of first of 4 children; INVALID_INDEX if leaf
28      first_particle_index     uint32_t    4     Index into host_leaf_particle_indices; INVALID_INDEX if not a populated leaf
32      particle_count           uint32_t    4     Particles at this leaf; 0 for internal/empty nodes
36      reserved_padding         std::byte  28     Explicit pad to 64 bytes
──────  ───────────────────────  ─────────  ─────
        TOTAL                               64
```

**Why `first_particle_index + particle_count` instead of a single index?**
A single particle index can represent only one particle per leaf, making
coincident-coordinate handling impossible.  The bucket-array design lets one
leaf node hold any number of particles at (or near) the same coordinates.

### `LinearQuadtree` (`include/SpatialTree.hpp`)

```
LinearQuadtree
├── host_nodes[]                   : vector<CompactQuadtreeNode>  — flat node array
├── host_leaf_particle_indices[]   : vector<uint32_t>             — leaf bucket array
├── node_count                     : uint32_t                     — valid entries in host_nodes
├── device_nodes*                  : CompactQuadtreeNode*         — GPU placeholder
└── device_leaf_particle_indices*  : uint32_t*                    — GPU placeholder
```

Leaf node `L` owns particles at indices:
```
host_leaf_particle_indices[ L.first_particle_index  ..
                             L.first_particle_index + L.particle_count )
```

The `INVALID_INDEX = 0xFFFFFFFFu` sentinel marks absent child/particle links.

---

## Build Phases

`build_linear_quadtree` executes exactly four phases in sequence:

```
Phase A  ──►  Phase B  ──►  Phase C  ──►  Phase D
 BBox         Alloc        Subdivide       Mass
```

### Phase A — Bounding Box

Single linear pass over `host_pos_x` / `host_pos_y`.  Derives a **square**
root bounding box (equal half-widths on both axes) centred on the coordinate
midpoint, enlarged by a small epsilon so all particles are strictly inside.

### Phase B — Pre-Allocation *(only heap activity in the entire build)*

Reserves five vectors before the hot loop:

| Vector | Capacity | Purpose |
|---|---|---|
| `host_nodes` | `8·N + 64` | Tree nodes (direct-index access via `resize`) |
| `host_leaf_particle_indices` | `N` | Particle scratch / final leaf buckets |
| `node_build_states` | `8·N + 64` | Per-node particle range and depth (local) |
| `work_stack` | `8·N + 64` | Node indices awaiting subdivision (local) |
| `partition_scratch` | `N` | Counting-sort scatter buffer (local) |

**Why `8·N + 64`?** The naive `4·N + 1` bound is not safe for clustered inputs
— a particle that forces repeated subdivision before any neighbour lands in
a different quadrant can create more nodes than that.  `8·N + 64` is an
empirically conservative upper bound; a runtime guard throws
`std::runtime_error` if it is ever exceeded.

### Phase C — Iterative Top-Down Subdivision

```
work_stack ← {root}

while work_stack not empty:
    node ← pop work_stack

    if particle_count ≤ max_particles_per_leaf
       OR depth ≥ max_tree_depth
       OR box_half_width ≤ geometric_epsilon:
        → node stays a leaf (termination / coincident-coord case)
        continue

    partition particles into 4 quadrants  ← counting sort (3 passes, O(N) total)
    allocate 4 child nodes  (capacity guard throws on overflow)
    mark node as internal
    for each child with particle_count > max_particles_per_leaf:
        push child onto work_stack
```

**Counting-sort partitioning** avoids any per-sort allocation:
1. Count particles per quadrant.
2. Compute prefix sums → scatter offsets.
3. Scatter particle indices into `partition_scratch`.
4. Copy back to `host_leaf_particle_indices`.

**Invariant preserved:** every child always receives a *strictly higher* index
than its parent.  Phase D exploits this to avoid a second tree walk.

**Geometric epsilon termination:** `box_half_width ≤ root_half_width × 2⁻²⁰`.
After ~20 binary halvings, floating-point arithmetic can no longer distinguish
child box boundaries; any further subdivision would be meaningless and
potentially infinite.  Nodes that hit this limit become leaves holding all
their particles — the exact mechanism that makes coincident coordinates safe.

### Phase D — Bottom-Up Mass Reduction

Iterates `host_nodes` in **reverse index order** (from `node_count-1` down to 0).

Because children always have higher indices than their parent (Phase C
invariant), every child's `total_mass` and `center_of_mass` are already
finalised by the time the parent is visited.  No separate post-order traversal
or visited-flag array is needed.

```
for node_idx from (node_count−1) down to 0:
    if leaf:
        total_mass  = Σ particle_mass_i
        center_of_mass = Σ (mass_i · pos_i) / total_mass
    else (internal):
        for each of 4 children:
            skip if empty quadrant (particle_count == 0 AND first_child_index == INVALID_INDEX)
            total_mass  += child.total_mass
            weighted_pos += child.total_mass · child.center_of_mass
        center_of_mass = weighted_pos / total_mass
```

---

## File Layout

```
.
├── CMakeLists.txt                 Build configuration (C++20, -Wall -Wextra -Wpedantic -Werror)
├── include/
│   ├── Storage.hpp                ParticleSystem (SoA + device placeholders)
│   └── SpatialTree.hpp            CompactQuadtreeNode, LinearQuadtree, build declaration
├── src/
│   └── SpatialTree.cpp            Full four-phase build implementation
└── tests/
    └── test_spatial_tree.cpp      7-test correctness suite
```

---

## Building

### Requirements

| Tool | Minimum version |
|---|---|
| CMake | 3.22 |
| C++ compiler | GCC 13, Clang 16, or MSVC 19.38 (VS 2022 17.8) |
| C++ standard | C++20 |

### Steps

```bash
# Configure (Release mode recommended for benchmarking)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Compile
cmake --build build
```

This produces:
- `build/libbh_spatial_lib.a` (or `.lib` on MSVC) — the static library
- `build/bh_spatial_tests[.exe]` — the test executable

---

## Running the Tests

```bash
# Via CTest (recommended — integrates with CI)
ctest --test-dir build --output-on-failure

# Or run directly
./build/bh_spatial_tests
```

Expected output:
```
=== bh_spatial::LinearQuadtree correctness tests ===

PASS  cache_line_size_static_assertion
PASS  single_particle
PASS  two_particles_opposite_quadrants
PASS  mass_conservation_uniform_distribution
PASS  coincident_particles_leaf_termination
PASS  rebuild_determinism
PASS  invalid_argument_exceptions

All tests PASSED.
```

### What Each Test Covers

| Test | Failure mode caught |
|---|---|
| `cache_line_size_static_assertion` | Struct layout drift (field reordering, padding change) |
| `single_particle` | Root-is-leaf edge case; exact CoM for trivial input |
| `two_particles_opposite_quadrants` | First real subdivision; root promoted to internal; CoM weighted correctly |
| `mass_conservation_uniform_distribution` | Phase D accumulation; partition completeness for 200-particle grid |
| `coincident_particles_leaf_termination` | Geometric epsilon stops infinite subdivision of N coincident points |
| `rebuild_determinism` | Bit-identical output across two builds of identical input |
| `invalid_argument_exceptions` | Parameter validation throws `std::invalid_argument` as documented |

---

## Design Decisions & Trade-offs

### Why iterative and not recursive?

Recursive construction risks stack overflow on adversarial inputs
(e.g. all particles at the same coordinate would cause recursion up to
`max_tree_depth` frames with no natural base case before the epsilon check).
An explicit pre-reserved work stack bounds memory usage to `O(depth)` with
a statically-known maximum.

### Why `8·N + 64` nodes?

Each particle insertion can cause at most a bounded number of extra
subdivisions, but that bound is *not* `4` for clustered inputs — two nearly
coincident particles may force many levels of single-child internal nodes
before they land in separate quadrants.  `8·N + 64` is a conservative
practical bound.  The runtime capacity guard throws `std::runtime_error`
rather than silently overrunning the buffer.

### Why are children always allocated contiguously?

Allocating all four children in consecutive index slots means fetching any
child on a 64-byte-aligned machine typically costs one cache-line read for
two children.  On the GPU, four consecutive 64-byte nodes fit in a single
256-byte L1 cache transaction — critical for warp-coherent tree traversal.

### Why does `host_leaf_particle_indices` double as a scratch buffer?

Using the same array for construction scratch and final output means:
- No extra copy from scratch to output at the end.
- The working permutation produced by counting-sort partitioning *is* the
  final bucket layout, requiring no post-build reformatting.
- Allocation stays within Phase B — no mid-build allocation needed.

### Why reverse-index iteration for Phase D?

The allocation order in Phase C guarantees `child.index > parent.index` for
every parent-child relationship.  A single reverse-order linear scan therefore
processes all children before their parent, giving correct bottom-up semantics
with `O(N)` work and no auxiliary visited-flag array or recursive DFS.

---

## GPU Readiness Roadmap

The `LinearQuadtree` and `CompactQuadtreeNode` types are deliberately designed
so that upgrading to GPU execution requires only a new translation unit — no
changes to the spatial-index data structures.

Planned follow-on phases:

1. **GPU upload** (`EngineGPU.cu`) — `cudaMalloc` + `cudaMemcpy` to mirror
   `host_nodes` and `host_leaf_particle_indices` to the device pointers already
   declared in `LinearQuadtree`.
2. **Force traversal kernel** — CUDA kernel operating over the exact same
   `CompactQuadtreeNode` layout on device memory.
3. **Benchmarking** — side-by-side throughput comparison of OpenMP
   (CPU) vs. CUDA (GPU) tree traversal using identical serialised tree data.

The `CompactQuadtreeNode` is `std::is_trivially_copyable`, `alignas(64)`,
and exactly 64 bytes — all of which are required for correct `cudaMemcpy`
behaviour and efficient coalesced GPU reads.

---

## License

Distributed under the **GNU General Public License v3**.
See [`LICENSE`](LICENSE) for terms.
