# CFD/VOF Solver TODO

This repository contains a custom CFD/VOF solver under active development.
The solver started as a 2D structured-grid, staggered/MAC-grid incompressible flow solver with VOF free-surface tracking, GPU acceleration, variable-density pressure projection, geometric multigrid pressure solvers, and surface tension modeling.
The solver has now been extended to 3D free-surface simulations such as water crown and milk crown problems.

The current priority is no longer basic 3D conversion. The immediate focus is reducing CUDA kernel argument size by switching large grid arguments from value passing to pointer passing. After that, the focus is curvature improvement with RDF, embedded-boundary support, validation, robustness, pressure-solver benchmarking, and preparation for long-running 3D simulations.

---

## Current Status

### Implemented

- [x] 2D structured MAC grid solver
- [x] Ghost cells and staggered velocity fields
- [x] SMAC projection method
- [x] VOF method with THINC / WLIC
- [x] Flux-direction-based upwind / THINC / WLIC switching near boundaries
- [x] MUSCL-TVD momentum advection with van Leer and minmod limiters
- [x] Variable-density / variable-coefficient Poisson equation
- [x] GPU CG / PCG pressure solver
- [x] Standalone GMG pressure solver
- [x] GMG-preconditioned pressure solver
- [x] Direction-selective / non-uniform GMG coarsening and validation
- [x] Full GPU port and PCG kernel fusion
- [x] CFL-based variable time step
- [x] Alpha substepping for VOF transport
- [x] Surface tension implementation
- [x] Face/cell indexing convention cleanup and validation
- [x] 3D MAC-grid extension
- [x] 3D boundary-cell attribute based boundary handling
- [x] Variable-viscosity stress-divergence viscous term
- [x] Basic validation: lid-driven cavity, OpenFOAM comparison, dam break, Zalesak slotted disk
- [x] Basic diagnostics: alpha conservation, divergence, CFL / dt, PCG residuals

### Main Remaining Work

- [ ] Switch large CUDA kernel arguments from value passing to pointer passing
- [ ] RDF-based curvature calculation for surface tension
- [ ] Embedded boundary method
- [ ] 3D validation and robustness checks
- [ ] Solver configuration cleanup
- [ ] Fixed-dt / variable-dt mode switching
- [ ] Numerical scheme switching
- [ ] Pressure-solver validation and benchmarking
- [ ] Stress-divergence viscous-term validation
- [ ] Surface-tension validation and improvement
- [ ] Additional benchmark cases
- [ ] Restart, binary output, and long-running simulation utilities

---

# Short-Term Priorities

1. Switch large CUDA kernel arguments from value passing to pointer passing.
2. Add a device-side self pointer to `Grid` / `G_StaggeredGrid` so kernels can receive a lightweight pointer.
3. Add RDF-based curvature calculation for surface tension.
4. Add embedded-boundary support using cell/face attributes.
5. Validate the 3D solver with small-grid tests, divergence checks, and simple projection tests.
6. Validate flux-direction-based THINC/WLIC switching near boundaries using dam-break and 3D wall cases.
7. Validate the stress-divergence viscous term for constant and variable viscosity cases.
8. Organize `SolverConfig`, parameter ownership, and fixed/CFL time-step switching.
9. Validate PCG, GMG, and GMG-preconditioned pressure solvers in 3D.
10. Add additional 3D benchmark cases and long-running output/restart utilities.

---

# Phase 0: Configuration and Design Cleanup

## 0.1 SolverConfig

Create a central configuration structure for numerical method selection and important runtime parameters.

TODO:

- [ ] Add enum-based switching for time stepping, pressure solver, momentum scheme, VOF scheme, limiter, viscous term, and surface tension model
- [ ] Move numerical options into `SolverConfig`
- [ ] Keep runtime selection outside inner loops and CUDA kernels

Recommended enum examples:

```cpp
enum class TimeStepMode{ Fixed, CFL };
enum class PressureSolverType{ PCG, GPU_PCG, GeometricMultigrid, GMGPreconditioned };
enum class MomentumSchemeType{ Upwind, MUSCL };
enum class VOFSchemeType{ Upwind, THINC, WLIC };
enum class SurfaceTensionMode{ Off, CSF, BalancedForce };
```

## 0.2 Time-Step Management

CFL-based variable time stepping is already implemented. Fixed time stepping should remain available for debugging and controlled comparisons.

TODO:

- [x] Implement CFL-based variable time step
- [x] Implement alpha substepping
- [ ] Add fixed-dt mode
- [ ] Add alpha-substep settings to `SolverConfig`
- [ ] Confirm physical-time output works correctly in both fixed and CFL modes

## 0.3 Parameter Ownership

Clarify where scalar parameters live to avoid stale values and CPU/GPU synchronization bugs.

TODO:

- [ ] Decide ownership of grid size, spacing, time variables, material properties, solver tolerances, and scheme parameters
- [ ] Separate responsibilities between `Grid`, `Solver`, `SolverConfig`, and `MaterialConfig`
- [ ] Decide how scalar parameters are passed to GPU kernels

Recommended organization:

```text
Grid:
    Nx, Ny, Nz, dx, dy, dz, pitches, field arrays

Solver:
    dt, t_now, step, output scheduling, time-step orchestration

SolverConfig:
    method switches, dt settings, CFL settings, tolerances, output settings

MaterialConfig:
    rho_l, rho_g, mu_l, mu_g, sigma
```

## 0.4 Face/Cell Indexing Cleanup

The face/cell indexing convention has been fully refactored and validated. The solver now uses the cleaner `f0 c0 f1 c1 f2` style consistently across the core staggered-grid arrays and related kernels.

TODO:

- [x] Update core face/cell indexing convention
- [x] Update related kernels and pitch definitions
- [x] Re-run main validation cases after the indexing change
- [x] Add lightweight debug checks for valid index ranges

---

## 0.5 GPU Kernel Argument Passing Cleanup

The grid structure has grown as the solver moved to 3D, added boundary flags, stress-divergence terms, GMG data, and additional VOF/surface-tension fields. Passing `G_StaggeredGrid` by value to every CUDA kernel is starting to exceed the kernel argument-size limit and also makes the launch interface heavier than necessary.

TODO:

- [ ] Change large kernel arguments from value passing to pointer passing
- [ ] Add a device-side self pointer such as `G_StaggeredGrid* d_self` to the host-side grid object
- [ ] Update kernels to receive `G_StaggeredGrid* grid` or a lightweight pointer wrapper
- [ ] Keep scalar arguments separate only when they are small and frequently changed
- [ ] Validate that existing kernels give identical results after the pointer-passing transition

Recommended direction:

```cpp
// Before
kernel<<<grid_dim, block_dim>>>(grid, dt);

// After
kernel<<<grid_dim, block_dim>>>(grid.d_self_, dt);
```

This should reduce CUDA launch argument size and avoid repeatedly copying the full grid descriptor into kernel parameters.

---

# Phase 1: Numerical Method Switching

Use coarse-grained runtime selection for large solver components and compile-time/template selection for inner-loop numerical schemes.

TODO:

- [ ] Organize pressure solvers behind a common interface
- [ ] Add factory-style pressure solver selection from `SolverConfig`
- [ ] Organize momentum and VOF schemes so they can be switched cleanly
- [ ] Avoid virtual calls inside cell loops and CUDA kernels

Design rule:

```text
Pressure solver selection:
    runtime enum / virtual interface is acceptable

Cell-wise schemes:
    enum switch outside loops, template/policy inside kernels
```

---

# Phase 2: Diagnostics and Logging

Basic diagnostics are implemented. The next step is to make logs easier to compare across schemes and solver settings.

TODO:

- [x] Log alpha mass, alpha min/max, divergence, CFL / dt, and PCG residuals
- [ ] Add compact per-step CSV logging
- [ ] Add timing breakdown for major solver stages
- [ ] Add key min/max diagnostics for pressure, velocity, density, viscosity, curvature, and surface-tension force

---

# Phase 3: Pressure Solver Validation and Benchmarking

GPU PCG, standalone GMG, and GMG-preconditioned pressure solvers are implemented. The remaining work is validation, tuning, and choosing the default.

TODO:

- [x] Implement GPU PCG pressure solver
- [x] Implement standalone GMG pressure solver
- [x] Implement GMG-preconditioned pressure solver
- [ ] Compare convergence, divergence after projection, pressure fields, and total solve time
- [ ] Test robustness for dam-break, static droplet, and high-density-ratio cases
- [ ] Tune smoother, restriction/prolongation, coefficient restriction, and pressure null-space handling
- [ ] Decide the default pressure solver

Target operator:

```text
A(p) = div( beta * grad(p) )
beta = dt / rho
```

## 3.5 Direction-Selective / Non-Uniform GMG Coarsening

Direction-selective / non-uniform GMG coarsening is implemented and validated. GMG levels can now coarsen only the directions that are still large enough, so thin directions are not over-coarsened.

TODO:

- [x] Add a threshold size for each direction before coarsening
- [x] Build GMG levels using per-direction coarsening decisions
- [x] Update restriction, prolongation, and operator construction for anisotropic level transitions
- [x] Validate convergence and speed against uniform coarsening

Example:

```text
64 x 64 x 16
-> 32 x 32 x 8
-> 16 x 16 x 8
->  8 x  8 x 8
```

Here, the z direction stops coarsening once it reaches the threshold size, while x and y continue to coarsen.

---

# Phase 4: Higher-Order Momentum Advection

MUSCL-TVD momentum advection is implemented and checked with both van Leer and minmod limiters. Upwind remains available as the robust baseline for comparison and debugging.

TODO:

- [x] Organize the current upwind scheme as the baseline implementation
- [x] Implement MUSCL-TVD momentum advection on GPU
- [x] Implement the van Leer limiter and validate it
- [x] Implement the minmod limiter and validate it

---

# Phase 4.5: VOF Boundary Treatment

THINC/WLIC now uses flux-direction-based fallback near boundaries. Instead of switching the entire near-boundary cell to first-order upwind, only the flux direction whose stencil is blocked by a boundary falls back to upwind or a limited treatment. Tangential and non-blocked directions can still use THINC/WLIC.

TODO:

- [x] Replace full-cell near-boundary upwind fallback with direction-aware fallback
- [x] Use boundary-direction / stencil validity checks to decide whether x/y/z fluxes can use THINC/WLIC
- [ ] Validate interface sharpness near walls in dam-break and 3D test cases

Current rule:

```text
For each flux direction:
    boundary-normal direction with invalid stencil -> upwind or limited fallback
    tangential / non-blocked direction             -> THINC/WLIC
```

---

# Phase 5: Viscous Stress-Divergence Term

The variable-viscosity viscous term has been replaced with the stress-divergence form. The remaining work is validation and tuning near interfaces and boundaries.

TODO:

- [x] Implement `div[ mu * (grad(u) + grad(u)^T) ]` for 2D/3D staggered grids
- [x] Organize viscosity interpolation to faces and corners/edges
- [ ] Validate the term for constant-viscosity cases against the old Laplacian form
- [ ] Test stability and accuracy at high viscosity ratio
- [ ] Check interaction with 3D boundary-cell attributes and free-surface cells

---

# Phase 6: Surface Tension Validation and Improvement

Surface tension is implemented. The next major improvement is RDF-based curvature calculation, followed by validation and spurious-current reduction.

TODO:

- [x] Implement surface-tension force coupling
- [ ] Add RDF-based curvature calculation
- [ ] Compare alpha-gradient curvature and RDF curvature
- [ ] Add static droplet, Laplace pressure, and spurious-current tests
- [ ] Check grid-resolution, density-ratio, and sigma sensitivity
- [ ] Improve balanced-force consistency and force placement if needed

Laplace pressure targets:

```text
2D circular droplet:    pressure jump = sigma / R
3D spherical droplet:   pressure jump = 2 * sigma / R
```

---

# Phase 7: Additional Validation Tests

TODO:

- [x] Lid-driven cavity
- [x] OpenFOAM comparison
- [x] Dam break
- [x] Zalesak slotted disk
- [ ] Static droplet / Laplace pressure
- [ ] Spurious-current test
- [ ] Rising bubble
- [ ] Rayleigh-Taylor instability
- [ ] High-density-ratio dam break
- [ ] Capillary wave or oscillating droplet

---

# Phase 8: 3D Solver

The solver has been extended from 2D to 3D. The focus is now validation, boundary robustness, memory/output design, and preparation for water crown / milk crown simulations.

## 8.0 3D Boundary-Cell Attribute System

Per-cell attributes are used to centralize boundary handling. Each cell can represent fluid, ghost/solid, or boundary-adjacent states, avoiding scattered index-based boundary checks.

TODO:

- [x] Add `cell_flag` / `cell_type` to `StaggeredGrid`
- [x] Build flags for fluid cells, ghost/solid cells, and boundary-adjacent cells
- [x] Use cell attributes in boundary conditions, divergence, Poisson, and velocity correction
- [ ] Validate with small-grid flag dumps, divergence checks, and simple projection tests
- [ ] Extend or rebuild the same flag logic consistently on GMG levels

Possible initial design:

```cpp
enum class CellFlag : unsigned char{
    Fluid     = 0,
    Ghost     = 1 << 0,
    Solid     = 1 << 1,
    BndXMinus = 1 << 2,
    BndXPlus  = 1 << 3,
    BndYMinus = 1 << 4,
    BndYPlus  = 1 << 5,
    BndZMinus = 1 << 6,
    BndZPlus  = 1 << 7
};
```

If more states are needed later, use `uint16_t`. Boundary-condition type information can be kept separate from geometric flags.

## 8.1 Core 3D Operators

TODO:

- [x] Decide the 3D MAC-grid layout and z-face array layout
- [x] Add `w` velocity and 3D divergence / gradient operators
- [x] Implement the 3D variable-coefficient Poisson stencil
- [x] Extend CFL calculation to 3D
- [x] Extend the viscous stress-divergence operator to 3D
- [ ] Validate 3D VOF, curvature, and surface-tension behavior
- [ ] Estimate memory usage and finalize output format for large 3D cases

3D CFL estimate:

```text
local_adv =
    max(|u_left|, |u_right|)/dx
  + max(|v_back|, |v_front|)/dy
  + max(|w_bottom|, |w_top|)/dz
```

## 8.5 Embedded Boundary Method

Add embedded-boundary support on the structured grid. The existing cell/face attribute system should be reused so that wall handling, Poisson stencils, VOF transport, and viscous terms remain centralized.

TODO:

- [ ] Add embedded-boundary geometry representation and cell/face classification
- [ ] Compute boundary volume/area information needed by the flow solver
- [ ] Apply embedded-boundary conditions to velocity, pressure, VOF, and viscous terms
- [ ] Validate with simple internal-wall and obstacle cases

---

# Phase 9: Long-Running Simulation Utilities

Large 3D simulations need robust output, restart, and logging.

TODO:

- [ ] Add restart functionality
- [ ] Add binary output and lightweight ParaView-compatible output
- [ ] Add parameter/log files and automatic simulation-condition saving
- [ ] Save output/restart intervals and git commit hash
- [ ] Transfer only output data from device to host

---

# Phase 10: Debugging and Quality Control

TODO:

- [ ] Add `CUDA_CHECK` and debug-mode kernel error checks
- [ ] Add small compute-sanitizer test cases
- [ ] Check CUDA kernel argument sizes after major grid-structure changes
- [ ] Make raw-pointer-owning classes safer against accidental copying
- [ ] Initialize and clear device pointers consistently
- [ ] Log array sizes and pitches at startup
- [x] Verify key face/cell index ranges in debug checks

---

# Design Summary

```text
Already implemented:
    GPU solver
    PCG kernel fusion
    WLIC
    flux-direction-based upwind / THINC / WLIC switching near boundaries
    MUSCL-TVD momentum advection with van Leer and minmod limiters
    CFL-based variable dt
    alpha substepping
    variable-density Poisson
    standalone GMG pressure solver
    GMG-preconditioned pressure solver
    direction-selective / non-uniform GMG coarsening
    surface tension
    face/cell indexing cleanup and validation
    3D MAC-grid extension
    3D boundary-cell attribute system
    variable-viscosity stress-divergence viscous term

Current priority:
    switch large CUDA kernel arguments from value passing to pointer passing
    RDF-based curvature calculation
    embedded boundary method
    validation and robustness of the 3D solver

Main remaining work:
    pointer-based CUDA kernel argument passing
    scheme/config switching
    parameter ownership cleanup
    pressure-solver validation
    RDF curvature and surface-tension validation
    embedded-boundary support
    stress-divergence viscous-term validation
    surface-tension validation
    additional 3D benchmarks
    restart / binary output
```
