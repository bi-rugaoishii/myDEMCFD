# CFD/VOF Solver TODO

This repository contains a custom CFD/VOF solver under active development.
The solver started as a 2D structured-grid, staggered/MAC-grid incompressible flow solver with VOF free-surface tracking, GPU acceleration, variable-density pressure projection, geometric multigrid pressure solvers, and surface tension modeling.
It is now being extended to 3D free-surface simulations such as water crown and milk crown problems.

The current highest priority is the 3D boundary handling design based on per-cell attributes, because divergence, pressure projection, velocity correction, VOF transport, viscous terms, and surface tension all depend on consistent boundary treatment.

---

## Current Status

### Implemented

- [x] 2D structured MAC grid solver
- [x] Ghost cells and staggered velocity fields
- [x] SMAC projection method
- [x] VOF method with THINC / WLIC
- [x] Variable-density / variable-coefficient Poisson equation
- [x] GPU CG / PCG pressure solver
- [x] Standalone GMG pressure solver
- [x] GMG-preconditioned pressure solver
- [x] Full GPU port and PCG kernel fusion
- [x] CFL-based variable time step
- [x] Alpha substepping for VOF transport
- [x] Surface tension implementation
- [x] Face/cell indexing convention cleanup
- [x] Basic validation: lid-driven cavity, OpenFOAM comparison, dam break, Zalesak slotted disk
- [x] Basic diagnostics: alpha conservation, divergence, CFL / dt, PCG residuals

### Main Remaining Work

- [ ] 3D boundary-cell attribute system
- [ ] 3D MAC-grid extension
- [ ] Solver configuration cleanup
- [ ] Fixed-dt / variable-dt mode switching
- [ ] Numerical scheme switching
- [ ] Pressure-solver validation and benchmarking
- [ ] MUSCL-TVD momentum advection
- [ ] Variable-viscosity stress-divergence viscous term
- [ ] Surface-tension validation and improvement
- [ ] Additional benchmark cases
- [ ] Restart, binary output, and long-running simulation utilities

---

# Short-Term Priorities

1. Complete the 3D boundary-cell attribute system.
2. Apply the cell attributes to boundary conditions, divergence, Poisson, and velocity correction.
3. Extend the core MAC-grid operators to 3D.
4. Validate the 3D boundary handling with small-grid tests and divergence checks.
5. Organize `SolverConfig`, parameter ownership, and fixed/CFL time-step switching.
6. Validate PCG, GMG, and GMG-preconditioned pressure solvers.
7. Add MUSCL-TVD momentum advection.
8. Correct the viscous term for variable viscosity.
9. Validate surface tension using static droplet, Laplace pressure, and spurious-current tests.
10. Add restart, binary output, and simulation logging.

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

The indexing convention has been refactored from the older `f0 c1 f1 c2 f2` style toward a cleaner `f0 c0 f1 c1 f2` style.

TODO:

- [x] Update core face/cell indexing convention
- [x] Update related kernels and pitch definitions
- [ ] Re-run main validation cases after the 3D boundary-cell attribute system is integrated
- [ ] Add lightweight debug checks for valid index ranges

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

---

# Phase 4: Higher-Order Momentum Advection

The current first-order upwind momentum advection is robust but diffusive. MUSCL-TVD should be added for splash and crown simulations.

TODO:

- [ ] Organize the current upwind scheme as the baseline implementation
- [ ] Implement MUSCL-TVD momentum advection on GPU
- [ ] Start with minmod, then add van Leer and MC limiters
- [ ] Compare upwind and MUSCL-TVD using dam-break and interface-flow tests

---

# Phase 5: Viscous Term Correction

The current viscous term is approximate. For variable-viscosity VOF flows, use the stress-divergence form.

TODO:

- [ ] Identify the current approximation
- [ ] Implement `div[ mu * (grad(u) + grad(u)^T) ]` for 2D/3D staggered grids
- [ ] Organize viscosity interpolation to faces and corners/edges
- [ ] Test stability and accuracy at high viscosity ratio

---

# Phase 6: Surface Tension Validation and Improvement

Surface tension is implemented. The next work is validation and spurious-current reduction.

TODO:

- [x] Implement surface-tension force coupling
- [ ] Add static droplet, Laplace pressure, and spurious-current tests
- [ ] Check grid-resolution, density-ratio, and sigma sensitivity
- [ ] Improve curvature and force placement if needed
- [ ] Investigate balanced-force consistency and height-function curvature

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

# Phase 8: 3D Extension

The long-term target is 3D water crown / milk crown simulation.
The most urgent part is the boundary-cell attribute system.

## 8.0 3D Boundary-Cell Attribute System

Use per-cell attributes to centralize boundary handling. Each cell should know whether it is fluid, ghost/solid, or adjacent to a physical boundary.

TODO:

- [ ] Add `cell_flag` / `cell_type` to `StaggeredGrid`
- [ ] Build flags for fluid cells, ghost/solid cells, and boundary-adjacent cells
- [ ] Use cell attributes in boundary conditions, divergence, Poisson, and velocity correction
- [ ] Extend the same logic to VOF, momentum, viscous terms, surface tension, and GMG levels
- [ ] Validate with small-grid flag dumps and simple projection tests

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

- [ ] Decide the 3D MAC-grid layout and z-face array layout
- [ ] Add `w` velocity and 3D divergence / gradient operators
- [ ] Implement the 3D variable-coefficient Poisson stencil
- [ ] Extend VOF, viscous term, curvature, surface tension, and CFL calculation to 3D
- [ ] Decide memory estimates and output format for large 3D cases

3D CFL estimate:

```text
local_adv =
    max(|u_left|, |u_right|)/dx
  + max(|v_back|, |v_front|)/dy
  + max(|w_bottom|, |w_top|)/dz
```

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
- [ ] Make raw-pointer-owning classes safer against accidental copying
- [ ] Initialize and clear device pointers consistently
- [ ] Log array sizes, pitches, and key index ranges at startup

---

# Design Summary

```text
Already implemented:
    GPU solver
    PCG kernel fusion
    WLIC
    CFL-based variable dt
    alpha substepping
    variable-density Poisson
    standalone GMG pressure solver
    GMG-preconditioned pressure solver
    surface tension
    face/cell indexing cleanup

Highest priority:
    3D boundary-cell attribute system

Main remaining work:
    3D MAC-grid extension
    scheme/config switching
    parameter ownership cleanup
    pressure-solver validation
    MUSCL-TVD momentum advection
    variable-viscosity viscous term
    surface-tension validation
    restart / binary output
```
