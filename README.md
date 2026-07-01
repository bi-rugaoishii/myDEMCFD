# CFD/VOF Solver TODO

This repository contains a custom CFD/VOF solver under active development.
The solver started as a 2D structured-grid, staggered/MAC-grid incompressible flow solver with VOF free-surface tracking, GPU acceleration, variable-density pressure projection, geometric multigrid pressure solvers, and surface tension modeling. It is now being extended to 3D, with boundary handling organized around per-cell attributes and boundary-adjacent cell flags.

The long-term goal is to extend the solver toward 3D free-surface simulations such as water crown and milk crown problems.

---

## Overview

The solver has already reached a fairly advanced 2D GPU-accelerated CFD/VOF stage.

Major implemented features include:

- SMAC projection method on a staggered/MAC grid
- VOF free-surface tracking
- THINC and WLIC interface advection
- Variable-density / variable-coefficient pressure Poisson equation
- GPU CG / PCG pressure solver
- Standalone geometric multigrid pressure solver
- GMG-preconditioned pressure solver
- Full GPU port of the solver
- PCG kernel fusion and GPU optimization
- CFL-based variable time step
- Alpha substepping for VOF transport
- Surface tension implementation
- Face/cell indexing convention cleanup

The main remaining work is no longer basic implementation, but rather:

- better software organization and scheme switching
- fixed-dt / variable-dt mode switching
- parameter ownership cleanup
- 3D cell-attribute / boundary-adjacent-cell based boundary handling
- pressure-solver validation and benchmarking
- higher-order momentum advection
- viscous-term correction for variable viscosity
- surface-tension validation and spurious-current reduction
- additional benchmark problems
- restart / binary output
- preparation for 3D water crown / milk crown simulations

---

## Current Status

### Implemented

- [x] 2D structured grid
- [x] Staggered grid / MAC grid
- [x] Ghost cells
- [x] Cell-centered `p`, `alpha`, `rho`, `mu`
- [x] x-face velocity `vx` / `u`
- [x] y-face velocity `vy` / `v`
- [x] SMAC method
- [x] VOF method
- [x] THINC
- [x] WLIC
- [x] Variable-density Poisson equation
- [x] Variable-coefficient Poisson operator
- [x] CG / PCG pressure solver
- [x] GPU CG / PCG
- [x] Geometric multigrid pressure solver
- [x] GMG-preconditioned pressure solver
- [x] Full GPU port of the solver
- [x] PCG iteration kernel fusion
- [x] GPU optimization
- [x] CFL-based variable time step
- [x] Alpha substepping for VOF transport
- [x] Surface tension
- [x] Lid-driven cavity flow
- [x] Comparison with OpenFOAM
- [x] Dam-break simulation
- [x] Zalesak slotted disk test
- [x] Upwind / THINC / WLIC comparison
- [x] Basic diagnostics for alpha conservation and divergence
- [x] Face/cell indexing convention cleanup

---

## Main Development Goals

The next major goals are:

1. Make numerical methods switchable.
2. Support both fixed time step and CFL-based variable time step modes.
3. Organize solver parameters such as `dt`, `Nx`, `Ny`, material properties, and scheme settings.
4. Complete the 3D boundary-cell attribute system for boundary-adjacent cells and boundary-condition handling.
5. Validate and benchmark GPU PCG, GMG, and GMG-preconditioned pressure solvers.
6. Improve momentum advection using MUSCL-TVD schemes.
7. Replace the approximate viscous term with the full variable-viscosity stress-divergence form.
8. Validate and improve surface tension, curvature calculation, and spurious-current behavior.
9. Add more validation test cases.
10. Prepare for 3D extension and long-running simulations.

---

# Phase 0: Configuration and Design Cleanup

## 0.1 SolverConfig

Create or reorganize `SolverConfig` so that the main numerical methods can be switched cleanly.

TODO:

- [ ] Define `TimeStepMode` as an `enum class`
- [ ] Define `PressureSolverType` as an `enum class`
- [ ] Define `MomentumSchemeType` as an `enum class`
- [ ] Define `VOFSchemeType` as an `enum class`
- [ ] Define `LimiterType` as an `enum class`
- [ ] Define `SurfaceTensionMode` as an `enum class`
- [ ] Define `ViscousTermMode` as an `enum class`

Example:

```cpp
enum class TimeStepMode{
    Fixed,
    CFL
};

enum class PressureSolverType{
    PCG,
    GPU_PCG,
    GeometricMultigrid,
    GMGPreconditioned
};

enum class MomentumSchemeType{
    Upwind,
    MUSCL
};

enum class VOFSchemeType{
    Upwind,
    THINC,
    WLIC
};

enum class SurfaceTensionMode{
    Off,
    CSF,
    BalancedForce
};
```

---

## 0.2 Fixed Time Step / CFL Time Step Switching

The CFL-based variable time step is implemented, but fixed time step mode should also remain available for debugging and controlled comparisons.

TODO:

- [x] Implement CFL-based variable time step
- [ ] Add fixed time step mode
- [ ] Add `dt_mode` to `SolverConfig`
- [ ] Add `dt_fixed` to `SolverConfig`
- [ ] Use `CFL_target`, `dt_min`, `dt_max`, and `dt_growth_max` only in CFL mode
- [ ] Check consistency between physical-time output and both time-step modes

CFL estimate for a 2D staggered grid:

```text
local_adv = max(|u_left|, |u_right|)/dx
          + max(|v_bottom|, |v_top|)/dy

max_adv = max over all cells local_adv

dt_cfl = CFL_target / (max_adv + eps)
```

---

## 0.2b Alpha Substepping for VOF Transport

Alpha substepping is implemented. It should remain configurable because VOF stability and interface quality can depend strongly on the alpha transport time step.

TODO:

- [x] Implement alpha substepping
- [ ] Add `alpha_substep_mode` or `alpha_substeps` to `SolverConfig`
- [ ] Support `alpha_substeps = 1` for debugging and comparison
- [ ] Support automatic alpha substep count based on alpha-CFL if needed
- [ ] Log the number of alpha substeps used in each main time step
- [ ] Confirm alpha mass conservation with and without substepping
- [ ] Compare dam-break and Zalesak results with different alpha substep counts

Recommended interpretation:

```text
Main time step:
    dt

Alpha transport substep:
    dt_alpha = dt / alpha_substeps
```

If an automatic mode is added later, the alpha substep count can be chosen so that:

```text
alpha_CFL <= alpha_CFL_target
```

---

## 0.3 Ownership of dt, Nx, Ny, and Other Parameters

The solver now has many scalar parameters. Their ownership should be clarified to avoid stale values, duplicated state, and CPU/GPU synchronization bugs.

TODO:

- [ ] Decide where `Nx`, `Ny`, `dx`, `dy`, `inv_dx`, and `inv_dy` belong
- [ ] Decide where `dt`, `t_now`, `step`, and `next_output_time` belong
- [ ] Decide where `rho_l`, `rho_g`, `mu_l`, and `mu_g` belong
- [ ] Decide where `sigma` belongs
- [ ] Decide where `CFL_target`, `dt_min`, `dt_max`, and `dt_growth_max` belong
- [ ] Decide where Poisson tolerance and max iteration belong
- [ ] Decide where THINC / WLIC parameters belong
- [ ] Separate responsibilities among `Grid`, `Solver`, `SolverConfig`, and `MaterialConfig`
- [ ] Decide how scalar parameters should be passed to GPU kernels

Recommended organization:

```text
StaggeredGrid2D:
    Nx, Ny, dx, dy, inv_dx, inv_dy
    pitch_p, pitch_vx, pitch_vy
    p, alpha, rho, mu, vx, vy, and other field arrays

SMACSolver:
    dt, t_now, step
    output_flag, next_output_time
    time-step orchestration

SolverConfig:
    dt_mode, dt_fixed
    CFL_target, dt_min, dt_max, dt_growth_max
    solver type, scheme type, limiter type
    poisson_tol, poisson_max_iter
    output_interval

MaterialConfig:
    rho_l, rho_g
    mu_l, mu_g
    sigma
```

---

## 0.4 Face/Cell Indexing Convention Cleanup Completed

The current face/cell indexing convention is based on a pattern like:

```text
f0 c1 f1 c2 f2
```

This should be refactored toward a cleaner and more consistent convention:

```text
f0 c0 f1 c1 f2 c2 ...
```

The goal is to make cell and face indices easier to reason about, reduce off-by-one errors, and make the 3D extension more natural.

TODO:

- [x] Document the current `f0 c1 f1 c2 f2` indexing convention
- [x] Define the target `f0 c0 f1 c1` indexing convention for 2D MAC grids
- [x] Decide the exact valid index ranges for cell-centered variables `p`, `alpha`, `rho`, and `mu`
- [x] Decide the exact valid index ranges for x-face velocity `vx` / `u`
- [x] Decide the exact valid index ranges for y-face velocity `vy` / `v`
- [x] Update pitch definitions if needed
- [x] Update boundary-condition kernels for the new face indexing
- [x] Update divergence, gradient, Poisson RHS, and velocity correction kernels
- [x] Update VOF flux kernels and WLIC/THINC face access
- [x] Update momentum advection and viscous-term kernels
- [x] Update surface-tension force placement if needed
- [ ] Add debug assertions or test kernels to verify valid index ranges
- [ ] Re-run lid-driven cavity, dam-break, static droplet, and Zalesak tests after the indexing change
- [ ] Keep the old indexing branch or commit available until the new convention is validated

Recommended principle:

```text
Cell index and face index should use the same logical numbering origin.
For example, cell c0 is between face f0 and face f1.
```

This makes the common finite-volume relation easier to read:

```text
cell c0 uses left face f0 and right face f1
cell c1 uses left face f1 and right face f2
```

Status: core implementation completed. Remaining work is mainly validation after the 3D boundary-cell attribute system is integrated.

---

# Phase 1: Switchable Numerical Methods

## 1.1 Coarse-Grained Algorithms: Virtual Interfaces

Large algorithmic components such as pressure solvers can use virtual functions, because they are called only a small number of times per time step.

TODO:

- [ ] Create a `PressureSolver` base class
- [ ] Organize `PCGPressureSolver`
- [ ] Organize `GPUPCGPressureSolver`
- [ ] Organize `GMGPressureSolver`
- [ ] Organize `GMGPreconditionedPressureSolver`
- [ ] Add `JacobiPressureSolver` and `SORPressureSolver` if needed
- [ ] Create `make_pressure_solver(config)`
- [ ] Call the pressure solver from `SMACSolver` through `pressure_solver_->solve(...)`

Design rule:

```text
Virtual functions are acceptable for coarse-grained solver components.
Do not use virtual calls inside cell loops or CUDA kernels.
```

---

## 1.2 Fine-Grained Schemes: Template / Policy Design

Cell-wise advection, reconstruction, interpolation, and limiter operations should not use virtual dispatch.

TODO:

- [ ] Organize momentum advection as `compute_ustar_impl<MomentumScheme>()`
- [ ] Organize VOF transport as `transport_alpha_impl<VOFScheme>()`
- [ ] Create or organize `UpwindMomentum`
- [ ] Create `MUSCLMomentum`
- [ ] Organize `UpwindVOF`
- [ ] Organize `THINCVOF`
- [ ] Organize `WLICVOF`
- [ ] Decide whether limiters should be template policies or selected by enum switch

Design rule:

```text
Coarse-grained components:
    virtual interface

Fine-grained cell-loop operations:
    template / policy

Runtime selection:
    enum class + switch outside loops

CUDA kernels:
    template-based selection
```

---

# Phase 2: Diagnostics and Logging

Basic diagnostics are already implemented. The next step is to organize logs for easier comparison and debugging.

Implemented diagnostics:

- [x] Total alpha mass
- [x] Alpha min/max
- [x] Clip check
- [x] Max divergence
- [x] PCG residual
- [x] PCG iteration count
- [x] CFL / dt

Additional TODO:

- [ ] Log `dt`, CFL, and max velocity every step
- [ ] Log pressure min/max
- [ ] Log velocity min/max
- [ ] Log rho min/max
- [ ] Log mu min/max
- [ ] Log kinetic energy
- [ ] Log maximum curvature after surface-tension computation
- [ ] Log maximum surface-tension force
- [ ] Measure total time per step
- [ ] Measure separate timings for Poisson, VOF, momentum, and surface-tension steps
- [ ] Add CSV log output

---

# Phase 3: Pressure Solver Validation and Benchmarking

Standalone GMG and a GMG-preconditioned pressure solver are now implemented. The remaining work is validation, tuning, and comparison against the existing GPU PCG solver.

The target pressure equation is the existing variable-coefficient form:

```text
A(p) = div( beta * grad(p) )

beta = dt / rho
```

## 3.1 Implemented Solvers

- [x] GPU PCG pressure solver
- [x] Standalone geometric multigrid pressure solver
- [x] GMG-preconditioned pressure solver

## 3.2 Validation

TODO:

- [ ] Verify GMG residual reduction per V-cycle
- [ ] Compare final pressure fields from GPU PCG, GMG, and GMG-preconditioned solver
- [ ] Compare corrected velocity divergence after projection
- [ ] Compare pressure-solver iteration counts
- [ ] Compare total pressure-solve time per time step
- [ ] Test on lid-driven cavity, dam-break, static droplet, and high-density-ratio cases
- [ ] Confirm that pressure null-space handling is consistent across solvers
- [ ] Confirm that pressure boundary conditions are applied consistently on all GMG levels

## 3.3 GMG Tuning

TODO:

- [ ] Tune the number of pre-smoothing steps
- [ ] Tune the number of post-smoothing steps
- [ ] Tune the weighted Jacobi relaxation factor
- [ ] Compare weighted Jacobi and red-black Gauss-Seidel if implemented
- [ ] Tune restriction and prolongation operators
- [ ] Test arithmetic vs harmonic restriction of beta
- [ ] Measure convergence factor per V-cycle
- [ ] Decide whether standalone GMG, GMG-preconditioned solver, or GPU PCG should be the default

Recommended initial smoother setting:

```text
weighted Jacobi
omega = 0.6 to 0.8
```

## 3.4 GMG Robustness

TODO:

- [ ] Verify GMG behavior at high density ratio
- [ ] Verify GMG behavior with strong alpha gradients
- [ ] Check convergence in dam-break with air/water density ratio
- [ ] Check convergence in surface-tension-dominated cases
- [ ] Check whether coarse-level operators remain stable with variable coefficients
- [ ] Confirm that the reference pressure or pressure mean removal is handled correctly on every level

---

# Phase 4: Higher-Order Momentum Advection

The current momentum advection scheme is first-order upwind. It is robust, but numerically diffusive. For jet, splash, and crown simulations, a higher-order scheme is needed.

## MUSCL-TVD

TODO:

- [ ] Organize the current upwind advection as `UpwindMomentum`
- [ ] Implement MUSCL reconstruction for GPU execution
- [ ] Implement the minmod limiter
- [ ] Implement the van Leer limiter
- [ ] Implement the MC limiter
- [ ] Compare upwind and MUSCL-TVD using the dam-break problem
- [ ] Test the combination of WLIC and MUSCL-TVD
- [ ] Consider conservative momentum transport if needed

Limiter examples:

```text
minmod:
    psi(r) = max(0, min(1, r))

van Leer:
    psi(r) = (r + |r|) / (1 + |r|)
```

Recommended order:

```text
1. minmod
2. van Leer
3. MC
```

---

# Phase 5: Viscous Term Correction

The current viscous term is an approximation. It is acceptable for single-phase constant-viscosity flows, but for VOF flows with variable viscosity, the viscous term should be written as the divergence of the viscous stress tensor.

Target form:

```text
viscous force = div[ mu * (grad(u) + grad(u)^T) ]
```

For the x-momentum equation:

```text
d/dx [ 2 mu du/dx ]
+
d/dy [ mu (du/dy + dv/dx) ]
```

For the y-momentum equation:

```text
d/dx [ mu (dv/dx + du/dy) ]
+
d/dy [ 2 mu dv/dy ]
```

TODO:

- [ ] Identify the exact approximation currently used for the viscous term
- [ ] Confirm that the current form matches `nu * laplacian(u)` for constant viscosity
- [ ] Replace the viscous term with the variable-viscosity stress-divergence form
- [ ] Evaluate the x-momentum viscous term on u-faces
- [ ] Evaluate the y-momentum viscous term on v-faces
- [ ] Organize mu interpolation to faces
- [ ] Organize mu interpolation to cell corners
- [ ] Implement the corrected viscous term as GPU kernels
- [ ] Test stability at high viscosity ratios
- [ ] Compare old and new viscous terms using the dam-break problem

---

# Phase 6: Surface Tension Validation and Improvement

Surface tension is now implemented. The remaining tasks are validation, robustness improvement, and reduction of spurious currents.

## 6.1 Implemented

- [x] Surface tension model
- [x] Surface tension force computation
- [x] Surface tension coupling to the flow solver

## 6.2 Validation

TODO:

- [ ] Add a static droplet test
- [ ] Verify Laplace pressure
- [ ] Measure spurious currents
- [ ] Compare results with and without surface tension
- [ ] Check sensitivity to grid resolution
- [ ] Check sensitivity to density ratio
- [ ] Check sensitivity to surface tension coefficient `sigma`
- [ ] Check interaction with CFL-based variable time step and alpha substepping
- [ ] Add capillary time-step diagnostics if needed

Laplace pressure targets:

```text
2D circular droplet:
    pressure jump = sigma / R

3D spherical droplet:
    pressure jump = 2 * sigma / R
```

## 6.3 Curvature and Force Improvements

TODO:

- [ ] Check curvature field near the interface
- [ ] Log maximum and minimum curvature
- [ ] Try alpha smoothing
- [ ] Try curvature smoothing
- [ ] Evaluate surface tension force on faces
- [ ] Make pressure-gradient and surface-tension discretizations consistent
- [ ] Implement or investigate a balanced-force formulation
- [ ] Investigate height-function curvature

Curvature estimate:

```text
normal = grad(alpha) / (|grad(alpha)| + eps)

kappa = - div(normal)
```

CSF form:

```text
surface tension force = sigma * kappa * grad(alpha)
```

---

# Phase 7: Additional Validation Tests

## Already tested

- [x] Lid-driven cavity
- [x] OpenFOAM comparison
- [x] Dam-break
- [x] Zalesak slotted disk
- [x] Upwind / THINC / WLIC comparison

## Future validation cases

- [ ] Static droplet
- [ ] Laplace pressure test
- [ ] Spurious-current test
- [ ] Rising bubble
- [ ] Rayleigh-Taylor instability
- [ ] High-density-ratio dam-break
- [ ] Capillary wave
- [ ] Oscillating droplet
- [ ] Shear flow with interface

---

# Phase 8: Preparation for 3D

The long-term target is 3D water crown / milk crown simulation.

## 8.0 3D Boundary-Cell Attribute System Highest Priority

The current 3D refactoring should prioritize boundary handling based on per-cell attributes.
Each cell should know whether it is a normal fluid cell, ghost/solid cell, or adjacent to a physical boundary.
This boundary metadata should be used consistently by boundary-condition kernels, divergence/RHS construction, Poisson stencils, velocity correction, VOF transport, viscous terms, surface tension, and later GMG coarse levels.

Recommended initial design:

```cpp
enum class CellFlag : unsigned char{
    Fluid      = 0,
    Ghost      = 1 << 0,
    Solid      = 1 << 1,
    BndXMinus  = 1 << 2,
    BndXPlus   = 1 << 3,
    BndYMinus  = 1 << 4,
    BndYPlus   = 1 << 5,
    BndZMinus  = 1 << 6,
    BndZPlus   = 1 << 7
};
```

If more states are needed later, use `uint16_t` instead of `unsigned char`.
The important point is to keep boundary checks readable and centralized, not scattered as ad-hoc index checks in every kernel.

TODO:

- [ ] Define the 3D cell attribute / cell flag representation
- [ ] Add `cell_flag` or `cell_type` array to `StaggeredGrid`
- [ ] Decide whether flags are stored as `unsigned char`, `uint8_t`, or `uint16_t`
- [ ] Define flags for fluid, ghost, solid/obstacle, and boundary-adjacent cells
- [ ] Define directional boundary-adjacent flags: `x-`, `x+`, `y-`, `y+`, `z-`, `z+`
- [ ] Add a function such as `build_cell_flags_3d()`
- [ ] Initialize flags from domain boundaries and ghost-cell ranges
- [ ] Keep boundary-condition type information separate from geometric flags if possible
- [ ] Use cell flags in 3D velocity boundary-condition kernels
- [ ] Use cell flags in 3D pressure / Neumann boundary-condition handling
- [ ] Use cell flags in 3D divergence and Poisson RHS construction
- [ ] Use cell flags in the 3D variable-coefficient Poisson stencil
- [ ] Use cell flags in the 3D velocity correction step
- [ ] Use cell flags in 3D VOF transport so that boundary faces do not create invalid fluxes
- [ ] Use cell flags in the 3D viscous stress-divergence operator
- [ ] Use cell flags in 3D surface-tension force placement if needed
- [ ] Add small-grid debug tests such as `Nx=2, Ny=2, Nz=2` and `Nx=4, Ny=3, Nz=2`
- [ ] Print or dump `cell_flag` for a small grid to verify boundary-adjacent cells
- [ ] Confirm that kernels no longer need scattered hard-coded boundary index checks
- [ ] Extend the same flag logic to embedded boundaries later if needed
- [ ] Decide how cell flags are restricted/coarsened for GMG levels

Priority: highest. This should be completed before large-scale 3D solver validation, because the 3D divergence, Poisson operator, velocity correction, and boundary-condition code will all depend on it.

TODO:

- [ ] Complete the 3D boundary-cell attribute system described in Phase 8.0
- [ ] Decide the 3D MAC-grid layout
- [ ] Add the z-velocity component `w`
- [ ] Design z-face arrays
- [ ] Organize the 3D divergence stencil
- [ ] Organize the 3D gradient stencil
- [ ] Organize the 3D variable-coefficient Poisson stencil
- [ ] Design 3D THINC/WLIC
- [ ] Design the 3D viscous stress-divergence operator
- [ ] Design 3D curvature calculation
- [ ] Extend surface tension to 3D
- [ ] Implement 3D CFL calculation
- [ ] Estimate GPU memory usage for 3D cases
- [ ] Decide the 3D output format

3D CFL estimate:

```text
local_adv =
    max(|u_left|, |u_right|)/dx
  + max(|v_back|, |v_front|)/dy
  + max(|w_bottom|, |w_top|)/dz
```

---

# Phase 9: Long-Running Simulation Utilities

Large 3D free-surface simulations require robust output, restart, and logging utilities.

TODO:

- [ ] Add restart functionality
- [ ] Add binary output
- [ ] Add a parameter file
- [ ] Add a log file
- [ ] Save simulation conditions automatically
- [ ] Save the git commit hash
- [ ] Add output-interval settings
- [ ] Add restart-interval settings
- [ ] Add backup output on abnormal termination
- [ ] Transfer only output data from device to host
- [ ] Investigate lighter ParaView-compatible output formats

---

# Phase 10: Debugging and Quality Control

The solver is now GPU-based, so debugging utilities are important.

TODO:

- [ ] Add `CUDA_CHECK` to all CUDA API calls
- [ ] In debug mode, check `cudaGetLastError()` after kernels
- [ ] In debug mode, call `cudaDeviceSynchronize()` where needed
- [ ] Create small test cases for `compute-sanitizer`
- [ ] Disable copying for host classes that own raw pointers
- [ ] Initialize all device pointers to `nullptr`
- [ ] Set device pointers to `nullptr` after `cudaFree`
- [ ] Log array sizes and pitch values at startup
- [ ] Clearly document the valid index range of each kernel

---

# Current Short-Term Priorities

1. Complete the 3D boundary-cell attribute system for boundary-adjacent cells.
2. Add `cell_flag` / `cell_type` to `StaggeredGrid` and initialize it for 3D ghost cells and physical boundaries.
3. Use the cell attributes in 3D boundary-condition kernels.
4. Use the cell attributes in 3D divergence, Poisson RHS, Poisson stencil, and velocity correction.
5. Use the cell attributes in VOF transport and momentum/viscous updates near boundaries.
6. Decide how the cell flags should be restricted or rebuilt on GMG coarse levels.
7. Validate the new 3D boundary handling on very small grids with printed flags.
8. Validate the new 3D boundary handling using divergence checks and simple projection tests.
9. Implement switching between fixed time step and CFL-based variable time step.
10. Add alpha-substep settings to `SolverConfig` and keep `alpha_substeps = 1` available for comparison.
11. Organize `SolverConfig`.
12. Clarify ownership of `dt`, `Nx`, `Ny`, `Nz`, grid spacing, material properties, and solver parameters.
13. Implement the basic switching structure for numerical methods.
14. Validate and benchmark GPU PCG, GMG, and GMG-preconditioned pressure solvers.
15. Decide which pressure solver should be the default.
16. Add MUSCL-TVD momentum advection.
17. Start with the minmod limiter.
18. Add van Leer and MC limiters.
19. Identify the current viscous-term approximation.
20. Replace the viscous term with the variable-viscosity stress-divergence form.
21. Validate the implemented surface tension model.
22. Add the static droplet test.
23. Measure spurious currents.
24. Investigate a balanced-force surface-tension formulation.
25. Add rising bubble and Rayleigh-Taylor tests.
26. Add restart and binary output.

---

# Design Summary

```text
Already implemented:
    GPU solver
    GPU optimization
    PCG kernel fusion
    WLIC
    CFL-based variable dt
    alpha substepping
    variable-density Poisson
    standalone GMG pressure solver
    GMG-preconditioned pressure solver
    surface tension

Main remaining work:
    3D boundary-cell attribute system
    scheme switching design
    fixed-dt / variable-dt switching
    alpha-substep configuration cleanup
    parameter ownership cleanup
    pressure-solver validation and benchmarking
    higher-order momentum advection
    viscous-term correction
    surface-tension validation and improvement
    additional validation
    restart / binary output
    3D preparation
```
