# CFD/VOF Solver TODO

This repository contains a custom CFD/VOF solver under active development.
The solver started as a 2D structured-grid, staggered/MAC-grid incompressible flow solver with VOF free-surface tracking, GPU acceleration, variable-density pressure projection, geometric multigrid pressure solvers, and surface-tension modeling.
The solver has now been extended to 3D free-surface simulations such as water crown and milk crown problems.

The current priority is JSON-based simulation configuration, RDF-based curvature calculation, embedded-boundary support, validation, robustness, pressure-solver benchmarking, and preparation for long-running 3D simulations.

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
- [x] Pointer-based CUDA kernel argument passing with grid self pointer
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

- [ ] Read calculation conditions from a JSON configuration file
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

1. Add JSON configuration-file loading for calculation conditions.
2. Add RDF-based curvature calculation for surface tension.
3. Add embedded-boundary support using cell/face attributes, including STL geometry support.
4. Validate the 3D solver with small-grid tests, divergence checks, and simple projection tests.
5. Clean up `SolverConfig`, parameter ownership, fixed/CFL time-step switching, and scheme selection.
6. Validate PCG, GMG, and GMG-preconditioned pressure solvers in 3D.
7. Validate the stress-divergence viscous term for constant and variable viscosity cases.
8. Validate and improve surface tension using static droplet, Laplace pressure, and spurious-current tests.
9. Add additional benchmark cases.
10. Add restart, binary output, and long-running simulation logging.

---

# Remaining Work Details

The sections below follow the same order as `Main Remaining Work`.
Completed items are kept later in `Completed Major Work`, not as active phases.

## 1. JSON Configuration File for Calculation Conditions

Add a JSON configuration file so simulation conditions can be changed without recompiling. This should cover grid size, domain size, time-step settings, material properties, solver options, output settings, and numerical scheme choices.

TODO:

- [ ] Define the JSON schema for calculation conditions
- [ ] Load JSON settings into `SolverConfig`, `MaterialConfig`, and grid/domain settings
- [ ] Validate required fields and provide safe defaults where appropriate
- [ ] Save a copy of the used configuration with simulation outputs

---

## 2. RDF-Based Curvature Calculation for Surface Tension

Add RDF-based curvature calculation to improve curvature quality and reduce spurious currents compared with direct alpha-gradient curvature.

TODO:

- [ ] Add RDF reconstruction near the interface
- [ ] Compute curvature from the reconstructed distance field
- [ ] Compare alpha-gradient curvature and RDF curvature
- [ ] Check effect on static droplet, Laplace pressure, and milk-crown behavior

---

## 3. Embedded Boundary Method

Add embedded-boundary support on the structured grid. The existing cell/face attribute system should be reused so that wall handling, Poisson stencils, VOF transport, and viscous terms remain centralized.

TODO:

- [ ] Add embedded-boundary geometry representation and cell/face classification
- [ ] Compute boundary volume/area information needed by the flow solver
- [ ] Apply embedded-boundary conditions to velocity, pressure, VOF, and viscous terms
- [ ] Validate with simple internal-wall and obstacle cases

---

## 4. 3D Validation and Robustness Checks

The solver has been extended from 2D to 3D. The remaining work is validation, boundary robustness, memory/output design, and preparation for water crown / milk crown simulations.

TODO:

- [ ] Validate small-grid indexing and boundary flags using printed/debug outputs
- [ ] Check divergence and projection behavior in simple 3D cases
- [ ] Validate 3D VOF, curvature, and surface-tension behavior
- [ ] Estimate memory usage and finalize output format for large 3D cases
- [ ] Check robustness of boundary flags on GMG levels if needed

---

## 5. Solver Configuration Cleanup

Clean up solver configuration and scalar parameter ownership so the growing 3D solver remains maintainable.

TODO:

- [ ] Move numerical options into `SolverConfig`
- [ ] Connect JSON-loaded settings to `SolverConfig` and `MaterialConfig`
- [ ] Clarify ownership of grid spacing, time variables, material properties, solver tolerances, and scheme parameters
- [ ] Decide how scalar parameters should be passed alongside the pointer-based grid argument

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

---

## 6. Fixed-dt / Variable-dt Mode Switching

CFL-based variable time stepping is implemented, but fixed time step mode should remain available for debugging, reproducibility, and controlled comparisons.

TODO:

- [ ] Add fixed time step mode to `SolverConfig`
- [ ] Keep CFL mode as the default for production runs if appropriate
- [ ] Check consistency between physical-time output and both time-step modes
- [ ] Keep alpha substepping configurable in both modes

---

## 7. Numerical Scheme Switching

Organize numerical method selection so schemes can be switched cleanly without scattering conditionals through the code.

TODO:

- [ ] Add enum-based switching for pressure solver, momentum scheme, VOF scheme, limiter, viscous term, and surface-tension model
- [ ] Organize pressure solvers behind a common interface
- [ ] Organize momentum and VOF schemes so they can be selected from `SolverConfig`
- [ ] Avoid virtual calls inside cell loops and CUDA kernels

---

## 8. Pressure-Solver Validation and Benchmarking

GPU PCG, standalone GMG, GMG-preconditioned pressure solvers, and direction-selective GMG coarsening are implemented. The remaining work is validation, tuning, and choosing the default.

TODO:

- [ ] Compare convergence, divergence after projection, pressure fields, and total solve time
- [ ] Test robustness for dam-break, static droplet, high-density-ratio, and 3D cases
- [ ] Tune smoother, restriction/prolongation, coefficient restriction, and pressure null-space handling
- [ ] Decide the default pressure solver

Target operator:

```text
A(p) = div( beta * grad(p) )
beta = dt / rho
```

---

## 9. Stress-Divergence Viscous-Term Validation

The variable-viscosity viscous term has been replaced with the stress-divergence form. The remaining work is validation and tuning near interfaces and boundaries.

TODO:

- [ ] Validate the term for constant-viscosity cases against the old Laplacian form
- [ ] Test stability and accuracy at high viscosity ratio
- [ ] Check interaction with 3D boundary-cell attributes and free-surface cells

---

## 10. Surface-Tension Validation and Improvement

Surface tension is implemented. The next work is validation, curvature improvement, and spurious-current reduction.

TODO:

- [ ] Add static droplet, Laplace pressure, and spurious-current tests
- [ ] Check grid-resolution, density-ratio, and sigma sensitivity
- [ ] Improve balanced-force consistency and force placement if needed
- [ ] Compare results before and after RDF curvature is added

Laplace pressure targets:

```text
2D circular droplet:    pressure jump = sigma / R
3D spherical droplet:   pressure jump = 2 * sigma / R
```

---

## 11. Additional Benchmark Cases

Add benchmark cases that are useful for free-surface, high-density-ratio, and surface-tension behavior.

TODO:

- [ ] Static droplet / Laplace pressure
- [ ] Spurious-current test
- [ ] Rising bubble
- [ ] Rayleigh-Taylor instability
- [ ] High-density-ratio dam break
- [ ] Capillary wave or oscillating droplet

---

## 12. Restart, Binary Output, and Long-Running Simulation Utilities

Large 3D simulations need robust output, restart, and logging.

TODO:

- [ ] Add restart functionality
- [ ] Add binary output and lightweight ParaView-compatible output
- [ ] Add parameter/log files and automatic simulation-condition saving
- [ ] Save output/restart intervals and git commit hash
- [ ] Transfer only output data from device to host

---

# Debugging and Quality Control

These are ongoing support tasks rather than main numerical-development phases.

TODO:

- [ ] Add `CUDA_CHECK` and debug-mode kernel error checks
- [ ] Add small compute-sanitizer test cases
- [ ] Check CUDA kernel argument sizes after major grid-structure changes
- [ ] Make raw-pointer-owning classes safer against accidental copying
- [ ] Initialize and clear device pointers consistently
- [ ] Log array sizes and pitches at startup
- [x] Verify key face/cell index ranges in debug checks

---

# Completed Major Work

The following items are implemented and are kept here only as a compact history, not as active phases.

- [x] Pointer-based CUDA kernel argument passing with grid self pointer
- [x] Face/cell indexing cleanup and validation
- [x] MUSCL-TVD momentum advection with van Leer and minmod limiters
- [x] Direction-selective / non-uniform GMG coarsening and validation
- [x] Core 3D MAC-grid extension
- [x] 3D boundary-cell attribute system
- [x] Variable-viscosity stress-divergence implementation
- [x] Flux-direction-based THINC/WLIC switching near boundaries

---

# Design Summary

```text
Already implemented:
    GPU solver
    pointer-based CUDA kernel argument passing with grid self pointer
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
    JSON-based calculation-condition setup
    RDF-based curvature calculation
    embedded boundary method
    validation and robustness of the 3D solver

Main remaining work:
    JSON configuration-file loading for calculation conditions
    RDF-based curvature calculation
    embedded boundary method
    3D validation and robustness checks
    solver configuration cleanup
    fixed-dt / variable-dt mode switching
    numerical scheme switching
    pressure-solver validation and benchmarking
    stress-divergence viscous-term validation
    surface-tension validation and improvement
    additional benchmark cases
    restart, binary output, and long-running simulation utilities
```
