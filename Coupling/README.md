# CFDDEM TODO

This document summarizes planned improvements for the CFD–DEM solver that are not currently under active development, but should be addressed in the future.

## Future TODO

### 1. Virtual Mass Force

Introduce the virtual mass force into the CFD–DEM coupling model.

Planned items:

- Evaluate the fluid material acceleration `Du_f/Dt`
- Add the virtual mass force on DEM particles
- Apply the corresponding reaction force to the fluid phase
- Check consistency with the existing drag and pressure-gradient force models
- Validate the model using particle-settling test cases

---

### 2. Checkpoint / Restart

Add a restart capability so that long simulations can be resumed from saved states.

Planned items:

- Save CFD fields
- Save DEM particle data
- Save simulation time and step number
- Save quantities required for time integration and coupling
- Restore the full simulation state from a checkpoint
- Verify that restarted calculations reproduce uninterrupted calculations

---

### 3. Runtime Monitoring with ttyplot

Investigate a lightweight runtime monitoring system using `ttyplot`.

Possible quantities to monitor:

- CFL number
- Linear solver residual
- Pressure solver iteration count
- `alpha_min` / `alpha_max`
- `epsilon_min` / `epsilon_max`
- Conserved liquid volume `∫ epsilon * alpha dV`
- CFD / DEM timestep
- Representative particle velocity
- Number of active particles

The goal is to monitor long-running simulations directly from the terminal without requiring a GUI visualization tool.

---

### 4. Independent Linear Algebra Solver

Separate the linear algebra solver from the CFD solver implementation.

Planned design:

- Separate matrix / coefficient construction from the iterative solver
- Separate RHS construction from the solver
- Provide reusable solver interfaces
- Allow different iterative methods to be selected independently

Possible solvers:

- Jacobi
- PCG
- GMG-preconditioned PCG

The linear algebra layer should eventually be reusable for equations other than the pressure Poisson equation.

---

## Current Priority

These items are intentionally deferred for now.

The current development priority is to stabilize the two-way CFD–DEM–VOF coupling, especially:

- VOF boundedness
- Conservation of `epsilon * alpha`
- Stable redistribution / correction
- Consistent CFD–DEM coupling

After the current VOF and two-way coupling issues are resolved, the items listed above can be addressed one by one.
