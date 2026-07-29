#pragma once
#include "SMACSolver.h"
#include "G_SMACSolver.h"
#include <cub/cub.cuh>

struct SimulationConfig;

namespace SimulationSetup{

void apply_boundary_conditions(
    SMACSolver& solver,
    const SimulationConfig& config);

void apply_initial_conditions(
    SMACSolver& solver,
    const SimulationConfig& config);

void apply_initial_conditions_device(
    G_SMACSolver& solver,
    const SimulationConfig& config);
} // namespace SimulationSetup
