#pragma once

#include <string>
#include <vector>

#include "../Enums.h"

struct ConfigVec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct HardwareConfig
{
    bool use_gpu = true;
};

struct OutputConfig
{
    std::string directory = "results";
};

struct GridConfig
{
    int Nx = 1;
    int Ny = 1;
    int Nz = 1;

    double size_x = 1.0;
    double size_y = 1.0;
    double size_z = 1.0;

    double dx() const
    {
        return size_x / static_cast<double>(Nx);
    }

    double dy() const
    {
        return size_y / static_cast<double>(Ny);
    }

    double dz() const
    {
        return size_z / static_cast<double>(Nz);
    }
};


struct PhaseConfig
{
    double density = 1.0;
    double viscosity = 0.0;
};

struct FluidConfig
{
    PhaseConfig phase0;
    PhaseConfig phase1;

    double surface_tension = 0.0;
    ConfigVec3 gravity;
};

struct TimeConfig
{

    double initial_dt = 1.0e-5;
    double maximum_dt = 1.0e-2;
    double end_time = 1.0;
    double interval = 0.05;

    double cfl = 0.4;
    double vof_cfl = 0.2;
};

struct PressureSolverConfig
{
    int gmg_levels = 4;
    double tol = 1e-8;
};

struct BoundaryConditionConfig
{
    int id = 0;
    BoundaryType type = BC_NOSLIP;
    ConfigVec3 velocity;
};

enum class ConfigBoundarySide
{
    Min,
    Max
};

struct BoundaryFaceConfig
{
    AXIS axis = AXIS_X;
    ConfigBoundarySide side = ConfigBoundarySide::Min;
    unsigned char boundary_id = 0;
};

enum class InitialConditionType
{
    VofBox,
    VofSphere,
    SolidBox,
    SolidCylinder,
    SolidCylinderInv,
};

struct InitialConditionConfig
{
    InitialConditionType type = InitialConditionType::VofBox;

    ConfigVec3 minimum;

    ConfigVec3 maximum{1.0,1.0,1.0};


    ConfigVec3 center;
    double radius;
    double vini;

    double alpha = 0.0;
    unsigned char solid_id = 0;
};

struct SimulationConfig
{
    HardwareConfig hardware;
    OutputConfig output;
    GridConfig grid;
    FluidConfig fluid;
    TimeConfig time;
    PressureSolverConfig pressure_solver;

    std::vector<BoundaryConditionConfig> boundary_conditions;
    std::vector<BoundaryFaceConfig> boundary_faces;
    std::vector<InitialConditionConfig> initial_conditions;

    void validate() const;

    int get_num_boundary_ids() const;
    bool is_pure_neumann() const;
    int get_alpha_substeps() const;
};
