#include "SimulationSetup.h"

#include "config/SimulationConfig.h"
#include "SMACSolver.h"

#include <stdexcept>

namespace{

int get_boundary_plane_index(
    const BoundaryFaceConfig& face,
    const GridConfig& grid)
{
    if (face.side == ConfigBoundarySide::Min) {
        return 1;
    }

    switch (face.axis) {
    case AXIS_X:
        return grid.Nx + 1;

    case AXIS_Y:
        return grid.Ny + 1;

    case AXIS_Z:
        return grid.Nz + 1;

    default:
        throw std::runtime_error(
            "invalid boundary axis");
    }
}

} // anonymous namespace

namespace SimulationSetup
{

void apply_boundary_conditions(
    SMACSolver& solver,
    const SimulationConfig& config)
{
    for (const auto& boundary :
         config.boundary_conditions) {

        solver.grid_.bc_.set_bctype(
            boundary.id,
            boundary.type);

        solver.grid_.bc_.set_boundary_velocity(
            boundary.id,
            boundary.velocity.x,
            boundary.velocity.y,
            boundary.velocity.z);
    }

    for (const auto& face :
         config.boundary_faces) {

        const int index =
            get_boundary_plane_index(
                face,
                config.grid);

        solver.grid_.set_boundary_id(
            face.axis,
            face.boundary_id,
            index);
    }
}

void apply_initial_conditions(
    SMACSolver& solver,
    const SimulationConfig& config)
{
    for (const auto& initial :
         config.initial_conditions) {

        switch (initial.type) {
        case InitialConditionType::VofBox:

            solver.grid_.place_vof(
                initial.minimum.x,
                initial.maximum.x,
                initial.minimum.y,
                initial.maximum.y,
                initial.minimum.z,
                initial.maximum.z,
                initial.alpha);

            break;

        case InitialConditionType::VofSphere:

            solver.set_sphere(
                initial.center.x,
                initial.center.y,
                initial.center.z,
                initial.radius,
                initial.vini);

            break;


        case InitialConditionType::SolidBox:

            solver.grid_.place_solid(
                initial.minimum.x,
                initial.maximum.x,
                initial.minimum.y,
                initial.maximum.y,
                initial.minimum.z,
                initial.maximum.z,
                initial.solid_id);

            break;


        case InitialConditionType::SolidCylinder:
            break;

        case InitialConditionType::SolidCylinderInv:
            break;

        default:
            throw std::runtime_error(
                "unsupported initial condition");
        }
    }
}

void apply_initial_conditions_device(
    G_SMACSolver& solver,
    const SimulationConfig& config)
{
    for (const auto& initial :
         config.initial_conditions) {

        switch (initial.type) {
        case InitialConditionType::VofBox:

            break;

        case InitialConditionType::VofSphere:
            break;

        case InitialConditionType::SolidBox:

            break;

        case InitialConditionType::SolidCylinder:
            printf("creating cylinder ibm\n");
            solver.make_cylinder_ibm(initial.center.x,initial.center.y,initial.center.z,initial.radius);
            printf("creating cylinder done\n");
            break;

        case InitialConditionType::SolidCylinderInv:
            printf("creating cylinder inv ibm\n");
            solver.make_cylinder_ibm_inv(initial.center.x,initial.center.y,initial.center.z,initial.radius);
            printf("creating cylinder done\n");
            break;


        default:
            throw std::runtime_error(
                    "unsupported initial condition");
        }
    }
}

} // namespace SimulationSetup
