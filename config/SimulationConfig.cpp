#include "SimulationConfig.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

[[noreturn]]
void config_error(const std::string& message)
{
    throw std::runtime_error(
        "configuration error: " + message);
}

void require_finite(
    double value,
    const std::string& name)
{
    if (!std::isfinite(value)) {
        config_error(name + " must be finite");
    }
}

void require_positive(
    double value,
    const std::string& name)
{
    require_finite(value, name);

    if (value <= 0.0) {
        config_error(name + " must be greater than zero");
    }
}

void require_nonnegative(
    double value,
    const std::string& name)
{
    require_finite(value, name);

    if (value < 0.0) {
        config_error(name + " must be nonnegative");
    }
}

void validate_vec3(
    const ConfigVec3& value,
    const std::string& name)
{
    require_finite(value.x, name + "[0]");
    require_finite(value.y, name + "[1]");
    require_finite(value.z, name + "[2]");
}

} // namespace

void SimulationConfig::validate() const
{
    if (grid.Nx <= 0) {
        config_error("grid.cells[0] must be positive");
    }

    if (grid.Ny <= 0) {
        config_error("grid.cells[1] must be positive");
    }

    if (grid.Nz <= 0) {
        config_error("grid.cells[2] must be positive");
    }

    require_positive(grid.size_x, "grid.size[0]");
    require_positive(grid.size_y, "grid.size[1]");
    require_positive(grid.size_z, "grid.size[2]");


    require_positive(
        fluid.phase0.density,
        "fluid.phase0.density");

    require_positive(
        fluid.phase1.density,
        "fluid.phase1.density");

    require_nonnegative(
        fluid.phase0.viscosity,
        "fluid.phase0.viscosity");

    require_nonnegative(
        fluid.phase1.viscosity,
        "fluid.phase1.viscosity");

    require_nonnegative(
        fluid.surface_tension,
        "fluid.surface_tension");

    validate_vec3(
        fluid.gravity,
        "fluid.gravity");

    require_positive(
        time.initial_dt,
        "time.initial_dt");

    require_positive(
        time.maximum_dt,
        "time.maximum_dt");

    require_positive(
        time.end_time,
        "time.end_time");

    require_positive(
        time.cfl,
        "time.cfl");

    require_positive(
        time.interval,
        "time.interval");

    require_positive(
        time.vof_cfl,
        "time.vof_cfl");

    if (time.initial_dt > time.maximum_dt) {
        config_error(
            "time.initial_dt must not exceed "
            "time.maximum_dt");
    }

    if (pressure_solver.gmg_levels <= 0) {
        config_error(
            "pressure_solver.gmg_levels must be positive");
    }


    if (output.directory.empty()) {
        config_error(
            "output.directory must not be empty");
    }

    std::set<int> defined_boundary_ids;

    for (const auto& boundary :
         boundary_conditions) {

        if (boundary.id <= 0) {
            config_error(
                "boundary condition IDs must be positive");
        }

        if (!defined_boundary_ids
                 .insert(boundary.id)
                 .second) {
            config_error(
                "boundary condition ID " +
                std::to_string(boundary.id) +
                " is defined more than once");
        }

        validate_vec3(
            boundary.velocity,
            "boundary velocity");
    }

    std::set<std::pair<int, int>>
        assigned_boundary_faces;

    for (const auto& face : boundary_faces) {
        if (face.axis != AXIS_X &&
            face.axis != AXIS_Y &&
            face.axis != AXIS_Z) {
            config_error(
                "boundary face has an invalid axis");
        }

        if (defined_boundary_ids.find(
                face.boundary_id) ==
            defined_boundary_ids.end()) {
            config_error(
                "boundary face refers to undefined "
                "boundary ID " +
                std::to_string(face.boundary_id));
        }

        const int side =
            face.side == ConfigBoundarySide::Min
                ? 0
                : 1;

        const std::pair<int, int> key{
            face.axis,
            side
        };

        if (!assigned_boundary_faces
                 .insert(key)
                 .second) {
            config_error(
                "the same outer boundary face is "
                "assigned more than once");
        }
    }

    for (const auto& initial :
         initial_conditions) {

        validate_vec3(
            initial.minimum,
            "initial condition minimum");

        validate_vec3(
            initial.maximum,
            "initial condition maximum");

        if (initial.minimum.x >
                initial.maximum.x ||
            initial.minimum.y >
                initial.maximum.y ||
            initial.minimum.z >
                initial.maximum.z) {
            config_error(
                "initial condition minimum must not "
                "exceed maximum");
        }

        if (initial.type ==
            InitialConditionType::VofBox) {

            require_finite(
                initial.alpha,
                "initial condition alpha");

            if (initial.alpha < 0.0 ||
                initial.alpha > 1.0) {
                config_error(
                    "VOF alpha must be between "
                    "zero and one");
            }
        }

        if (initial.type ==
            InitialConditionType::SolidBox) {

            if (initial.solid_id <= 0) {
                config_error(
                    "solid_id must be positive");
            }
        }
    }
}

int SimulationConfig::get_num_boundary_ids() const
{
    int maximum_id = 0;

    for (const auto& boundary :
         boundary_conditions) {
        maximum_id =
            std::max(maximum_id, boundary.id);
    }

    // Boundary ID 0も配列内に存在するため+1する。
    return maximum_id + 1;
}

bool SimulationConfig::is_pure_neumann() const
{
    for (const auto& boundary :
         boundary_conditions) {

        if (boundary.type == BC_OUTLET ||
            boundary.type == BC_INFLOW) {
            return false;
        }
    }

    return true;
}

int SimulationConfig::get_alpha_substeps() const
{
    const double ratio =
        time.cfl / time.vof_cfl;

    return std::max(
        1,
        static_cast<int>(std::ceil(ratio)));
}
