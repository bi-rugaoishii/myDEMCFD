#include "ConfigLoader.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

using json = nlohmann::json;

[[noreturn]]
void json_error(
    const std::string& path,
    const std::string& message)
{
    throw std::runtime_error(
        "configuration error at \"" +
        path + "\": " + message);
}

const json& require_member(
    const json& object,
    const std::string& key,
    const std::string& parent_path)
{
    if (!object.is_object()) {
        json_error(
            parent_path,
            "must be a JSON object");
    }

    const auto it = object.find(key);

    if (it == object.end()) {
        json_error(
            parent_path + "." + key,
            "required value is missing");
    }

    return *it;
}

const json& require_object(
    const json& parent,
    const std::string& key,
    const std::string& parent_path)
{
    const json& value =
        require_member(parent, key, parent_path);

    if (!value.is_object()) {
        json_error(
            parent_path + "." + key,
            "must be an object");
    }

    return value;
}

const json& require_array(
    const json& parent,
    const std::string& key,
    const std::string& parent_path)
{
    const json& value =
        require_member(parent, key, parent_path);

    if (!value.is_array()) {
        json_error(
            parent_path + "." + key,
            "must be an array");
    }

    return value;
}

std::string read_string(
    const json& parent,
    const std::string& key,
    const std::string& parent_path)
{
    const json& value =
        require_member(parent, key, parent_path);

    if (!value.is_string()) {
        json_error(
            parent_path + "." + key,
            "must be a string");
    }

    return value.get<std::string>();
}

bool read_bool(
    const json& parent,
    const std::string& key,
    const std::string& parent_path)
{
    const json& value =
        require_member(parent, key, parent_path);

    if (!value.is_boolean()) {
        json_error(
            parent_path + "." + key,
            "must be true or false");
    }

    return value.get<bool>();
}

double read_number_value(
    const json& value,
    const std::string& path)
{
    if (!value.is_number()) {
        json_error(
            path,
            "must be a number");
    }

    return value.get<double>();
}

double read_number(
    const json& parent,
    const std::string& key,
    const std::string& parent_path)
{
    const json& value =
        require_member(parent, key, parent_path);

    return read_number_value(
        value,
        parent_path + "." + key);
}

int read_integer_value(
    const json& value,
    const std::string& path)
{
    if (!value.is_number_integer() &&
        !value.is_number_unsigned()) {
        json_error(
            path,
            "must be an integer");
    }

    long long result = 0;

    if (value.is_number_unsigned()) {
        const unsigned long long unsigned_result =
            value.get<unsigned long long>();

        if (unsigned_result >
            static_cast<unsigned long long>(
                std::numeric_limits<int>::max())) {
            json_error(
                path,
                "is outside the range of int");
        }

        result =
            static_cast<long long>(
                unsigned_result);
    } else {
        result = value.get<long long>();
    }

    if (result <
            std::numeric_limits<int>::min() ||
        result >
            std::numeric_limits<int>::max()) {
        json_error(
            path,
            "is outside the range of int");
    }

    return static_cast<int>(result);
}

int read_integer(
    const json& parent,
    const std::string& key,
    const std::string& parent_path)
{
    const json& value =
        require_member(parent, key, parent_path);

    return read_integer_value(
        value,
        parent_path + "." + key);
}

ConfigVec3 read_vec3_value(
    const json& value,
    const std::string& path)
{
    if (!value.is_array()) {
        json_error(
            path,
            "must be an array");
    }

    if (value.size() != 3) {
        json_error(
            path,
            "must contain exactly three numbers");
    }

    ConfigVec3 result;

    result.x = read_number_value(
        value.at(0),
        path + "[0]");

    result.y = read_number_value(
        value.at(1),
        path + "[1]");

    result.z = read_number_value(
        value.at(2),
        path + "[2]");

    return result;
}

ConfigVec3 read_vec3(
    const json& parent,
    const std::string& key,
    const std::string& parent_path)
{
    const json& value =
        require_member(parent, key, parent_path);

    return read_vec3_value(
        value,
        parent_path + "." + key);
}

BoundaryType parse_boundary_type(
    const std::string& value,
    const std::string& path)
{
    if (value == "noslip") {
        return BC_NOSLIP;
    }

    if (value == "slip") {
        return BC_SLIP;
    }

    if (value == "inflow") {
        return BC_INFLOW;
    }

    if (value == "outlet") {
        return BC_OUTLET;
    }

    json_error(
        path,
        "unknown boundary type \"" +
        value + "\"");
}

int parse_axis(
    const std::string& value,
    const std::string& path)
{
    if (value == "x") {
        return AXIS_X;
    }

    if (value == "y") {
        return AXIS_Y;
    }

    if (value == "z") {
        return AXIS_Z;
    }

    json_error(
        path,
        "unknown axis \"" + value + "\"");
}

ConfigBoundarySide parse_boundary_side(
    const std::string& value,
    const std::string& path)
{
    if (value == "min") {
        return ConfigBoundarySide::Min;
    }

    if (value == "max") {
        return ConfigBoundarySide::Max;
    }

    json_error(
        path,
        "boundary side must be \"min\" or \"max\"");
}

HardwareConfig parse_hardware(
    const json& value)
{
    HardwareConfig result;

    result.use_gpu =
        read_bool(value, "use_gpu", "hardware");

    return result;
}

OutputConfig parse_output(
    const json& value)
{
    OutputConfig result;

    result.directory =
        read_string(
            value,
            "directory",
            "output");

    return result;
}

GridConfig parse_grid(
    const json& value)
{
    GridConfig result;

    const json& cells =
        require_array(
            value,
            "cells",
            "grid");

    if (cells.size() != 3) {
        json_error(
            "grid.cells",
            "must contain exactly three integers");
    }

    result.Nx =
        read_integer_value(
            cells.at(0),
            "grid.cells[0]");

    result.Ny =
        read_integer_value(
            cells.at(1),
            "grid.cells[1]");

    result.Nz =
        read_integer_value(
            cells.at(2),
            "grid.cells[2]");

    const ConfigVec3 size =
        read_vec3(
            value,
            "size",
            "grid");

    result.size_x = size.x;
    result.size_y = size.y;
    result.size_z = size.z;

    const ConfigVec3 origin =
        read_vec3(
            value,
            "origin",
            "grid");

    result.origin_x_ = origin.x;
    result.origin_y_ = origin.y;
    result.origin_z_ = origin.z;

    return result;
}


PhaseConfig parse_phase(
    const json& value,
    const std::string& path)
{
    PhaseConfig result;

    result.density =
        read_number(
            value,
            "density",
            path);

    result.viscosity =
        read_number(
            value,
            "viscosity",
            path);

    return result;
}

FluidConfig parse_fluid(
    const json& value)
{
    FluidConfig result;

    result.phase0 =
        parse_phase(
            require_object(
                value,
                "phase0",
                "fluid"),
            "fluid.phase0");

    result.phase1=
        parse_phase(
            require_object(
                value,
                "phase1",
                "fluid"),
            "fluid.liquid");

    result.surface_tension =
        read_number(
            value,
            "surface_tension",
            "fluid");

    result.gravity =
        read_vec3(
            value,
            "gravity",
            "fluid");

    return result;
}

TimeConfig parse_time(
    const json& value)
{
    TimeConfig result;


    result.initial_dt =
        read_number(
            value,
            "initial_dt",
            "time");

    result.maximum_dt =
        read_number(
            value,
            "maximum_dt",
            "time");

    result.end_time =
        read_number(
            value,
            "end_time",
            "time");

    result.cfl =
        read_number(
            value,
            "cfl",
            "time");

    result.vof_cfl =
        read_number(
            value,
            "vof_cfl",
            "time");

    result.interval =
        read_number(
            value,
            "interval",
            "time");


    return result;
}

PressureSolverConfig parse_pressure_solver(
    const json& value)
{
    PressureSolverConfig result;

    result.gmg_levels =
        read_integer(
            value,
            "gmg_levels",
            "pressure_solver");

    result.tol = read_number(value, "tol","pressure_solver");
    printf("tol = %f\n",result.tol);

    return result;
}

BoundaryConditionConfig parse_boundary_condition(
    const json& value,
    std::size_t index)
{
    const std::string path =
        "boundary_conditions[" +
        std::to_string(index) + "]";

    BoundaryConditionConfig result;

    result.id =
        read_integer(
            value,
            "id",
            path);

    result.type =
        parse_boundary_type(
            read_string(
                value,
                "type",
                path),
            path + ".type");

    // 速度を省略した場合はゼロ。
    if (value.contains("velocity")) {
        result.velocity =
            read_vec3(
                value,
                "velocity",
                path);
    }

    return result;
}

BoundaryFaceConfig parse_boundary_face(
    const json& value,
    std::size_t index)
{
    const std::string path =
        "boundary_faces[" +
        std::to_string(index) + "]";

    BoundaryFaceConfig result;

    result.axis =
        (AXIS)parse_axis(
            read_string(
                value,
                "axis",
                path),
            path + ".axis");

    result.side =
        parse_boundary_side(
            read_string(
                value,
                "side",
                path),
            path + ".side");

    result.boundary_id =
        read_integer(
            value,
            "boundary_id",
            path);

    return result;
}

InitialConditionConfig parse_initial_condition(
    const json& value,
    std::size_t index){

    const std::string path =
        "initial_conditions[" +
        std::to_string(index) + "]";

    InitialConditionConfig result;

    const std::string type =
        read_string(
            value,
            "type",
            path);


    if (type == "vof_box") {
        result.type =
            InitialConditionType::VofBox;

        result.alpha =
            read_number(
                    value,
                    "alpha",
                    path);

        result.minimum =
            read_vec3(
                    value,
                    "minimum",
                    path);

        result.maximum =
            read_vec3(
                    value,
                    "maximum",
                    path);


        return result;
    }

    if (type == "vof_sphere") {
        result.type =
            InitialConditionType::VofSphere;

        result.vini =
            read_number(
                    value,
                    "vini",
                    path);

        result.alpha =
            read_number(
                    value,
                    "alpha",
                    path);

        result.radius =
            read_number(
                    value,
                    "radius",
                    path);

        result.center =
            read_vec3(
                    value,
                    "center",
                    path);

        return result;
    }

    if (type == "solid_box") {
        result.type =
            InitialConditionType::SolidBox;

        result.minimum =
            read_vec3(
                    value,
                    "minimum",
                    path);

        result.maximum =
            read_vec3(
                    value,
                    "maximum",
                    path);


        result.solid_id =
            read_integer(
                    value,
                    "solid_id",
                    path);

        return result;
    }

    if (type == "solid_cylinder_inv") {
        result.type =
            InitialConditionType::SolidCylinderInv;

        result.center =
            read_vec3(
                    value,
                    "center",
                    path);

        result.radius =
            read_number(
                    value,
                    "radius",
                    path);

        result.solid_id =
            read_integer(
                    value,
                    "solid_id",
                    path);

        return result;
    }

    if (type == "solid_cylinder") {
        result.type =
            InitialConditionType::SolidCylinder;

        result.center =
            read_vec3(
                    value,
                    "center",
                    path);

        result.radius =
            read_number(
                    value,
                    "radius",
                    path);

        result.solid_id =
            read_integer(
                    value,
                    "solid_id",
                    path);

        return result;
    }


    json_error(
            path + ".type",
            "unknown initial condition type \"" +
            type + "\"");
}

SimulationConfig parse_root(
        const json& root)
{
    if (!root.is_object()) {
        json_error(
                "$",
                "root must be a JSON object");
    }

    SimulationConfig result;

    result.hardware =
        parse_hardware(
                require_object(
                    root,
                    "hardware",
                    "$"));

    result.output =
        parse_output(
                require_object(
                    root,
                    "output",
                    "$"));

    result.grid =
        parse_grid(
                require_object(
                    root,
                    "grid",
                    "$"));


    result.fluid =
        parse_fluid(
                require_object(
                    root,
                    "fluid",
                    "$"));

    result.time =
        parse_time(
                require_object(
                    root,
                    "time",
                    "$"));

    result.pressure_solver =
        parse_pressure_solver(
                require_object(
                    root,
                    "pressure_solver",
                    "$"));

    const json& boundary_conditions =
        require_array(
                root,
                "boundary_conditions",
                "$");

    result.boundary_conditions.reserve(
            boundary_conditions.size());

    for (std::size_t i = 0;
            i < boundary_conditions.size();
            ++i) {
        result.boundary_conditions.push_back(
                parse_boundary_condition(
                    boundary_conditions.at(i),
                    i));
    }

    const json& boundary_faces =
        require_array(
                root,
                "boundary_faces",
                "$");

    result.boundary_faces.reserve(
            boundary_faces.size());

    for (std::size_t i = 0;
            i < boundary_faces.size();
            ++i) {
        result.boundary_faces.push_back(
                parse_boundary_face(
                    boundary_faces.at(i),
                    i));
    }

    const json& initial_conditions =
        require_array(
                root,
                "initial_conditions",
                "$");

    result.initial_conditions.reserve(
            initial_conditions.size());

    for (std::size_t i = 0;
            i < initial_conditions.size();
            ++i) {
        result.initial_conditions.push_back(
                parse_initial_condition(
                    initial_conditions.at(i),
                    i));
    }

    return result;
}

} // namespace

SimulationConfig load_simulation_config(
        const std::string& filename)
{
    std::ifstream file(filename);

    if (!file) {
        throw std::runtime_error(
                "failed to open configuration file: " +
                filename);
    }

    try {
        json root;
        file >> root;

        SimulationConfig config =
            parse_root(root);

        config.validate();

        return config;
    }
    catch (const nlohmann::json::parse_error& error) {
        throw std::runtime_error(
                "JSON parse error in \"" +
                filename + "\": " + error.what());
    }
    catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
                "JSON error in \"" +
                filename + "\": " + error.what());
    }
}
