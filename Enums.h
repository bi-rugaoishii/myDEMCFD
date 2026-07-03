#pragma once

enum FaceType: unsigned char{
    F_INTERIOR,
    F_WALL_NOSLIP,
    F_WALL_SLIP,
    F_INLET,
    F_OUTLET,
    F_GHOST,
    F_NEAR_BOUNDARY
};

enum CellType: unsigned char{
    C_INTERIOR,
    C_NEAR_BOUNDARY,
    C_GHOST
};
