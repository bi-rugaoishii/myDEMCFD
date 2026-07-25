#pragma once

enum FaceType: unsigned char{
    F_INTERIOR,
    F_BOUNDARY,
    F_GHOST,
    F_NEAR_BOUNDARY
};

enum BoundaryType: unsigned char{
    BC_NOSLIP,
    BC_SLIP,
    BC_INFLOW,
    BC_OUTFLOW,
    BC_OUTLET
};

enum CellType: unsigned char{
    C_INTERIOR,
    C_NEAR_BOUNDARY,
    C_GHOST,
    C_SOLID
};

enum PCG_TYPES: unsigned char{
    STANDARD_PCG,
    PURENEUMANN_STANDARD_PCG,
    GMG_PCG,
    PURENEUMANN_GMG_PCG
};

enum AXIS: unsigned char{
    AXIS_X,
    AXIS_Y,
    AXIS_Z
};
