#pragma once

enum FaceType: unsigned char{
    INTERIOR,
    WALL_NOSLIP,
    WALL_SLIP,
    INLET,
    OUTLET,
    GHOST
};
