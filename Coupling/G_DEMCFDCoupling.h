#pragma once
#include "../CFD/G_StaggeredGrid.h"
#include "../DEM/ParticleSystem.h"
#include <cmath>

struct TrilinearStencil{
    int i0;
    int j0;
    int k0;

    double tx;
    double ty;
    double tz;
};

template<int OFFSET_X2, int OFFSET_Y2, int OFFSET_Z2>
__device__ __forceinline__ TrilinearStencil d_get_stencil(const G_StaggeredGrid* grid, const ParticleSys<DeviceMemory>* ps, int pid){
    static_assert(OFFSET_X2 == 0 || OFFSET_X2 == 1,
        "OFFSET_X2 must be 0 or 1");

    static_assert(OFFSET_Y2 == 0 || OFFSET_Y2 == 1,
        "OFFSET_Y2 must be 0 or 1");

    static_assert(OFFSET_Z2 == 0 || OFFSET_Z2 == 1,
        "OFFSET_Z2 must be 0 or 1");

    constexpr double offset_x =
        0.5 * static_cast<double>(OFFSET_X2);

    constexpr double offset_y =
        0.5 * static_cast<double>(OFFSET_Y2);

    constexpr double offset_z =
        0.5 * static_cast<double>(OFFSET_Z2);

    int bi = pid*DIM;

    const double xp = ps->x[bi+0];
    const double yp = ps->x[bi+1];
    const double zp = ps->x[bi+2];

    const double qx = (xp - grid->origin_x_) * grid->inv_dx_ - offset_x;
    const double qy = (yp - grid->origin_y_) * grid->inv_dy_ - offset_y;
    const double qz = (zp - grid->origin_z_) * grid->inv_dz_ - offset_z;


    const double floor_x = floor(qx);
    const double floor_y = floor(qy);
    const double floor_z = floor(qz);

    TrilinearStencil stencil;

    stencil.i0 = static_cast<int>(floor_x) + 1;
    stencil.j0 = static_cast<int>(floor_y) + 1;
    stencil.k0 = static_cast<int>(floor_z) + 1;

    stencil.tx = qx - floor_x;
    stencil.ty = qy - floor_y;
    stencil.tz = qz - floor_z;

    return stencil;
}

__device__ __forceinline__
double d_interpolate_trilinear(MyArray<double,3> field, const TrilinearStencil& stencil){
    const int i0 = stencil.i0;
    const int j0 = stencil.j0;
    const int k0 = stencil.k0;

    const double tx = stencil.tx;
    const double ty = stencil.ty;
    const double tz = stencil.tz;

    const double wx0 = 1.0 - tx;
    const double wy0 = 1.0 - ty;
    const double wz0 = 1.0 - tz;

    const double c00 =
        wx0 * field(i0,     j0, k0) +
        tx  * field(i0 + 1, j0, k0);

    const double c10 =
        wx0 * field(i0,     j0 + 1, k0) +
        tx  * field(i0 + 1, j0 + 1, k0);

    const double c01 =
        wx0 * field(i0,     j0, k0 + 1) +
        tx  * field(i0 + 1, j0, k0 + 1);

    const double c11 =
        wx0 * field(i0,     j0 + 1, k0 + 1) +
        tx  * field(i0 + 1, j0 + 1, k0 + 1);

    const double c0 =
        wy0 * c00 +
        ty  * c10;

    const double c1 = wy0 * c01 + ty  * c11;

    return  wz0 * c0 + tz  * c1;
}

__device__ __forceinline__
bool same_stencil_base(const TrilinearStencil& stencil, int old_i0, int old_j0, int old_k0){
    return stencil.i0 == old_i0 && stencil.j0 == old_j0 && stencil.k0 == old_k0;
}

struct G_DEMCFDCoupling{

    void interpolate_fluid_to_particle(G_StaggeredGrid& grid, ParticleSys<DeviceMemory>& ps, int gridSize, int blockSize);

    void get_index_of_Cell(G_StaggeredGrid& grid, ParticleSys<DeviceMemory>& ps, int gridSize,int blockSize);

};
