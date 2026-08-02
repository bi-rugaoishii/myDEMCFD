#include "G_DEMCFDCoupling.h"
#include <cmath>

static __device__ __forceinline__ void d_get_cfd_cell_index(G_StaggeredGrid *grid, ParticleSys<DeviceMemory>* ps, int i){


    int bi=i*DIM;

    const int Nx = grid->Nx_;
    const int Ny = grid->Ny_;
    const int Nz = grid->Nz_;

    const double xp = ps->x[bi+0];
    const double yp = ps->x[bi+1];
    const double zp = ps->x[bi+2];

    const double originx = grid->origin_x_;
    const double originy = grid->origin_y_;
    const double originz = grid->origin_z_;

    const double inv_dx = grid->inv_dx_;
    const double inv_dy = grid->inv_dy_;
    const double inv_dz = grid->inv_dz_;

    const double sx = (xp - originx) * inv_dx;
    const double sy = (yp - originy) * inv_dy;
    const double sz = (zp - originz) * inv_dz;

    const int ix_raw = static_cast<int>(floor(sx));
    const int iy_raw = static_cast<int>(floor(sy));
    const int iz_raw = static_cast<int>(floor(sz));

    /* debug */
    if(i==0){
        printf("x= %f y=%f z=%f, cfdidx = %d, cfdidy = %d, cfdidz = %d, \n", xp, yp, zp, ix_raw+1, iy_raw+1, iz_raw+1);
        printf("originx = %f, originy = %f, originz = %f\n", originx, originy, originz);
        printf("dx = %f, dy = %f, dz = %f\n", 1./inv_dx, 1./inv_dy, 1./inv_dz);
    }

    if(ix_raw < 0 || ix_raw >= Nx ||
       iy_raw < 0 || iy_raw >= Ny ||
       iz_raw < 0 || iz_raw >= Nz){

        ps->is_in_CFD[i]= false;

        /*debug*/
    if(i==0){
        printf("not in\n");
    }

        return;
    }

    // Real CFD cells start from index 1.
    ps->cfd_cellid_x_[i] = ix_raw + 1;
    ps->cfd_cellid_y_[i] = iy_raw + 1;
    ps->cfd_cellid_z_[i] = iz_raw + 1;

    /* debug */
    if(i==0){
        printf("is in\n");
    }

    ps->is_in_CFD[i] = true;

}

static __global__ void k_get_cfd_cell_index(G_StaggeredGrid *grid, ParticleSys<DeviceMemory>* ps){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps->N || ps->isActive[i]!=1) return;

    d_get_cfd_cell_index(grid,ps,i);

}

void G_DEMCFDCoupling::get_index_of_Cell(G_StaggeredGrid& grid, ParticleSys<DeviceMemory>& ps, int gridSize, int blockSize){

    k_get_cfd_cell_index<<<gridSize, blockSize>>>(grid.d_ptr_, ps.d_self);

}


