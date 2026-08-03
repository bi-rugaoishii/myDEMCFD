#include "G_DEMCFDCoupling.h"
#include <cmath>

static __device__ __forceinline__ void d_save_stencil_base(ParticleSys<DeviceMemory>* ps, int pid, const TrilinearStencil& vx_stencil, const TrilinearStencil& vy_stencil, const TrilinearStencil& vz_stencil){

    ps->cfd_vx_x0_[pid] = vx_stencil.i0;
    ps->cfd_vx_y0_[pid] = vx_stencil.j0;
    ps->cfd_vx_z0_[pid] = vx_stencil.k0;

    ps->cfd_vy_x0_[pid] = vy_stencil.i0;
    ps->cfd_vy_y0_[pid] = vy_stencil.j0;
    ps->cfd_vy_z0_[pid] = vy_stencil.k0;

    ps->cfd_vz_x0_[pid] = vz_stencil.i0;
    ps->cfd_vz_y0_[pid] = vz_stencil.j0;
    ps->cfd_vz_z0_[pid] = vz_stencil.k0;
}

static __device__ __forceinline__ bool d_is_inside_cfd(G_StaggeredGrid *grid, const double xp,const double yp, const double zp){



    const int Nx = grid->Nx_;
    const int Ny = grid->Ny_;
    const int Nz = grid->Nz_;


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


    if(ix_raw < 0 || ix_raw >= Nx ||
       iy_raw < 0 || iy_raw >= Ny ||
       iz_raw < 0 || iz_raw >= Nz){

        return false;
    }

    return true;

}

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


    if(ix_raw < 0 || ix_raw >= Nx ||
       iy_raw < 0 || iy_raw >= Ny ||
       iz_raw < 0 || iz_raw >= Nz){

        ps->cfd_is_in_CFD_[i]= false;
        return;
    }

    // Real CFD cells start from index 1.
    ps->cfd_cellid_x_[i] = ix_raw + 1;
    ps->cfd_cellid_y_[i] = iy_raw + 1;
    ps->cfd_cellid_z_[i] = iz_raw + 1;

    ps->cfd_is_in_CFD_[i] = true;

}

static __global__ void k_get_cfd_cell_index(G_StaggeredGrid *grid, ParticleSys<DeviceMemory>* ps){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps->N || ps->isActive[i]!=1) return;

    d_get_cfd_cell_index(grid,ps,i);

}

__global__ void k_interpolate_fluid_to_particle(G_StaggeredGrid* grid, ParticleSys<DeviceMemory>* ps){
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;

    if (pid >= ps->N || ps->isActive[pid]!=1) return;

    int bi=pid*DIM;

    const double xp = ps->x[bi+0];
    const double yp = ps->x[bi+1];
    const double zp = ps->x[bi+2];

    if(!d_is_inside_cfd(grid, xp, yp, zp)){
        ps->cfd_vx_[pid] = 0.0;
        ps->cfd_vy_[pid] = 0.0;
        ps->cfd_vz_[pid] = 0.0;
        ps->cfd_rho_[pid] = 0.0;
        ps->cfd_mu_[pid] = 0.0;
        return;
    }

    // Cell-centered field: offset = (0.5, 0.5, 0.5)
    const TrilinearStencil cell_stencil = d_get_stencil<1,1,1>(grid, ps, pid);

    // Vx field: offset = (0.0, 0.5, 0.5)
    const TrilinearStencil vx_stencil = d_get_stencil<0,1,1>(grid, ps, pid);

    // Vy field: offset = (0.5, 0.0, 0.5)
    const TrilinearStencil vy_stencil = d_get_stencil<1,0,1>(grid, ps, pid);

    // Vz field: offset = (0.5, 0.5, 0.0)
    const TrilinearStencil vz_stencil = d_get_stencil<1,1,0>(grid, ps, pid);

    const MyArray<double,3> vx = grid->f_vx_;
    const MyArray<double,3> vy = grid->f_vy_;
    const MyArray<double,3> vz = grid->f_vz_;
    const MyArray<double,3> rho = grid->rho_;
    const MyArray<double,3> mu = grid->mu_;

    const double fluid_vx = d_interpolate_trilinear(vx, vx_stencil);
    const double fluid_vy = d_interpolate_trilinear(vy, vy_stencil);
    const double fluid_vz = d_interpolate_trilinear(vz, vz_stencil);
    const double fluid_rho = d_interpolate_trilinear(rho, cell_stencil);
    const double fluid_mu = d_interpolate_trilinear(mu, cell_stencil);

    d_save_stencil_base(ps, pid, vx_stencil, vy_stencil, vz_stencil);

    ps->cfd_vx_[pid] = fluid_vx;
    ps->cfd_vy_[pid] = fluid_vy;
    ps->cfd_vz_[pid] = fluid_vz;
    ps->cfd_rho_[pid] = fluid_rho;
    ps->cfd_mu_[pid] = fluid_mu;

    /* debug */
    printf("position = %f %f %f, rho = %f\n",xp,yp,zp, fluid_rho);
}
void G_DEMCFDCoupling::get_index_of_Cell(G_StaggeredGrid& grid, ParticleSys<DeviceMemory>& ps, int gridSize, int blockSize){

    k_get_cfd_cell_index<<<gridSize, blockSize>>>(grid.d_ptr_, ps.d_self);

}

void G_DEMCFDCoupling::interpolate_fluid_to_particle(G_StaggeredGrid& grid, ParticleSys<DeviceMemory>& ps, int gridSize, int blockSize){

    k_interpolate_fluid_to_particle<<<gridSize, blockSize>>>(grid.d_ptr_, ps.d_self);

}



