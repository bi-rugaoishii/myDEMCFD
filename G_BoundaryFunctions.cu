#include "G_BoundaryFunctions.h"


__global__ void k_update_vxstar_boundary(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    unsigned char ftype = grid->f_xtype_(ix,iy,iz);
    MyArray<double,3>  vx_star = grid->f_vx_star_;


    if(ftype == F_BOUNDARY){
        vx_star(ix,iy,iz) = d_get_vxstar_xface(grid,ix,iy,iz);
    }else{
        return;
    }
}

__global__ void k_update_vystar_boundary(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    unsigned char ftype = grid->f_ytype_(ix,iy,iz);


    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    MyArray<double,3>  vy_star = grid->f_vy_star_;


    if(ftype == F_BOUNDARY){
        vy_star(ix,iy,iz) = d_get_vystar_yface(grid,ix,iy,iz);
    }else{
        return;
    }
}

__global__ void k_update_vzstar_boundary(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    unsigned char ftype = grid->f_ztype_(ix,iy,iz);


    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    MyArray<double,3>  vz_star = grid->f_vz_star_;


    if(ftype == F_BOUNDARY){
        vz_star(ix,iy,iz) = d_get_vzstar_zface(grid,ix,iy,iz);
    }else{
        return;
    }
}

__global__ void k_update_vx_boundary(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    unsigned char ftype = grid->f_xtype_(ix,iy,iz);
    MyArray<double,3>  vx = grid->f_vx_;


    if(ftype == F_BOUNDARY){
        vx(ix,iy,iz) = d_get_vx_xface(grid,ix,iy,iz);
    }else{
        return;
    }
}

__global__ void k_update_vy_boundary(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    unsigned char ftype = grid->f_ytype_(ix,iy,iz);


    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    MyArray<double,3>  vy = grid->f_vy_;


    if(ftype == F_BOUNDARY){
        vy(ix,iy,iz) = d_get_vy_yface(grid,ix,iy,iz);
    }else{
        return;
    }
}

__global__ void k_update_vz_boundary(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    unsigned char ftype = grid->f_ztype_(ix,iy,iz);


    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    MyArray<double,3>  vz = grid->f_vz_;


    if(ftype == F_BOUNDARY){
        vz(ix,iy,iz) = d_get_vz_zface(grid,ix,iy,iz);
    }else{
        return;
    }
}
