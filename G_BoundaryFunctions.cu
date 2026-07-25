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

__global__ void k_update_vx_outlet(SMACSolver solv, G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    double dt= solv.dt_;
    MyArray<double,3> p = grid->p_delta_;

    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    unsigned char ftype = grid->f_xtype_(ix,iy,iz);
    MyArray<double,3>  vx_star = grid->f_vx_star_;
    MyArray<double,3>  vx = grid->f_vx_;
    MyArray<double,3> f_bx = grid->f_bx_;
    double inv_dx = grid->inv_dx_;


    if(ftype == F_BOUNDARY){
        int bid = grid->f_xbcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);

        if(bcType == BC_OUTLET){
            int int_id_shift = grid->f_xinternal_id_(ix,iy,iz);

            int sign = int_id_shift < 0 ? 1:-1 ;
            vx(ix,iy,iz) = vx_star(ix,iy,iz) + sign*f_bx(ix,iy,iz)*dt*p(ix+int_id_shift,iy,iz)*inv_dx;
            /* debug*/
            //printf("%d %d %d vx = %f int_id_shift = %d\n", ix,iy,iz,vx(ix,iy,iz),int_id_shift);
            return;
        }
    }else{
        return;
    }
}


__global__ void k_update_vy_outlet(SMACSolver solv, G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    double dt= solv.dt_;
    MyArray<double,3> p = grid->p_delta_;

    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    unsigned char ftype = grid->f_ytype_(ix,iy,iz);
    MyArray<double,3>  vy_star = grid->f_vy_star_;
    MyArray<double,3>  vy = grid->f_vy_;
    MyArray<double,3> f_by = grid->f_by_;
    double inv_dy = grid->inv_dy_;


    if(ftype == F_BOUNDARY){
        int bid = grid->f_ybcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);

        if(bcType == BC_OUTLET){
            int int_id_shift = grid->f_yinternal_id_(ix,iy,iz);

            int sign = int_id_shift < 0 ? -1:1 ;
            vy(ix,iy,iz) = vy_star(ix,iy,iz) + sign*f_by(ix,iy,iz)*dt*p(ix,iy+int_id_shift,iz)*inv_dy;
            return;
        }
    }else{
        return;
    }
}

__global__ void k_update_vz_outlet(SMACSolver solv, G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    double dt= solv.dt_;
    MyArray<double,3> p = grid->p_delta_;

    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    unsigned char ftype = grid->f_ztype_(ix,iy,iz);
    MyArray<double,3>  vz_star = grid->f_vz_star_;
    MyArray<double,3>  vz = grid->f_vz_;
    MyArray<double,3> f_bz = grid->f_bz_;
    double inv_dz = grid->inv_dz_;


    if(ftype == F_BOUNDARY){
        int bid = grid->f_zbcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);

        if(bcType == BC_OUTLET){
            int int_id_shift = grid->f_zinternal_id_(ix,iy,iz);
            int sign = int_id_shift < 0 ? -1:1 ;

            vz(ix,iy,iz) = vz_star(ix,iy,iz) + sign*f_bz(ix,iy,iz)*dt*p(ix,iy,iz+int_id_shift)*inv_dz;
            return;
        }
    }else{
        return;
    }
}

