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



    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    unsigned char ftype = grid->f_ytype_(ix,iy,iz);
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



    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    unsigned char ftype = grid->f_ztype_(ix,iy,iz);
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



    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    unsigned char ftype = grid->f_ytype_(ix,iy,iz);
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



    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    unsigned char ftype = grid->f_ztype_(ix,iy,iz);
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

            int sign = int_id_shift < 0 ? 1:-1 ;
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
            int sign = int_id_shift < 0 ? 1:-1 ;

            vz(ix,iy,iz) = vz_star(ix,iy,iz) + sign*f_bz(ix,iy,iz)*dt*p(ix,iy,iz+int_id_shift)*inv_dz;
            return;
        }
    }else{
        return;
    }
}

static __device__ __forceinline__ double d_get_velocity_ghost_value(
    unsigned char bcType,
    double boundaryVelocity,
    double lambda,
    double boundaryValue,
    double innerValue){

    if(bcType == BC_INFLOW){
        return 2.0*(1.0-lambda)*boundaryVelocity-innerValue;
    }else if(bcType == BC_NOSLIP){
        return -innerValue;
    }else if(bcType == BC_OUTLET){
        return 2.0*boundaryValue-innerValue;
    }else if(bcType == BC_SLIP){
        return innerValue;
    }

    return boundaryValue;
}

__global__ void k_update_vx_ghost(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >= Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    MyArray<double,3> vx = grid->f_vx_;

    // Lower y boundary, excluding corners.
    if(iy == 0 && iz >= 1 && iz <= Nz){
        int bid = grid->f_xbcid_(ix,1,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);
        double lambda = grid->f_ibm_area_solid_fraction_x_(ix,1,iz);

        vx(ix,iy,iz) = d_get_velocity_ghost_value(
            bcType,
            grid->bc_.vx_(bid),
            lambda,
            vx(ix,1,iz),
            vx(ix,2,iz));
    }else if(iy == Ny+1 && iz >= 1 && iz <= Nz){
        int bid = grid->f_xbcid_(ix,Ny,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);
        double lambda = grid->f_ibm_area_solid_fraction_x_(ix,Ny,iz);

        vx(ix,iy,iz) = d_get_velocity_ghost_value(
            bcType,
            grid->bc_.vx_(bid),
            lambda,
            vx(ix,Ny,iz),
            vx(ix,Ny-1,iz));
    }else if(iz == 0 && iy >= 1 && iy <= Ny){
        int bid = grid->f_xbcid_(ix,iy,1);
        unsigned char bcType = grid->bc_.bcType_(bid);
        double lambda = grid->f_ibm_area_solid_fraction_x_(ix,iy,1);

        vx(ix,iy,iz) = d_get_velocity_ghost_value(
            bcType,
            grid->bc_.vx_(bid),
            lambda,
            vx(ix,iy,1),
            vx(ix,iy,2));
    }else if(iz == Nz+1 && iy >= 1 && iy <= Ny){
        int bid = grid->f_xbcid_(ix,iy,Nz);
        unsigned char bcType = grid->bc_.bcType_(bid);
        double lambda = grid->f_ibm_area_solid_fraction_x_(ix,iy,Nz);

        vx(ix,iy,iz) = d_get_velocity_ghost_value(
            bcType,
            grid->bc_.vx_(bid),
            lambda,
            vx(ix,iy,Nz),
            vx(ix,iy,Nz-1));
    }
}

__global__ void k_update_vx_ghost_corner(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >= Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    bool is_y_ghost = iy == 0 || iy == Ny+1;
    bool is_z_ghost = iz == 0 || iz == Nz+1;

    if(!is_y_ghost || !is_z_ghost) return;

    MyArray<double,3> vx = grid->f_vx_;

    int inner_iy = iy == 0 ? 1 : Ny;
    int inner_iz = iz == 0 ? 1 : Nz;

    double y_ghost_value = vx(ix,iy,inner_iz);
    double z_ghost_value = vx(ix,inner_iy,iz);

    vx(ix,iy,iz) = 0.5*(y_ghost_value+z_ghost_value);
}

__global__ void k_update_vy_ghost(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >= Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    MyArray<double,3> vy = grid->f_vy_;

    // Lower x boundary, excluding corners.
    if(ix == 0 && iz >= 1 && iz <= Nz){
        int bid = grid->f_ybcid_(1,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);
        double lambda = grid->f_ibm_area_solid_fraction_y_(1,iy,iz);

        vy(ix,iy,iz) = d_get_velocity_ghost_value(
            bcType,
            grid->bc_.vy_(bid),
            lambda,
            vy(1,iy,iz),
            vy(2,iy,iz));
    }else if(ix == Nx+1 && iz >= 1 && iz <= Nz){
        int bid = grid->f_ybcid_(Nx,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);
        double lambda = grid->f_ibm_area_solid_fraction_y_(Nx,iy,iz);

        vy(ix,iy,iz) = d_get_velocity_ghost_value(
            bcType,
            grid->bc_.vy_(bid),
            lambda,
            vy(Nx,iy,iz),
            vy(Nx-1,iy,iz));
    }else if(iz == 0 && ix >= 1 && ix <= Nx){
        int bid = grid->f_ybcid_(ix,iy,1);
        unsigned char bcType = grid->bc_.bcType_(bid);
        double lambda = grid->f_ibm_area_solid_fraction_y_(ix,iy,1);

        vy(ix,iy,iz) = d_get_velocity_ghost_value(
            bcType,
            grid->bc_.vy_(bid),
            lambda,
            vy(ix,iy,1),
            vy(ix,iy,2));
    }else if(iz == Nz+1 && ix >= 1 && ix <= Nx){
        int bid = grid->f_ybcid_(ix,iy,Nz);
        unsigned char bcType = grid->bc_.bcType_(bid);
        double lambda = grid->f_ibm_area_solid_fraction_y_(ix,iy,Nz);

        vy(ix,iy,iz) = d_get_velocity_ghost_value(
            bcType,
            grid->bc_.vy_(bid),
            lambda,
            vy(ix,iy,Nz),
            vy(ix,iy,Nz-1));
    }
}

__global__ void k_update_vy_ghost_corner(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >= Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    bool is_x_ghost = ix == 0 || ix == Nx+1;
    bool is_z_ghost = iz == 0 || iz == Nz+1;

    if(!is_x_ghost || !is_z_ghost) return;

    MyArray<double,3> vy = grid->f_vy_;

    int inner_ix = ix == 0 ? 1 : Nx;
    int inner_iz = iz == 0 ? 1 : Nz;

    double x_ghost_value = vy(ix,iy,inner_iz);
    double z_ghost_value = vy(inner_ix,iy,iz);

    vy(ix,iy,iz) = 0.5*(x_ghost_value+z_ghost_value);
}


__global__ void k_update_vz_ghost(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >= Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    MyArray<double,3> vz = grid->f_vz_;

    // Lower x boundary, excluding corners.
    if(ix == 0 && iy >= 1 && iy <= Ny){
        int bid = grid->f_zbcid_(1,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);
        double lambda = grid->f_ibm_area_solid_fraction_z_(1,iy,iz);

        vz(ix,iy,iz) = d_get_velocity_ghost_value(
            bcType,
            grid->bc_.vz_(bid),
            lambda,
            vz(1,iy,iz),
            vz(2,iy,iz));
    }else if(ix == Nx+1 && iy >= 1 && iy <= Ny){
        int bid = grid->f_zbcid_(Nx,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);
        double lambda = grid->f_ibm_area_solid_fraction_z_(Nx,iy,iz);

        vz(ix,iy,iz) = d_get_velocity_ghost_value(
            bcType,
            grid->bc_.vz_(bid),
            lambda,
            vz(Nx,iy,iz),
            vz(Nx-1,iy,iz));
    }else if(iy == 0 && ix >= 1 && ix <= Nx){
        int bid = grid->f_zbcid_(ix,1,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);
        double lambda = grid->f_ibm_area_solid_fraction_z_(ix,1,iz);

        vz(ix,iy,iz) = d_get_velocity_ghost_value(
            bcType,
            grid->bc_.vz_(bid),
            lambda,
            vz(ix,1,iz),
            vz(ix,2,iz));
    }else if(iy == Ny+1 && ix >= 1 && ix <= Nx){
        int bid = grid->f_zbcid_(ix,Ny,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);
        double lambda = grid->f_ibm_area_solid_fraction_z_(ix,Ny,iz);

        vz(ix,iy,iz) = d_get_velocity_ghost_value(
            bcType,
            grid->bc_.vz_(bid),
            lambda,
            vz(ix,Ny,iz),
            vz(ix,Ny-1,iz));
    }
}

__global__ void k_update_vz_ghost_corner(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >= Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    bool is_x_ghost = ix == 0 || ix == Nx+1;
    bool is_y_ghost = iy == 0 || iy == Ny+1;

    if(!is_x_ghost || !is_y_ghost) return;

    MyArray<double,3> vz = grid->f_vz_;

    int inner_ix = ix == 0 ? 1 : Nx;
    int inner_iy = iy == 0 ? 1 : Ny;

    double x_ghost_value = vz(ix,inner_iy,iz);
    double y_ghost_value = vz(inner_ix,iy,iz);

    vz(ix,iy,iz) = 0.5*(x_ghost_value+y_ghost_value);
}
