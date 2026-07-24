#include "G_SMACSolver.h"
#include "G_StaggeredGrid.h"
#include "PCG_Scalars.h"
#include "hardCodedParameters.h"
#include "MyArray.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <algorithm>
#include <cub/cub.cuh>

static constexpr int IBM_SAMPLE_N = 8;
static constexpr double IBM_SAMPLE_TOTAL = IBM_SAMPLE_N*IBM_SAMPLE_N*IBM_SAMPLE_N;
static constexpr double INV_IBM_SAMPLE_TOTAL = 1./IBM_SAMPLE_TOTAL;
static constexpr double IBM_SAMPLE_FACT = 1./(double)(IBM_SAMPLE_N);

struct CylinderIBM{
    double xc_;
    double yc_;
    double zc_;

    double r_;
    double rsq_;
};

static __device__ __forceinline__ bool d_is_inside_cylinder(double x, double y, double z,CylinderIBM cyl){
    double rsq = cyl.rsq_;
    double dx = x-cyl.xc_;
    double dy = y-cyl.yc_;

    
/* assuming cylinder h is in z direction */

    return dx*dx+dy*dy <= rsq;
}

static __global__ void k_get_solid_fraction(G_StaggeredGrid *grid, CylinderIBM cyl){

    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    const double xc = grid->x_(ix,iy,iz);  
    const double yc = grid->y_(ix,iy,iz);  
    const double zc = grid->z_(ix,iy,iz);  

    const double xini = xc - grid->dx_*0.5;
    const double yini = yc - grid->dy_*0.5;
    const double zini = zc - grid->dz_*0.5;

    const double sample_dx = grid->dx_ * IBM_SAMPLE_FACT;
    const double sample_dy = grid->dy_ * IBM_SAMPLE_FACT;
    const double sample_dz = grid->dz_ * IBM_SAMPLE_FACT;

    int numpts = 0;

    for (int incz=0; incz<IBM_SAMPLE_N; incz++){
        for (int incy=0; incy<IBM_SAMPLE_N; incy++){
            for (int incx=0; incx<IBM_SAMPLE_N; incx++){
                const double x_pt = xini + sample_dx*(incx+0.5);
                const double y_pt = yini + sample_dy*(incy+0.5);
                const double z_pt = zini + sample_dz*(incz+0.5);

                if(d_is_inside_cylinder(x_pt,y_pt,z_pt,cyl)){
                    numpts += 1;
                }
            }
        }
    }

    double solidfraction = (double)numpts*INV_IBM_SAMPLE_TOTAL;

    /* clip */
    if(solidfraction>1.-1.e-2){
        solidfraction = 1.0;
    }


    grid->ibm_solid_fraction_(ix,iy,iz)= solidfraction;
}

/* solid fraction of velocity control volume */
static __global__ void k_get_solid_fraction_staggered_x(G_StaggeredGrid *grid, CylinderIBM cyl){

    int ix = blockIdx.x*blockDim.x + threadIdx.x+2;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix > Nx || iy > Ny || iz > Nz){
        return;
    }

    const double xc = grid->x_(ix,iy,iz)-grid->dx_*0.5;  
    const double yc = grid->y_(ix,iy,iz);  
    const double zc = grid->z_(ix,iy,iz);  

    const double xini = xc - grid->dx_*0.5;
    const double yini = yc - grid->dy_*0.5;
    const double zini = zc - grid->dz_*0.5;

    const double sample_dx = grid->dx_ * IBM_SAMPLE_FACT;
    const double sample_dy = grid->dy_ * IBM_SAMPLE_FACT;
    const double sample_dz = grid->dz_ * IBM_SAMPLE_FACT;

    int numpts = 0;

    for (int incz=0; incz<IBM_SAMPLE_N; incz++){
        for (int incy=0; incy<IBM_SAMPLE_N; incy++){
            for (int incx=0; incx<IBM_SAMPLE_N; incx++){
                const double x_pt = xini + sample_dx*(incx+0.5);
                const double y_pt = yini + sample_dy*(incy+0.5);
                const double z_pt = zini + sample_dz*(incz+0.5);

                if(d_is_inside_cylinder(x_pt,y_pt,z_pt,cyl)){
                    numpts += 1;
                }
            }
        }
    }

    double solidfraction = (double)numpts*INV_IBM_SAMPLE_TOTAL;

    /* clip */
    if(solidfraction>1.-1.e-2){
        solidfraction = 1.0;
    }


    grid->f_ibm_solid_fraction_x_(ix,iy,iz)= solidfraction;
}

static __global__ void k_get_solid_fraction_staggered_y(G_StaggeredGrid *grid, CylinderIBM cyl){

    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+2;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix > Nx || iy > Ny || iz > Nz){
        return;
    }

    const double xc = grid->x_(ix,iy,iz);  
    const double yc = grid->y_(ix,iy,iz)-grid->dy_*0.5;  
    const double zc = grid->z_(ix,iy,iz);  

    const double xini = xc - grid->dx_*0.5;
    const double yini = yc - grid->dy_*0.5;
    const double zini = zc - grid->dz_*0.5;

    const double sample_dx = grid->dx_ * IBM_SAMPLE_FACT;
    const double sample_dy = grid->dy_ * IBM_SAMPLE_FACT;
    const double sample_dz = grid->dz_ * IBM_SAMPLE_FACT;

    int numpts = 0;

    for (int incz=0; incz<IBM_SAMPLE_N; incz++){
        for (int incy=0; incy<IBM_SAMPLE_N; incy++){
            for (int incx=0; incx<IBM_SAMPLE_N; incx++){
                const double x_pt = xini + sample_dx*(incx+0.5);
                const double y_pt = yini + sample_dy*(incy+0.5);
                const double z_pt = zini + sample_dz*(incz+0.5);

                if(d_is_inside_cylinder(x_pt,y_pt,z_pt,cyl)){
                    numpts += 1;
                }
            }
        }
    }

    double solidfraction = (double)numpts*INV_IBM_SAMPLE_TOTAL;

    /* clip */
    if(solidfraction>1.-1.e-2){
        solidfraction = 1.0;
    }


    grid->f_ibm_solid_fraction_y_(ix,iy,iz)= solidfraction;
}

static __global__ void k_get_solid_fraction_staggered_z(G_StaggeredGrid *grid, CylinderIBM cyl){

    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+2;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix > Nx || iy > Ny || iz > Nz){
        return;
    }

    const double xc = grid->x_(ix,iy,iz);  
    const double yc = grid->y_(ix,iy,iz);  
    const double zc = grid->z_(ix,iy,iz)-grid->dz_*0.5;  

    const double xini = xc - grid->dx_*0.5;
    const double yini = yc - grid->dy_*0.5;
    const double zini = zc - grid->dz_*0.5;

    const double sample_dx = grid->dx_ * IBM_SAMPLE_FACT;
    const double sample_dy = grid->dy_ * IBM_SAMPLE_FACT;
    const double sample_dz = grid->dz_ * IBM_SAMPLE_FACT;

    int numpts = 0;

    for (int incz=0; incz<IBM_SAMPLE_N; incz++){
        for (int incy=0; incy<IBM_SAMPLE_N; incy++){
            for (int incx=0; incx<IBM_SAMPLE_N; incx++){
                const double x_pt = xini + sample_dx*(incx+0.5);
                const double y_pt = yini + sample_dy*(incy+0.5);
                const double z_pt = zini + sample_dz*(incz+0.5);

                if(d_is_inside_cylinder(x_pt,y_pt,z_pt,cyl)){
                    numpts += 1;
                }
            }
        }
    }

    double solidfraction = (double)numpts*INV_IBM_SAMPLE_TOTAL;

    /* clip */
    if(solidfraction>1.-1.e-2){
        solidfraction = 1.0;
    }


    grid->f_ibm_solid_fraction_z_(ix,iy,iz)= solidfraction;
}

static __global__ void k_set_solid_cell_from_ibm(G_StaggeredGrid *grid){

    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    MyArray<unsigned char,3>& ctype= grid->celltype_;
    MyArray<unsigned char,3>& fxtype= grid->f_xtype_;
    MyArray<unsigned char,3>& fytype= grid->f_ytype_;
    MyArray<unsigned char,3>& fztype= grid->f_ztype_;
    MyArray<double,3>& sf = grid->ibm_solid_fraction_;

    if(sf(ix,iy,iz)>1.-EPS){
        ctype(ix,iy,iz) = C_SOLID;
        fxtype(ix,iy,iz) = F_BOUNDARY;
        fxtype(ix+1,iy,iz) = F_BOUNDARY;

        fytype(ix,iy,iz) = F_BOUNDARY;
        fytype(ix,iy+1,iz) = F_BOUNDARY;

        fztype(ix,iy,iz) = F_BOUNDARY;
        fztype(ix,iy,iz+1) = F_BOUNDARY;
    }
}

static __global__ void k_set_face_internal_direction(G_StaggeredGrid *grid){
    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    MyArray<unsigned char,3>& ctype= grid->celltype_;
    MyArray<unsigned char,3>& f_xtype = grid->f_xtype_;
    MyArray<unsigned char,3>& f_ytype = grid->f_ytype_;
    MyArray<unsigned char,3>& f_ztype = grid->f_ztype_;

    MyArray<int,3>& f_xinternal_id = grid->f_xinternal_id_;
    MyArray<int,3>& f_yinternal_id = grid->f_yinternal_id_;
    MyArray<int,3>& f_zinternal_id = grid->f_zinternal_id_;

    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    /* == get direction of internal faces == */
    /* == assuming no internal wall exists == */

    // valid x-faces
    if(iz<=Nz && iy <=Ny && ix <= Nx+1){
        if(f_xtype(ix,iy,iz)==F_BOUNDARY){
            unsigned char ctypep = ctype(ix,iy,iz);
            if(ctypep == C_INTERIOR ||ctypep==  C_NEAR_BOUNDARY){
                f_xinternal_id(ix,iy,iz)=0;
            }else{
                f_xinternal_id(ix,iy,iz)=-1;
            }
        }

    }

    // valid y-faces
    if(iz<=Nz && iy <=Ny+1 && ix <= Nx){
        if(f_ytype(ix,iy,iz)==F_BOUNDARY){
            unsigned char ctypep = ctype(ix,iy,iz);
            if(ctypep == C_INTERIOR || ctypep== C_NEAR_BOUNDARY){
                f_yinternal_id(ix,iy,iz)=0;
            }else{
                f_yinternal_id(ix,iy,iz)=-1;
            }
        }
    }

    // valid z-faces
    if(iz<=Nz+1 && iy <=Ny && ix <= Nx){
        if(f_ztype(ix,iy,iz)==F_BOUNDARY){
            unsigned char ctypep = ctype(ix,iy,iz);
            if(ctypep == C_INTERIOR || ctypep== C_NEAR_BOUNDARY){
                f_zinternal_id(ix,iy,iz)=0;
            }else{
                f_zinternal_id(ix,iy,iz)=-1;
            }
        }
    }

}

void G_SMACSolver::make_cylinder_ibm(double xc, double yc, double zc, double r){
    CylinderIBM cyl;

    cyl.xc_ = xc;
    cyl.yc_ = yc;
    cyl.zc_ = zc;
    cyl.r_ = r;
    cyl.rsq_ = r*r;

    k_get_solid_fraction<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,cyl);
    k_get_solid_fraction_staggered_x<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,cyl);
    k_get_solid_fraction_staggered_y<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,cyl);
    k_get_solid_fraction_staggered_z<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,cyl);
}

void G_SMACSolver::set_solid_cell(){
    k_set_solid_cell_from_ibm<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
    k_set_face_internal_direction<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
}
