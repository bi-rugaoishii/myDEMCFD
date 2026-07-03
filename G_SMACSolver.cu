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


void G_SMACSolver::set_calc_properties(double rho, double dt,double u_lid, double nu, double sizex, double sizey,double sizez, int Nx, int Ny, int Nz){
    rho_=rho;
    u_lid_=u_lid;
    nu_=nu;
    grid_.sizex_=sizex;
    grid_.sizey_=sizey;
    grid_.sizez_=sizez;
    grid_.Nx_=Nx;
    grid_.Ny_=Ny;
    grid_.Nz_=Nz;
    dt_=dt;
    inv_dt_=1./dt;

    grid_.dx_ = sizex/(double)Nx;
    grid_.dy_ = sizey/(double)Ny;
    grid_.dz_ = sizez/(double)Nz;
    grid_.inv_dx_ = 1./grid_.dx_;
    grid_.inv_dy_ = 1./grid_.dy_;
    grid_.inv_dz_ = 1./grid_.dz_;
    grid_.inv_2dx_ = 1./(2.*grid_.dx_);
    grid_.inv_2dy_ = 1./(2.*grid_.dy_);
    grid_.inv_2dz_ = 1./(2.*grid_.dz_);
    grid_.inv_dx2_ = 1./(grid_.dx_*grid_.dx_);
    grid_.inv_dy2_ = 1./(grid_.dy_*grid_.dy_);
    grid_.inv_dz2_ = 1./(grid_.dz_*grid_.dz_);

    grid_.v_b_1_ = 2.*u_lid;
    grid_.v_b_2_ = 0.;
}

/* ========================= */
/* ===== alpha related ===== */
/* ========================= */

/* == function for thinc == */
static __device__ __forceinline__ double sgn(double a){
    double result;
    if(a<0){
        result =-1.;
    }else{
        result =1.;
    }

    return result;
}

static  __device__ __forceinline__ double integrate_thinc(double a, double b, double gamma,double xi){
    double result = 0.5*(b-a)+gamma/(2.*BETA)*(log(cosh(BETA*(b-xi)))-log(cosh(BETA*(a-xi))));

    return result;
}


static __device__ __forceinline__ double find_xi0_analytic(double alpha, double gamma){
    double A=exp(BETA*gamma*(2.*alpha-1.));

    double result = 1./(2.*BETA)*log((exp(BETA)-A)/(A-exp(-BETA)));

    return result;
}


static __global__ void k_alpha_flux_accum(G_StaggeredGrid grid_){
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;


    MyArray<double,3>& Fz_accum  = grid_.f_Fz_accum_;
    MyArray<double,3>& Fz  = grid_.f_Fz_;

    MyArray<double,3>& Fx_accum  = grid_.f_Fx_accum_;
    MyArray<double,3>& Fx  = grid_.f_Fx_;
    MyArray<double,3>& Fy_accum  = grid_.f_Fy_accum_;
    MyArray<double,3>& Fy  = grid_.f_Fy_;


    if(iy>=Ny+2 || ix>= Nx+3 || iz>=Nz+2){
        /* do nothing */
    }else{
        Fx_accum(ix,iy,iz) += Fx(ix,iy,iz);
    }

    if(iy>=Ny+3 || ix>= Nx+2 || iz>=Nz+2){
        /* do nothing */
    }else{
        Fy_accum(ix,iy,iz) += Fy(ix,iy,iz);
    }

    if(iy>=Ny+2 || ix>= Nx+2 || iz>=Nz+3){
        /* do nothing */
    }else{
        Fz_accum(ix,iy,iz) += Fz(ix,iy,iz);
    }
}


static __global__ void k_alpha_flux_thincwlic_x(G_StaggeredGrid grid, double dt){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+2;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;

    MyArray<double,3>& a  = grid.alpha_;
    MyArray<double,3>& vx = grid.f_vx_;
    MyArray<double,3>& Fx = grid.f_Fx_;
    MyArray<unsigned char,3>& f_xtype = grid.f_xtype_;
    MyArray<unsigned char,3>& celltype = grid.celltype_;

    double inv_dx  = grid.inv_dx_;
    double inv_2dx = grid.inv_2dx_;
    double inv_2dy = grid.inv_2dy_;
    double inv_2dz = grid.inv_2dz_;

    double dtbydx = dt*inv_dx;

    // x-face index:
    // ix = 2..Nx, iy = 1..Ny, iz = 1..Nz
    if(ix > Nx || iy > Ny || iz > Nz){
        return;
    }

    // x ghost  face
    if (f_xtype(ix,iy,iz)==F_GHOST || f_xtype(ix,iy,iz)==F_WALL_NOSLIP){
        Fx(ix,iy,iz) = 0.0;
        return ;
    }


    double vxf = vx(ix,iy,iz);

    int donorInd = vxf > 0.0 ? ix-1 : ix;


    double axf = a(donorInd,iy,iz);
    unsigned char ctype = celltype(donorInd,iy,iz);


    /* if near boundary */
    if (ctype == C_NEAR_BOUNDARY) {
        Fx(ix,iy,iz) = vxf*axf*dtbydx;
        return;
    }


    double gamma_x = a(donorInd+1,iy,iz) - a(donorInd-1,iy,iz);

    if(axf < EPS || axf > 1.0-EPS || fabs(gamma_x) < 1e-6){
        Fx(ix,iy,iz) = vxf*axf*dtbydx;
        return;
    }

    double nx = -gamma_x*inv_2dx;
    double ny = -(a(donorInd,iy+1,iz) - a(donorInd,iy-1,iz))*inv_2dy;
    double nz = -(a(donorInd,iy,iz+1) - a(donorInd,iy,iz-1))*inv_2dz;

    double nx_abs = fabs(nx);
    double ny_abs = fabs(ny);
    double nz_abs = fabs(nz);

    double s = nx_abs + ny_abs + nz_abs + EPS;
    double inv_s = 1.0/s;

    double wx = nx_abs*inv_s;
    double wy = ny_abs*inv_s;
    double wz = 1.-(wx+wy);

    double gamma = sgn(gamma_x);

    double xi0 = find_xi0_analytic(a(donorInd,iy,iz), gamma);
    double lambda = vxf*dtbydx;

    double Fx_thinc;

    if(vxf > 0.0){
        Fx_thinc = integrate_thinc(1.0-lambda, 1.0, gamma, xi0);
    }else{
        Fx_thinc = -integrate_thinc(0.0, -lambda, gamma, xi0);
    }

    double Fx_upwind = lambda*axf;

    Fx(ix,iy,iz) = wx*Fx_thinc + (wy+wz)*Fx_upwind;
}

static __global__ void k_alpha_flux_thincwlic_y(G_StaggeredGrid grid, double dt){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+2;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;

    MyArray<double,3>& a  = grid.alpha_;
    MyArray<double,3>& vy = grid.f_vy_;
    MyArray<double,3>& Fy = grid.f_Fy_;
    MyArray<unsigned char,3>& f_ytype = grid.f_ytype_;
    MyArray<unsigned char,3>& celltype = grid.celltype_;

    double inv_dy  = grid.inv_dy_;
    double inv_2dx = grid.inv_2dx_;
    double inv_2dy = grid.inv_2dy_;
    double inv_2dz = grid.inv_2dz_;

    double dtbydy = dt*inv_dy;

    // y-face index:
    // ix = 1..Nx, iy = 2..Ny, iz = 1..Nz

    if(ix > Nx || iy > Ny || iz > Nz){
        return;
    }

    // y ghost  face
    if (f_ytype(ix,iy,iz)==F_GHOST || f_ytype(ix,iy,iz)==F_WALL_NOSLIP){
        Fy(ix,iy,iz) = 0.0;
        return ;
    }

    double vyf = vy(ix,iy,iz);

    int donorInd = vyf > 0.0 ? iy-1 : iy;


    double ayf = a(ix,donorInd,iz);

    unsigned char ctype = celltype(ix,donorInd,iz);


    // upwind near boundary

    if (ctype == C_NEAR_BOUNDARY) {
        Fy(ix,iy,iz) = vyf*ayf*dtbydy;
        return;
    }


    double gamma_y = a(ix,donorInd+1,iz) - a(ix,donorInd-1,iz);

    if(ayf < EPS || ayf > 1.0-EPS || fabs(gamma_y) < 1e-6){
        Fy(ix,iy,iz) = vyf*ayf*dtbydy;
        return;
    }

    double nx = -(a(ix+1,donorInd,iz) - a(ix-1,donorInd,iz))*inv_2dx;
    double ny = -gamma_y*inv_2dy;
    double nz = -(a(ix,donorInd,iz+1) - a(ix,donorInd,iz-1))*inv_2dz;

    double nx_abs = fabs(nx);
    double ny_abs = fabs(ny);
    double nz_abs = fabs(nz);

    double s = nx_abs + ny_abs + nz_abs + EPS;
    double inv_s = 1.0/s;

    double wx = nx_abs*inv_s;
    double wy = ny_abs*inv_s;
    double wz = 1.-(wx+wy);

    double gamma = sgn(gamma_y);

    double xi0 = find_xi0_analytic(a(ix,donorInd,iz), gamma);
    double lambda = vyf*dtbydy;

    double Fy_thinc;

    if(vyf > 0.0){
        Fy_thinc = integrate_thinc(1.0-lambda, 1.0, gamma, xi0);
    }else{
        Fy_thinc = -integrate_thinc(0.0, -lambda, gamma, xi0);
    }

    double Fy_upwind = lambda*ayf;

    Fy(ix,iy,iz) = wy*Fy_thinc + (wx+wz)*Fy_upwind;
}

static __global__ void k_alpha_flux_thincwlic_z(G_StaggeredGrid grid, double dt){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+2;

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;

    MyArray<double,3>& a  = grid.alpha_;
    MyArray<double,3>& vz = grid.f_vz_;
    MyArray<double,3>& Fz = grid.f_Fz_;
    MyArray<unsigned char,3>& f_ztype = grid.f_ztype_;
    MyArray<unsigned char,3>& celltype = grid.celltype_;

    double inv_dz  = grid.inv_dz_;
    double inv_2dx = grid.inv_2dx_;
    double inv_2dy = grid.inv_2dy_;
    double inv_2dz = grid.inv_2dz_;

    double dtbydz = dt*inv_dz;

    // z-face index:
    // ix = 1..Nx, iy = 1..Ny, iz = 2..Nz
    if(ix > Nx || iy > Ny || iz > Nz){
        return;
    }

    // z ghost  face
    if (f_ztype(ix,iy,iz)==F_GHOST || f_ztype(ix,iy,iz)==F_WALL_NOSLIP){
        Fz(ix,iy,iz) = 0.0;
        return ;
    }

    double vzf = vz(ix,iy,iz);

    int donorInd = vzf > 0.0 ? iz-1 : iz;


    double azf = a(ix,iy,donorInd);
    unsigned char ctype = celltype(ix,iy,donorInd);


    // upwind near boundary
    if (ctype == C_NEAR_BOUNDARY) {
        Fz(ix,iy,iz) = vzf*azf*dtbydz;
        return;
    }

    double gamma_z = a(ix,iy,donorInd+1) - a(ix,iy,donorInd-1);

    if(azf < EPS || azf > 1.0-EPS || fabs(gamma_z) < 1e-6){
        Fz(ix,iy,iz) = vzf*azf*dtbydz;
        return;
    }

    double nx = -(a(ix+1,iy,donorInd) - a(ix-1,iy,donorInd))*inv_2dx;
    double ny = -(a(ix,iy+1,donorInd) - a(ix,iy-1,donorInd))*inv_2dy;
    double nz = -gamma_z*inv_2dz;

    double nx_abs = fabs(nx);
    double ny_abs = fabs(ny);
    double nz_abs = fabs(nz);

    double s = nx_abs + ny_abs + nz_abs + EPS;
    double inv_s = 1.0/s;

    double wx = nx_abs*inv_s;
    double wy = ny_abs*inv_s;
    double wz = 1-(wx+wy);

    double gamma = sgn(gamma_z);

    double xi0 = find_xi0_analytic(a(ix,iy,donorInd), gamma);
    double lambda = vzf*dtbydz;

    double Fz_thinc;

    if(vzf > 0.0){
        Fz_thinc = integrate_thinc(1.0-lambda, 1.0, gamma, xi0);
    }else{
        Fz_thinc = -integrate_thinc(0.0, -lambda, gamma, xi0);
    }

    double Fz_upwind = lambda*azf;

    Fz(ix,iy,iz) = wz*Fz_thinc + (wx+wy)*Fz_upwind;
}

static __global__ void k_transport_alpha(G_StaggeredGrid grid_){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

    MyArray<double,3>& a = grid_.alpha_;
    MyArray<double,3>& a_new = grid_.alpha_new_;
    const MyArray<double,3>& Fx = grid_.f_Fx_;
    const MyArray<double,3>& Fy = grid_.f_Fy_;
    const MyArray<double,3>& Fz = grid_.f_Fz_;



    if (iy >=a.sizey_-1 || ix >= a.sizex_-1 || iz>= a.sizez_-1) return;

    double flux = 0.;

    /* == x direction == */
    flux += Fx(ix+1,iy,iz) - Fx(ix,iy,iz);

    /* == y direction == */
    flux += Fy(ix,iy+1,iz) - Fy(ix,iy,iz);

    /* == z direction == */
    flux += Fz(ix,iy,iz+1) - Fz(ix,iy,iz);

    a_new(ix,iy,iz) = a(ix,iy,iz) - flux;

    /* clipping */
    if (a_new(ix,iy,iz)<EPS){
        a_new(ix,iy,iz)=0.;
    }else if (a_new(ix,iy,iz)>1.){
        a_new(ix,iy,iz)=1.;
    }

}

void G_SMACSolver::transport_alpha(){
    k_transport_alpha<<<grid_dim_, block_dim_>>>(grid_);

    /* == swap == */
    std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
}


static __global__ void k_update_cell_properties_by_alpha(G_StaggeredGrid grid_,double rho0, double rho1, double mu0, double mu1){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;


    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;
    int Nz=grid_.Nz_;

    MyArray<double,3> a = grid_.alpha_;
    MyArray<double,3> rho = grid_.rho_;
    MyArray<double,3> inv_rho = grid_.inv_rho_;
    MyArray<double,3> mu = grid_.mu_;

    MyArray<unsigned char,3>& celltype = grid_.celltype_;

    if (ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    unsigned char ctype = celltype(ix,iy,iz);

    if(ctype != C_INTERIOR) return;

    /* == update rho == */
    double alpha = a(ix,iy,iz);
    rho(ix,iy,iz)= (1.-alpha)*rho0+alpha*rho1;

    /* == update inv_rho == */
    inv_rho(ix,iy,iz)= 1./rho(ix,iy,iz);

    /* == update mu == */
    mu(ix,iy,iz) = (1.-alpha)*mu0+alpha*mu1;
}

static __global__ void k_update_x_face_properties_by_alpha(G_StaggeredGrid grid_,double rho0, double rho1){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;
    int Nz=grid_.Nz_;


    if (ix >=Nx+3 || iy >= Ny+2 || iz >= Nz+2) return;


    MyArray<double,3> rho = grid_.rho_;
    MyArray<double,3> mu = grid_.mu_;

    /* == update inv rho at face== */
    /* == x faces == */
    MyArray<double,3>  f_bx = grid_.f_bx_;


    /* == update rho at face== */
    MyArray<double,3>  f_rhox = grid_.f_rhox_;
    f_rhox(ix,iy,iz) = 0.5*(rho(ix,iy,iz)+rho(ix-1,iy,iz));

    //f_bx[ind] = (inv_rho[ind1]+inv_rho[ind0])*0.5;
    f_bx(ix,iy,iz) = 1./f_rhox(ix,iy,iz);

    /* == update mu at face== */
    MyArray<double,3>  f_mux = grid_.f_mux_;
    f_mux(ix,iy,iz) = 0.5*(mu(ix,iy,iz)+mu(ix-1,iy,iz));
}


static __global__ void k_update_y_face_properties_by_alpha(G_StaggeredGrid grid_,double rho0, double rho1){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;
    int Nz=grid_.Nz_;


    if (ix >=Nx+2 || iy >= Ny+3 || iz >= Nz+2) return;


    MyArray<double,3> rho = grid_.rho_;
    MyArray<double,3> mu = grid_.mu_;

    /* == update inv rho at face== */
    /* == x faces == */
    MyArray<double,3>  f_by = grid_.f_by_;


    /* == update rho at face== */
    MyArray<double,3>  f_rhoy = grid_.f_rhoy_;
    f_rhoy(ix,iy,iz) = 0.5*(rho(ix,iy,iz)+rho(ix,iy-1,iz));

    f_by(ix,iy,iz) = 1./f_rhoy(ix,iy,iz);

    /* == update mu at face== */
    MyArray<double,3>  f_muy = grid_.f_muy_;
    f_muy(ix,iy,iz) = 0.5*(mu(ix,iy,iz)+mu(ix,iy-1,iz));
}

static __global__ void k_update_z_face_properties_by_alpha(G_StaggeredGrid grid_,double rho0, double rho1){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;
    int Nz=grid_.Nz_;


    if (ix >=Nx+2 || iy >= Ny+3 || iz >= Nz+2) return;


    MyArray<double,3> rho = grid_.rho_;
    MyArray<double,3> mu = grid_.mu_;

    /* == update inv rho at face== */
    /* == x faces == */
    MyArray<double,3>  f_bz = grid_.f_bz_;


    /* == update rho at face== */
    MyArray<double,3>  f_rhoz = grid_.f_rhoz_;
    f_rhoz(ix,iy,iz) = 0.5*(rho(ix,iy,iz)+rho(ix,iy,iz-1));

    f_bz(ix,iy,iz) = 1./f_rhoz(ix,iy,iz);

    /* == update mu at face== */
    MyArray<double,3>  f_muz = grid_.f_muz_;
    f_muz(ix,iy,iz) = 0.5*(mu(ix,iy,iz)+mu(ix,iy,iz-1));
}



void G_SMACSolver::update_properties_by_alpha(SMACSolver solv){

    const double rho1 = solv.rho1_;
    const double rho0 = solv.rho0_;

    const double mu1 = solv.mu1_;
    const double mu0 = solv.mu0_;

    std::swap(grid_.rho_old_.data_,grid_.rho_.data_);
    k_update_cell_properties_by_alpha<<<grid_dim_,block_dim_>>>(grid_,rho0, rho1, mu0, mu1);
    k_update_x_face_properties_by_alpha<<<grid_dim_,block_dim_>>>(grid_,rho0, rho1);
    k_update_y_face_properties_by_alpha<<<grid_dim_,block_dim_>>>(grid_,rho0, rho1);
    k_update_z_face_properties_by_alpha<<<grid_dim_,block_dim_>>>(grid_,rho0, rho1);

}

void G_SMACSolver::alpha_flux_thincwlic(double dt){

    //k_alpha_flux_upwind<<<grid_dim_,block_dim_>>>(grid_, dt);
    k_alpha_flux_thincwlic_x<<<grid_dim_,block_dim_ >>>(grid_, dt);
    k_alpha_flux_thincwlic_y<<<grid_dim_,block_dim_ >>>(grid_, dt);
    k_alpha_flux_thincwlic_z<<<grid_dim_,block_dim_ >>>(grid_, dt);
}

void G_SMACSolver::alpha_flux_accum(){

    k_alpha_flux_accum<<<grid_dim_,block_dim_>>>(grid_);
}

void G_SMACSolver::clear_alpha_flux_accum(){
    MyArray<double,3> &Fx_accum = grid_.f_Fx_accum_;
    MyArray<double,3> &Fy_accum = grid_.f_Fy_accum_;
    MyArray<double,3> &Fz_accum = grid_.f_Fz_accum_;

    cudaMemset(Fx_accum.data_, 0, sizeof(double) * Fx_accum.size_);
    cudaMemset(Fy_accum.data_, 0, sizeof(double) * Fy_accum.size_);
    cudaMemset(Fz_accum.data_, 0, sizeof(double) * Fz_accum.size_);
}

void G_SMACSolver::clear_alpha_flux(){
    MyArray<double,3> &Fx = grid_.f_Fx_;
    MyArray<double,3> &Fy = grid_.f_Fy_;
    MyArray<double,3> &Fz = grid_.f_Fz_;

    cudaMemset(Fx.data_, 0, sizeof(double) * Fx.size_);
    cudaMemset(Fy.data_, 0, sizeof(double) * Fy.size_);
    cudaMemset(Fz.data_, 0, sizeof(double) * Fz.size_);
}

/* === vstar calculation === */
static __global__ void k_get_vof_vstar_rhouu_upwind_consistent(SMACSolver solv,G_StaggeredGrid grid_){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;

    if(ix >=Nx+3 || iy >= Ny+3 || iz >= Nz+3) return;

    double inv_dx = grid_.inv_dx_;
    double inv_dy = grid_.inv_dy_;
    double inv_dz = grid_.inv_dz_;
    double inv_dx2 = grid_.inv_dx2_;
    double inv_dy2 = grid_.inv_dy2_;
    double inv_dz2 = grid_.inv_dz2_;
    double dt= solv.dt_;

    double gx = solv.gx_;
    double gy = solv.gy_;
    double gz = solv.gz_;



    MyArray<double,3>  mfx = grid_.f_mfx_;
    MyArray<double,3>  mfy = grid_.f_mfy_;
    MyArray<double,3>  mfz = grid_.f_mfz_;

    MyArray<double,3>  rho = grid_.rho_;
    MyArray<double,3>  rho_old = grid_.rho_old_;
    MyArray<double,3>  mu = grid_.mu_;

    MyArray<double,3>  vx = grid_.vx_;
    MyArray<double,3>  vy = grid_.vy_;
    MyArray<double,3>  vz = grid_.vz_;

    MyArray<double,3>  vx_star = grid_.vx_star_;
    MyArray<double,3>  vy_star = grid_.vy_star_;
    MyArray<double,3>  vz_star = grid_.vz_star_;

    MyArray<unsigned char,3>& celltype = grid_.celltype_;
    MyArray<unsigned char,3>& f_xtype = grid_.f_xtype_;
    MyArray<unsigned char,3>& f_ytype = grid_.f_ytype_;
    MyArray<unsigned char,3>& f_ztype = grid_.f_ztype_;

    /* == check cell types == */

    /* == vx == */
    if(f_xtype(ix,iy,iz) == F_BOUNDARY || f_xtype(ix,iy,iz)== F_GHOST){
        /* do nothing */
    }else{
        unsigned char ftype_121 = f_xtype(ix,iy+1,iz);
        unsigned char ftype_101 = f_xtype(ix,iy-1,iz);
        unsigned char ftype_112 = f_xtype(ix,iy,iz+1);
        unsigned char ftype_110 = f_xtype(ix,iy,iz-1);

        
        double vx_211 =vx(ix+1,iy,iz);
        double vx_111 =vx(ix,iy,iz);
        double vx_011 =vx(ix-1,iy,iz);
        double vx_121 = ftype_121 != F_INTERIOR? -1.*vx(ix,iy,iz):vx(ix,iy+1,iz);
        double vx_101 = ftype_101 != F_INTERIOR? -1.*vx(ix,iy,iz):vx(ix,iy-1,iz);


        /* === vx === */
        double tmp_vx = 0.;

        /* viscous */
        double tau_xp = mu(ix,iy,iz)*(vx_211-vx_111);
        double tau_xm = mu(ix-1,iy,iz)*(vx_111-vx_011);

        tmp_vx =  2.*(tau_xp - tau_xm)*inv_dx2;

        double mu_yp = 0.5*(f_muy(ix+1,iy+1,iz+1)+f_muy(ix,iy+1,iz+1));
        double tau_yp = mu_yp*((vy_222-vy_122)*inv_dx+(vx_121-vx_111)*inv_dy);

        double mu_ym = 0.5*(f_muy(j+1,i)+f_muy(j,i));
        double tau_ym = mu_ym*((vy_21-vy_11)*inv_dx+(vx_11-vx_10)*inv_dy);

        tmp_vx +=  (tau_yp-tau_ym)*inv_dy;

        /* convection */
        double vx_xp= 0.5*(mfx[i*pitch_vx+j]+mfx[i*pitch_vx+j+1]);
        double vx_xm= 0.5*(mfx[i*pitch_vx+j-1]+mfx[i*pitch_vx+j]);


        /* == upwind == */
        double ux_xp= vx_xp > 0. ? vx[i*pitch_vx+j]: vx[i*pitch_vx+j+1];
        double ux_xm= vx_xm > 0. ? vx[i*pitch_vx+j-1]: vx[i*pitch_vx+j];

        double M_xp = vx_xp * ux_xp;
        double M_xm = vx_xm * ux_xm;

        tmp_vx -= (M_xp - M_xm)*inv_dx;

        /* == y direction == */
        double vy_yp= 0.5*(mfy[(i+1)*pitch_vy+j]+mfy[(i+1)*pitch_vy+j-1]);
        double vy_ym= 0.5*(mfy[(i)*pitch_vy+j]+mfy[(i)*pitch_vy+j-1]);


        /* == upwind == */
        double uy_yp= vy_yp > 0. ? vx[i*pitch_vx+j]: vx[(i+1)*pitch_vx+j];
        double uy_ym= vy_ym > 0. ? vx[(i-1)*pitch_vx+j]: vx[(i)*pitch_vx+j];

        double M_yp = vy_yp * uy_yp;
        double M_ym = vy_ym * uy_ym;

        tmp_vx -= (M_yp - M_ym)*inv_dy;


        double f_inv_rho = 1./(0.5*(rho[i*pitch_c+j]+rho[i*pitch_c+j-1]));
        double f_rho_old= (0.5*(rho_old[i*pitch_c+j]+rho_old[i*pitch_c+j-1]));

        vx_star[i*(pitch_vx)+j]=f_inv_rho*vx_11*f_rho_old+dt*(f_inv_rho*tmp_vx+gx);

    }

    /* == vy == */
    if(i >= Ny+1|| j >= Nx+1 || i<2 || j<1){
        /* do nothing */
    }else{

        double vy_21 = vy[i*(pitch_vy)+j+1];
        double vy_11 = vy[i*(pitch_vy)+j];
        double vy_11_2 = 2.*vy_11;
        double vy_01 = vy[(i)*(pitch_vy)+j-1];
        double vy_10 = vy[(i-1)*(pitch_vy)+j];
        double vy_12 = vy[(i+1)*(pitch_vy)+j];

        double tmp_vy = 0.;
        /* viscous */

        tmp_vy +=  (vy_21-vy_11_2+vy_01)*inv_dx2;
        tmp_vy +=  (vy_12-vy_11_2+vy_10)*inv_dy2;

        /* calculate face nu */
        double f_mu = (mu[(i)*(pitch_c)+j]+mu[(i-1)*pitch_c+j])*0.5;
        tmp_vy *= f_mu;

        /* convection */
        double vy_yp = 0.5*(mfy[i*pitch_vy+j]+mfy[(i+1)*pitch_vy+j]);
        double vy_ym = 0.5*(mfy[i*pitch_vy+j]+mfy[(i-1)*pitch_vy+j]);

        /* == upwind == */
        double uy_yp= vy_yp > 0. ? vy[i*pitch_vy+j]: vy[(i+1)*pitch_vy+j];
        double uy_ym= vy_ym > 0. ? vy[(i-1)*pitch_vy+j]: vy[i*pitch_vy+j];

        double M_yp = vy_yp * uy_yp;
        double M_ym = vy_ym * uy_ym;

        tmp_vy -= (M_yp - M_ym)*inv_dy;

        /* == x direction == */
        double vx_xp = 0.5*(mfx[i*pitch_vx+j+1]+mfx[(i-1)*pitch_vx+j+1]);
        double vx_xm = 0.5*(mfx[i*pitch_vx+j]+mfx[(i-1)*pitch_vx+j]);

        /* == upwind == */
        double ux_xp= vx_xp > 0. ? vy[i*pitch_vy+j]: vy[(i)*pitch_vy+j+1];
        double ux_xm= vx_xm > 0. ? vy[(i)*pitch_vy+j-1]: vy[(i)*pitch_vy+j];

        double M_xp = vx_xp * ux_xp;
        double M_xm = vx_xm * ux_xm;

        tmp_vy -= (M_xp - M_xm)*inv_dx;


        double f_inv_rho = 1./(0.5*(rho[i*pitch_c+j]+rho[(i-1)*pitch_c+j]));
        double f_rho_old= (0.5*(rho_old[i*pitch_c+j]+rho_old[(i-1)*pitch_c+j]));
        vy_star[i*(pitch_vy)+j]=f_inv_rho*vy_11*f_rho_old+dt*(f_inv_rho*tmp_vy+gy);
    }
}


static __global__  void  k_calc_cfl(G_StaggeredGrid grid_,double dt){

    int ix=blockIdx.x*blockDim.x+threadIdx.x+1; //+1 for ghost cell
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1; 
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1; 

    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;
    int Nz=grid_.Nz_;

    if (ix >=Nx+1 || iy >=Ny+1 || iz >= Nz+1) return;


    MyArray<double,3> vx = grid_.f_vx_;
    MyArray<double,3> vy = grid_.f_vy_;
    MyArray<double,3> vz = grid_.f_vz_;
    MyArray<double,3> cfl = grid_.cfl_;
    double inv_dx = grid_.inv_dx_;
    double inv_dy = grid_.inv_dy_;
    double inv_dz = grid_.inv_dz_;


    double vx_xp = fabs(vx(ix+1,iy,iz));
    double vx_xm = fabs(vx(ix,iy,iz));
    double vy_yp = fabs(vy(ix,iy+1,iz));
    double vy_ym = fabs(vy(ix,iy,iz));
    double vz_zp = fabs(vz(ix,iy,iz+1));
    double vz_zm = fabs(vz(ix,iy,iz));

    double max_vx= vx_xp>vx_xm ? vx_xp: vx_xm;
    double max_vy= vy_yp>vy_ym ? vy_yp: vy_ym;
    double max_vz= vz_zp>vz_zm ? vz_zp: vz_zm;

    //cfl has size of Nx*Ny (without ghost) and hence needs the shift
    cfl(ix-1,iy-1,iz-1) = dt*(max_vx*inv_dx+max_vy*inv_dy+max_vz*inv_dz);
}

double G_SMACSolver::calc_cfl(){
    double dt= dt_;

    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;
    int Nz=grid_.Nz_;

    k_calc_cfl<<<grid_dim_,block_dim_>>>(grid_,dt);
    cub::DeviceReduce::Max(cub_temp_storage_,cub_temp_storage_bytes_,grid_.cfl_.data_,&d_pcg_scalars_[CFL],Nx*Ny*Nz);
    double h_cfl;
    cudaMemcpy(&h_cfl,&d_pcg_scalars_[CFL],sizeof(double),cudaMemcpyDeviceToHost);
    printf("CFL = %3.2e\n", h_cfl);
    return h_cfl;
}

void G_SMACSolver::solver_free(){
    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) cudaFree(grid_.name.data_);
    #include "memberList/gridMembers.def"
    #undef MEMBER


    /* gpu only members */
    cudaFree(d_pcg_scalars_);
    cudaFree(d_r2_);
    cudaFree(d_dot_);
    cudaFree(cub_temp_storage_);

}

void G_SMACSolver::solver_malloc(){
    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;
    int Nz=grid_.Nz_;

    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) grid_.name.sizex_= sizex;\
    grid_.name.sizey_= sizey;\
    grid_.name.sizez_= sizez;\
    grid_.name.size_ = sizex*sizey*sizez;
    #include "memberList/gridMembers.def"
    #undef MEMBER

    #define MEMBER(type, name, sizex,sizey, sizez, isSAVE) cudaMalloc((void**)&grid_.name.data_, sizeof(type)*(sizex*sizey*sizez));
    #include "memberList/gridMembers.def"
    #undef MEMBER



    /* gpu only */
    cudaMalloc((void**)&d_pcg_scalars_, sizeof(double)*NUM_SCALARS);
    cudaMalloc((void**)&d_r2_, sizeof(double));
    cudaMalloc((void**)&d_dot_, sizeof(double));

    void* tmp=nullptr;
    cub::DeviceReduce::Sum(tmp, cub_temp_storage_bytes_, grid_.pcg_r_.data_, d_r2_,Nx*Ny*Nz);
    cudaMalloc((void**)&cub_temp_storage_, cub_temp_storage_bytes_);
}





/* == gpu memory related ==*/
void G_SMACSolver::cpuTogpu(StaggeredGrid h_grid){
    int Nx = h_grid.Nx_;
    int Ny = h_grid.Ny_;
    int Nz = h_grid.Nz_;

    #define MEMBER(type,name,sizex,sizey,sizez,SAVE_FLAG) cudaMemcpy(grid_.name.data_,h_grid.name.data_,(sizex*sizey*sizez)*sizeof(type),cudaMemcpyHostToDevice); 
    #include "memberList/gridMembers.def"
    #undef MEMBER


}

void G_SMACSolver::gpuTocpu(StaggeredGrid h_grid){
    int Nx = h_grid.Nx_;
    int Ny = h_grid.Ny_;
    int Nz = h_grid.Nz_;

    #define MEMBER(type,name,sizex,sizey,sizez,SAVE_FLAG) cudaMemcpy(h_grid.name.data_,grid_.name.data_,(sizex*sizey*sizez)*sizeof(type),cudaMemcpyDeviceToHost); 
    #include "memberList/gridMembers.def"
    #undef MEMBER



}
