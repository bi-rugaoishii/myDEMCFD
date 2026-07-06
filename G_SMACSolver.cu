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
    if (f_xtype(ix,iy,iz)==F_GHOST || f_xtype(ix,iy,iz)==F_BOUNDARY){
        Fx(ix,iy,iz) = 0.0;
        return ;
    }


    double vxf = vx(ix,iy,iz);

    int donorInd = vxf > 0.0 ? ix-1 : ix;


    double axf = a(donorInd,iy,iz);

    unsigned char ctyped = celltype(donorInd,iy,iz);
    unsigned char ctypep = celltype(donorInd+1,iy,iz);
    unsigned char ctypem = celltype(donorInd-1,iy,iz);


    /* check if donor cell is near boundary in flow direction*/
    /* if then give upwind*/
    if(ctyped != C_INTERIOR){
        Fx(ix,iy,iz) = 0.;
    }

    if (ctypep != C_INTERIOR || ctypem != C_INTERIOR ) {
        Fx(ix,iy,iz) = vxf*axf*dtbydx;
        return;
    }


    double gamma_x = a(donorInd+1,iy,iz) - a(donorInd-1,iy,iz);

    if(axf < EPS || axf > 1.0-EPS || fabs(gamma_x) < 1e-6){
        Fx(ix,iy,iz) = vxf*axf*dtbydx;
        return;
    }

    double nx = -gamma_x*inv_2dx;

    /* == check if stencil cells are boundary == */
    bool is_boundaryy = false;
    bool is_boundaryz = false;

    unsigned char ctypeyp = celltype(donorInd,iy+1,iz);
    unsigned char ctypeym= celltype(donorInd,iy-1,iz);

    if(ctypeyp != C_INTERIOR || ctypeym != C_INTERIOR){
        is_boundaryy = true;
    }

    unsigned char ctypezp = celltype(donorInd,iy,iz+1);
    unsigned char ctypezm= celltype(donorInd,iy,iz-1);

    if(ctypezp != C_INTERIOR || ctypezm != C_INTERIOR){
        is_boundaryz = true;
    }

    double ny = is_boundaryy? 0.0:-(a(donorInd,iy+1,iz) - a(donorInd,iy-1,iz))*inv_2dy;
    double nz = is_boundaryz? 0.0:-(a(donorInd,iy,iz+1) - a(donorInd,iy,iz-1))*inv_2dz;

    double nx_abs = fabs(nx);
    double ny_abs = fabs(ny);
    double nz_abs = fabs(nz);

    double s = nx_abs + ny_abs + nz_abs + EPS;
    double inv_s = 1.0/s;

    double wx = nx_abs*inv_s;

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

    Fx(ix,iy,iz) = wx*Fx_thinc + (1.-wx)*Fx_upwind;
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
    if (f_ytype(ix,iy,iz)==F_GHOST || f_ytype(ix,iy,iz)==F_BOUNDARY){
        Fy(ix,iy,iz) = 0.0;
        return ;
    }

    double vyf = vy(ix,iy,iz);

    int donorInd = vyf > 0.0 ? iy-1 : iy;


    double ayf = a(ix,donorInd,iz);


    unsigned char ctyped = celltype(ix,donorInd,iz);
    unsigned char ctypep = celltype(ix,donorInd+1,iz);
    unsigned char ctypem = celltype(ix,donorInd-1,iz);

    // upwind near boundary

    if(ctyped != C_INTERIOR){
        Fy(ix,iy,iz) = 0.;
    }

    if (ctypep != C_INTERIOR || ctypem != C_INTERIOR ) {
        Fy(ix,iy,iz) = vyf*ayf*dtbydy;
        return;
    }


    double gamma_y = a(ix,donorInd+1,iz) - a(ix,donorInd-1,iz);

    if(ayf < EPS || ayf > 1.0-EPS || fabs(gamma_y) < 1e-6){
        Fy(ix,iy,iz) = vyf*ayf*dtbydy;
        return;
    }

    double ny = -gamma_y*inv_2dy;

    /* == check if stencil cells are boundary == */
    bool is_boundaryx = false;
    bool is_boundaryz = false;

    unsigned char ctypexp = celltype(ix+1,donorInd,iz);
    unsigned char ctypexm= celltype(ix-1,donorInd,iz);

    if(ctypexp != C_INTERIOR || ctypexm != C_INTERIOR){
        is_boundaryx = true;
    }

    unsigned char ctypezp = celltype(ix,donorInd,iz+1);
    unsigned char ctypezm= celltype(ix,donorInd,iz-1);

    if(ctypezp != C_INTERIOR || ctypezm != C_INTERIOR){
        is_boundaryz = true;
    }

    double nx = is_boundaryx? 0.0: -(a(ix+1,donorInd,iz) - a(ix-1,donorInd,iz))*inv_2dx;
    double nz =  is_boundaryz? 0.0:-(a(ix,donorInd,iz+1) - a(ix,donorInd,iz-1))*inv_2dz;


    double nx_abs = fabs(nx);
    double ny_abs = fabs(ny);
    double nz_abs = fabs(nz);

    double s = nx_abs + ny_abs + nz_abs + EPS;
    double inv_s = 1.0/s;

    double wy = ny_abs*inv_s;

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

    Fy(ix,iy,iz) = wy*Fy_thinc + (1.-wy)*Fy_upwind;
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
    if (f_ztype(ix,iy,iz)==F_GHOST || f_ztype(ix,iy,iz)==F_BOUNDARY){
        Fz(ix,iy,iz) = 0.0;
        return ;
    }

    double vzf = vz(ix,iy,iz);

    int donorInd = vzf > 0.0 ? iz-1 : iz;


    double azf = a(ix,iy,donorInd);

    unsigned char ctyped = celltype(ix,iy,donorInd);
    unsigned char ctypep = celltype(ix,iy,donorInd+1);
    unsigned char ctypem = celltype(ix,iy,donorInd-1);


    /* check if donor cell is near boundary in flow direction*/
    /* if then give upwind*/
    if(ctyped != C_INTERIOR){
        Fz(ix,iy,iz) = 0.;
    }

    if (ctypep != C_INTERIOR || ctypem != C_INTERIOR) {
        Fz(ix,iy,iz) = vzf*azf*dtbydz;
        return;
    }

    double gamma_z = a(ix,iy,donorInd+1) - a(ix,iy,donorInd-1);

    if(azf < EPS || azf > 1.0-EPS || fabs(gamma_z) < 1e-6){
        Fz(ix,iy,iz) = vzf*azf*dtbydz;
        return;
    }

    double nz = -gamma_z*inv_2dz;

    /* == check if stencil cells are boundary == */
    bool is_boundaryx = false;
    bool is_boundaryy = false;

    unsigned char ctypeyp = celltype(ix,iy+1,donorInd);
    unsigned char ctypeym= celltype(ix,iy-1,donorInd);

    if(ctypeyp != C_INTERIOR || ctypeym != C_INTERIOR){
       is_boundaryy = true;
    }

    unsigned char ctypexp = celltype(ix+1,iy,donorInd);
    unsigned char ctypexm = celltype(ix-1,iy,donorInd);

    if(ctypexp != C_INTERIOR || ctypexm != C_INTERIOR){
        is_boundaryx = true;
    }


    double nx =is_boundaryx? 0.0: -(a(ix+1,iy,donorInd) - a(ix-1,iy,donorInd))*inv_2dx;
    double ny =is_boundaryy? 0.0: -(a(ix,iy+1,donorInd) - a(ix,iy-1,donorInd))*inv_2dy;

    double nx_abs = fabs(nx);
    double ny_abs = fabs(ny);
    double nz_abs = fabs(nz);

    double s = nx_abs + ny_abs + nz_abs + EPS;
    double inv_s = 1.0/s;

    double wz = nz_abs*inv_s;

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

    Fz(ix,iy,iz) = wz*Fz_thinc + (1.-wz)*Fz_upwind;
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

    if(ctype == C_INTERIOR || ctype == C_NEAR_BOUNDARY){

        /* == update rho == */
        double alpha = a(ix,iy,iz);
        rho(ix,iy,iz)= (1.-alpha)*rho0+alpha*rho1;

        /* == update inv_rho == */
        inv_rho(ix,iy,iz)= 1./rho(ix,iy,iz);

        /* == update mu == */
        mu(ix,iy,iz) = (1.-alpha)*mu0+alpha*mu1;
    }
}

static __global__ void k_update_x_face_properties_by_alpha(G_StaggeredGrid grid,double rho0, double rho1){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx=grid.Nx_;
    int Ny=grid.Ny_;
    int Nz=grid.Nz_;


    if (ix >=Nx+3 || iy >= Ny+2 || iz >= Nz+2) return;


    MyArray<double,3> rho = grid.rho_;
    MyArray<double,3> mu = grid.mu_;

    /* == update inv rho at face== */

    /* == x faces == */
    MyArray<double,3>  f_bx = grid.f_bx_;
    unsigned char f_xtype= grid.f_xtype_(ix,iy,iz);

    if(f_xtype!=F_INTERIOR){
        f_bx(ix,iy,iz)=0.;
        return;
    }



    /* == update rho at face== */
    MyArray<double,3>  f_rhox = grid.f_rhox_;
    f_rhox(ix,iy,iz) = 0.5*(rho(ix,iy,iz)+rho(ix-1,iy,iz));

    //f_bx[ind] = (inv_rho[ind1]+inv_rho[ind0])*0.5;
    f_bx(ix,iy,iz) = 1./f_rhox(ix,iy,iz);

    /* == update mu at face== */
    MyArray<double,3>  f_mux = grid.f_mux_;
    f_mux(ix,iy,iz) = 0.5*(mu(ix,iy,iz)+mu(ix-1,iy,iz));
}


static __global__ void k_update_y_face_properties_by_alpha(G_StaggeredGrid grid,double rho0, double rho1){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx=grid.Nx_;
    int Ny=grid.Ny_;
    int Nz=grid.Nz_;


    if (ix >=Nx+2 || iy >= Ny+3 || iz >= Nz+2) return;


    MyArray<double,3> rho = grid.rho_;
    MyArray<double,3> mu = grid.mu_;


    /* == update inv rho at face== */
    MyArray<double,3>  f_by = grid.f_by_;

    unsigned char f_ytype= grid.f_ytype_(ix,iy,iz);
    if(f_ytype!=F_INTERIOR){
        f_by(ix,iy,iz)=0.;
        return;
    }


    /* == update rho at face== */
    MyArray<double,3>  f_rhoy = grid.f_rhoy_;
    f_rhoy(ix,iy,iz) = 0.5*(rho(ix,iy,iz)+rho(ix,iy-1,iz));

    f_by(ix,iy,iz) = 1./f_rhoy(ix,iy,iz);

    /* == update mu at face== */
    MyArray<double,3>  f_muy = grid.f_muy_;
    f_muy(ix,iy,iz) = 0.5*(mu(ix,iy,iz)+mu(ix,iy-1,iz));
}

static __global__ void k_update_z_face_properties_by_alpha(G_StaggeredGrid grid,double rho0, double rho1){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx=grid.Nx_;
    int Ny=grid.Ny_;
    int Nz=grid.Nz_;


    if (ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+3) return;


    MyArray<double,3> rho = grid.rho_;
    MyArray<double,3> mu = grid.mu_;

    /* == update inv rho at face== */
    /* == x faces == */
    MyArray<double,3>  f_bz = grid.f_bz_;
    unsigned char f_ztype= grid.f_ztype_(ix,iy,iz);

    if(f_ztype!=F_INTERIOR){
        f_bz(ix,iy,iz)=0.;
        return;
    }



    /* == update rho at face== */
    MyArray<double,3>  f_rhoz = grid.f_rhoz_;
    f_rhoz(ix,iy,iz) = 0.5*(rho(ix,iy,iz)+rho(ix,iy,iz-1));

    f_bz(ix,iy,iz) = 1./f_rhoz(ix,iy,iz);

    /* == update mu at face== */
    MyArray<double,3>  f_muz = grid.f_muz_;
    f_muz(ix,iy,iz) = 0.5*(mu(ix,iy,iz)+mu(ix,iy,iz-1));
}



void G_SMACSolver::update_properties_by_alpha(){

    const double rho1 = rho1_;
    const double rho0 = rho0_;

    const double mu1 = mu1_;
    const double mu0 = mu0_;

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

static __global__ void k_compute_mass_flux_from_alpha_flux(SMACSolver solv,G_StaggeredGrid grid_){
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int ix = blockIdx.x*blockDim.x + threadIdx.x;

    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;

    MyArray<double,3> vx  = grid_.f_vx_;
    MyArray<double,3> vy  = grid_.f_vy_;
    MyArray<double,3> vz  = grid_.f_vz_;
    MyArray<double,3> Fx  = grid_.f_Fx_accum_;
    MyArray<double,3> Fy  = grid_.f_Fy_accum_;
    MyArray<double,3> Fz  = grid_.f_Fz_accum_;
    MyArray<double,3> mfx = grid_.f_mfx_;
    MyArray<double,3> mfy = grid_.f_mfy_;
    MyArray<double,3> mfz = grid_.f_mfz_;

    double inv_dt = solv.inv_dt_;
    double dx = grid_.dx_;
    double dy = grid_.dy_;
    double dz = grid_.dz_;

    double rho0 = solv.rho0_;
    double drho = solv.rho1_ - solv.rho0_;

    double dxbydt = dx*inv_dt;
    double dybydt = dy*inv_dt;
    double dzbydt = dz*inv_dt;

    if(iy>=Ny+2 || ix>= Nx+3 || iz >= Nz+2){
        /* do nothing */
    }else{
        if(grid_.f_xtype_(ix,iy,iz) == F_INTERIOR){
            double q = vx(ix,iy,iz);                    // u_f
            double alpha_q = Fx(ix,iy,iz)* dxbydt; // u_f * alpha_f
            mfx(ix,iy,iz) = rho0 * q + drho * alpha_q; // rho*u
        }else{
            mfx(ix,iy,iz) = 0.0; // rho*u
        }

    }

    if(iy>=Ny+3 || ix>= Nx+2 || iz>= Nz+2){
        /* do nothing */
    }else{
        if(grid_.f_ytype_(ix,iy,iz) == F_INTERIOR){
            double q = vy(ix,iy,iz);                    // u_f
            double alpha_q = Fy(ix,iy,iz)* dybydt; // u_f * alpha_f
            mfy(ix,iy,iz) = rho0 * q + drho * alpha_q; // rho*u
        }else{
            mfy(ix,iy,iz) = 0.0; // rho*u
        }
    }

    if(iy>=Ny+2 || ix>= Nx+2 || iz>= Nz+3){
        /* do nothing */
    }else{
        if(grid_.f_ztype_(ix,iy,iz) == F_INTERIOR){
            double q = vz(ix,iy,iz);                    // u_f
            double alpha_q = Fz(ix,iy,iz)* dzbydt; // u_f * alpha_f
            mfz(ix,iy,iz) = rho0 * q + drho * alpha_q; // rho*u
        }else{
            mfz(ix,iy,iz) = 0.0; // rho*u
        }
    }
}

void G_SMACSolver::compute_mass_flux_from_alpha_flux(SMACSolver solv){
    k_compute_mass_flux_from_alpha_flux<<<grid_dim_,block_dim_>>>(solv,grid_);

}

/* === boundary condition related == */
static __global__ void k_update_x_face_boundary_properties(G_StaggeredGrid grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    int Nx=grid.Nx_;
    int Ny=grid.Ny_;
    int Nz=grid.Nz_;


    if (ix >=Nx+3 || iy >= Ny+2 || iz >= Nz+2) return;

    MyArray<double,3> f_rhox = grid.f_rhox_;
    MyArray<double,3> f_mux = grid.f_mux_;

    if(grid.f_xtype_(ix,iy,iz) == F_BOUNDARY){
        int int_id = grid.f_xinternal_id_(ix,iy,iz);
        f_rhox(ix,iy,iz) = grid.rho_(ix+int_id,iy,iz);
        f_mux(ix,iy,iz) = grid.mu_(ix+int_id,iy,iz);
    };

}

static __global__ void k_update_y_face_boundary_properties(G_StaggeredGrid grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    int Nx=grid.Nx_;
    int Ny=grid.Ny_;
    int Nz=grid.Nz_;


    if (ix >=Nx+2 || iy >= Ny+3 || iz >= Nz+2) return;

    MyArray<double,3> f_rhoy = grid.f_rhoy_;
    MyArray<double,3> f_muy = grid.f_muy_;

    if(grid.f_ytype_(ix,iy,iz) == F_BOUNDARY){
        int int_id = grid.f_yinternal_id_(ix,iy,iz);
        f_rhoy(ix,iy,iz) = grid.rho_(ix,iy+int_id,iz);
        f_muy(ix,iy,iz) = grid.mu_(ix,iy+int_id,iz);
    };

}

static __global__ void k_update_z_face_boundary_properties(G_StaggeredGrid grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    int Nx=grid.Nx_;
    int Ny=grid.Ny_;
    int Nz=grid.Nz_;


    if (ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+3) return;

    MyArray<double,3> f_rhoz = grid.f_rhoz_;
    MyArray<double,3> f_muz = grid.f_muz_;

    if(grid.f_ztype_(ix,iy,iz) == F_BOUNDARY){
        int int_id = grid.f_zinternal_id_(ix,iy,iz);
        f_rhoz(ix,iy,iz) = grid.rho_(ix,iy,iz+int_id);
        f_muz(ix,iy,iz) = grid.mu_(ix,iy,iz+int_id);
    };

}


void G_SMACSolver::update_boundary_faces(){

    k_update_x_face_boundary_properties<<<grid_dim_,block_dim_>>>(grid_);
    k_update_y_face_boundary_properties<<<grid_dim_,block_dim_>>>(grid_);
    k_update_z_face_boundary_properties<<<grid_dim_,block_dim_>>>(grid_);

}

/*===============================================*/
/* == face boundary related device functions ===*/
/*===============================================*/

static __device__ __forceinline__ double d_get_vx_xface(G_StaggeredGrid& grid,int ix,int iy, int iz){

    unsigned char ftype = grid.f_xtype_(ix,iy,iz);

    if(ftype == F_INTERIOR){
        return grid.f_vx_(ix,iy,iz);
    }

    if(ftype == F_BOUNDARY){
        int bid = grid.f_xbcid_(ix,iy,iz);
        return grid.bc_.vx_(bid);
    }

    return 0.;
}

static __device__ __forceinline__ double d_get_vy_yface(G_StaggeredGrid& grid,int ix,int iy, int iz){

    unsigned char ftype = grid.f_ytype_(ix,iy,iz);

    if(ftype == F_INTERIOR){
        return grid.f_vy_(ix,iy,iz);
    }

    if(ftype == F_BOUNDARY){
        int bid = grid.f_ybcid_(ix,iy,iz);
        return grid.bc_.vy_(bid);
    }

    return 0.;

}

static __device__ __forceinline__ double d_get_vz_zface(G_StaggeredGrid &grid,int ix,int iy, int iz){

    unsigned char ftype = grid.f_ztype_(ix,iy,iz);

    if(ftype == F_INTERIOR){
        return grid.f_vz_(ix,iy,iz);
    }

    if(ftype == F_BOUNDARY){
        int bid = grid.f_zbcid_(ix,iy,iz);
        return grid.bc_.vz_(bid);
    }

    return 0.;

}


__device__ __forceinline__
double d_get_vx_ydir(G_StaggeredGrid &grid,int ix,int iy,int iz,int sy){
    double vx_inside = grid.f_vx_(ix,iy,iz);

    int iy2 = iy + sy;

    if (grid.f_xtype_(ix,iy2,iz) == F_INTERIOR) {
        return grid.f_vx_(ix,iy2,iz);
    }

    int iyf = sy > 0 ? iy + 1 : iy;

    double vx_wall = 0.0;
    int count = 0;

    G_BoundaryCondition &bc=grid.bc_;

    if (grid.f_ytype_(ix,iyf,iz) == F_BOUNDARY) {
        unsigned char bid = grid.f_ybcid_(ix,iyf,iz);
        vx_wall += bc.vx_(bid);
        count++;
    }

    if (grid.f_ytype_(ix-1,iyf,iz) == F_BOUNDARY) {
        unsigned char bid = grid.f_ybcid_(ix-1,iyf,iz);
        vx_wall += bc.vx_(bid);
        count++;
    }


    if (count > 0) {
        vx_wall /= (double)count;
    }

    unsigned char bid = grid.f_ybcid_(ix,iyf,iz);
    unsigned char btype = bc.bcType_(bid);

    if(btype == BC_SLIP){
        return vx_inside;
    }else{   //assuming NO_SLIP for now
        return 2.0*vx_wall - vx_inside;

    }

}

__device__ __forceinline__
double d_get_vx_zdir(G_StaggeredGrid &grid,int ix,int iy,int iz,int sz){
    double vx_inside = grid.f_vx_(ix,iy,iz);

    int iz2 = iz + sz;

    if (grid.f_xtype_(ix,iy,iz2) == F_INTERIOR) {
        return grid.f_vx_(ix,iy,iz2);
    }

    int izf = sz > 0 ? iz + 1 : iz;

    double vx_wall = 0.0;
    int count = 0;

    G_BoundaryCondition &bc=grid.bc_;

    if (grid.f_ztype_(ix,iy,izf) == F_BOUNDARY) {
        unsigned char bid = grid.f_zbcid_(ix,iy,izf);

        vx_wall += bc.vx_(bid);
        count++;
    }

    if (grid.f_ztype_(ix-1,iy,izf) == F_BOUNDARY) {
        unsigned char bid = grid.f_zbcid_(ix-1,iy,izf);
        vx_wall += bc.vx_(bid);
        count++;
    }

    if (count > 0) {
        vx_wall /= (double)count;
    }

    unsigned char bid = grid.f_zbcid_(ix,iy,izf);
    unsigned char btype = bc.bcType_(bid);

    if(btype == BC_SLIP){
        return vx_inside;
    }else{   //assuming NO_SLIP for now
        return 2.0*vx_wall - vx_inside;

    }
}

__device__ __forceinline__
double d_get_vy_xdir(G_StaggeredGrid &grid,int ix,int iy,int iz,int sx){
    double vy_inside = grid.f_vy_(ix,iy,iz);

    int ix2 = ix + sx;

    if (grid.f_ytype_(ix2,iy,iz) == F_INTERIOR) {
        return grid.f_vy_(ix2,iy,iz);
    }

    int ixf = sx > 0 ? ix + 1 : ix;

    double vy_wall = 0.0;
    int count = 0;

    G_BoundaryCondition &bc=grid.bc_;

    if (grid.f_xtype_(ixf,iy,iz) == F_BOUNDARY) {
        unsigned char bid = grid.f_xbcid_(ixf,iy,iz);

        vy_wall += bc.vy_(bid);
        count++;
    }

    if (grid.f_xtype_(ixf,iy-1,iz) == F_BOUNDARY) {
        unsigned char bid = grid.f_xbcid_(ixf,iy-1,iz);
        vy_wall += bc.vy_(bid);
        count++;
    }

    if (count > 0) {
        vy_wall /= (double)count;
    }

    unsigned char bid = grid.f_xbcid_(ixf,iy,iz);
    unsigned char btype = bc.bcType_(bid);

    if(btype == BC_SLIP){
        return vy_inside;
    }else{   //assuming NO_SLIP for now
        return 2.0*vy_wall - vy_inside;

    }
}

__device__ __forceinline__
double d_get_vy_zdir(G_StaggeredGrid &grid,int ix,int iy,int iz,int sz){
    double vy_inside = grid.f_vy_(ix,iy,iz);

    int iz2 = iz + sz;

    if (grid.f_ytype_(ix,iy,iz2) == F_INTERIOR) {
        return grid.f_vy_(ix,iy,iz2);
    }

    int izf = sz > 0 ? iz + 1 : iz;

    double vy_wall = 0.0;
    int count = 0;

    G_BoundaryCondition &bc=grid.bc_;

    if (grid.f_ztype_(ix,iy,izf) == F_BOUNDARY) {
        unsigned char bid = grid.f_zbcid_(ix,iy,izf);

        vy_wall += bc.vy_(bid);
        count++;
    }

    if (grid.f_ztype_(ix,iy-1,izf) == F_BOUNDARY) {
        unsigned char bid = grid.f_zbcid_(ix,iy-1,izf);
        vy_wall += bc.vy_(bid);
        count++;
    }

    if (count > 0) {
        vy_wall /= (double)count;
    }

    unsigned char bid = grid.f_zbcid_(ix,iy,izf);
    unsigned char btype = bc.bcType_(bid);

    if(btype == BC_SLIP){
        return vy_inside;
    }else{   //assuming NO_SLIP for now
        return 2.0*vy_wall - vy_inside;
    }

}

__device__ __forceinline__
double d_get_vz_xdir(G_StaggeredGrid &grid,int ix,int iy,int iz,int sx){
    double vz_inside = grid.f_vz_(ix,iy,iz);

    int ix2 = ix + sx;

    if (grid.f_ztype_(ix2,iy,iz) == F_INTERIOR) {
        return grid.f_vz_(ix2,iy,iz);
    }

    int ixf = sx > 0 ? ix + 1 : ix;

    double vz_wall = 0.0;
    int count = 0;

    G_BoundaryCondition &bc=grid.bc_;

    if (grid.f_xtype_(ixf,iy,iz) == F_BOUNDARY) {
        unsigned char bid = grid.f_xbcid_(ixf,iy,iz);

        vz_wall += bc.vz_(bid);
        count++;
    }

    if (grid.f_xtype_(ixf,iy,iz-1) == F_BOUNDARY) {
        unsigned char bid = grid.f_xbcid_(ixf,iy,iz-1);
        vz_wall += bc.vz_(bid);
        count++;
    }

    if (count > 0) {
        vz_wall /= (double)count;
    }

    unsigned char bid = grid.f_xbcid_(ixf,iy,iz);
    unsigned char btype = bc.bcType_(bid);

    if(btype == BC_SLIP){
        return vz_inside;
    }else{   //assuming NO_SLIP for now
        return 2.0*vz_wall - vz_inside;
    }
}

__device__ __forceinline__
double d_get_vz_ydir(G_StaggeredGrid &grid,int ix,int iy,int iz,int sy){
    double vz_inside = grid.f_vz_(ix,iy,iz);

    int iy2 = iy + sy;

    if (grid.f_ztype_(ix,iy2,iz) == F_INTERIOR) {
        return grid.f_vz_(ix,iy2,iz);
    }

    int iyf = sy > 0 ? iy + 1 : iy;

    double vz_wall = 0.0;
    int count = 0;

    G_BoundaryCondition &bc=grid.bc_;

    if (grid.f_ytype_(ix,iyf,iz) == F_BOUNDARY) {
        unsigned char bid = grid.f_ybcid_(ix,iyf,iz);

        vz_wall += bc.vz_(bid);
        count++;
    }

    if (grid.f_ytype_(ix,iyf,iz-1) == F_BOUNDARY) {
        unsigned char bid = grid.f_ybcid_(ix,iyf,iz-1);
        vz_wall += bc.vz_(bid);
        count++;
    }

    if (count > 0) {
        vz_wall /= (double)count;
    }

    unsigned char bid =grid.f_ybcid_(ix,iyf,iz);
    unsigned char btype = bc.bcType_(bid);

    if(btype == BC_SLIP){
        return vz_inside;
    }else{   //assuming NO_SLIP for now
        return 2.0*vz_wall - vz_inside;
    }
}


/* === vstar calculation === */
static __global__ void k_get_vof_vstar_rhouu_upwind_consistent(SMACSolver solv,G_StaggeredGrid grid_){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;

    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

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


    MyArray<double,3>  p = grid_.p_;

    MyArray<double,3>  mfx = grid_.f_mfx_;
    MyArray<double,3>  mfy = grid_.f_mfy_;
    MyArray<double,3>  mfz = grid_.f_mfz_;

    MyArray<double,3>  rho = grid_.rho_;
    MyArray<double,3>  rho_old = grid_.rho_old_;
    MyArray<double,3>  mu = grid_.mu_;

    MyArray<double,3>  f_mux = grid_.f_mux_;
    MyArray<double,3>  f_muy = grid_.f_muy_;
    MyArray<double,3>  f_muz = grid_.f_muz_;

    MyArray<double,3>  vx = grid_.f_vx_;
    MyArray<double,3>  vy = grid_.f_vy_;
    MyArray<double,3>  vz = grid_.f_vz_;

    MyArray<double,3>  vx_star = grid_.f_vx_star_;
    MyArray<double,3>  vy_star = grid_.f_vy_star_;
    MyArray<double,3>  vz_star = grid_.f_vz_star_;

    MyArray<unsigned char,3>& f_xtype = grid_.f_xtype_;
    MyArray<unsigned char,3>& f_ytype = grid_.f_ytype_;
    MyArray<unsigned char,3>& f_ztype = grid_.f_ztype_;

    /* == check cell types == */

    /* == vx == */
    if(f_xtype(ix,iy,iz) != F_INTERIOR){
        /* do nothing */
    }else{


        double vx_211 =d_get_vx_xface(grid_,ix+1,iy,iz);
        double vx_111 =vx(ix,iy,iz);
        double vx_011 =d_get_vx_xface(grid_,ix-1,iy,iz);
        double vx_101 =d_get_vx_ydir(grid_,ix,iy,iz,-1);
        double vx_110 =d_get_vx_zdir(grid_,ix,iy,iz,-1);
        double vx_121 =d_get_vx_ydir(grid_,ix,iy,iz,+1);
        double vx_112 =d_get_vx_zdir(grid_,ix,iy,iz,+1);

        double vy_121 =d_get_vy_yface(grid_,ix,iy+1,iz);
        double vy_021 =d_get_vy_yface(grid_,ix-1,iy+1,iz);
        double vy_111 = d_get_vy_yface(grid_,ix,iy,iz);
        double vy_011 = d_get_vy_yface(grid_,ix-1,iy,iz);

        double vz_112 =d_get_vz_zface(grid_,ix,iy,iz+1);
        double vz_012 =d_get_vz_zface(grid_,ix-1,iy,iz+1);
        double vz_111 =d_get_vz_zface(grid_,ix,iy,iz); 
        double vz_011 =d_get_vz_zface(grid_,ix-1,iy,iz);




        /* === vx === */
        double tmp_vx = 0.;

        /* viscous */
        double tau_xp = mu(ix,iy,iz)*(vx_211-vx_111);
        double tau_xm = mu(ix-1,iy,iz)*(vx_111-vx_011);

        tmp_vx =  2.*(tau_xp - tau_xm)*inv_dx2;

        double mu_yp = 0.5*(f_muy(ix,iy+1,iz)+f_muy(ix-1,iy+1,iz));
        double tau_yp = mu_yp*((vy_121-vy_021)*inv_dx+(vx_121-vx_111)*inv_dy);

        double mu_ym = 0.5*(f_muy(ix,iy,iz)+f_muy(ix-1,iy,iz));
        double tau_ym = mu_ym*((vy_111-vy_011)*inv_dx+(vx_111-vx_101)*inv_dy);

        tmp_vx +=  (tau_yp-tau_ym)*inv_dy;

        double mu_zp = 0.5*(f_muz(ix,iy,iz+1)+f_muz(ix-1,iy,iz+1));
        double tau_zp = mu_zp*((vz_112-vz_012)*inv_dx+(vx_112-vx_111)*inv_dz);

        double mu_zm = 0.5*(f_muz(ix,iy,iz)+f_muz(ix-1,iy,iz));
        double tau_zm = mu_zm*((vz_111-vz_011)*inv_dx+(vx_111-vx_110)*inv_dz);

        tmp_vx +=  (tau_zp-tau_zm)*inv_dz;


        /* convection */
        double vx_xp= 0.5*(mfx(ix,iy,iz)+mfx(ix+1,iy,iz));
        double vx_xm= 0.5*(mfx(ix-1,iy,iz)+mfx(ix,iy,iz));


        /* == upwind == */
        double ux_xp= vx_xp > 0. ? vx_111:vx_211;
        double ux_xm= vx_xm > 0. ? vx_011: vx_111;

        double M_xp = vx_xp * ux_xp;
        double M_xm = vx_xm * ux_xm;

        tmp_vx -= (M_xp - M_xm)*inv_dx;


        /* == y direction == */
        double vy_yp= 0.5*(mfy(ix-1,iy+1,iz)+mfy(ix,iy+1,iz));
        double vy_ym= 0.5*(mfy(ix,iy,iz)+mfy(ix-1,iy,iz));


        /* == upwind == */
        double uy_yp= vy_yp > 0. ? vx_111: vx_121;
        double uy_ym= vy_ym > 0. ? vx_101: vx_111;

        double M_yp = vy_yp * uy_yp;
        double M_ym = vy_ym * uy_ym;

        tmp_vx -= (M_yp - M_ym)*inv_dy;


        /* == z direction == */
        double vz_zp= 0.5*(mfz(ix-1,iy,iz+1)+mfz(ix,iy,iz+1));
        double vz_zm= 0.5*(mfz(ix,iy,iz)+mfz(ix-1,iy,iz));


        /* == upwind == */
        double uz_zp= vz_zp > 0. ? vx_111: vx_112;
        double uz_zm= vz_zm > 0. ? vx_110: vx_111;

        double M_zp = vz_zp * uz_zp;
        double M_zm = vz_zm * uz_zm;

        tmp_vx -= (M_zp - M_zm)*inv_dz;



        double f_inv_rho = 1./(0.5*(rho(ix,iy,iz)+rho(ix-1,iy,iz)));
        double f_rho_old= (0.5*(rho_old(ix,iy,iz)+rho_old(ix-1,iy,iz)));

        /* == add pressure gradient == */
        tmp_vx -=  (p(ix,iy,iz) - p(ix-1,iy,iz))*inv_dx;


        vx_star(ix,iy,iz)=f_inv_rho*vx_111*f_rho_old+dt*(f_inv_rho*tmp_vx+gx);
    }

    /* == vy == */
    if(f_ytype(ix,iy,iz) != F_INTERIOR){
        /* do nothing */
    }else{


        double vy_121 =d_get_vy_yface(grid_,ix,iy+1,iz);
        double vy_111 =vy(ix,iy,iz);
        double vy_101 =d_get_vy_yface(grid_,ix,iy-1,iz);
        double vy_011 =d_get_vy_xdir(grid_,ix,iy,iz,-1);
        double vy_110 =d_get_vy_zdir(grid_,ix,iy,iz,-1);
        double vy_211 = d_get_vy_xdir(grid_,ix,iy,iz,+1); 
        double vy_112 = d_get_vy_zdir(grid_,ix,iy,iz,+1); 

        double vx_211 =d_get_vx_xface(grid_,ix+1,iy,iz);
        double vx_201 =d_get_vx_xface(grid_,ix+1,iy-1,iz);
        double vx_111 = d_get_vx_xface(grid_,ix,iy,iz);
        double vx_101 = d_get_vx_xface(grid_,ix,iy-1,iz);

        double vz_112 =d_get_vz_zface(grid_,ix,iy,iz+1);
        double vz_102 =d_get_vz_zface(grid_,ix,iy-1,iz+1);
        double vz_111 = d_get_vz_zface(grid_,ix,iy,iz);
        double vz_101 = d_get_vz_zface(grid_,ix,iy-1,iz);




        double tmp_vy = 0.;

        /* viscous */
        double tau_yp = mu(ix,iy,iz)*(vy_121-vy_111);
        double tau_ym = mu(ix,iy-1,iz)*(vy_111-vy_101);

        tmp_vy =  2.*(tau_yp - tau_ym)*inv_dy2;

        double mu_xp = 0.5*(f_mux(ix+1,iy,iz)+f_mux(ix+1,iy-1,iz));
        double tau_xp = mu_xp*((vx_211-vx_201)*inv_dy+(vy_211-vy_111)*inv_dx);

        double mu_xm = 0.5*(f_mux(ix,iy,iz)+f_mux(ix,iy-1,iz));
        double tau_xm = mu_xm*((vx_111-vx_101)*inv_dy+(vy_111-vy_011)*inv_dx);

        tmp_vy +=  (tau_xp-tau_xm)*inv_dx;

        double mu_zp = 0.5*(f_muz(ix,iy,iz+1)+f_muz(ix,iy-1,iz+1));
        double tau_zp = mu_zp*((vz_112-vz_102)*inv_dy+(vy_112-vy_111)*inv_dz);

        double mu_zm = 0.5*(f_muz(ix,iy,iz)+f_muz(ix,iy-1,iz));
        double tau_zm = mu_zm*((vz_111-vz_101)*inv_dy+(vy_111-vy_110)*inv_dz);

        tmp_vy +=  (tau_zp-tau_zm)*inv_dz;

        /* convection */
        double vx_xp= 0.5*(mfx(ix+1,iy,iz)+mfx(ix+1,iy-1,iz));
        double vx_xm= 0.5*(mfx(ix,iy,iz)+mfx(ix,iy-1,iz));


        /* == upwind == */
        double ux_xp= vx_xp > 0. ? vy_111: vy_211;
        double ux_xm= vx_xm > 0. ? vy_011: vy_111;

        double M_xp = vx_xp * ux_xp;
        double M_xm = vx_xm * ux_xm;

        tmp_vy -= (M_xp - M_xm)*inv_dx;

        /* == y direction == */
        double vy_yp= 0.5*(mfy(ix,iy+1,iz)+mfy(ix,iy,iz));
        double vy_ym= 0.5*(mfy(ix,iy,iz)+mfy(ix,iy-1,iz));


        /* == upwind == */
        double uy_yp= vy_yp > 0. ? vy_111: vy_121;
        double uy_ym= vy_ym > 0. ? vy_101: vy_111;

        double M_yp = vy_yp * uy_yp;
        double M_ym = vy_ym * uy_ym;

        tmp_vy -= (M_yp - M_ym)*inv_dy;

        /* == z direction == */
        double vz_zp= 0.5*(mfz(ix,iy,iz+1)+mfz(ix,iy-1,iz+1));
        double vz_zm= 0.5*(mfz(ix,iy,iz)+mfz(ix,iy-1,iz));


        /* == upwind == */
        double uz_zp= vz_zp > 0. ? vy_111: vy_112;
        double uz_zm= vz_zm > 0. ? vy_110: vy_111;

        double M_zp = vz_zp * uz_zp;
        double M_zm = vz_zm * uz_zm;

        tmp_vy -= (M_zp - M_zm)*inv_dz;


        double f_inv_rho = 1./(0.5*(rho(ix,iy,iz)+rho(ix,iy-1,iz)));
        double f_rho_old= (0.5*(rho_old(ix,iy,iz)+rho_old(ix,iy-1,iz)));

        /* == add pressure gradient == */
        tmp_vy -=  (p(ix,iy,iz) - p(ix,iy-1,iz))*inv_dy;
        vy_star(ix,iy,iz)=f_inv_rho*vy_111*f_rho_old+dt*(f_inv_rho*tmp_vy+gy);

    }

    /* == vz == */
    if(f_ztype(ix,iy,iz) != F_INTERIOR){
        /* do nothing */
    }else{

        double vz_111 =vz(ix,iy,iz);
        double vz_112 = d_get_vz_zface(grid_,ix,iy,iz+1);
        double vz_110 =d_get_vz_zface(grid_,ix,iy,iz-1);
        double vz_211 =d_get_vz_xdir(grid_,ix,iy,iz,+1); 
        double vz_011 =d_get_vz_xdir(grid_,ix,iy,iz,-1);
        double vz_121 =d_get_vz_ydir(grid_,ix,iy,iz,+1);
        double vz_101 =d_get_vz_ydir(grid_,ix,iy,iz,-1);

        double vx_211 =d_get_vx_xface(grid_,ix+1,iy,iz);
        double vx_210 =d_get_vx_xface(grid_,ix+1,iy,iz-1);
        double vx_111 = d_get_vx_xface(grid_,ix,iy,iz); 
        double vx_110 = d_get_vx_xface(grid_,ix,iy,iz-1);

        double vy_121 =d_get_vy_yface(grid_,ix,iy+1,iz);
        double vy_120 =d_get_vy_yface(grid_,ix,iy+1,iz-1);
        double vy_111 = d_get_vy_yface(grid_,ix,iy,iz); 
        double vy_110 = d_get_vy_yface(grid_,ix,iy,iz-1);




        double tmp_vz = 0.;

        /* viscous */
        double tau_zp = mu(ix,iy,iz)*(vz_112-vz_111);
        double tau_zm = mu(ix,iy,iz-1)*(vz_111-vz_110);

        tmp_vz =  2.*(tau_zp - tau_zm)*inv_dz2;

        double mu_xp = 0.5*(f_mux(ix+1,iy,iz)+f_mux(ix+1,iy,iz-1));
        double tau_xp = mu_xp*((vx_211-vx_210)*inv_dz+(vz_211-vz_111)*inv_dx);

        double mu_xm = 0.5*(f_mux(ix,iy,iz)+f_mux(ix,iy,iz-1));
        double tau_xm = mu_xm*((vx_111-vx_110)*inv_dz+(vz_111-vz_011)*inv_dx);

        tmp_vz +=  (tau_xp-tau_xm)*inv_dx;

        double mu_yp = 0.5*(f_muy(ix,iy+1,iz)+f_muy(ix,iy+1,iz-1));
        double tau_yp = mu_yp*((vy_121-vy_120)*inv_dz+(vz_121-vz_111)*inv_dy);

        double mu_ym = 0.5*(f_muy(ix,iy,iz)+f_muy(ix,iy,iz-1));
        double tau_ym = mu_ym*((vy_111-vy_110)*inv_dz+(vz_111-vz_101)*inv_dy);

        tmp_vz +=  (tau_yp-tau_ym)*inv_dy;

        /* convection */
        double vx_xp= 0.5*(mfx(ix+1,iy,iz)+mfx(ix+1,iy,iz-1));
        double vx_xm= 0.5*(mfx(ix,iy,iz)+mfx(ix,iy,iz-1));


        /* == upwind == */
        double ux_xp= vx_xp > 0. ? vz_111: vz_211;
        double ux_xm= vx_xm > 0. ? vz_011: vz_111;

        double M_xp = vx_xp * ux_xp;
        double M_xm = vx_xm * ux_xm;

        tmp_vz -= (M_xp - M_xm)*inv_dx;

        /* == y direction == */
        double vy_yp= 0.5*(mfy(ix,iy+1,iz)+mfy(ix,iy+1,iz-1));
        double vy_ym= 0.5*(mfy(ix,iy,iz)+mfy(ix,iy,iz-1));


        /* == upwind == */
        double uy_yp= vy_yp > 0. ? vz_111: vz_121;
        double uy_ym= vy_ym > 0. ? vz_101: vz_111;

        double M_yp = vy_yp * uy_yp;
        double M_ym = vy_ym * uy_ym;

        tmp_vz -= (M_yp - M_ym)*inv_dy;

        /* == z direction == */
        double vz_zp= 0.5*(mfz(ix,iy,iz+1)+mfz(ix,iy,iz));
        double vz_zm= 0.5*(mfz(ix,iy,iz)+mfz(ix,iy,iz-1));


        /* == upwind == */
        double uz_zp= vz_zp > 0. ? vz_111: vz_112;
        double uz_zm= vz_zm > 0. ? vz_110: vz_111;

        double M_zp = vz_zp * uz_zp;
        double M_zm = vz_zm * uz_zm;

        tmp_vz -= (M_zp - M_zm)*inv_dz;


        double f_inv_rho = 1./(0.5*(rho(ix,iy,iz)+rho(ix,iy,iz-1)));
        double f_rho_old= (0.5*(rho_old(ix,iy,iz)+rho_old(ix,iy,iz-1)));

        /* == add pressure gradient == */
        tmp_vz -=  (p(ix,iy,iz) - p(ix,iy,iz-1))*inv_dz;

        vz_star(ix,iy,iz)=f_inv_rho*vz_111*f_rho_old+dt*(f_inv_rho*tmp_vz+gz);

    }
}

void G_SMACSolver::get_vof_vstar_rhouu_upwind_consistent(SMACSolver solv){
    k_get_vof_vstar_rhouu_upwind_consistent<<<grid_dim_,block_dim_>>>(solv,grid_);

}

static __global__ void k_correct_vof_velocity(SMACSolver solv, G_StaggeredGrid grid_){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;



    MyArray<double,3> p = grid_.p_delta_;
    MyArray<double,3> p_new = grid_.p_;
    MyArray<double,3> vx = grid_.f_vx_;
    MyArray<double,3> vy = grid_.f_vy_;
    MyArray<double,3> vz = grid_.f_vz_;
    MyArray<double,3> vx_star = grid_.f_vx_star_;
    MyArray<double,3> vy_star = grid_.f_vy_star_;
    MyArray<double,3> vz_star = grid_.f_vz_star_;
    MyArray<double,3> f_bx = grid_.f_bx_;
    MyArray<double,3> f_by = grid_.f_by_;
    MyArray<double,3> f_bz = grid_.f_bz_;

    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;

    double inv_dx = grid_.inv_dx_;
    double inv_dy = grid_.inv_dy_;
    double inv_dz = grid_.inv_dz_;


    double dt= solv.dt_;

    /* == fix vx == */
    if(iy>=Ny+2 || ix>= Nx+3 || iz >= Nz+2){
        /* do nothing */
    }else{
        if(grid_.f_xtype_(ix,iy,iz) == F_INTERIOR){
            vx(ix,iy,iz) = vx_star(ix,iy,iz) + f_bx(ix,iy,iz)*dt*(-(p(ix,iy,iz)-p(ix-1,iy,iz))*inv_dx);


        }
    }

    /* == fix vy == */
    if(iy>=Ny+3 || ix>= Nx+2 || iz >= Nz+2){
        /* do nothing */
    }else{
        if(grid_.f_ytype_(ix,iy,iz) == F_INTERIOR){
            vy(ix,iy,iz) = vy_star(ix,iy,iz) + f_by(ix,iy,iz)*dt*(-(p(ix,iy,iz)-p(ix,iy-1,iz))*inv_dy);
        }
    }

    /* == fix vz == */
    if(iy>=Ny+2 || ix>= Nx+2 || iz >= Nz+3){
        /* do nothing */
    }else{
        if(grid_.f_ztype_(ix,iy,iz) == F_INTERIOR){
            vz(ix,iy,iz) = vz_star(ix,iy,iz) + f_bz(ix,iy,iz)*dt*(-(p(ix,iy,iz)-p(ix,iy,iz-1))*inv_dz);
        }
    }

    if (ix < Nx+1 && iy < Ny+1 && iz < Nz+1){
        p_new(ix,iy,iz) += p(ix,iy,iz);
    }
}

void G_SMACSolver::correct_vof_velocity(SMACSolver solv){
    k_correct_vof_velocity<<<grid_dim_,block_dim_>>>(solv,grid_);
    cudaMemset(grid_.p_delta_.data_, 0, sizeof(double) * grid_.p_delta_.size_);
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

    #define MEMBER(type, name, sizex,sizey, sizez, isSAVE) cudaFree(grid_.bc_.name.data_);
    #include "memberList/boundaryConditionMembers.def"
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

    #define MEMBER(type, name, sizex,sizey, sizez, isSAVE) cudaMalloc((void**)&grid_.bc_.name.data_, sizeof(type)*(grid_.bc_.num_boundary_id_));
    #include "memberList/boundaryConditionMembers.def"
    #undef MEMBER


    /* gpu only */
    cudaMalloc((void**)&d_pcg_scalars_, sizeof(double)*NUM_SCALARS);
    cudaMalloc((void**)&d_r2_, sizeof(double));
    cudaMalloc((void**)&d_dot_, sizeof(double));

    void* tmp=nullptr;
    cub::DeviceReduce::Sum(tmp, cub_temp_storage_bytes_, grid_.pcg_r_.data_, d_r2_,Nx*Ny*Nz);
    cudaMalloc((void**)&cub_temp_storage_, cub_temp_storage_bytes_);
}


void G_SMACSolver::solve_poisson(){
    pressure_solver_->solve(*this);
}



/* == gpu memory related ==*/
void G_SMACSolver::cpuTogpu(StaggeredGrid h_grid){
    int Nx = h_grid.Nx_;
    int Ny = h_grid.Ny_;
    int Nz = h_grid.Nz_;

    #define MEMBER(type,name,sizex,sizey,sizez,SAVE_FLAG) cudaMemcpy(grid_.name.data_,h_grid.name.data_,(sizex*sizey*sizez)*sizeof(type),cudaMemcpyHostToDevice); 
    #include "memberList/gridMembers.def"
    #undef MEMBER

    #define MEMBER(type, name, sizex,sizey, sizez, isSAVE) cudaMemcpy(grid_.bc_.name.data_, h_grid.bc_.name.data_,sizeof(type)*(grid_.bc_.num_boundary_id_),cudaMemcpyHostToDevice);
    #include "memberList/boundaryConditionMembers.def"
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
