#include "G_SMACSolver.h"
#include "G_BoundaryFunctions.h"
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


/* =============================
   ======== set properties =====
   ============================*/

void G_SMACSolver::set_calc_properties(double origin_x, double origin_y, double origin_z, double sizex, double sizey,double sizez, int Nx, int Ny, int Nz){

    grid_.origin_x_ = origin_x;
    grid_.origin_y_ = origin_y;
    grid_.origin_z_ = origin_z;

    grid_.sizex_=sizex;
    grid_.sizey_=sizey;
    grid_.sizez_=sizez;

    grid_.Nx_=Nx;
    grid_.Ny_=Ny;
    grid_.Nz_=Nz;

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

/* === vstar calculation === */
__device__ __forceinline__ double d_vanleer(double deltap, double deltam){
    double deltaprod = deltap*deltam;

    double s_u = 0.; // higher order term 

    if(deltaprod>0.){


        s_u = 2.*deltaprod/(deltap + deltam);

    }else{
        s_u = 0.;
    }

    return s_u;

}

__device__ __forceinline__ double d_minmod(double deltap, double deltam){
    double deltaprod = deltap*deltam;

    double s_u = 0.; // higher order term 

    if(deltaprod>0.){

        if (fabs(deltap) < fabs(deltam)) {
            s_u = deltap;
        }
        else {
            s_u = deltam;
        }


    }else{
        s_u = 0.;
    }

    return s_u;

}

static __global__ void k_get_vof_vstar_rhouu_consistent_x(SMACSolver solv,G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    double inv_dx = grid->inv_dx_;
    double inv_dy = grid->inv_dy_;
    double inv_dz = grid->inv_dz_;
    double inv_dx2 = grid->inv_dx2_;
    double dt= solv.dt_;

    double gx = solv.gx_;


    MyArray<double,3>  p = grid->p_;

    MyArray<double,3>  mfx = grid->f_mfx_;
    MyArray<double,3>  mfy = grid->f_mfy_;
    MyArray<double,3>  mfz = grid->f_mfz_;

    MyArray<double,3>  rho_old = grid->rho_old_;
    MyArray<double,3>  mu = grid->mu_;

    MyArray<double,3>  f_muy = grid->f_muy_;
    MyArray<double,3>  f_muz = grid->f_muz_;

    MyArray<double,3>  vx = grid->f_vx_;

    MyArray<double,3>  f_bx = grid->f_bx_;


    MyArray<double,3>  vx_star = grid->f_vx_star_;

    MyArray<unsigned char,3> f_xtype = grid->f_xtype_;

    /* == check cell types == */

    /* == vx == */
    if(f_xtype(ix,iy,iz) != F_INTERIOR){
        /* do nothing */
    }else{


        double vx_211 =d_get_vx_xface(grid,ix+1,iy,iz);
        double vx_111 =vx(ix,iy,iz);
        double vx_011 =d_get_vx_xface(grid,ix-1,iy,iz);
        double vx_101 =d_get_vx_ydir(grid,ix,iy,iz,-1);
        double vx_110 =d_get_vx_zdir(grid,ix,iy,iz,-1);
        double vx_121 =d_get_vx_ydir(grid,ix,iy,iz,+1);
        double vx_112 =d_get_vx_zdir(grid,ix,iy,iz,+1);

        double vy_121 =d_get_vy_yface(grid,ix,iy+1,iz);
        double vy_021 =d_get_vy_yface(grid,ix-1,iy+1,iz);
        double vy_111 = d_get_vy_yface(grid,ix,iy,iz);
        double vy_011 = d_get_vy_yface(grid,ix-1,iy,iz);

        double vz_112 =d_get_vz_zface(grid,ix,iy,iz+1);
        double vz_012 =d_get_vz_zface(grid,ix-1,iy,iz+1);
        double vz_111 =d_get_vz_zface(grid,ix,iy,iz); 
        double vz_011 =d_get_vz_zface(grid,ix-1,iy,iz);




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
        /*
           double ux_xp= vx_xp > 0. ? vx_111:vx_211;
           double ux_xm= vx_xm > 0. ? vx_011: vx_111;

           double M_xp = vx_xp * ux_xp;
           double M_xm = vx_xm * ux_xm;

           tmp_vx -= (M_xp - M_xm)*inv_dx;
         */

        /* == muscl vanleer == */
        int ind_upwind;

        ind_upwind = vx_xp > 0. ? ix: ix+1;
        double ux_xp = 0.;

        /*== check if has stencil == */
        if(f_xtype(ind_upwind,iy,iz) != F_INTERIOR ){
            ux_xp = vx_xp > 0. ? vx_111:vx_211;
        }else{
            double deltap = d_get_vx_xface(grid,ind_upwind+1,iy,iz) - d_get_vx_xface(grid,ind_upwind,iy,iz);
            double deltam = d_get_vx_xface(grid,ind_upwind,iy,iz) - d_get_vx_xface(grid,ind_upwind-1,iy,iz); 

            double s_u = d_minmod(deltap,deltam); // higher order term 

            int dir = sgn(vx_xp);

            double upwind_v = vx_xp > 0. ? vx_111:vx_211;
            ux_xp = upwind_v + dir*0.5*s_u;
        }

        double M_xp = vx_xp * ux_xp;



        /* == muscl vanleer == */
        ind_upwind = vx_xm > 0. ? ix-1: ix;
        double ux_xm = 0.;

        /*== check if has stencil == */
        if(f_xtype(ind_upwind,iy,iz) != F_INTERIOR ){
            ux_xm = vx_xm > 0. ? vx_011:vx_111;
        }else{
            double deltap = d_get_vx_xface(grid,ind_upwind+1,iy,iz) - d_get_vx_xface(grid,ind_upwind,iy,iz);
            double deltam = d_get_vx_xface(grid,ind_upwind,iy,iz) - d_get_vx_xface(grid,ind_upwind-1,iy,iz); 

            double s_u = d_minmod(deltap,deltam); // higher order term 

            int dir = sgn(vx_xm);

            double upwind_v = vx_xm > 0. ? vx_011:vx_111;
            ux_xm = upwind_v + dir*0.5*s_u;
        }


        double M_xm = vx_xm * ux_xm;

        tmp_vx -= (M_xp - M_xm)*inv_dx;

        /* == y direction == */
        double vy_yp= 0.5*(mfy(ix-1,iy+1,iz)+mfy(ix,iy+1,iz));
        double vy_ym= 0.5*(mfy(ix,iy,iz)+mfy(ix-1,iy,iz));


        /* == upwind == */
        /*
           double uy_yp= vy_yp > 0. ? vx_111: vx_121;
           double uy_ym= vy_ym > 0. ? vx_101: vx_111;

           double M_yp = vy_yp * uy_yp;
           double M_ym = vy_ym * uy_ym;

           tmp_vx -= (M_yp - M_ym)*inv_dy;
         */

        /* == muscl vanleer == */
        ind_upwind = vy_yp > 0. ? iy: iy+1;
        double uy_yp = 0.;

        /*== check if has stencil == */
        if(f_xtype(ix,ind_upwind,iz) != F_INTERIOR){
            uy_yp = vy_yp > 0. ? vx_111:vx_121;
        }else{

            double deltap = d_get_vx_ydir(grid,ix,ind_upwind,iz,+1) - vx(ix,ind_upwind,iz);
            double deltam = vx(ix,ind_upwind,iz) - d_get_vx_ydir(grid,ix,ind_upwind,iz,-1); 

            double s_u = d_minmod(deltap,deltam); // higher order term 


            int dir = sgn(vy_yp);
            double upwind_v = vy_yp > 0. ? vx_111:vx_121;
            uy_yp = upwind_v + dir*0.5*s_u;
        }


        double M_yp = vy_yp * uy_yp;

        /* == muscl vanleer == */
        ind_upwind = vy_ym > 0. ? iy-1: iy;
        double uy_ym = 0.;

        /*== check if has stencil == */
        if(f_xtype(ix,ind_upwind,iz) != F_INTERIOR){
            uy_ym = vy_ym > 0. ? vx_101:vx_111;
        }else{
            double deltap = d_get_vx_ydir(grid,ix,ind_upwind,iz,+1) - vx(ix,ind_upwind,iz);
            double deltam = vx(ix,ind_upwind,iz) - d_get_vx_ydir(grid,ix,ind_upwind,iz,-1); 

            double s_u = d_minmod(deltap,deltam); // higher order term 
            int dir = sgn(vy_ym);

            double upwind_v = vy_ym > 0. ? vx_101:vx_111;
            uy_ym = upwind_v + dir*0.5*s_u;
        }


        double M_ym = vy_ym * uy_ym;

        tmp_vx -= (M_yp - M_ym)*inv_dy;


        /* == z direction == */
        double vz_zp= 0.5*(mfz(ix-1,iy,iz+1)+mfz(ix,iy,iz+1));
        double vz_zm= 0.5*(mfz(ix,iy,iz)+mfz(ix-1,iy,iz));


        /* == upwind == */
        /*
           double uz_zp= vz_zp > 0. ? vx_111: vx_112;
           double uz_zm= vz_zm > 0. ? vx_110: vx_111;
         */


        /* == muscl vanleer == */
        ind_upwind = vz_zp > 0. ? iz: iz+1;
        double uz_zp = 0.;

        /*== check if has stencil == */
        if(f_xtype(ix,iy,ind_upwind) != F_INTERIOR){
            uz_zp = vz_zp > 0. ? vx_111:vx_112;
        }else{

            double deltap = d_get_vx_zdir(grid,ix,iy,ind_upwind,+1) - vx(ix,iy,ind_upwind);
            double deltam = vx(ix,iy,ind_upwind) - d_get_vx_zdir(grid,ix,iy,ind_upwind,-1); 

            double s_u = d_minmod(deltap,deltam); // higher order term 


            int dir = sgn(vz_zp);
            double upwind_v = vz_zp > 0. ? vx_111:vx_112;
            uz_zp = upwind_v + dir*0.5*s_u;
        }


        double M_zp = vz_zp * uz_zp;

        /* == muscl vanleer == */
        ind_upwind = vz_zm > 0. ? iz-1: iz;
        double uz_zm = 0.;

        /*== check if has stencil == */
        if(f_xtype(ix,iy,ind_upwind) != F_INTERIOR){
            uz_zm = vz_zm > 0. ? vx_110:vx_111;
        }else{
            double deltap = d_get_vx_zdir(grid,ix,iy,ind_upwind,+1) - vx(ix,iy,ind_upwind);
            double deltam = vx(ix,iy,ind_upwind) - d_get_vx_zdir(grid,ix,iy,ind_upwind,-1); 

            double s_u = d_minmod(deltap,deltam); // higher order term 

            int dir = sgn(vz_zm);

            double upwind_v = vz_zm > 0. ? vx_110:vx_111;
            uz_zm = upwind_v + dir*0.5*s_u;
        }


        double M_zm = vz_zm * uz_zm;


        tmp_vx -= (M_zp - M_zm)*inv_dz;



        double f_inv_rho = f_bx(ix,iy,iz);
        double f_rho_old= (0.5*(rho_old(ix,iy,iz)+rho_old(ix-1,iy,iz)));

        /* == add pressure gradient == */
        tmp_vx -=  (p(ix,iy,iz) - p(ix-1,iy,iz))*inv_dx;

        const double inv_volume = grid->inv_dx_*grid->inv_dy_*grid->inv_dz_;

        const double coupling_momentum_density = grid->f_coupling_impulse_x_(ix,iy,iz)*inv_volume;

        vx_star(ix,iy,iz)=f_inv_rho*(vx_111*f_rho_old+coupling_momentum_density)+dt*(f_inv_rho*tmp_vx+gx);

        /* == add ibm == */
        double solid_frac = grid->f_ibm_solid_fraction_x_(ix,iy,iz);

        /* == assuming v_ibm = 0 for now. it actually is (1-frac)*v_star + frac *v_ibm*/
        vx_star(ix,iy,iz) = (1.0-solid_frac)*vx_star(ix,iy,iz);

    }
}

static __global__ void k_get_vof_vstar_rhouu_consistent_y(SMACSolver solv,G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    double inv_dx = grid->inv_dx_;
    double inv_dy = grid->inv_dy_;
    double inv_dz = grid->inv_dz_;
    double inv_dy2 = grid->inv_dy2_;
    double dt= solv.dt_;

    double gy = solv.gy_;


    MyArray<double,3>  p = grid->p_;

    MyArray<double,3>  mfx = grid->f_mfx_;
    MyArray<double,3>  mfy = grid->f_mfy_;
    MyArray<double,3>  mfz = grid->f_mfz_;

    MyArray<double,3>  rho_old = grid->rho_old_;
    MyArray<double,3>  mu = grid->mu_;

    MyArray<double,3>  f_mux = grid->f_mux_;
    MyArray<double,3>  f_muz = grid->f_muz_;

    MyArray<double,3>  vy = grid->f_vy_;

    MyArray<double,3>  f_by = grid->f_by_;


    MyArray<double,3>  vy_star = grid->f_vy_star_;

    MyArray<unsigned char,3> f_ytype = grid->f_ytype_;

    /* == check cell types == */
    /* == vy == */
    if(f_ytype(ix,iy,iz) != F_INTERIOR){
        /* do nothing */
    }else{


        double vy_121 =d_get_vy_yface(grid,ix,iy+1,iz);
        double vy_111 =vy(ix,iy,iz);
        double vy_101 =d_get_vy_yface(grid,ix,iy-1,iz);
        double vy_011 =d_get_vy_xdir(grid,ix,iy,iz,-1);
        double vy_110 =d_get_vy_zdir(grid,ix,iy,iz,-1);
        double vy_211 = d_get_vy_xdir(grid,ix,iy,iz,+1); 
        double vy_112 = d_get_vy_zdir(grid,ix,iy,iz,+1); 

        double vx_211 =d_get_vx_xface(grid,ix+1,iy,iz);
        double vx_201 =d_get_vx_xface(grid,ix+1,iy-1,iz);
        double vx_111 = d_get_vx_xface(grid,ix,iy,iz);
        double vx_101 = d_get_vx_xface(grid,ix,iy-1,iz);

        double vz_112 =d_get_vz_zface(grid,ix,iy,iz+1);
        double vz_102 =d_get_vz_zface(grid,ix,iy-1,iz+1);
        double vz_111 = d_get_vz_zface(grid,ix,iy,iz);
        double vz_101 = d_get_vz_zface(grid,ix,iy-1,iz);




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
        //  double ux_xp= vx_xp > 0. ? vy_111: vy_211;
        //  double ux_xm= vx_xm > 0. ? vy_011: vy_111;

        /* == muscl vanleer == */
        int ind_upwind;

        ind_upwind = vx_xp > 0. ? ix: ix+1;
        double ux_xp = 0.;

        /*== check if has stencil == */
        if(f_ytype(ind_upwind,iy,iz) != F_INTERIOR ){
            ux_xp = vx_xp > 0. ? vy_111:vy_211;
        }else{
            double deltap = d_get_vy_xdir(grid,ind_upwind,iy,iz,+1) - vy(ind_upwind,iy,iz);
            double deltam = vy(ind_upwind,iy,iz) - d_get_vy_xdir(grid,ind_upwind,iy,iz,-1); 

            double s_u = d_minmod(deltap,deltam); // higher order term 

            int dir = sgn(vx_xp);

            double upwind_v = vx_xp > 0. ? vy_111:vy_211;
            ux_xp = upwind_v + dir*0.5*s_u;
        }

        double M_xp = vx_xp * ux_xp;


        /* == muscl vanleer == */
        ind_upwind = vx_xm > 0. ? ix-1: ix;
        double ux_xm = 0.;

        /*== check if has stencil == */
        if(f_ytype(ind_upwind,iy,iz) != F_INTERIOR ){
            ux_xm = vx_xm > 0. ? vy_011:vy_111;
        }else{
            double deltap = d_get_vy_xdir(grid,ind_upwind,iy,iz,+1) - vy(ind_upwind,iy,iz);
            double deltam = vy(ind_upwind,iy,iz) - d_get_vy_xdir(grid,ind_upwind,iy,iz,-1); 

            double s_u = d_minmod(deltap,deltam); // higher order term 

            int dir = sgn(vx_xm);

            double upwind_v = vx_xm > 0. ? vy_011:vy_111;
            ux_xm = upwind_v + dir*0.5*s_u;
        }

        double M_xm = vx_xm * ux_xm;

        tmp_vy -= (M_xp - M_xm)*inv_dx;

        /* == y direction == */
        double vy_yp= 0.5*(mfy(ix,iy+1,iz)+mfy(ix,iy,iz));
        double vy_ym= 0.5*(mfy(ix,iy,iz)+mfy(ix,iy-1,iz));


        /* == upwind == */
        /*
           double uy_yp= vy_yp > 0. ? vy_111: vy_121;
           double uy_ym= vy_ym > 0. ? vy_101: vy_111;
         */

        /* == muscl vanleer == */
        ind_upwind = vy_yp > 0. ? iy: iy+1;
        double uy_yp = 0.;

        /*== check if has stencil == */
        if(f_ytype(ix,ind_upwind,iz) != F_INTERIOR){
            uy_yp = vy_yp > 0. ? vy_111:vy_121;
        }else{

            double deltap = d_get_vy_yface(grid,ix,ind_upwind+1,iz) - vy(ix,ind_upwind,iz);
            double deltam = vy(ix,ind_upwind,iz) - d_get_vy_yface(grid,ix,ind_upwind-1,iz); 

            double s_u = d_minmod(deltap,deltam); // higher order term 


            int dir = sgn(vy_yp);
            double upwind_v = vy_yp > 0. ? vy_111:vy_121;
            uy_yp = upwind_v + dir*0.5*s_u;
        }


        double M_yp = vy_yp * uy_yp;

        /* == muscl vanleer == */
        ind_upwind = vy_ym > 0. ? iy-1: iy;
        double uy_ym = 0.;

        /*== check if has stencil == */
        if(f_ytype(ix,ind_upwind,iz) != F_INTERIOR){
            uy_ym = vy_ym > 0. ? vy_101:vy_111;
        }else{
            double deltap = d_get_vy_yface(grid,ix,ind_upwind+1,iz) - vy(ix,ind_upwind,iz);
            double deltam = vy(ix,ind_upwind,iz) - d_get_vy_yface(grid,ix,ind_upwind-1,iz); 


            double s_u = d_minmod(deltap,deltam); // higher order term 

            int dir = sgn(vy_ym);

            double upwind_v = vy_ym > 0. ? vy_101:vy_111;
            uy_ym = upwind_v + dir*0.5*s_u;
        }

        double M_ym = vy_ym * uy_ym;

        tmp_vy -= (M_yp - M_ym)*inv_dy;

        /* == z direction == */
        double vz_zp= 0.5*(mfz(ix,iy,iz+1)+mfz(ix,iy-1,iz+1));
        double vz_zm= 0.5*(mfz(ix,iy,iz)+mfz(ix,iy-1,iz));


        /* == upwind == */
        /*
           double uz_zp= vz_zp > 0. ? vy_111: vy_112;
           double uz_zm= vz_zm > 0. ? vy_110: vy_111;
         */

        /* == muscl vanleer == */
        ind_upwind = vz_zp > 0. ? iz: iz+1;
        double uz_zp = 0.;

        /*== check if has stencil == */
        if(f_ytype(ix,iy,ind_upwind) != F_INTERIOR){
            uz_zp = vz_zp > 0. ? vy_111:vy_112;
        }else{

            double deltap = d_get_vy_zdir(grid,ix,iy,ind_upwind,+1) - vy(ix,iy,ind_upwind);
            double deltam = vy(ix,iy,ind_upwind) - d_get_vy_zdir(grid,ix,iy,ind_upwind,-1); 

            double s_u = d_minmod(deltap,deltam); // higher order term 


            int dir = sgn(vz_zp);
            double upwind_v = vz_zp > 0. ? vy_111:vy_112;
            uz_zp = upwind_v + dir*0.5*s_u;
        }


        double M_zp = vz_zp * uz_zp;

        /* == muscl vanleer == */
        ind_upwind = vz_zm > 0. ? iz-1: iz;
        double uz_zm = 0.;

        /*== check if has stencil == */
        if(f_ytype(ix,iy,ind_upwind) != F_INTERIOR){
            uz_zm = vz_zm > 0. ? vy_110:vy_111;
        }else{
            double deltap = d_get_vy_zdir(grid,ix,iy,ind_upwind,+1) - vy(ix,iy,ind_upwind);
            double deltam = vy(ix,iy,ind_upwind) - d_get_vy_zdir(grid,ix,iy,ind_upwind,-1); 

            double s_u = d_minmod(deltap,deltam); // higher order term 

            int dir = sgn(vz_zm);

            double upwind_v = vz_zm > 0. ? vy_110:vy_111;
            uz_zm = upwind_v + dir*0.5*s_u;
        }



        double M_zm = vz_zm * uz_zm;

        tmp_vy -= (M_zp - M_zm)*inv_dz;


        double f_inv_rho = f_by(ix,iy,iz);
        double f_rho_old= (0.5*(rho_old(ix,iy,iz)+rho_old(ix,iy-1,iz)));

        /* == add pressure gradient == */
        tmp_vy -=  (p(ix,iy,iz) - p(ix,iy-1,iz))*inv_dy;

        const double inv_volume = grid->inv_dx_*grid->inv_dy_*grid->inv_dz_;

        const double coupling_momentum_density = grid->f_coupling_impulse_y_(ix,iy,iz)*inv_volume;

        vy_star(ix,iy,iz)=f_inv_rho*(vy_111*f_rho_old+coupling_momentum_density)+dt*(f_inv_rho*tmp_vy+gy);

        /* == add ibm == */
        double solid_frac = grid->f_ibm_solid_fraction_y_(ix,iy,iz);

        /* == assuming v_ibm = 0 for now. it actually is (1-frac)*v_star + frac *v_ibm*/
        vy_star(ix,iy,iz) = (1.0-solid_frac)*vy_star(ix,iy,iz);

    }

    /* debug*/
    /*
       int jp = iy;
       int jm = iy - 1;

       double rho_m = grid->rho_(ix, jm, iz);
       double rho_p = grid->rho_(ix, jp, iz);
       double beta_f = grid->f_by_( ix, iy, iz);

       double gravity_y = -9.81;
       double dpdy =
       (p(ix, jp, iz) - p(ix, jm, iz)) * inv_dy;

       double pressure_acc = -beta_f * dpdy;
       double total_acc = pressure_acc -9.81;

       if(total_acc > 1e-6){

       printf(
       "iy=%d rho_m=%.15e rho_p=%.15e "
       "beta=%.15e pm=%.15e pp=%.15e "
       "pacc=%.15e gy=%.15e total=%.15e\n",
       iy,
       rho_m,
       rho_p,
       beta_f,
       p(ix, jm, iz),
       p(ix, jp, iz),
       pressure_acc,
       gravity_y,
       total_acc);
       }
     */
}



static __global__ void k_get_vof_vstar_rhouu_consistent_z(SMACSolver solv,G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    double inv_dx = grid->inv_dx_;
    double inv_dy = grid->inv_dy_;
    double inv_dz = grid->inv_dz_;
    double inv_dz2 = grid->inv_dz2_;
    double dt= solv.dt_;

    double gz = solv.gz_;


    MyArray<double,3>  p = grid->p_;

    MyArray<double,3>  mfx = grid->f_mfx_;
    MyArray<double,3>  mfy = grid->f_mfy_;
    MyArray<double,3>  mfz = grid->f_mfz_;

    MyArray<double,3>  rho_old = grid->rho_old_;
    MyArray<double,3>  mu = grid->mu_;

    MyArray<double,3>  f_mux = grid->f_mux_;
    MyArray<double,3>  f_muy = grid->f_muy_;

    MyArray<double,3>  vz = grid->f_vz_;

    MyArray<double,3>  f_bz = grid->f_bz_;


    MyArray<double,3>  vz_star = grid->f_vz_star_;

    MyArray<unsigned char,3> f_ztype = grid->f_ztype_;

    /* == check cell types == */

    /* == vz == */
    if(f_ztype(ix,iy,iz) != F_INTERIOR){
        /* do nothing */
    }else{

        double vz_111 =vz(ix,iy,iz);
        double vz_112 = d_get_vz_zface(grid,ix,iy,iz+1);
        double vz_110 =d_get_vz_zface(grid,ix,iy,iz-1);
        double vz_211 =d_get_vz_xdir(grid,ix,iy,iz,+1); 
        double vz_011 =d_get_vz_xdir(grid,ix,iy,iz,-1);
        double vz_121 =d_get_vz_ydir(grid,ix,iy,iz,+1);
        double vz_101 =d_get_vz_ydir(grid,ix,iy,iz,-1);

        double vx_211 =d_get_vx_xface(grid,ix+1,iy,iz);
        double vx_210 =d_get_vx_xface(grid,ix+1,iy,iz-1);
        double vx_111 = d_get_vx_xface(grid,ix,iy,iz); 
        double vx_110 = d_get_vx_xface(grid,ix,iy,iz-1);

        double vy_121 =d_get_vy_yface(grid,ix,iy+1,iz);
        double vy_120 =d_get_vy_yface(grid,ix,iy+1,iz-1);
        double vy_111 = d_get_vy_yface(grid,ix,iy,iz); 
        double vy_110 = d_get_vy_yface(grid,ix,iy,iz-1);




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
        /*
           double ux_xp= vx_xp > 0. ? vz_111: vz_211;
           double ux_xm= vx_xm > 0. ? vz_011: vz_111;
         */

        /* == muscl vanleer == */
        int ind_upwind;

        ind_upwind = vx_xp > 0. ? ix: ix+1;
        double ux_xp = 0.;

        /*== check if has stencil == */
        if(f_ztype(ind_upwind,iy,iz) != F_INTERIOR ){
            ux_xp = vx_xp > 0. ? vz_111:vz_211;
        }else{
            double deltap = d_get_vz_xdir(grid,ind_upwind,iy,iz,+1) - vz(ind_upwind,iy,iz);
            double deltam = vz(ind_upwind,iy,iz) - d_get_vz_xdir(grid,ind_upwind,iy,iz,-1); 

            double s_u = d_minmod(deltap,deltam); // higher order term 

            int dir = sgn(vx_xp);

            double upwind_v = vx_xp > 0. ? vz_111:vz_211;
            ux_xp = upwind_v + dir*0.5*s_u;
        }


        double M_xp = vx_xp * ux_xp;

        /* == muscl vanleer == */
        ind_upwind = vx_xm > 0. ? ix-1: ix;
        double ux_xm = 0.;

        /*== check if has stencil == */
        if(f_ztype(ind_upwind,iy,iz) != F_INTERIOR ){
            ux_xm = vx_xm > 0. ? vz_011:vz_111;
        }else{
            double deltap = d_get_vz_xdir(grid,ind_upwind,iy,iz,+1) - vz(ind_upwind,iy,iz);
            double deltam = vz(ind_upwind,iy,iz) - d_get_vz_xdir(grid,ind_upwind,iy,iz,-1); 

            double s_u = d_minmod(deltap,deltam); // higher order term 
            int dir = sgn(vx_xm);

            double upwind_v = vx_xm > 0. ? vz_011:vz_111;
            ux_xm = upwind_v + dir*0.5*s_u;
        }


        double M_xm = vx_xm * ux_xm;

        tmp_vz -= (M_xp - M_xm)*inv_dx;

        /* == y direction == */
        double vy_yp= 0.5*(mfy(ix,iy+1,iz)+mfy(ix,iy+1,iz-1));
        double vy_ym= 0.5*(mfy(ix,iy,iz)+mfy(ix,iy,iz-1));


        /* == upwind == */
        /*
           double uy_yp= vy_yp > 0. ? vz_111: vz_121;
           double uy_ym= vy_ym > 0. ? vz_101: vz_111;
         */


        /* == muscl vanleer == */
        ind_upwind = vy_yp > 0. ? iy: iy+1;
        double uy_yp = 0.;

        /*== check if has stencil == */
        if(f_ztype(ix,ind_upwind,iz) != F_INTERIOR){
            uy_yp = vy_yp > 0. ? vz_111:vz_121;
        }else{

            double deltap = d_get_vz_ydir(grid,ix,ind_upwind,iz,+1) - vz(ix,ind_upwind,iz);
            double deltam = vz(ix,ind_upwind,iz) - d_get_vz_ydir(grid,ix,ind_upwind,iz,-1); 


            double s_u = d_minmod(deltap,deltam); // higher order term 


            int dir = sgn(vy_yp);
            double upwind_v = vy_yp > 0. ? vz_111:vz_121;
            uy_yp = upwind_v + dir*0.5*s_u;
        }

        double M_yp = vy_yp * uy_yp;

        /* == muscl vanleer == */
        ind_upwind = vy_ym > 0. ? iy-1: iy;
        double uy_ym = 0.;

        /*== check if has stencil == */
        if(f_ztype(ix,ind_upwind,iz) != F_INTERIOR){
            uy_ym = vy_ym > 0. ? vz_101:vz_111;
        }else{
            double deltap = d_get_vz_ydir(grid,ix,ind_upwind,iz,+1) - vz(ix,ind_upwind,iz);
            double deltam = vz(ix,ind_upwind,iz) - d_get_vz_ydir(grid,ix,ind_upwind,iz,-1); 


            double s_u = d_minmod(deltap,deltam); // higher order term 

            int dir = sgn(vy_ym);

            double upwind_v = vy_ym > 0. ? vz_101:vz_111;
            uy_ym = upwind_v + dir*0.5*s_u;
        }



        double M_ym = vy_ym * uy_ym;

        tmp_vz -= (M_yp - M_ym)*inv_dy;

        /* == z direction == */
        double vz_zp= 0.5*(mfz(ix,iy,iz+1)+mfz(ix,iy,iz));
        double vz_zm= 0.5*(mfz(ix,iy,iz)+mfz(ix,iy,iz-1));


        /* == upwind == */
        /*
           double uz_zp= vz_zp > 0. ? vz_111: vz_112;
           double uz_zm= vz_zm > 0. ? vz_110: vz_111;
         */

        /* == muscl vanleer == */
        ind_upwind = vz_zp > 0. ? iz: iz+1;
        double uz_zp = 0.;

        /*== check if has stencil == */
        if(f_ztype(ix,iy,ind_upwind) != F_INTERIOR){
            uz_zp = vz_zp > 0. ? vz_111:vz_112;
        }else{

            double deltap = d_get_vz_zface(grid,ix,iy,ind_upwind+1) - vz(ix,iy,ind_upwind);
            double deltam = vz(ix,iy,ind_upwind) - d_get_vz_zface(grid,ix,iy,ind_upwind-1); 
            double s_u = d_minmod(deltap,deltam); // higher order term 

            int dir = sgn(vz_zp);
            double upwind_v = vz_zp > 0. ? vz_111:vz_112;
            uz_zp = upwind_v + dir*0.5*s_u;
        }


        double M_zp = vz_zp * uz_zp;

        /* == muscl vanleer == */
        ind_upwind = vz_zm > 0. ? iz-1: iz;
        double uz_zm = 0.;

        /*== check if has stencil == */
        if(f_ztype(ix,iy,ind_upwind) != F_INTERIOR){
            uz_zm = vz_zm > 0. ? vz_110:vz_111;
        }else{
            double deltap = d_get_vz_zface(grid,ix,iy,ind_upwind+1) - vz(ix,iy,ind_upwind);
            double deltam = vz(ix,iy,ind_upwind) - d_get_vz_zface(grid,ix,iy,ind_upwind-1); 

            double s_u = d_minmod(deltap,deltam); // higher order term 

            int dir = sgn(vz_zm);

            double upwind_v = vz_zm > 0. ? vz_110:vz_111;
            uz_zm = upwind_v + dir*0.5*s_u;
        }


        double M_zm = vz_zm * uz_zm;



        tmp_vz -= (M_zp - M_zm)*inv_dz;


        double f_inv_rho = f_bz(ix,iy,iz);
        double f_rho_old= (0.5*(rho_old(ix,iy,iz)+rho_old(ix,iy,iz-1)));

        /* == add pressure gradient == */
        tmp_vz -=  (p(ix,iy,iz) - p(ix,iy,iz-1))*inv_dz;

        const double inv_volume = grid->inv_dx_*grid->inv_dy_*grid->inv_dz_;

        const double coupling_momentum_density = grid->f_coupling_impulse_z_(ix,iy,iz)*inv_volume;

        vz_star(ix,iy,iz)=f_inv_rho*(vz_111*f_rho_old+coupling_momentum_density)+dt*(f_inv_rho*tmp_vz+gz);

        /* == add ibm == */
        double solid_frac = grid->f_ibm_solid_fraction_z_(ix,iy,iz);

        /* == assuming v_ibm = 0 for now. it actually is (1-frac)*v_star + frac *v_ibm*/
        vz_star(ix,iy,iz) = (1.0-solid_frac)*vz_star(ix,iy,iz);

    }

}

/* debug */
__global__ void k_check_vxstar_boundary(G_StaggeredGrid* grid){
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
        int bid = grid->f_xbcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);
        if(bcType == BC_OUTLET){
            /* debug*/
            printf("%d %d %d vx_star = %f \n", ix,iy,iz,vx_star(ix,iy,iz));
            printf("%d %d %d vx_star = %f \n", ix-1,iy,iz,vx_star(ix,iy,iz));
        }

    }else{
        return;
    }
}

void G_SMACSolver::update_vstar_boundary(){
    k_update_vxstar_boundary<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);

    /*debug*/
    //k_check_vxstar_boundary<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);

    k_update_vystar_boundary<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
    k_update_vzstar_boundary<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
}

void G_SMACSolver::update_v_boundary(){
    k_update_vx_boundary<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
    k_update_vy_boundary<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
    k_update_vz_boundary<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
}



void G_SMACSolver::get_vof_vstar_rhouu_consistent(SMACSolver solv){


    k_get_vof_vstar_rhouu_consistent_x<<<grid_dim_,block_dim_>>>(solv,grid_.d_ptr_);
    k_get_vof_vstar_rhouu_consistent_y<<<grid_dim_,block_dim_>>>(solv,grid_.d_ptr_);
    k_get_vof_vstar_rhouu_consistent_z<<<grid_dim_,block_dim_>>>(solv,grid_.d_ptr_);

}


static __global__ void k_correct_vof_velocity(SMACSolver solv, G_StaggeredGrid* grid){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;



    MyArray<unsigned char,3> ct= grid->celltype_;
    MyArray<double,3> p = grid->p_delta_;
    MyArray<double,3> p_new = grid->p_;
    MyArray<double,3> vx = grid->f_vx_;
    MyArray<double,3> vy = grid->f_vy_;
    MyArray<double,3> vz = grid->f_vz_;
    MyArray<double,3> vx_star = grid->f_vx_star_;
    MyArray<double,3> vy_star = grid->f_vy_star_;
    MyArray<double,3> vz_star = grid->f_vz_star_;
    MyArray<double,3> f_inv_rhox = grid->f_inv_rhox_;
    MyArray<double,3> f_inv_rhoy = grid->f_inv_rhoy_;
    MyArray<double,3> f_inv_rhoz = grid->f_inv_rhoz_;

    MyArray<double,3> f_sx = grid->f_sx_;
    MyArray<double,3> f_sy = grid->f_sy_;
    MyArray<double,3> f_sz = grid->f_sz_;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    double inv_dx = grid->inv_dx_;
    double inv_dy = grid->inv_dy_;
    double inv_dz = grid->inv_dz_;


    double dt= solv.dt_;

    /* == fix vx == */
    if(iy>=Ny+2 || ix>= Nx+3 || iz >= Nz+2){
        /* do nothing */
    }else{
        if(grid->f_xtype_(ix,iy,iz) == F_INTERIOR){
            vx(ix,iy,iz) = vx_star(ix,iy,iz) + f_inv_rhox(ix,iy,iz)*dt*(f_sx(ix,iy,iz)-(p(ix,iy,iz)-p(ix-1,iy,iz))*inv_dx);


        }
    }

    /* == fix vy == */
    if(iy>=Ny+3 || ix>= Nx+2 || iz >= Nz+2){
        /* do nothing */
    }else{
        if(grid->f_ytype_(ix,iy,iz) == F_INTERIOR){
            vy(ix,iy,iz) = vy_star(ix,iy,iz) + f_inv_rhoy(ix,iy,iz)*dt*(f_sy(ix,iy,iz)-(p(ix,iy,iz)-p(ix,iy-1,iz))*inv_dy);
        }
    }

    /* == fix vz == */
    if(iy>=Ny+2 || ix>= Nx+2 || iz >= Nz+3){
        /* do nothing */
    }else{
        if(grid->f_ztype_(ix,iy,iz) == F_INTERIOR){
            vz(ix,iy,iz) = vz_star(ix,iy,iz) + f_inv_rhoz(ix,iy,iz)*dt*(f_sz(ix,iy,iz)-(p(ix,iy,iz)-p(ix,iy,iz-1))*inv_dz);
        }
    }

    /* debug */
    if (ix < Nx+1 && iy < Ny+1 && iz < Nz+1){
        if(ct(ix,iy,iz) == C_INTERIOR){
            p_new(ix,iy,iz) += p(ix,iy,iz);
        }
    }
}

static __global__ void k_update_cell_boundary_pressure(SMACSolver solv, G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;


    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;

    double dx = grid->dx_;
    double dy = grid->dy_;
    double dz = grid->dz_;

    MyArray<double,3> p= grid->p_;
    MyArray<double,3> rho= grid->rho_;

    double gx = solv.gx_;
    double gy = solv.gy_;
    double gz = solv.gz_;

    if (ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;


    if(ix == 0){

        /* == update p == */
        p(ix,iy,iz)= p(ix+1,iy,iz) - rho(ix+1,iy,iz)*gx*dx;
    }

    if(ix == Nx+1){
        /* == update p == */
        p(ix,iy,iz)= p(ix-1,iy,iz) + rho(ix-1,iy,iz)*gx*dx;

    }

    if(iy == 0){
        /* == update p == */
        p(ix,iy,iz)= p(ix,iy+1,iz) - rho(ix,iy+1,iz)*gy*dy;

    }

    if(iy == Ny+1){
        /* == update p == */
        p(ix,iy,iz)= p(ix,iy-1,iz) + rho(ix,iy-1,iz)*gy*dy;

    }

    if(iz == 0){
        /* == update p == */
        p(ix,iy,iz)= p(ix,iy,iz+1) - rho(ix,iy,iz+1)*gz*dz;


    }

    if(iz == Nz+1){
        /* == update p == */
        p(ix,iy,iz)= p(ix,iy,iz-1) + rho(ix,iy,iz-1)*gz*dz;

    }

    MyArray<unsigned char,3>& celltype = grid->celltype_;

    if(celltype(ix,iy,iz) == C_SOLID){

        if(celltype(ix-1,iy,iz) == C_INTERIOR){
            p(ix,iy,iz)= p(ix-1,iy,iz) + rho(ix-1,iy,iz)*gx*dx;
        }else if(celltype(ix+1,iy,iz) == C_INTERIOR){
            p(ix,iy,iz)= p(ix+1,iy,iz) - rho(ix+1,iy,iz)*gx*dx;
        }else if(celltype(ix,iy-1,iz) == C_INTERIOR){
            p(ix,iy,iz)= p(ix,iy-1,iz) + rho(ix,iy-1,iz)*gy*dy;
        }else if(celltype(ix,iy+1,iz) == C_INTERIOR){
            p(ix,iy,iz)= p(ix,iy+1,iz) - rho(ix,iy+1,iz)*gy*dy;
        }else if(celltype(ix,iy,iz-1) == C_INTERIOR){
            p(ix,iy,iz)= p(ix,iy,iz-1) + rho(ix,iy,iz-1)*gz*dz;
        }else if(celltype(ix,iy,iz+1) == C_INTERIOR){
            p(ix,iy,iz)= p(ix,iy,iz+1) - rho(ix,iy,iz+1)*gz*dz;
        }else{
            p(ix,iy,iz)=0.;
        }
    }
}

static __global__ void k_update_cell_ghost_pressure(SMACSolver solv, G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >= Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    bool is_ghost =
        ix == 0 || ix == Nx+1 ||
        iy == 0 || iy == Ny+1 ||
        iz == 0 || iz == Nz+1;

    if(!is_ghost) return;

    MyArray<double,3> p = grid->p_;

    double dx = grid->dx_;
    double dy = grid->dy_;
    double dz = grid->dz_;

    MyArray<double,3> rho= grid->rho_;

    double gx = solv.gx_;
    double gy = solv.gy_;
    double gz = solv.gz_;

    int src_ix = ix == 0 ? 1 : ix == Nx+1 ? Nx : ix;
    int src_iy = iy == 0 ? 1 : iy == Ny+1 ? Ny : iy;
    int src_iz = iz == 0 ? 1 : iz == Nz+1 ? Nz : iz;


    double rx = (double)(ix - src_ix)*dx;
    double ry = (double)(iy - src_iy)*dy;
    double rz = (double)(iz - src_iz)*dz;

    p(ix,iy,iz) = p(src_ix,src_iy,src_iz) + rho(src_ix,src_iy,src_iz)*(rx*gx +ry*gy + rz*gz);
}

void G_SMACSolver::correct_vof_velocity(SMACSolver solv){
    k_correct_vof_velocity<<<grid_dim_,block_dim_>>>(solv,grid_.d_ptr_);


    cudaMemset(grid_.p_delta_.data_, 0, sizeof(double) * grid_.p_delta_.size_);
}

static __global__ void k_make_face_gradp_x(G_StaggeredGrid* grid){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    double inv_dx = grid->inv_dx_;
    MyArray<double,3>  p = grid->p_;
    MyArray<double,3>  f_gradp = grid->f_gradp_x_;

    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;


    f_gradp(ix,iy,iz) = (p(ix,iy,iz)-p(ix-1,iy,iz))*inv_dx;

}

static __global__ void k_make_face_gradp_y(G_StaggeredGrid* grid){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    double inv_dy = grid->inv_dy_;
    MyArray<double,3>  p = grid->p_;
    MyArray<double,3>  f_gradp = grid->f_gradp_y_;

    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;


    f_gradp(ix,iy,iz) = (p(ix,iy,iz)-p(ix,iy-1,iz))*inv_dy;

}

static __global__ void k_make_face_gradp_z(G_StaggeredGrid* grid){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    double inv_dz = grid->inv_dz_;
    MyArray<double,3>  p = grid->p_;
    MyArray<double,3>  f_gradp = grid->f_gradp_z_;

    if(ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;


    f_gradp(ix,iy,iz) = (p(ix,iy,iz)-p(ix,iy,iz-1))*inv_dz;

}

void G_SMACSolver::make_face_gradp(){

    k_make_face_gradp_x<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
    k_make_face_gradp_y<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
    k_make_face_gradp_z<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);

}

void G_SMACSolver::update_boundary_ghost(SMACSolver solv){

    k_update_cell_boundary_pressure<<<grid_dim_,block_dim_>>>(solv, grid_.d_ptr_);
    k_update_cell_ghost_pressure<<<grid_dim_,block_dim_>>>(solv, grid_.d_ptr_);

    k_update_vx_outlet<<<grid_dim_,block_dim_>>>(solv,grid_.d_ptr_);
    k_update_vy_outlet<<<grid_dim_,block_dim_>>>(solv,grid_.d_ptr_);
    k_update_vz_outlet<<<grid_dim_,block_dim_>>>(solv,grid_.d_ptr_);

    k_update_vx_ghost<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
    k_update_vx_ghost_corner<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);

    k_update_vy_ghost<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
    k_update_vy_ghost_corner<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);

    k_update_vz_ghost<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
    k_update_vz_ghost_corner<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);

}

static __global__  void  k_calc_cfl(G_StaggeredGrid* grid,double dt){

    int ix=blockIdx.x*blockDim.x+threadIdx.x+1; //+1 for ghost cell
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1; 
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1; 

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;

    if (ix >=Nx+1 || iy >=Ny+1 || iz >= Nz+1) return;


    MyArray<double,3> vx = grid->f_vx_;
    MyArray<double,3> vy = grid->f_vy_;
    MyArray<double,3> vz = grid->f_vz_;
    MyArray<double,3> cfl = grid->cfl_;
    MyArray<double,3> cfl_visc = grid->cfl_visc_;
    MyArray<double,3> inv_rho = grid->inv_rho_;
    MyArray<double,3> mu = grid->mu_;
    double inv_dx = grid->inv_dx_;
    double inv_dy = grid->inv_dy_;
    double inv_dz = grid->inv_dz_;
    double inv_dx2 = grid->inv_dx2_;
    double inv_dy2 = grid->inv_dy2_;
    double inv_dz2 = grid->inv_dz2_;


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
    cfl_visc(ix-1,iy-1,iz-1) = inv_rho(ix,iy,iz)*mu(ix,iy,iz)*dt*(inv_dx2 + inv_dy2 + inv_dz2);
}

double G_SMACSolver::calc_cfl(){
    double dt= dt_;

    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;
    int Nz=grid_.Nz_;

    k_calc_cfl<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,dt);
    cub::DeviceReduce::Max(cub_temp_storage_,cub_temp_storage_bytes_,grid_.cfl_.data_,&d_pcg_scalars_[CFL],Nx*Ny*Nz);
    double h_cfl;
    cudaMemcpy(&h_cfl,&d_pcg_scalars_[CFL],sizeof(double),cudaMemcpyDeviceToHost);

    cub::DeviceReduce::Max(cub_temp_storage_,cub_temp_storage_bytes_,grid_.cfl_visc_.data_,&d_pcg_scalars_[CFL],Nx*Ny*Nz);
    double h_cfl_visc;
    cudaMemcpy(&h_cfl_visc,&d_pcg_scalars_[CFL],sizeof(double),cudaMemcpyDeviceToHost);

    double max_cfl = h_cfl_visc > h_cfl? h_cfl_visc : h_cfl;

    printf("conv CFL=%3.2e, visc CFL = %3.2e, CFL = %3.2e\n",h_cfl,h_cfl_visc, max_cfl);
    return max_cfl;

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
    cudaFree(grid_.d_ptr_);

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

    /* use this temporarly to make big array */
    double *tmp_malloc=nullptr;
    cudaMalloc((void**)&tmp_malloc, sizeof(double)*(Nx+3)*(Ny+3)*(Nz+3));


    size_t max_temp_bytes = 0;
    size_t sum_temp_bytes = 0;

    cub::DeviceReduce::Sum(nullptr, sum_temp_bytes, tmp_malloc, d_r2_,(Nx+3)*(Ny+3)*(Nz+3));
    cub::DeviceReduce::Max(nullptr, max_temp_bytes, tmp_malloc, d_r2_,(Nx+3)*(Ny+3)*(Nz+3));

    cub_temp_storage_bytes_ = std::max(max_temp_bytes, sum_temp_bytes);

    cudaMalloc((void**)&cub_temp_storage_, cub_temp_storage_bytes_);

    cudaFree(tmp_malloc);

    /* malloc struct */
    cudaMalloc((void**)&grid_.d_ptr_,sizeof(G_StaggeredGrid));
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

    cudaMemcpy(grid_.d_ptr_,&grid_,sizeof(G_StaggeredGrid),cudaMemcpyHostToDevice);

}

void G_SMACSolver::gpuTocpu(StaggeredGrid h_grid){
    int Nx = h_grid.Nx_;
    int Ny = h_grid.Ny_;
    int Nz = h_grid.Nz_;

    #define MEMBER(type,name,sizex,sizey,sizez,SAVE_FLAG) cudaMemcpy(h_grid.name.data_,grid_.name.data_,(sizex*sizey*sizez)*sizeof(type),cudaMemcpyDeviceToHost); 
    #include "memberList/gridMembers.def"
    #undef MEMBER



}

/* ==== two way coupling === */
static __global__ void k_get_vof_vstar_rhouu_consistent_x_two_way(SMACSolver solv,G_StaggeredGrid* grid){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;

    if(ix>=Nx+2 || iy>=Ny+2 || iz>=Nz+2) return;

    double inv_dx=grid->inv_dx_;
    double inv_dy=grid->inv_dy_;
    double inv_dz=grid->inv_dz_;
    double inv_dx2=grid->inv_dx2_;
    double dt=solv.dt_;
    double gx=solv.gx_;

    MyArray<double,3> p=grid->p_;
    MyArray<double,3> mfx=grid->f_mfx_;
    MyArray<double,3> mfy=grid->f_mfy_;
    MyArray<double,3> mfz=grid->f_mfz_;
    MyArray<double,3> rho_old=grid->rho_old_;
    MyArray<double,3> mu=grid->mu_;
    MyArray<double,3> f_muy=grid->f_muy_;
    MyArray<double,3> f_muz=grid->f_muz_;
    MyArray<double,3> vx=grid->f_vx_;
    MyArray<double,3> f_inv_rhox=grid->f_inv_rhox_;
    MyArray<double,3> vx_star=grid->f_vx_star_;
    MyArray<double,3> eps_old=grid->void_fraction_old_;
    MyArray<double,3> eps_new=grid->void_fraction_;
    MyArray<unsigned char,3> f_xtype=grid->f_xtype_;

    if(f_xtype(ix,iy,iz)!=F_INTERIOR) return;

    double vx_211=d_get_vx_xface(grid,ix+1,iy,iz);
    double vx_111=vx(ix,iy,iz);
    double vx_011=d_get_vx_xface(grid,ix-1,iy,iz);
    double vx_101=d_get_vx_ydir(grid,ix,iy,iz,-1);
    double vx_110=d_get_vx_zdir(grid,ix,iy,iz,-1);
    double vx_121=d_get_vx_ydir(grid,ix,iy,iz,+1);
    double vx_112=d_get_vx_zdir(grid,ix,iy,iz,+1);

    double vy_121=d_get_vy_yface(grid,ix,iy+1,iz);
    double vy_021=d_get_vy_yface(grid,ix-1,iy+1,iz);
    double vy_111=d_get_vy_yface(grid,ix,iy,iz);
    double vy_011=d_get_vy_yface(grid,ix-1,iy,iz);

    double vz_112=d_get_vz_zface(grid,ix,iy,iz+1);
    double vz_012=d_get_vz_zface(grid,ix-1,iy,iz+1);
    double vz_111=d_get_vz_zface(grid,ix,iy,iz);
    double vz_011=d_get_vz_zface(grid,ix-1,iy,iz);

    double eps_old_x=0.5*(eps_old(ix-1,iy,iz)+eps_old(ix,iy,iz));
    double eps_new_x=0.5*(eps_new(ix-1,iy,iz)+eps_new(ix,iy,iz));

    double tmp_vx=0.0;

    /* viscous: d(epsilon*tau_xx)/dx */
    double tau_xp=mu(ix,iy,iz)*(vx_211-vx_111);
    double tau_xm=mu(ix-1,iy,iz)*(vx_111-vx_011);
    double eps_xx_p=eps_old(ix,iy,iz);
    double eps_xx_m=eps_old(ix-1,iy,iz);

    tmp_vx=2.0*(eps_xx_p*tau_xp-eps_xx_m*tau_xm)*inv_dx2;

    /* viscous: d(epsilon*tau_xy)/dy */
    double mu_yp=0.5*(f_muy(ix,iy+1,iz)+f_muy(ix-1,iy+1,iz));
    double tau_yp=mu_yp*((vy_121-vy_021)*inv_dx+(vx_121-vx_111)*inv_dy);
    double mu_ym=0.5*(f_muy(ix,iy,iz)+f_muy(ix-1,iy,iz));
    double tau_ym=mu_ym*((vy_111-vy_011)*inv_dx+(vx_111-vx_101)*inv_dy);

    double eps_xy_p=0.25*(eps_old(ix-1,iy,iz)+eps_old(ix,iy,iz)+eps_old(ix-1,iy+1,iz)+eps_old(ix,iy+1,iz));
    double eps_xy_m=0.25*(eps_old(ix-1,iy-1,iz)+eps_old(ix,iy-1,iz)+eps_old(ix-1,iy,iz)+eps_old(ix,iy,iz));

    tmp_vx+=(eps_xy_p*tau_yp-eps_xy_m*tau_ym)*inv_dy;

    /* viscous: d(epsilon*tau_xz)/dz */
    double mu_zp=0.5*(f_muz(ix,iy,iz+1)+f_muz(ix-1,iy,iz+1));
    double tau_zp=mu_zp*((vz_112-vz_012)*inv_dx+(vx_112-vx_111)*inv_dz);
    double mu_zm=0.5*(f_muz(ix,iy,iz)+f_muz(ix-1,iy,iz));
    double tau_zm=mu_zm*((vz_111-vz_011)*inv_dx+(vx_111-vx_110)*inv_dz);

    double eps_xz_p=0.25*(eps_old(ix-1,iy,iz)+eps_old(ix,iy,iz)+eps_old(ix-1,iy,iz+1)+eps_old(ix,iy,iz+1));
    double eps_xz_m=0.25*(eps_old(ix-1,iy,iz-1)+eps_old(ix,iy,iz-1)+eps_old(ix-1,iy,iz)+eps_old(ix,iy,iz));

    tmp_vx+=(eps_xz_p*tau_zp-eps_xz_m*tau_zm)*inv_dz;

    /* convection x */
    double vx_xp=0.5*(mfx(ix,iy,iz)+mfx(ix+1,iy,iz));
    double vx_xm=0.5*(mfx(ix-1,iy,iz)+mfx(ix,iy,iz));

    int ind_upwind=vx_xp>0.0?ix:ix+1;
    double ux_xp=0.0;

    if(f_xtype(ind_upwind,iy,iz)!=F_INTERIOR){
        ux_xp=vx_xp>0.0?vx_111:vx_211;
    }else{
        double deltap=d_get_vx_xface(grid,ind_upwind+1,iy,iz)-d_get_vx_xface(grid,ind_upwind,iy,iz);
        double deltam=d_get_vx_xface(grid,ind_upwind,iy,iz)-d_get_vx_xface(grid,ind_upwind-1,iy,iz);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(vx_xp);
        double upwind_v=vx_xp>0.0?vx_111:vx_211;
        ux_xp=upwind_v+dir*0.5*s_u;
    }

    double M_xp=vx_xp*ux_xp;

    ind_upwind=vx_xm>0.0?ix-1:ix;
    double ux_xm=0.0;

    if(f_xtype(ind_upwind,iy,iz)!=F_INTERIOR){
        ux_xm=vx_xm>0.0?vx_011:vx_111;
    }else{
        double deltap=d_get_vx_xface(grid,ind_upwind+1,iy,iz)-d_get_vx_xface(grid,ind_upwind,iy,iz);
        double deltam=d_get_vx_xface(grid,ind_upwind,iy,iz)-d_get_vx_xface(grid,ind_upwind-1,iy,iz);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(vx_xm);
        double upwind_v=vx_xm>0.0?vx_011:vx_111;
        ux_xm=upwind_v+dir*0.5*s_u;
    }

    double M_xm=vx_xm*ux_xm;

    tmp_vx-=(M_xp-M_xm)*inv_dx;

    /* convection y */
    double vy_yp=0.5*(mfy(ix-1,iy+1,iz)+mfy(ix,iy+1,iz));
    double vy_ym=0.5*(mfy(ix,iy,iz)+mfy(ix-1,iy,iz));

    ind_upwind=vy_yp>0.0?iy:iy+1;
    double uy_yp=0.0;

    if(f_xtype(ix,ind_upwind,iz)!=F_INTERIOR){
        uy_yp=vy_yp>0.0?vx_111:vx_121;
    }else{
        double deltap=d_get_vx_ydir(grid,ix,ind_upwind,iz,+1)-vx(ix,ind_upwind,iz);
        double deltam=vx(ix,ind_upwind,iz)-d_get_vx_ydir(grid,ix,ind_upwind,iz,-1);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(vy_yp);
        double upwind_v=vy_yp>0.0?vx_111:vx_121;
        uy_yp=upwind_v+dir*0.5*s_u;
    }

    double M_yp=vy_yp*uy_yp;

    ind_upwind=vy_ym>0.0?iy-1:iy;
    double uy_ym=0.0;

    if(f_xtype(ix,ind_upwind,iz)!=F_INTERIOR){
        uy_ym=vy_ym>0.0?vx_101:vx_111;
    }else{
        double deltap=d_get_vx_ydir(grid,ix,ind_upwind,iz,+1)-vx(ix,ind_upwind,iz);
        double deltam=vx(ix,ind_upwind,iz)-d_get_vx_ydir(grid,ix,ind_upwind,iz,-1);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(vy_ym);
        double upwind_v=vy_ym>0.0?vx_101:vx_111;
        uy_ym=upwind_v+dir*0.5*s_u;
    }

    double M_ym=vy_ym*uy_ym;

    tmp_vx-=(M_yp-M_ym)*inv_dy;

    /* convection z */
    double vz_zp=0.5*(mfz(ix-1,iy,iz+1)+mfz(ix,iy,iz+1));
    double vz_zm=0.5*(mfz(ix,iy,iz)+mfz(ix-1,iy,iz));

    ind_upwind=vz_zp>0.0?iz:iz+1;
    double uz_zp=0.0;

    if(f_xtype(ix,iy,ind_upwind)!=F_INTERIOR){
        uz_zp=vz_zp>0.0?vx_111:vx_112;
    }else{
        double deltap=d_get_vx_zdir(grid,ix,iy,ind_upwind,+1)-vx(ix,iy,ind_upwind);
        double deltam=vx(ix,iy,ind_upwind)-d_get_vx_zdir(grid,ix,iy,ind_upwind,-1);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(vz_zp);
        double upwind_v=vz_zp>0.0?vx_111:vx_112;
        uz_zp=upwind_v+dir*0.5*s_u;
    }

    double M_zp=vz_zp*uz_zp;

    ind_upwind=vz_zm>0.0?iz-1:iz;
    double uz_zm=0.0;

    if(f_xtype(ix,iy,ind_upwind)!=F_INTERIOR){
        uz_zm=vz_zm>0.0?vx_110:vx_111;
    }else{
        double deltap=d_get_vx_zdir(grid,ix,iy,ind_upwind,+1)-vx(ix,iy,ind_upwind);
        double deltam=vx(ix,iy,ind_upwind)-d_get_vx_zdir(grid,ix,iy,ind_upwind,-1);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(vz_zm);
        double upwind_v=vz_zm>0.0?vx_110:vx_111;
        uz_zm=upwind_v+dir*0.5*s_u;
    }

    double M_zm=vz_zm*uz_zm;

    tmp_vx-=(M_zp-M_zm)*inv_dz;

    /* old pressure: -epsilon^n grad(p^n) */
    tmp_vx-=eps_old_x*(p(ix,iy,iz)-p(ix-1,iy,iz))*inv_dx;

    /* old and new face mass */
    double f_inv_rho=f_inv_rhox(ix,iy,iz);
    double f_rho_old=0.5*(rho_old(ix,iy,iz)+rho_old(ix-1,iy,iz));

    const double inv_volume=grid->inv_dx_*grid->inv_dy_*grid->inv_dz_;
    const double coupling_momentum_density=grid->f_coupling_impulse_x_(ix,iy,iz)*inv_volume;

    /* (epsilon*rho*u)^n -> (epsilon*rho*u)* */
    double inv_eps_rho=f_inv_rho/eps_new_x;
    vx_star(ix,iy,iz)=inv_eps_rho*(eps_old_x*f_rho_old*vx_111+dt*tmp_vx+coupling_momentum_density)+dt*gx;

    /* IBM */
    double solid_frac=grid->f_ibm_solid_fraction_x_(ix,iy,iz);
    vx_star(ix,iy,iz)=(1.0-solid_frac)*vx_star(ix,iy,iz);

}

static __global__ void k_get_vof_vstar_rhouu_consistent_y_two_way(SMACSolver solv,G_StaggeredGrid* grid){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;

    if(ix>=Nx+2 || iy>=Ny+2 || iz>=Nz+2) return;

    double inv_dx=grid->inv_dx_;
    double inv_dy=grid->inv_dy_;
    double inv_dz=grid->inv_dz_;
    double inv_dy2=grid->inv_dy2_;
    double dt=solv.dt_;
    double gy=solv.gy_;

    MyArray<double,3> p=grid->p_;
    MyArray<double,3> mfx=grid->f_mfx_;
    MyArray<double,3> mfy=grid->f_mfy_;
    MyArray<double,3> mfz=grid->f_mfz_;
    MyArray<double,3> rho_old=grid->rho_old_;
    MyArray<double,3> mu=grid->mu_;
    MyArray<double,3> f_mux=grid->f_mux_;
    MyArray<double,3> f_muz=grid->f_muz_;
    MyArray<double,3> vy=grid->f_vy_;
    MyArray<double,3> f_inv_rhoy=grid->f_inv_rhoy_;
    MyArray<double,3> vy_star=grid->f_vy_star_;
    MyArray<double,3> eps_old=grid->void_fraction_old_;
    MyArray<double,3> eps_new=grid->void_fraction_;
    MyArray<unsigned char,3> f_ytype=grid->f_ytype_;

    if(f_ytype(ix,iy,iz)!=F_INTERIOR) return;

    double vy_xp=d_get_vy_xdir(grid,ix,iy,iz,+1);
    double vy_xm=d_get_vy_xdir(grid,ix,iy,iz,-1);
    double vy_yp=d_get_vy_yface(grid,ix,iy+1,iz);
    double vy_111=vy(ix,iy,iz);
    double vy_ym=d_get_vy_yface(grid,ix,iy-1,iz);
    double vy_zp=d_get_vy_zdir(grid,ix,iy,iz,+1);
    double vy_zm=d_get_vy_zdir(grid,ix,iy,iz,-1);

    double vx_xp_p=d_get_vx_xface(grid,ix+1,iy,iz);
    double vx_xp_m=d_get_vx_xface(grid,ix+1,iy-1,iz);
    double vx_xm_p=d_get_vx_xface(grid,ix,iy,iz);
    double vx_xm_m=d_get_vx_xface(grid,ix,iy-1,iz);

    double vz_zp_p=d_get_vz_zface(grid,ix,iy,iz+1);
    double vz_zp_m=d_get_vz_zface(grid,ix,iy-1,iz+1);
    double vz_zm_p=d_get_vz_zface(grid,ix,iy,iz);
    double vz_zm_m=d_get_vz_zface(grid,ix,iy-1,iz);

    double eps_old_y=0.5*(eps_old(ix,iy-1,iz)+eps_old(ix,iy,iz));
    double eps_new_y=0.5*(eps_new(ix,iy-1,iz)+eps_new(ix,iy,iz));

    double tmp_vy=0.0;

    /* viscous: d(epsilon*tau_yy)/dy */
    double tau_yp=mu(ix,iy,iz)*(vy_yp-vy_111);
    double tau_ym=mu(ix,iy-1,iz)*(vy_111-vy_ym);
    double eps_yy_p=eps_old(ix,iy,iz);
    double eps_yy_m=eps_old(ix,iy-1,iz);

    tmp_vy=2.0*(eps_yy_p*tau_yp-eps_yy_m*tau_ym)*inv_dy2;

    /* viscous: d(epsilon*tau_yx)/dx */
    double mu_xp=0.5*(f_mux(ix+1,iy,iz)+f_mux(ix+1,iy-1,iz));
    double tau_xp=mu_xp*((vx_xp_p-vx_xp_m)*inv_dy+(vy_xp-vy_111)*inv_dx);
    double mu_xm=0.5*(f_mux(ix,iy,iz)+f_mux(ix,iy-1,iz));
    double tau_xm=mu_xm*((vx_xm_p-vx_xm_m)*inv_dy+(vy_111-vy_xm)*inv_dx);

    double eps_yx_p=0.25*(eps_old(ix,iy-1,iz)+eps_old(ix+1,iy-1,iz)+eps_old(ix,iy,iz)+eps_old(ix+1,iy,iz));
    double eps_yx_m=0.25*(eps_old(ix-1,iy-1,iz)+eps_old(ix,iy-1,iz)+eps_old(ix-1,iy,iz)+eps_old(ix,iy,iz));

    tmp_vy+=(eps_yx_p*tau_xp-eps_yx_m*tau_xm)*inv_dx;

    /* viscous: d(epsilon*tau_yz)/dz */
    double mu_zp=0.5*(f_muz(ix,iy,iz+1)+f_muz(ix,iy-1,iz+1));
    double tau_zp=mu_zp*((vz_zp_p-vz_zp_m)*inv_dy+(vy_zp-vy_111)*inv_dz);
    double mu_zm=0.5*(f_muz(ix,iy,iz)+f_muz(ix,iy-1,iz));
    double tau_zm=mu_zm*((vz_zm_p-vz_zm_m)*inv_dy+(vy_111-vy_zm)*inv_dz);

    double eps_yz_p=0.25*(eps_old(ix,iy-1,iz)+eps_old(ix,iy,iz)+eps_old(ix,iy-1,iz+1)+eps_old(ix,iy,iz+1));
    double eps_yz_m=0.25*(eps_old(ix,iy-1,iz-1)+eps_old(ix,iy,iz-1)+eps_old(ix,iy-1,iz)+eps_old(ix,iy,iz));

    tmp_vy+=(eps_yz_p*tau_zp-eps_yz_m*tau_zm)*inv_dz;

    /* convection x */
    double mx_xp=0.5*(mfx(ix+1,iy-1,iz)+mfx(ix+1,iy,iz));
    double mx_xm=0.5*(mfx(ix,iy-1,iz)+mfx(ix,iy,iz));

    int ind_upwind=mx_xp>0.0?ix:ix+1;
    double uy_xp=0.0;

    if(f_ytype(ind_upwind,iy,iz)!=F_INTERIOR){
        uy_xp=mx_xp>0.0?vy_111:vy_xp;
    }else{
        double deltap=d_get_vy_xdir(grid,ind_upwind,iy,iz,+1)-vy(ind_upwind,iy,iz);
        double deltam=vy(ind_upwind,iy,iz)-d_get_vy_xdir(grid,ind_upwind,iy,iz,-1);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(mx_xp);
        double upwind_v=mx_xp>0.0?vy_111:vy_xp;
        uy_xp=upwind_v+dir*0.5*s_u;
    }

    double M_xp=mx_xp*uy_xp;

    ind_upwind=mx_xm>0.0?ix-1:ix;
    double uy_xm=0.0;

    if(f_ytype(ind_upwind,iy,iz)!=F_INTERIOR){
        uy_xm=mx_xm>0.0?vy_xm:vy_111;
    }else{
        double deltap=d_get_vy_xdir(grid,ind_upwind,iy,iz,+1)-vy(ind_upwind,iy,iz);
        double deltam=vy(ind_upwind,iy,iz)-d_get_vy_xdir(grid,ind_upwind,iy,iz,-1);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(mx_xm);
        double upwind_v=mx_xm>0.0?vy_xm:vy_111;
        uy_xm=upwind_v+dir*0.5*s_u;
    }

    double M_xm=mx_xm*uy_xm;

    tmp_vy-=(M_xp-M_xm)*inv_dx;

    /* convection y */
    double my_yp=0.5*(mfy(ix,iy,iz)+mfy(ix,iy+1,iz));
    double my_ym=0.5*(mfy(ix,iy-1,iz)+mfy(ix,iy,iz));

    ind_upwind=my_yp>0.0?iy:iy+1;
    double uy_yp=0.0;

    if(f_ytype(ix,ind_upwind,iz)!=F_INTERIOR){
        uy_yp=my_yp>0.0?vy_111:vy_yp;
    }else{
        double deltap=d_get_vy_yface(grid,ix,ind_upwind+1,iz)-d_get_vy_yface(grid,ix,ind_upwind,iz);
        double deltam=d_get_vy_yface(grid,ix,ind_upwind,iz)-d_get_vy_yface(grid,ix,ind_upwind-1,iz);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(my_yp);
        double upwind_v=my_yp>0.0?vy_111:vy_yp;
        uy_yp=upwind_v+dir*0.5*s_u;
    }

    double M_yp=my_yp*uy_yp;

    ind_upwind=my_ym>0.0?iy-1:iy;
    double uy_ym=0.0;

    if(f_ytype(ix,ind_upwind,iz)!=F_INTERIOR){
        uy_ym=my_ym>0.0?vy_ym:vy_111;
    }else{
        double deltap=d_get_vy_yface(grid,ix,ind_upwind+1,iz)-d_get_vy_yface(grid,ix,ind_upwind,iz);
        double deltam=d_get_vy_yface(grid,ix,ind_upwind,iz)-d_get_vy_yface(grid,ix,ind_upwind-1,iz);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(my_ym);
        double upwind_v=my_ym>0.0?vy_ym:vy_111;
        uy_ym=upwind_v+dir*0.5*s_u;
    }

    double M_ym=my_ym*uy_ym;

    tmp_vy-=(M_yp-M_ym)*inv_dy;

    /* convection z */
    double mz_zp=0.5*(mfz(ix,iy-1,iz+1)+mfz(ix,iy,iz+1));
    double mz_zm=0.5*(mfz(ix,iy-1,iz)+mfz(ix,iy,iz));

    ind_upwind=mz_zp>0.0?iz:iz+1;
    double uy_zp=0.0;

    if(f_ytype(ix,iy,ind_upwind)!=F_INTERIOR){
        uy_zp=mz_zp>0.0?vy_111:vy_zp;
    }else{
        double deltap=d_get_vy_zdir(grid,ix,iy,ind_upwind,+1)-vy(ix,iy,ind_upwind);
        double deltam=vy(ix,iy,ind_upwind)-d_get_vy_zdir(grid,ix,iy,ind_upwind,-1);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(mz_zp);
        double upwind_v=mz_zp>0.0?vy_111:vy_zp;
        uy_zp=upwind_v+dir*0.5*s_u;
    }

    double M_zp=mz_zp*uy_zp;

    ind_upwind=mz_zm>0.0?iz-1:iz;
    double uy_zm=0.0;

    if(f_ytype(ix,iy,ind_upwind)!=F_INTERIOR){
        uy_zm=mz_zm>0.0?vy_zm:vy_111;
    }else{
        double deltap=d_get_vy_zdir(grid,ix,iy,ind_upwind,+1)-vy(ix,iy,ind_upwind);
        double deltam=vy(ix,iy,ind_upwind)-d_get_vy_zdir(grid,ix,iy,ind_upwind,-1);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(mz_zm);
        double upwind_v=mz_zm>0.0?vy_zm:vy_111;
        uy_zm=upwind_v+dir*0.5*s_u;
    }

    double M_zm=mz_zm*uy_zm;

    tmp_vy-=(M_zp-M_zm)*inv_dz;

    /* old pressure */
    tmp_vy-=eps_old_y*(p(ix,iy,iz)-p(ix,iy-1,iz))*inv_dy;

    double f_inv_rho=f_inv_rhoy(ix,iy,iz);
    double f_rho_old=0.5*(rho_old(ix,iy,iz)+rho_old(ix,iy-1,iz));

    const double inv_volume=grid->inv_dx_*grid->inv_dy_*grid->inv_dz_;
    const double coupling_momentum_density=grid->f_coupling_impulse_y_(ix,iy,iz)*inv_volume;

    double inv_eps_rho=f_inv_rho/eps_new_y;
    vy_star(ix,iy,iz)=inv_eps_rho*(eps_old_y*f_rho_old*vy_111+dt*tmp_vy+coupling_momentum_density)+dt*gy;

    double solid_frac=grid->f_ibm_solid_fraction_y_(ix,iy,iz);
    vy_star(ix,iy,iz)=(1.0-solid_frac)*vy_star(ix,iy,iz);
}

static __global__ void k_get_vof_vstar_rhouu_consistent_z_two_way(SMACSolver solv,G_StaggeredGrid* grid){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;

    if(ix>=Nx+2 || iy>=Ny+2 || iz>=Nz+2) return;

    double inv_dx=grid->inv_dx_;
    double inv_dy=grid->inv_dy_;
    double inv_dz=grid->inv_dz_;
    double inv_dz2=grid->inv_dz2_;
    double dt=solv.dt_;
    double gz=solv.gz_;

    MyArray<double,3> p=grid->p_;
    MyArray<double,3> mfx=grid->f_mfx_;
    MyArray<double,3> mfy=grid->f_mfy_;
    MyArray<double,3> mfz=grid->f_mfz_;
    MyArray<double,3> rho_old=grid->rho_old_;
    MyArray<double,3> mu=grid->mu_;
    MyArray<double,3> f_mux=grid->f_mux_;
    MyArray<double,3> f_muy=grid->f_muy_;
    MyArray<double,3> vz=grid->f_vz_;
    MyArray<double,3> f_inv_rhoz=grid->f_inv_rhoz_;
    MyArray<double,3> vz_star=grid->f_vz_star_;
    MyArray<double,3> eps_old=grid->void_fraction_old_;
    MyArray<double,3> eps_new=grid->void_fraction_;
    MyArray<unsigned char,3> f_ztype=grid->f_ztype_;

    if(f_ztype(ix,iy,iz)!=F_INTERIOR) return;

    double vz_xp=d_get_vz_xdir(grid,ix,iy,iz,+1);
    double vz_xm=d_get_vz_xdir(grid,ix,iy,iz,-1);
    double vz_yp=d_get_vz_ydir(grid,ix,iy,iz,+1);
    double vz_ym=d_get_vz_ydir(grid,ix,iy,iz,-1);
    double vz_zp=d_get_vz_zface(grid,ix,iy,iz+1);
    double vz_111=vz(ix,iy,iz);
    double vz_zm=d_get_vz_zface(grid,ix,iy,iz-1);

    double vx_xp_p=d_get_vx_xface(grid,ix+1,iy,iz);
    double vx_xp_m=d_get_vx_xface(grid,ix+1,iy,iz-1);
    double vx_xm_p=d_get_vx_xface(grid,ix,iy,iz);
    double vx_xm_m=d_get_vx_xface(grid,ix,iy,iz-1);

    double vy_yp_p=d_get_vy_yface(grid,ix,iy+1,iz);
    double vy_yp_m=d_get_vy_yface(grid,ix,iy+1,iz-1);
    double vy_ym_p=d_get_vy_yface(grid,ix,iy,iz);
    double vy_ym_m=d_get_vy_yface(grid,ix,iy,iz-1);

    double eps_old_z=0.5*(eps_old(ix,iy,iz-1)+eps_old(ix,iy,iz));
    double eps_new_z=0.5*(eps_new(ix,iy,iz-1)+eps_new(ix,iy,iz));

    double tmp_vz=0.0;

    /* viscous: d(epsilon*tau_zz)/dz */
    double tau_zp=mu(ix,iy,iz)*(vz_zp-vz_111);
    double tau_zm=mu(ix,iy,iz-1)*(vz_111-vz_zm);
    double eps_zz_p=eps_old(ix,iy,iz);
    double eps_zz_m=eps_old(ix,iy,iz-1);

    tmp_vz=2.0*(eps_zz_p*tau_zp-eps_zz_m*tau_zm)*inv_dz2;

    /* viscous: d(epsilon*tau_zx)/dx */
    double mu_xp=0.5*(f_mux(ix+1,iy,iz)+f_mux(ix+1,iy,iz-1));
    double tau_xp=mu_xp*((vx_xp_p-vx_xp_m)*inv_dz+(vz_xp-vz_111)*inv_dx);
    double mu_xm=0.5*(f_mux(ix,iy,iz)+f_mux(ix,iy,iz-1));
    double tau_xm=mu_xm*((vx_xm_p-vx_xm_m)*inv_dz+(vz_111-vz_xm)*inv_dx);

    double eps_zx_p=0.25*(eps_old(ix,iy,iz-1)+eps_old(ix+1,iy,iz-1)+eps_old(ix,iy,iz)+eps_old(ix+1,iy,iz));
    double eps_zx_m=0.25*(eps_old(ix-1,iy,iz-1)+eps_old(ix,iy,iz-1)+eps_old(ix-1,iy,iz)+eps_old(ix,iy,iz));

    tmp_vz+=(eps_zx_p*tau_xp-eps_zx_m*tau_xm)*inv_dx;

    /* viscous: d(epsilon*tau_zy)/dy */
    double mu_yp=0.5*(f_muy(ix,iy+1,iz)+f_muy(ix,iy+1,iz-1));
    double tau_yp=mu_yp*((vy_yp_p-vy_yp_m)*inv_dz+(vz_yp-vz_111)*inv_dy);
    double mu_ym=0.5*(f_muy(ix,iy,iz)+f_muy(ix,iy,iz-1));
    double tau_ym=mu_ym*((vy_ym_p-vy_ym_m)*inv_dz+(vz_111-vz_ym)*inv_dy);

    double eps_zy_p=0.25*(eps_old(ix,iy,iz-1)+eps_old(ix,iy+1,iz-1)+eps_old(ix,iy,iz)+eps_old(ix,iy+1,iz));
    double eps_zy_m=0.25*(eps_old(ix,iy-1,iz-1)+eps_old(ix,iy,iz-1)+eps_old(ix,iy-1,iz)+eps_old(ix,iy,iz));

    tmp_vz+=(eps_zy_p*tau_yp-eps_zy_m*tau_ym)*inv_dy;

    /* convection x */
    double mx_xp=0.5*(mfx(ix+1,iy,iz-1)+mfx(ix+1,iy,iz));
    double mx_xm=0.5*(mfx(ix,iy,iz-1)+mfx(ix,iy,iz));

    int ind_upwind=mx_xp>0.0?ix:ix+1;
    double uz_xp=0.0;

    if(f_ztype(ind_upwind,iy,iz)!=F_INTERIOR){
        uz_xp=mx_xp>0.0?vz_111:vz_xp;
    }else{
        double deltap=d_get_vz_xdir(grid,ind_upwind,iy,iz,+1)-vz(ind_upwind,iy,iz);
        double deltam=vz(ind_upwind,iy,iz)-d_get_vz_xdir(grid,ind_upwind,iy,iz,-1);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(mx_xp);
        double upwind_v=mx_xp>0.0?vz_111:vz_xp;
        uz_xp=upwind_v+dir*0.5*s_u;
    }

    double M_xp=mx_xp*uz_xp;

    ind_upwind=mx_xm>0.0?ix-1:ix;
    double uz_xm=0.0;

    if(f_ztype(ind_upwind,iy,iz)!=F_INTERIOR){
        uz_xm=mx_xm>0.0?vz_xm:vz_111;
    }else{
        double deltap=d_get_vz_xdir(grid,ind_upwind,iy,iz,+1)-vz(ind_upwind,iy,iz);
        double deltam=vz(ind_upwind,iy,iz)-d_get_vz_xdir(grid,ind_upwind,iy,iz,-1);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(mx_xm);
        double upwind_v=mx_xm>0.0?vz_xm:vz_111;
        uz_xm=upwind_v+dir*0.5*s_u;
    }

    double M_xm=mx_xm*uz_xm;

    tmp_vz-=(M_xp-M_xm)*inv_dx;

    /* convection y */
    double my_yp=0.5*(mfy(ix,iy+1,iz-1)+mfy(ix,iy+1,iz));
    double my_ym=0.5*(mfy(ix,iy,iz-1)+mfy(ix,iy,iz));

    ind_upwind=my_yp>0.0?iy:iy+1;
    double uz_yp=0.0;

    if(f_ztype(ix,ind_upwind,iz)!=F_INTERIOR){
        uz_yp=my_yp>0.0?vz_111:vz_yp;
    }else{
        double deltap=d_get_vz_ydir(grid,ix,ind_upwind,iz,+1)-vz(ix,ind_upwind,iz);
        double deltam=vz(ix,ind_upwind,iz)-d_get_vz_ydir(grid,ix,ind_upwind,iz,-1);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(my_yp);
        double upwind_v=my_yp>0.0?vz_111:vz_yp;
        uz_yp=upwind_v+dir*0.5*s_u;
    }

    double M_yp=my_yp*uz_yp;

    ind_upwind=my_ym>0.0?iy-1:iy;
    double uz_ym=0.0;

    if(f_ztype(ix,ind_upwind,iz)!=F_INTERIOR){
        uz_ym=my_ym>0.0?vz_ym:vz_111;
    }else{
        double deltap=d_get_vz_ydir(grid,ix,ind_upwind,iz,+1)-vz(ix,ind_upwind,iz);
        double deltam=vz(ix,ind_upwind,iz)-d_get_vz_ydir(grid,ix,ind_upwind,iz,-1);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(my_ym);
        double upwind_v=my_ym>0.0?vz_ym:vz_111;
        uz_ym=upwind_v+dir*0.5*s_u;
    }

    double M_ym=my_ym*uz_ym;

    tmp_vz-=(M_yp-M_ym)*inv_dy;

    /* convection z */
    double mz_zp=0.5*(mfz(ix,iy,iz)+mfz(ix,iy,iz+1));
    double mz_zm=0.5*(mfz(ix,iy,iz-1)+mfz(ix,iy,iz));

    ind_upwind=mz_zp>0.0?iz:iz+1;
    double uz_zp=0.0;

    if(f_ztype(ix,iy,ind_upwind)!=F_INTERIOR){
        uz_zp=mz_zp>0.0?vz_111:vz_zp;
    }else{
        double deltap=d_get_vz_zface(grid,ix,iy,ind_upwind+1)-d_get_vz_zface(grid,ix,iy,ind_upwind);
        double deltam=d_get_vz_zface(grid,ix,iy,ind_upwind)-d_get_vz_zface(grid,ix,iy,ind_upwind-1);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(mz_zp);
        double upwind_v=mz_zp>0.0?vz_111:vz_zp;
        uz_zp=upwind_v+dir*0.5*s_u;
    }

    double M_zp=mz_zp*uz_zp;

    ind_upwind=mz_zm>0.0?iz-1:iz;
    double uz_zm=0.0;

    if(f_ztype(ix,iy,ind_upwind)!=F_INTERIOR){
        uz_zm=mz_zm>0.0?vz_zm:vz_111;
    }else{
        double deltap=d_get_vz_zface(grid,ix,iy,ind_upwind+1)-d_get_vz_zface(grid,ix,iy,ind_upwind);
        double deltam=d_get_vz_zface(grid,ix,iy,ind_upwind)-d_get_vz_zface(grid,ix,iy,ind_upwind-1);
        double s_u=d_minmod(deltap,deltam);
        int dir=sgn(mz_zm);
        double upwind_v=mz_zm>0.0?vz_zm:vz_111;
        uz_zm=upwind_v+dir*0.5*s_u;
    }

    double M_zm=mz_zm*uz_zm;

    tmp_vz-=(M_zp-M_zm)*inv_dz;

    /* old pressure */
    tmp_vz-=eps_old_z*(p(ix,iy,iz)-p(ix,iy,iz-1))*inv_dz;

    double f_inv_rho=f_inv_rhoz(ix,iy,iz);
    double f_rho_old=0.5*(rho_old(ix,iy,iz)+rho_old(ix,iy,iz-1));

    const double inv_volume=grid->inv_dx_*grid->inv_dy_*grid->inv_dz_;
    const double coupling_momentum_density=grid->f_coupling_impulse_z_(ix,iy,iz)*inv_volume;

    double inv_eps_rho=f_inv_rho/eps_new_z;
    vz_star(ix,iy,iz)=inv_eps_rho*(eps_old_z*f_rho_old*vz_111+dt*tmp_vz+coupling_momentum_density)+dt*gz;

    double solid_frac=grid->f_ibm_solid_fraction_z_(ix,iy,iz);
    vz_star(ix,iy,iz)=(1.0-solid_frac)*vz_star(ix,iy,iz);
}

void G_SMACSolver::get_vof_vstar_rhouu_consistent_two_way(SMACSolver solv){


    k_get_vof_vstar_rhouu_consistent_x_two_way<<<grid_dim_,block_dim_>>>(solv,grid_.d_ptr_);
    k_get_vof_vstar_rhouu_consistent_y_two_way<<<grid_dim_,block_dim_>>>(solv,grid_.d_ptr_);
    k_get_vof_vstar_rhouu_consistent_z_two_way<<<grid_dim_,block_dim_>>>(solv,grid_.d_ptr_);

}

