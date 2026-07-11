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

    MyArray<double,3>  rho_old = grid_.rho_old_;
    MyArray<double,3>  mu = grid_.mu_;

    MyArray<double,3>  f_mux = grid_.f_mux_;
    MyArray<double,3>  f_muy = grid_.f_muy_;
    MyArray<double,3>  f_muz = grid_.f_muz_;

    MyArray<double,3>  vx = grid_.f_vx_;
    MyArray<double,3>  vy = grid_.f_vy_;
    MyArray<double,3>  vz = grid_.f_vz_;

    MyArray<double,3>  f_bx = grid_.f_bx_;
    MyArray<double,3>  f_by = grid_.f_by_;
    MyArray<double,3>  f_bz = grid_.f_bz_;


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
            double deltap = d_get_vx_xface(grid_,ind_upwind+1,iy,iz) - d_get_vx_xface(grid_,ind_upwind,iy,iz);
            double deltam = d_get_vx_xface(grid_,ind_upwind,iy,iz) - d_get_vx_xface(grid_,ind_upwind-1,iy,iz); 

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
            double deltap = d_get_vx_xface(grid_,ind_upwind+1,iy,iz) - d_get_vx_xface(grid_,ind_upwind,iy,iz);
            double deltam = d_get_vx_xface(grid_,ind_upwind,iy,iz) - d_get_vx_xface(grid_,ind_upwind-1,iy,iz); 

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

            double deltap = d_get_vx_ydir(grid_,ix,ind_upwind,iz,+1) - vx(ix,ind_upwind,iz);
            double deltam = vx(ix,ind_upwind,iz) - d_get_vx_ydir(grid_,ix,ind_upwind,iz,-1); 

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
            double deltap = d_get_vx_ydir(grid_,ix,ind_upwind,iz,+1) - vx(ix,ind_upwind,iz);
            double deltam = vx(ix,ind_upwind,iz) - d_get_vx_ydir(grid_,ix,ind_upwind,iz,-1); 

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

            double deltap = d_get_vx_zdir(grid_,ix,iy,ind_upwind,+1) - vx(ix,iy,ind_upwind);
            double deltam = vx(ix,iy,ind_upwind) - d_get_vx_zdir(grid_,ix,iy,ind_upwind,-1); 

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
            double deltap = d_get_vx_zdir(grid_,ix,iy,ind_upwind,+1) - vx(ix,iy,ind_upwind);
            double deltam = vx(ix,iy,ind_upwind) - d_get_vx_zdir(grid_,ix,iy,ind_upwind,-1); 

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
            double deltap = d_get_vy_xdir(grid_,ind_upwind,iy,iz,+1) - vy(ind_upwind,iy,iz);
            double deltam = vy(ind_upwind,iy,iz) - d_get_vy_xdir(grid_,ind_upwind,iy,iz,-1); 

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
            double deltap = d_get_vy_xdir(grid_,ind_upwind,iy,iz,+1) - vy(ind_upwind,iy,iz);
            double deltam = vy(ind_upwind,iy,iz) - d_get_vy_xdir(grid_,ind_upwind,iy,iz,-1); 

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

            double deltap = d_get_vy_yface(grid_,ix,ind_upwind+1,iz) - vy(ix,ind_upwind,iz);
            double deltam = vy(ix,ind_upwind,iz) - d_get_vy_yface(grid_,ix,ind_upwind-1,iz); 

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
            double deltap = d_get_vy_yface(grid_,ix,ind_upwind+1,iz) - vy(ix,ind_upwind,iz);
            double deltam = vy(ix,ind_upwind,iz) - d_get_vy_yface(grid_,ix,ind_upwind-1,iz); 


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

            double deltap = d_get_vy_zdir(grid_,ix,iy,ind_upwind,+1) - vy(ix,iy,ind_upwind);
            double deltam = vy(ix,iy,ind_upwind) - d_get_vy_zdir(grid_,ix,iy,ind_upwind,-1); 

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
            double deltap = d_get_vy_zdir(grid_,ix,iy,ind_upwind,+1) - vy(ix,iy,ind_upwind);
            double deltam = vy(ix,iy,ind_upwind) - d_get_vy_zdir(grid_,ix,iy,ind_upwind,-1); 

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
            double deltap = d_get_vz_xdir(grid_,ind_upwind,iy,iz,+1) - vz(ind_upwind,iy,iz);
            double deltam = vz(ind_upwind,iy,iz) - d_get_vz_xdir(grid_,ind_upwind,iy,iz,-1); 

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
            double deltap = d_get_vz_xdir(grid_,ind_upwind,iy,iz,+1) - vz(ind_upwind,iy,iz);
            double deltam = vz(ind_upwind,iy,iz) - d_get_vz_xdir(grid_,ind_upwind,iy,iz,-1); 

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

            double deltap = d_get_vz_ydir(grid_,ix,ind_upwind,iz,+1) - vz(ix,ind_upwind,iz);
            double deltam = vz(ix,ind_upwind,iz) - d_get_vz_ydir(grid_,ix,ind_upwind,iz,-1); 


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
            double deltap = d_get_vz_ydir(grid_,ix,ind_upwind,iz,+1) - vz(ix,ind_upwind,iz);
            double deltam = vz(ix,ind_upwind,iz) - d_get_vz_ydir(grid_,ix,ind_upwind,iz,-1); 


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

            double deltap = d_get_vz_zface(grid_,ix,iy,ind_upwind+1) - vz(ix,iy,ind_upwind);
            double deltam = vz(ix,iy,ind_upwind) - d_get_vz_zface(grid_,ix,iy,ind_upwind-1); 
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
            double deltap = d_get_vz_zface(grid_,ix,iy,ind_upwind+1) - vz(ix,iy,ind_upwind);
            double deltam = vz(ix,iy,ind_upwind) - d_get_vz_zface(grid_,ix,iy,ind_upwind-1); 

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

    MyArray<double,3> f_sx = grid_.f_sx_;
    MyArray<double,3> f_sy = grid_.f_sy_;
    MyArray<double,3> f_sz = grid_.f_sz_;

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
            vx(ix,iy,iz) = vx_star(ix,iy,iz) + f_bx(ix,iy,iz)*dt*(f_sx(ix,iy,iz)-(p(ix,iy,iz)-p(ix-1,iy,iz))*inv_dx);


        }
    }

    /* == fix vy == */
    if(iy>=Ny+3 || ix>= Nx+2 || iz >= Nz+2){
        /* do nothing */
    }else{
        if(grid_.f_ytype_(ix,iy,iz) == F_INTERIOR){
            vy(ix,iy,iz) = vy_star(ix,iy,iz) + f_by(ix,iy,iz)*dt*(f_sy(ix,iy,iz)-(p(ix,iy,iz)-p(ix,iy-1,iz))*inv_dy);
        }
    }

    /* == fix vz == */
    if(iy>=Ny+2 || ix>= Nx+2 || iz >= Nz+3){
        /* do nothing */
    }else{
        if(grid_.f_ztype_(ix,iy,iz) == F_INTERIOR){
            vz(ix,iy,iz) = vz_star(ix,iy,iz) + f_bz(ix,iy,iz)*dt*(f_sz(ix,iy,iz)-(p(ix,iy,iz)-p(ix,iy,iz-1))*inv_dz);
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
    MyArray<double,3> cfl_visc = grid_.cfl_visc_;
    MyArray<double,3> inv_rho = grid_.inv_rho_;
    MyArray<double,3> mu = grid_.mu_;
    double inv_dx = grid_.inv_dx_;
    double inv_dy = grid_.inv_dy_;
    double inv_dz = grid_.inv_dz_;
    double inv_dx2 = grid_.inv_dx2_;
    double inv_dy2 = grid_.inv_dy2_;
    double inv_dz2 = grid_.inv_dz2_;


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

    k_calc_cfl<<<grid_dim_,block_dim_>>>(grid_,dt);
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
