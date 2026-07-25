#pragma once
#include "MyArray.h"
#include "G_StaggeredGrid.h"
/*===============================================*/
/* == face boundary related device functions ===*/
/*===============================================*/

 __device__ __forceinline__ double d_get_vx_xface(G_StaggeredGrid* grid,int ix,int iy, int iz){

    unsigned char ftype = grid->f_xtype_(ix,iy,iz);

    if(ftype == F_INTERIOR){
        return grid->f_vx_(ix,iy,iz);
    }

    if(ftype == F_BOUNDARY){
        int bid = grid->f_xbcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);



        if(bcType == BC_INFLOW){
            return grid->bc_.vx_(bid);
        }else if(bcType == BC_NOSLIP){
            return 0.;
        }else if(bcType == BC_OUTLET){

           int int_id_shift = grid->f_xinternal_id_(ix,iy,iz);
           double v_boundary = 0.;
           if(int_id_shift < 0){
               v_boundary = grid->f_vx_(ix-1,iy,iz);
           }else{
               v_boundary = grid->f_vx_(ix+1,iy,iz);
           }


           return v_boundary;
        }
    }

    return 0.;
}

 __device__ __forceinline__ double d_get_vy_yface(G_StaggeredGrid* grid,int ix,int iy, int iz){

    unsigned char ftype = grid->f_ytype_(ix,iy,iz);

    if(ftype == F_INTERIOR){
        return grid->f_vy_(ix,iy,iz);
    }

    if(ftype == F_BOUNDARY){
        int bid = grid->f_ybcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);

        if(bcType == BC_INFLOW){
            return grid->bc_.vy_(bid);
        }else if(bcType == BC_NOSLIP){
            return 0.;
        }else if(bcType == BC_OUTLET){

           int int_id_shift = grid->f_yinternal_id_(ix,iy,iz);
           double v_boundary = 0.;
           if(int_id_shift < 0){
               v_boundary = grid->f_vy_(ix,iy-1,iz);
           }else{
               v_boundary = grid->f_vy_(ix,iy+1,iz);
           }

           return v_boundary;
        }
    }

    return 0.;

}

 __device__ __forceinline__ double d_get_vz_zface(G_StaggeredGrid* grid,int ix,int iy, int iz){

    unsigned char ftype = grid->f_ztype_(ix,iy,iz);

    if(ftype == F_INTERIOR){
        return grid->f_vz_(ix,iy,iz);
    }

    if(ftype == F_BOUNDARY){
        int bid = grid->f_zbcid_(ix,iy,iz);

        unsigned char bcType = grid->bc_.bcType_(bid);

        if(bcType == BC_INFLOW){
            return grid->bc_.vz_(bid);
        }else if(bcType == BC_NOSLIP){
            return 0.;
        }else if(bcType == BC_OUTLET){

           int int_id_shift = grid->f_zinternal_id_(ix,iy,iz);
           double v_boundary = 0.;
           if(int_id_shift < 0){
               v_boundary = grid->f_vz_(ix,iy,iz-1);
           }else{
               v_boundary = grid->f_vz_(ix,iy,iz+1);
           }

           return v_boundary;
        }
    }

    return 0.;

}


__device__ __forceinline__
double d_get_vx_ydir(G_StaggeredGrid* grid,int ix,int iy,int iz,int sy){
    double vx_inside = grid->f_vx_(ix,iy,iz);

    int iy2 = iy + sy;

    if (grid->f_xtype_(ix,iy2,iz) == F_INTERIOR) {
        return grid->f_vx_(ix,iy2,iz);
    }

    int iyf = sy > 0 ? iy + 1 : iy;

    double vx_wall = 0.0;
    int count = 0;

    G_BoundaryCondition& bc=grid->bc_;

    if (grid->f_ytype_(ix,iyf,iz) == F_BOUNDARY) {
        unsigned char bid = grid->f_ybcid_(ix,iyf,iz);
        vx_wall += bc.vx_(bid);
        count++;
    }

    if (grid->f_ytype_(ix-1,iyf,iz) == F_BOUNDARY) {
        unsigned char bid = grid->f_ybcid_(ix-1,iyf,iz);
        vx_wall += bc.vx_(bid);
        count++;
    }


    if (count > 0) {
        vx_wall /= (double)count;
    }

    unsigned char bid = grid->f_ybcid_(ix,iyf,iz);
    unsigned char btype = bc.bcType_(bid);

    if(btype == BC_SLIP){
        return vx_inside;
    }else{   //assuming NO_SLIP for now
        return 2.0*vx_wall - vx_inside;

    }

}

__device__ __forceinline__
double d_get_vx_zdir(G_StaggeredGrid* grid,int ix,int iy,int iz,int sz){
    double vx_inside = grid->f_vx_(ix,iy,iz);

    int iz2 = iz + sz;

    if (grid->f_xtype_(ix,iy,iz2) == F_INTERIOR) {
        return grid->f_vx_(ix,iy,iz2);
    }

    int izf = sz > 0 ? iz + 1 : iz;

    double vx_wall = 0.0;
    int count = 0;

    G_BoundaryCondition& bc=grid->bc_;

    if (grid->f_ztype_(ix,iy,izf) == F_BOUNDARY) {
        unsigned char bid = grid->f_zbcid_(ix,iy,izf);

        vx_wall += bc.vx_(bid);
        count++;
    }

    if (grid->f_ztype_(ix-1,iy,izf) == F_BOUNDARY) {
        unsigned char bid = grid->f_zbcid_(ix-1,iy,izf);
        vx_wall += bc.vx_(bid);
        count++;
    }

    if (count > 0) {
        vx_wall /= (double)count;
    }

    unsigned char bid = grid->f_zbcid_(ix,iy,izf);
    unsigned char btype = bc.bcType_(bid);

    if(btype == BC_SLIP){
        return vx_inside;
    }else{   //assuming NO_SLIP for now
        return 2.0*vx_wall - vx_inside;

    }
}

__device__ __forceinline__
double d_get_vy_xdir(G_StaggeredGrid* grid,int ix,int iy,int iz,int sx){
    double vy_inside = grid->f_vy_(ix,iy,iz);

    int ix2 = ix + sx;

    if (grid->f_ytype_(ix2,iy,iz) == F_INTERIOR) {
        return grid->f_vy_(ix2,iy,iz);
    }

    int ixf = sx > 0 ? ix + 1 : ix;

    double vy_wall = 0.0;
    int count = 0;

    G_BoundaryCondition& bc=grid->bc_;

    if (grid->f_xtype_(ixf,iy,iz) == F_BOUNDARY) {
        unsigned char bid = grid->f_xbcid_(ixf,iy,iz);

        vy_wall += bc.vy_(bid);
        count++;
    }

    if (grid->f_xtype_(ixf,iy-1,iz) == F_BOUNDARY) {
        unsigned char bid = grid->f_xbcid_(ixf,iy-1,iz);
        vy_wall += bc.vy_(bid);
        count++;
    }

    if (count > 0) {
        vy_wall /= (double)count;
    }

    unsigned char bid = grid->f_xbcid_(ixf,iy,iz);
    unsigned char btype = bc.bcType_(bid);

    if(btype == BC_SLIP){
        return vy_inside;
    }else{   //assuming NO_SLIP for now
        return 2.0*vy_wall - vy_inside;

    }
}

__device__ __forceinline__
double d_get_vy_zdir(G_StaggeredGrid* grid,int ix,int iy,int iz,int sz){
    double vy_inside = grid->f_vy_(ix,iy,iz);

    int iz2 = iz + sz;

    if (grid->f_ytype_(ix,iy,iz2) == F_INTERIOR) {
        return grid->f_vy_(ix,iy,iz2);
    }

    int izf = sz > 0 ? iz + 1 : iz;

    double vy_wall = 0.0;
    int count = 0;

    G_BoundaryCondition& bc=grid->bc_;

    if (grid->f_ztype_(ix,iy,izf) == F_BOUNDARY) {
        unsigned char bid = grid->f_zbcid_(ix,iy,izf);

        vy_wall += bc.vy_(bid);
        count++;
    }

    if (grid->f_ztype_(ix,iy-1,izf) == F_BOUNDARY) {
        unsigned char bid = grid->f_zbcid_(ix,iy-1,izf);
        vy_wall += bc.vy_(bid);
        count++;
    }

    if (count > 0) {
        vy_wall /= (double)count;
    }

    unsigned char bid = grid->f_zbcid_(ix,iy,izf);
    unsigned char btype = bc.bcType_(bid);

    if(btype == BC_SLIP){
        return vy_inside;
    }else{   //assuming NO_SLIP for now
        return 2.0*vy_wall - vy_inside;
    }

}

__device__ __forceinline__
double d_get_vz_xdir(G_StaggeredGrid* grid,int ix,int iy,int iz,int sx){
    double vz_inside = grid->f_vz_(ix,iy,iz);

    int ix2 = ix + sx;

    if (grid->f_ztype_(ix2,iy,iz) == F_INTERIOR) {
        return grid->f_vz_(ix2,iy,iz);
    }

    int ixf = sx > 0 ? ix + 1 : ix;

    double vz_wall = 0.0;
    int count = 0;

    G_BoundaryCondition& bc=grid->bc_;

    if (grid->f_xtype_(ixf,iy,iz) == F_BOUNDARY) {
        unsigned char bid = grid->f_xbcid_(ixf,iy,iz);

        vz_wall += bc.vz_(bid);
        count++;
    }

    if (grid->f_xtype_(ixf,iy,iz-1) == F_BOUNDARY) {
        unsigned char bid = grid->f_xbcid_(ixf,iy,iz-1);
        vz_wall += bc.vz_(bid);
        count++;
    }

    if (count > 0) {
        vz_wall /= (double)count;
    }

    unsigned char bid = grid->f_xbcid_(ixf,iy,iz);
    unsigned char btype = bc.bcType_(bid);

    if(btype == BC_SLIP){
        return vz_inside;
    }else{   //assuming NO_SLIP for now
        return 2.0*vz_wall - vz_inside;
    }
}

__device__ __forceinline__
double d_get_vz_ydir(G_StaggeredGrid* grid,int ix,int iy,int iz,int sy){
    double vz_inside = grid->f_vz_(ix,iy,iz);

    int iy2 = iy + sy;

    if (grid->f_ztype_(ix,iy2,iz) == F_INTERIOR) {
        return grid->f_vz_(ix,iy2,iz);
    }

    int iyf = sy > 0 ? iy + 1 : iy;

    double vz_wall = 0.0;
    int count = 0;

    G_BoundaryCondition& bc=grid->bc_;

    if (grid->f_ytype_(ix,iyf,iz) == F_BOUNDARY) {
        unsigned char bid = grid->f_ybcid_(ix,iyf,iz);

        vz_wall += bc.vz_(bid);
        count++;
    }

    if (grid->f_ytype_(ix,iyf,iz-1) == F_BOUNDARY) {
        unsigned char bid = grid->f_ybcid_(ix,iyf,iz-1);
        vz_wall += bc.vz_(bid);
        count++;
    }

    if (count > 0) {
        vz_wall /= (double)count;
    }

    unsigned char bid =grid->f_ybcid_(ix,iyf,iz);
    unsigned char btype = bc.bcType_(bid);

    if(btype == BC_SLIP){
        return vz_inside;
    }else{   //assuming NO_SLIP for now
        return 2.0*vz_wall - vz_inside;
    }
}

 __device__ __forceinline__ double d_get_vxstar_xface(G_StaggeredGrid* grid ,int ix,int iy, int iz){

    unsigned char ftype = grid->f_xtype_(ix,iy,iz);

    if(ftype == F_INTERIOR){
        return grid->f_vx_star_(ix,iy,iz);
    }

    if(ftype == F_BOUNDARY){
        int bid = grid->f_xbcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);



        if(bcType == BC_INFLOW){
            return grid->bc_.vx_(bid);
        }else if(bcType == BC_NOSLIP){
            return 0.;
        }else if(bcType == BC_OUTLET){

           int int_id_shift = grid->f_xinternal_id_(ix,iy,iz);
           double v_boundary = 0.;
           if(int_id_shift < 0){
               v_boundary = grid->f_vx_star_(ix-1,iy,iz);
           }else{
               v_boundary = grid->f_vx_star_(ix+1,iy,iz);
           }


           return v_boundary;
        }
    }

    return 0.;
}

 __device__ __forceinline__ double d_get_vystar_yface(G_StaggeredGrid* grid ,int ix,int iy, int iz){

    unsigned char ftype = grid->f_ytype_(ix,iy,iz);

    if(ftype == F_INTERIOR){
        return grid->f_vy_star_(ix,iy,iz);
    }

    if(ftype == F_BOUNDARY){
        int bid = grid->f_ybcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);



        if(bcType == BC_INFLOW){
            return grid->bc_.vy_(bid);
        }else if(bcType == BC_NOSLIP){
            return 0.;
        }else if(bcType == BC_OUTLET){

           int int_id_shift = grid->f_yinternal_id_(ix,iy,iz);
           double v_boundary = 0.;
           if(int_id_shift < 0){
               v_boundary = grid->f_vy_star_(ix,iy-1,iz);
           }else{
               v_boundary = grid->f_vy_star_(ix,iy+1,iz);
           }


           return v_boundary;
        }
    }

    return 0.;
}



 __device__ __forceinline__ double d_get_vzstar_zface(G_StaggeredGrid* grid ,int ix,int iy, int iz){

    unsigned char ftype = grid->f_ztype_(ix,iy,iz);

    if(ftype == F_INTERIOR){
        return grid->f_vz_star_(ix,iy,iz);
    }

    if(ftype == F_BOUNDARY){
        int bid = grid->f_zbcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);



        if(bcType == BC_INFLOW){
            return grid->bc_.vz_(bid);
        }else if(bcType == BC_NOSLIP){
            return 0.;
        }else if(bcType == BC_OUTLET){

           int int_id_shift = grid->f_zinternal_id_(ix,iy,iz);
           double v_boundary = 0.;
           if(int_id_shift < 0){
               v_boundary = grid->f_vz_star_(ix,iy,iz-1);
           }else{
               v_boundary = grid->f_vz_star_(ix,iy,iz+1);
           }


           return v_boundary;
        }
    }

    return 0.;
}

__global__ void k_update_vxstar_boundary(G_StaggeredGrid* grid);
__global__ void k_update_vystar_boundary(G_StaggeredGrid* grid);
__global__ void k_update_vzstar_boundary(G_StaggeredGrid* grid);

__global__ void k_update_vx_boundary(G_StaggeredGrid* grid);
__global__ void k_update_vy_boundary(G_StaggeredGrid* grid);
__global__ void k_update_vz_boundary(G_StaggeredGrid* grid);
