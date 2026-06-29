#pragma once
#include "MyArray.h"
/* === grid class === */

struct StaggeredGrid{
    double dx_, dy_,dz_;
    double inv_dx_, inv_dy_,inv_dz_;
    double inv_2dx_, inv_2dy_,inv_2dz_;
    double inv_dx2_, inv_dy2_,inv_dz2_;
    int Nx_,Ny_,Nz_;
    double sizex_, sizey_,sizez_;

    /* boundary condition */
    /* for faces */
    double v_b_1_; // velocity at the boundary
    double v_b_2_; // velocity at the boundary

    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) type* name;
    #include "memberList/gridMembers.def"
    #undef MEMBER


    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) MyArray<type,3> wew_##name;
    #include "memberList/gridMembers.def"
    #undef MEMBER

    void place_vof(double minx,double maxx, double miny, double maxy,double alpha);
    void get_cell_coord();
};

inline void StaggeredGrid::get_cell_coord(){
    int Nx = Nx_;
    int Ny = Ny_;
    int Nz = Nz_;
    double dx=dx_;
    double dy=dy_;
    double dz=dz_;
    MyArray<double,3> x = wew_x_;
    MyArray<double,3> y = wew_y_;
    MyArray<double,3> z = wew_z_;

    for(int iz=0; iz<Nz+2; iz++){
        for(int iy=0; iy<Ny+2; iy++){
            for(int ix=0; ix<Nx+2; ix++){

                x(ix,iy,iz)=dx*(double)(ix)-0.5*dx;
                y(ix,iy,iz)=dy*(double)(iy)-0.5*dy;
                z(ix,iy,iz)=dz*(double)(iz)-0.5*dz;
            }
        }
    }
}

inline void StaggeredGrid::place_vof(double minx,double maxx, double miny, double maxy,double minz,double maxz,double alpha){
    int Nx = Nx_;
    int Ny = Ny_;
    int Nz = Nz_;

    for(int iz=1; j<Nx+1; j++){
        for(int iy=1; iy<Ny+1; iy++){
            for(int ix=1; ix<Nx+1; ix++){
                if(wew_x_(ix,iy,iz)<maxx && wew_x_(ix,iy,iz)>minx){
                    if(wew_y_(ix,iy,iz)<maxy && wew_y_(ix,iy,iz)>miny){
                        if(wew_z_(ix,iy,iz)<maxz && wew_z_(ix,iy,iz)>minz){
                            wew_alpha_(ix,iy,iz)=alpha;
                        }
                    }
                }
            }
        }

    }
}


