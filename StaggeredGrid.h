#pragma once
#include "MyArray.h"
#include "BoundaryCondition.h"
/* === grid class === */

struct StaggeredGrid{
    double dx_, dy_,dz_;
    double inv_dx_, inv_dy_,inv_dz_;
    double inv_2dx_, inv_2dy_,inv_2dz_;
    double inv_dx2_, inv_dy2_,inv_dz2_;
    int Nx_,Ny_,Nz_;
    double sizex_, sizey_,sizez_;

    /* boundary condition */
    BoundaryCondition bc_;



    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) MyArray<type,3> name;
    #include "memberList/gridMembers.def"
    #undef MEMBER

    void set_boundary_velocity();
    void set_num_bc_id(int num_bc_id);
    void place_vof(double minx,double maxx, double miny, double maxy,double minz, double maxz, double alpha);
    void get_cell_coord();
};

inline void StaggeredGrid::set_boundary_velocity(){
    /* == temporarly for debug == */
    int bc_id = 1;
    bc_.vx_(bc_id)=2.0;
}

inline void StaggeredGrid::set_num_bc_id(int num_bc_id){
    bc_.num_boundary_id_ = num_bc_id;
}

inline void StaggeredGrid::get_cell_coord(){
    int Nx = Nx_;
    int Ny = Ny_;
    int Nz = Nz_;
    double dx=dx_;
    double dy=dy_;
    double dz=dz_;
    MyArray<double,3> x = x_;
    MyArray<double,3> y = y_;
    MyArray<double,3> z = z_;

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

    for(int iz=1; iz<Nz+1; iz++){
        for(int iy=1; iy<Ny+1; iy++){
            for(int ix=1; ix<Nx+1; ix++){
                if(x_(ix,iy,iz)<maxx && x_(ix,iy,iz)>minx){
                    if(y_(ix,iy,iz)<maxy && y_(ix,iy,iz)>miny){
                        if(z_(ix,iy,iz)<maxz && z_(ix,iy,iz)>minz){
                            alpha_(ix,iy,iz)=alpha;
                        }
                    }
                }
            }
        }

    }
}


