#pragma once
#include <cstdio>
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

    void set_boundary_id(AXIS dir,unsigned char bid, int index); 
    void set_num_bc_id(int num_bc_id);
    void place_vof(double minx,double maxx, double miny, double maxy,double minz, double maxz, double alpha);
    void place_solid(double minx,double maxx, double miny, double maxy,double minz, double maxz, unsigned char bid);
    void get_cell_coord();
};

inline void StaggeredGrid::place_solid(double minx,double maxx, double miny, double maxy,double minz, double maxz, unsigned char bid){ 

    int Nx = Nx_;
    int Ny = Ny_;
    int Nz = Nz_;

    for(int iz=1; iz<Nz+1; iz++){
        for(int iy=1; iy<Ny+1; iy++){
            for(int ix=1; ix<Nx+1; ix++){
                if(x_(ix,iy,iz)<maxx && x_(ix,iy,iz)>minx){
                    if(y_(ix,iy,iz)<maxy && y_(ix,iy,iz)>miny){
                        if(z_(ix,iy,iz)<maxz && z_(ix,iy,iz)>minz){
                            celltype_(ix,iy,iz)=C_SOLID;

                            f_xtype_(ix,iy,iz)=F_BOUNDARY;
                            f_xbcid_(ix,iy,iz)=bid;

                            f_xtype_(ix+1,iy,iz)=F_BOUNDARY;
                            f_xbcid_(ix+1,iy,iz)=bid;

                            f_ytype_(ix,iy,iz)=F_BOUNDARY;
                            f_ybcid_(ix,iy,iz)=bid;

                            f_ytype_(ix,iy+1,iz)=F_BOUNDARY;
                            f_ybcid_(ix,iy+1,iz)=bid;

                            f_ztype_(ix,iy,iz)=F_BOUNDARY;
                            f_zbcid_(ix,iy,iz)=bid;

                            f_ztype_(ix,iy,iz+1)=F_BOUNDARY;
                            f_zbcid_(ix,iy,iz+1)=bid;

                        }
                    }
                }
            }
        }


    } 
}


inline void StaggeredGrid::set_boundary_id(AXIS dir,unsigned char bid, int index){ 

    if(dir==AXIS_X){

        if(index >= f_xbcid_.sizex_){
            printf("index %d is over sizex %d!!!!!\n", index, f_xbcid_.sizex_);
        }

        for(int iz=1; iz<=Nz_; iz++){
            for(int iy=1; iy<=Ny_; iy++){
                f_xbcid_(index,iy,iz) = bid;
            }
        }
    }

    if(dir==AXIS_Y){

        if(index >= f_ybcid_.sizey_){
            printf("index %d is over sizey %d!!!!!\n", index, f_ybcid_.sizey_);
        }

        for(int iz=1; iz<=Nz_; iz++){
            for(int ix=1; ix<=Nx_; ix++){
                f_ybcid_(ix,index,iz) = bid;
            }
        }
    }

    if(dir==AXIS_Z){
        if(index >= f_zbcid_.sizez_){
            printf("index %d is over sizez %d!!!!!\n", index, f_zbcid_.sizez_);
        }

        for(int iy=1; iy<=Ny_; iy++){
            for(int ix=1; ix<=Nx_; ix++){
                f_zbcid_(ix,iy,index) = bid;
            }
        }
    }

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


