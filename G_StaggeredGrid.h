#pragma once
#include "MyArray.h"
#include "Enums.h"
/* === grid class === */

struct G_StaggeredGrid{
    double dx_, dy_,dz_;
    double inv_dx_, inv_dy_, inv_dz_;
    double inv_2dx_, inv_2dy_, inv_2dz_;
    double inv_dx2_, inv_dy2_, inv_dz2_;
    int Nx_,Ny_,Nz_;
    double sizex_, sizey_,sizez_;

    /* boundary condition */
    /* for faces */
    double v_b_1_; // velocity at the boundary
    double v_b_2_; // velocity at the boundary



    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) MyArray<type,3> name;
    #include "memberList/gridMembers.def"
    #undef MEMBER

};

