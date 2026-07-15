#pragma once
#include "MyArray.h"
#include "Enums.h"
#include "G_BoundaryCondition.h"
/* === grid class === */

struct G_StaggeredGrid{
    double dx_, dy_,dz_;
    double inv_dx_, inv_dy_, inv_dz_;
    double inv_2dx_, inv_2dy_, inv_2dz_;
    double inv_dx2_, inv_dy2_, inv_dz2_;
    int Nx_,Ny_,Nz_;
    double sizex_, sizey_,sizez_;

    G_StaggeredGrid *d_ptr_;

    /* members */

    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) MyArray<type,3> name;
    #include "memberList/gridMembers.def"
    #undef MEMBER

    /* boundary condition */
    G_BoundaryCondition bc_;

};

