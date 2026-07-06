#pragma once
#include "Enums.h"

/* == boundary condition class ==*/

struct G_BoundaryCondition{
    int num_boundary_id_;

    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) MyArray<type,3> name;
    #include "memberList/boundaryConditionMembers.def"
    #undef MEMBER

    void set_bctype(int side,  BoundaryType Type); 
    void set_boundary_velocity(int bid, double vx, double vy, double vz);

};

inline void G_BoundaryCondition::set_bctype(int bid, BoundaryType Type){
    bcType_(bid)=Type;
}

inline void G_BoundaryCondition::set_boundary_velocity(int bid, double vx, double vy, double vz){
    vx_(bid)=vx;
    vy_(bid)=vy;
    vz_(bid)=vz;
}
