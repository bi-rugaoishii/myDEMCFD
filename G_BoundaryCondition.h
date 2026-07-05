#pragma once
#include "Enums.h"

/* == boundary condition class ==*/

struct G_BoundaryCondition{
    int num_boundary_id_;

    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) MyArray<type,3> name;
    #include "memberList/boundaryConditionMembers.def"
    #undef MEMBER

};
