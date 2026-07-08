#include "../G_StaggeredGrid.h"
#include "../G_SMACSolver.h"
#include "../MyArray.h"
#include <stdio.h>

struct G_Levels{

    double inv_dx2_;
    double inv_dy2_;
    double inv_dz2_;

    bool is_x_coarse_=true;
    bool is_y_coarse_=true;
    bool is_z_coarse_=true;

    #define MEMBER(type, name, xshift,yshift,zshift, isSAVE) MyArray<type,3> name;
    #include "../memberList/levelMembers.def"
    #undef MEMBER


    void free();


};
