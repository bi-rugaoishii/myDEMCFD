#include "G_Levels.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>


void G_Levels::free(){

    #define MEMBER(type, name, xshift,yshift,zshift, isSAVE) cudaFree(name.data_);
    #include "../memberList/levelMembers.def"
    #undef MEMBER
}

