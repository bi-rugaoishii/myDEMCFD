#pragma once
#include <math.h>
#include "MyArray.h"
#include <cuda_runtime.h>

__global__ void k_debug(G_StaggeredGrid grid){
    printf("bc1 = %f\n",grid.bc_.vx_(1));
}

__global__ void k_print_array(MyArray<double,3> array){
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int ix = blockIdx.x*blockDim.x + threadIdx.x;

    if(ix >= array.sizex_ || iy >= array.sizey_ || iz >= array.sizez_ ) return;

    printf("%d %d %d %e\n",ix,iy,iz,array(ix,iy,iz));
}
