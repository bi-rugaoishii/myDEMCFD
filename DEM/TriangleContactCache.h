
#ifndef _TRIANGLECONTACTCACHE_H_
#define _TRIANGLECONTACTCACHE_H_

#include "Vec3.h"

typedef struct TriangleContactCache{
    Vec3 n;
    double dist;
    int hitAt; //-1 = face, else either at face or vertices with its ID


    #if USE_GPU
    __host__ __device__
    TriangleContactCache(){}

    #endif

}TriangleContactCache;

#endif
