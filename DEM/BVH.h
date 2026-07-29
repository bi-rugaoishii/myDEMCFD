#pragma once
#ifndef _BVH_H_
#define _BVH_H_

#include "TriangleMesh.h"
#include "ParticleSystem.h"
#include "TriangleContactCache.h"

/*
 * ==============================
 * for GPU
 * ==============================
 */
typedef struct DeviceBVH{
    #define MEMBER(type,name,size) type* name;
    #include "memberList/BVHMember_common.def"
    #undef MEMBER

    int nodeCount;
} DeviceBVH;

typedef struct BVH{
    DeviceBVH d_bvh;
    DeviceBVH* d_bvhPtr;

    #define MEMBER(type,name,size) type* name;
    #include "memberList/BVHMember_common.def"
    #undef MEMBER

    int nodeCount;
} BVH;

void initializeBVH(BVH *bvh, int numTriangles, int isGPUon);

void update_neighborlist_wall_nobvh(ParticleSys<HostMemory> *p,TriangleMesh *mesh,BoundingBox *box,double skinR);
TriangleContactCache dist_triangle_neighbor(ParticleSys<HostMemory>* ps, int i, TriangleMesh* mesh, int j,double skinR);

void update_neighborlist_wall(ParticleSys<HostMemory> *p,TriangleMesh *mesh, BVH *bvh,double skinR);

void copyToDeviceBVH(BVH *bvh, int numTriangles);

void free_BVH(BVH *bvh, int isGPUon);
void computeNodeAABB(int start, int end,TriangleMesh *mesh, BVH *bvh,int k);



int buildBVH(BVH* bvh,TriangleMesh *mesh);

/* == for device == */
__global__ void k_update_neighborlist_wall(ParticleSys<DeviceMemory> *p, DeviceTriangleMesh *mesh, DeviceBVH *bvh,double skinR);

__device__ __forceinline__ int d_sphereAABBOverlapNeighbor(ParticleSys<DeviceMemory>* p,int i,DeviceBVH *bvh, int j, double skinR);

__device__ __forceinline__ TriangleContactCache d_dist_triangle_neighbor(ParticleSys<DeviceMemory>* ps, int i, DeviceTriangleMesh* mesh, int j,double skinR);
#endif
