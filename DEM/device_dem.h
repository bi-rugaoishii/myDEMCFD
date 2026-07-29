#pragma once
#ifndef _DEVICE_DEM_H_
#define _DEVICE_DEM_H_

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include "ParticleSystem.h"
#include "BoundingBox.h"
#include "BVH.h"
#include "Vec3.h"
#include "ContactCache.h"
#include "TriangleContactCache.h"


/* =========== triangle related ====== */
__device__ __forceinline__
TriangleContactCache d_dist_triangle(ParticleSys<DeviceMemory>* ps, int i, DeviceTriangleMesh* mesh, int j);

__device__ __forceinline__ void d_particle_collision_verlet(ParticleSys<DeviceMemory>* p, int i ,DeviceBoundingBox *box);

__device__ __forceinline__
void d_wall_collision_triangles(ParticleSys<DeviceMemory>* p,int i,DeviceBoundingBox *box, DeviceTriangleMesh* mesh);

__device__ __forceinline__
void d_wall_collision_verlet(ParticleSys<DeviceMemory>* p,int i,DeviceTriangleMesh* mesh);


__device__ __forceinline__
void updateAcceleration(ParticleSys<DeviceMemory>* p,int i);

__device__ __forceinline__
void updateVelocity(ParticleSys<DeviceMemory>* p,int i);


/* 位置更新（オイラー法） */
__device__ __forceinline__
void updatePosition(ParticleSys<DeviceMemory>* p,
                    int i);

__global__
void k_wall_collision_triangles_naive(ParticleSys<HostMemory>* p,DeviceTriangleMesh* mesh);

/* 床衝突処理 */
__device__ __forceinline__
void resolveFloorCollision(ParticleSys<DeviceMemory>* p,
                           int i,
                           double restitution);

__device__ __forceinline__
ContactCache d_calc_normal_force_wall(ParticleSys<DeviceMemory> *p,int i,int j,Vec3 n,double delMag);

__device__ __forceinline__
ContactCache d_calc_normal_force(ParticleSys<DeviceMemory>* p,int i,int j,Vec3 n,double delMag);

__device__ __forceinline__
void d_calc_tangential_force_wall(ParticleSys<DeviceMemory> *p,int i,int j,ContactCache c);

__device__ __forceinline__
void d_calc_tangential_force(ParticleSys<DeviceMemory> *p,int i,int j,ContactCache c);

__device__ __forceinline__
void d_update_history_wall(ParticleSys<DeviceMemory> *p,int i);


__device__ __forceinline__
void d_update_history(ParticleSys<DeviceMemory> *p,int i);


__device__ __forceinline__
void d_particle_collision_cell_linked(ParticleSys<DeviceMemory>* p, int i, DeviceBoundingBox* box);


__device__ __forceinline__
void d_particle_collision_cell_linked_noVec(ParticleSys<DeviceMemory>* p, int i, DeviceBoundingBox* box);

/* ============== check Out of Bounds ============  */
__global__ void dk_checkOoB(ParticleSys<DeviceMemory> *p, DeviceBoundingBox* box);


/*
============================================================
カーネル
============================================================
*/
__global__ void integrateKernel(ParticleSys<DeviceMemory>* p);

__global__ void check_g_kernel(ParticleSys<DeviceMemory>* p,DeviceTriangleMesh *mesh);

__global__ void k_integrate(ParticleSys<DeviceMemory>* p);

__global__ void k_shouldRefreshNeighborList(ParticleSys<DeviceMemory> *p, DeviceBoundingBox* box);

__global__ void k_collision_verlet_verlet(ParticleSys<DeviceMemory>* p, DeviceBoundingBox* box,DeviceTriangleMesh* mesh);

__global__ void k_collision_triangle(ParticleSys<DeviceMemory>* p, DeviceBoundingBox* box,DeviceTriangleMesh* mesh);

__global__ void k_collision(ParticleSys<DeviceMemory>* p, DeviceBoundingBox* box);

/*
   ======================================================
   main routine 
   ======================================================
*/

void device_dem_naive(ParticleSys<DeviceMemory> *p,BoundingBox *box, TriangleMesh *mesh, BVH *bvh, int gridSize, int blockSize);

void device_dem_verlet_verlet_withSort(ParticleSys<DeviceMemory> *p,ParticleSys<DeviceMemory> *tmpPs, BoundingBox *box,TriangleMesh *mesh, BVH *bvh, int gridSize, int blockSize);

void device_dem_verlet_verlet(ParticleSys<DeviceMemory> *p, BoundingBox *box,TriangleMesh *mesh, BVH *bvh, int gridSize, int blockSize);

void device_dem_verlet_triangles(ParticleSys<DeviceMemory> *p, BoundingBox *box,TriangleMesh *mesh, int gridSize, int blockSize);


void device_dem_triangles(ParticleSys<DeviceMemory> *p, BoundingBox *box,TriangleMesh *mesh, int gridSize, int blockSize);

void device_dem(ParticleSys<DeviceMemory> *p, BoundingBox *box, int gridSize, int blockSize);
#endif
