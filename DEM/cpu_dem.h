#ifndef _CPU_DEM_H_
#define _CPU_DEM_H_
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include "ParticleSystem.h"
#include "BoundingBox.h"
#include "Vec3.h"
#include "ContactCache.h"
#include "TriangleContactCache.h"
#include "BVH.h"
#define DIM 3

/* === triangle related === */
void wall_collision_verlet(ParticleSys<HostMemory>* p,TriangleMesh* mesh);
void wall_collision_BVH(ParticleSys<HostMemory>* p,BVH* bvh);
void wall_collision_triangles(ParticleSys<HostMemory>* ps, BoundingBox *box,TriangleMesh* mesh);
void cpu_dem_sort_triangles(ParticleSys<HostMemory>* ps, ParticleSys<HostMemory> *tmpPs, BoundingBox* box,TriangleMesh *mesh, int step);

void cpu_dem_verlet_verlet(ParticleSys<HostMemory>* p, ParticleSys<HostMemory> *tmpP, BoundingBox* box,TriangleMesh *mesh,BVH *bvh, int step);

/* calculated distance between a triangle */
/* i is index of a particle, j is index of a triangle*/
/* returns dist 1e10 if no collision */
TriangleContactCache dist_triangle(ParticleSys<HostMemory>* ps,int i, TriangleMesh* mesh, int j);


void integrateCPU(ParticleSys<HostMemory> *ps, BoundingBox *box);
void cpu_dem_naive_triangle(ParticleSys<HostMemory> *ps, BoundingBox *box, TriangleMesh *mesh);
void particle_collision_naive(ParticleSys<HostMemory>* ps);


void cpu_dem_verlet_BVH(ParticleSys<HostMemory>* p, ParticleSys<HostMemory> *tmpP, BoundingBox* box,TriangleMesh *mesh,BVH *bvh, int step);

void particle_collision_verlet(ParticleSys<HostMemory>* p, BoundingBox *box);

void particle_collision_cell_linked(ParticleSys<HostMemory>* ps, BoundingBox *box);
void particle_collision_cell_linked_fastUpdate(ParticleSys<HostMemory>* p, BoundingBox *box);
void particle_collision_cell_linked_withSort_fastUpdate(ParticleSys<HostMemory>* p,ParticleSys<HostMemory>* tmpPs, BoundingBox *box);

/* =========== verlet list related =============== */
int shouldRefreshNeighborList(ParticleSys<HostMemory> *p, BoundingBox* box);

void checkOoB(ParticleSys<HostMemory> *p, ParticleSys<HostMemory> *tmpP, BoundingBox* box);

void cpu_dem_verlet_triangles(ParticleSys<HostMemory>* p, ParticleSys<HostMemory> *tmpP, BoundingBox* box,TriangleMesh *mesh, int step);
void cpu_dem_nosort_triangle(ParticleSys<HostMemory>* ps, ParticleSys<HostMemory> *tmpPs, BoundingBox* box, TriangleMesh* mesh);
void particle_collision_cell_linked_withSort(ParticleSys<HostMemory>* p,ParticleSys<HostMemory>* tmpPs, BoundingBox *box);

void particle_collision_cell_linked_noVec3(ParticleSys<HostMemory>* ps, BoundingBox *box);

inline ContactCache calc_normal_force(ParticleSys<HostMemory> *p,int i,int j,Vec3 n,double delMag);


inline ContactCache calc_normal_force_wall(ParticleSys<HostMemory> *p,int i,int j,Vec3 n,double delMag);

inline void calc_tangential_force_wall(ParticleSys<HostMemory> *p,int i,int j,ContactCache c);

inline void calc_tangential_force(ParticleSys<HostMemory> *p,int i,int j,ContactCache c);

inline void update_history(ParticleSys<HostMemory> *p,int i);
inline void update_history_wall(ParticleSys<HostMemory> *p,int i);

void cpu_dem_sort(ParticleSys<HostMemory> *ps, ParticleSys<HostMemory> *tmpPs, BoundingBox* box, int step);
void wall_collision_naive(ParticleSys<HostMemory>* ps);

inline void velocity(ParticleSys<HostMemory>* ps);
inline void position(ParticleSys<HostMemory>* ps);
#endif
