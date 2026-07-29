#pragma once
#ifndef _TRIANGLEMESH_H_
#define _TRIANGLEMESH_H_
#include "Vec3.h"
#include "BoundingBox.h"
#include "cJSON_d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ======== used for merging ========== */
typedef struct {
    long long ix, iy, iz;  // 量子化キー
    int index;             // unique vertex index
    int used;              // 0=empty, 1=occupied
} HashEntry;

typedef struct {
    HashEntry* table;
    size_t size;
} VertexHash;

void vertex_hash_init(VertexHash* h, size_t size);
int vertex_hash_get_or_insert(VertexHash* h, double x, double y, double z, double eps, int* nextIndex);

int get_edge_index(int v0, int v1, int* edge,int eIndex);
void sort3(int *x, int *y, int *z);
/* =====================================
   Device Triangle Mesh
===================================== */

typedef struct DeviceTriangleMesh {

    #define MEMBER(type,name, size) type* name; 
    #include "memberList/TriangleMeshMember_common.def"
    #undef MEMBER


    /* === non array === */
    int nVert; /* number of vertices */
    int nTri; /* number of triangles */
    int nShift; /*  shift of id  from vertex to edge  */

    /* bounding box as a whole triangle mesh */
    double gminx, gmaxx;
    double gminy, gmaxy;
    double gminz, gmaxz;

} DeviceTriangleMesh;

/* =====================================
   double precision triangle mesh
===================================== */
typedef struct TriangleMesh {
    int *sortedIndex;


    /* morton key */
    uint32_t *mortonKey;

    /* center coord */
    double* cx; 
    double* cy;
    double* cz;

    #define MEMBER(type,name,size) type* name; 
    #include "memberList/TriangleMeshMember_common.def"
    #undef MEMBER

    /* === non array === */
    int nVert; /* number of vertices */
    int nTri; /* number of triangles */
    int nShift; /*  shift of id  from vertex to edge  */

    /* bounding box as a whole triangle mesh */
    double gminx, gmaxx;
    double gminy, gmaxy;
    double gminz, gmaxz;

    DeviceTriangleMesh d_mesh;
    DeviceTriangleMesh *d_meshPtr;

} TriangleMesh;

/* ==== device related functions ====*/
#if USE_GPU
void deviceMallocCopyTriangleMesh(TriangleMesh *mesh);
#endif

void free_TriangleMesh(TriangleMesh* mesh,int isGPUon);
static int count_ascii_stl_triangles(FILE* fp);
int load_ascii_stl_double(const cJSON* filename, TriangleMesh* mesh);

#endif
