#pragma once
#include "hardCodedParameters.h"
#include <stdint.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include "Vec3.h"
#include "BoundingBox.h"
#include "TriangleMesh.h"
#include "cJSON_d.h"
struct TriangleMesh;
struct BoundingBox;
/* ==================== Memory Structs =================== */
struct HostMemory{
    template <typename T>
    static T* allocate(int N){
        return static_cast<T*>(malloc(sizeof(T)*N));
    }

    template <typename T>
    static void deallocate(T* ptr){
        free(ptr);
    }
};

struct DeviceMemory{
    template <typename T>
        static T* allocate(int N){
            T* ptr;
            cudaMalloc((void**)&ptr, sizeof(T)*N);
            return ptr;
        }

    template <typename T>
        static void deallocate(T* ptr){
            cudaFree(ptr);
        }

};

/* 
   =======================================
   interface 
   =======================================
   */

struct ParticleSysShared {
    double dt;
    double mu;
    int N;
    int steps;

    double time_factor;
    double length_factor;
    double mass_factor;

    #define MEMBER(type,name,Np,SAVE_FLAG) type* name;
    #include "memberList/ParticleSystemMember_common.def"
    #undef MEMBER
};

template <typename Memory>
struct ParticleSys;

template <>
struct ParticleSys<DeviceMemory> : ParticleSysShared {
    void* tmp_storage;
    size_t tmp_bytes;
    int* refreshVerletFlag;
    ParticleSys<DeviceMemory>* d_self;
};

template <>
struct ParticleSys<HostMemory> : ParticleSysShared {
};

/* ==================== Memory Allocation =================== */
void allocate(ParticleSys<HostMemory> *ps);
void allocate(ParticleSys<DeviceMemory> *ps);


void deallocate(ParticleSys<HostMemory> *ps);
void deallocate(ParticleSys<DeviceMemory> *ps);



/*
   ============================================================
   Host→Device転送
   ============================================================
   */
void copyToDevice(ParticleSys<HostMemory> *ps, ParticleSys<DeviceMemory> *d_ps);


/*
   ============================================================
   Device→Host転送
   ============================================================
   */
void copyFromDevice(ParticleSys<DeviceMemory>* d_ps, ParticleSys<HostMemory>* ps);


/*
   ============================================================
   初期化
   ============================================================
   */
void initializeTmpParticles(ParticleSys<HostMemory>* ps,cJSON *json_inlet, double r,double m,double k,double res);
void initializeParticles(ParticleSys<HostMemory>* ps,cJSON *json_inlet, double r,double m, double k, double res);

void nondimensionalize(ParticleSys<HostMemory>* ps, BoundingBox* box, TriangleMesh *mesh);



