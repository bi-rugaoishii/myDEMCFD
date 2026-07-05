#pragma once
#include "../SMACSolver.h"
#include "../G_StaggeredGrid.h"
#include "G_PressureSolverBase.h"
#include "../MyArray.h"
#include "../hardCodedParameters.h"
#include "../PCG_Scalars.h"
#include "G_GMGSolver.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <algorithm>
#include <cub/cub.cuh>

struct G_PCGSolver:G_PressureSolverBase{
    using Base=G_PressureSolverBase;


    G_PCGSolver();
    ~G_PCGSolver();


    void solve(G_SMACSolver& solv) override;
    void solve_pcg(G_SMACSolver& solv);
    void set_gmg(G_GMGSolver& gmg) ;

    G_GMGSolver* gmg_;
};


#define PCG_PROFILE 0

struct PCGProfile{
    float total_ms;
    float init_ms;
    float Ap_kernel_ms;
    float pAp_reduce_ms;
    float update_prz_ms;
    float rz_reduce_ms;
    float residual_check_ms;
    float update_dir_ms;
    float swap_rz_ms;
    float final_ms;
    int iter;
};

static void reset_pcg_profile(PCGProfile& p){
    p.total_ms=0.0f;
    p.init_ms=0.0f;
    p.Ap_kernel_ms=0.0f;
    p.pAp_reduce_ms=0.0f;
    p.update_prz_ms=0.0f;
    p.rz_reduce_ms=0.0f;
    p.residual_check_ms=0.0f;
    p.update_dir_ms=0.0f;
    p.swap_rz_ms=0.0f;
    p.final_ms=0.0f;
    p.iter=0;
}

#if PCG_PROFILE
static void pcg_prof_start(cudaEvent_t ev_start){
    cudaEventRecord(ev_start,0);
}

static void pcg_prof_stop(cudaEvent_t ev_start,cudaEvent_t ev_stop,float& accum_ms){
    float ms=0.0f;
    cudaEventRecord(ev_stop,0);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ms,ev_start,ev_stop);
    accum_ms+=ms;
}
#endif
