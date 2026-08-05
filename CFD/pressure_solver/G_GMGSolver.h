#pragma once
#define MAX_LEVELS 16
#include "../SMACSolver.h"
#include "../G_StaggeredGrid.h"
#include "G_PressureSolverBase.h"
#include "../MyArray.h"
#include "../hardCodedParameters.h"
#include "../PCG_Scalars.h"
#include "G_Levels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <algorithm>
#include <cub/cub.cuh>

struct G_GMGSolver:G_PressureSolverBase{
    using Base=G_PressureSolverBase;

    int num_levels_;
    int min_size_=6;
    int num_iter_coarse_ =30; 
    int num_iter_fine_ = 4;



    void coarse_zero_clear();
    void create_coeffs(G_StaggeredGrid& grid);
    void v_cycle(G_StaggeredGrid& grid);
    void v_cycle_as_preconditioner(G_StaggeredGrid& grid, MyArray<double,3> &q,MyArray<double,3> &rhs);
   void solve(G_SMACSolver& solv) override;
    void initialize(G_SMACSolver& solv,int max_levels);
    void propagate();
    void restrict();
    void calc_res();
    void free_levels();

    G_Levels levels_[MAX_LEVELS];


};
