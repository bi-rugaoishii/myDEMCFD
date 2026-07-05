#pragma once
#include "../SMACSolver.h"
#include "../G_SMACSolver.h"
#include "../G_StaggeredGrid.h"
#include "../PCG_Scalars.h"
#include <cub/cub.cuh>

struct G_SMACSolver;

__global__ void base_k_divide(double* const num,const double div);
__global__ void base_k_mult_elementwise_array_to_tmp(MyArray<double,3> b,MyArray<double,3> q,MyArray<double,3> result,int Nx, int Ny, int Nz);
__global__ void base_k_fix_pressure_reference(G_StaggeredGrid grid_);
__global__ void base_k_make_poisson_rhs(G_StaggeredGrid grid_,double inv_dt);
__global__ void k_shift_pressure_reference(G_StaggeredGrid grid_);

__global__ void base_k_copy_to_tmp(MyArray<double,3> q,MyArray<double,3> tmp, int Nx, int Ny, int Nz);

__global__ void base_k_mult_elementwise_array(MyArray<double,3> b,MyArray<double,3> q,MyArray<double,3> result,int Nx, int Ny, int Nz);

__global__ void k_set_boundary_array(double* const q, int Nx, int Ny);
__global__ void base_k_add_scalar_to_array(const double a, const double* const b,double* const q);

struct G_PressureSolverBase{
    virtual ~G_PressureSolverBase(){}
    virtual void solve(G_SMACSolver& solv)=0;
    void subtract_cell_mean(G_StaggeredGrid& grid,MyArray<double,3> p);
    void copyData(G_SMACSolver& solv);

    /* == for cub == */
    dim3 block_dim_;
    dim3 grid_dim_;
    double *d_pcg_scalars_;
    void* cub_temp_storage_;
    size_t cub_temp_storage_bytes_;
    double* d_r2_;
    double* d_dot_;

};
