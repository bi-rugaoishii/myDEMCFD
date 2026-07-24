#pragma once
#include "G_StaggeredGrid.h"
#include "pressure_solver/G_PressureSolverBase.h"
#include "StaggeredGrid.h"
#include "SMACSolver.h"
#include <cuda_runtime.h>

struct G_PressureSolverBase;
struct G_SMACSolver{
    G_StaggeredGrid grid_;
    double nu_;
    double rho_;
    double rho1_,rho0_;
    double mu1_,mu0_;
    double t_now_;
    double dt_;
    double inv_dt_;
    double u_lid_;
    double gx_,gy_,gz_; //gravity
    
    double *d_pcg_scalars_;



    dim3 block_dim_;
    dim3 grid_dim_; //grid size for cuda

    void set_calc_properties(double rho, double dt,double u_lid, double nu, double sizex, double sizey, double sizez, int Nx, int Ny, int Nz);

    /* == surface tension related == */
    void calc_surface_tension();


    /* == functions == */


    void solve_poisson();
    void solve_vof_poisson_pcg(SMACSolver solv);
    void solve_vof_poisson_pcg_fused_kernel(SMACSolver solv);

    void correct_vof_velocity(SMACSolver solv);
    void check_divergence();

    void transport_alpha();

    void clear_alpha_flux();
    void clear_alpha_flux_accum();
    void compute_mass_flux_from_alpha_flux(SMACSolver solv);
    double calc_alpha_area() ;
    void alpha_flux_accum();
    void alpha_flux_thincwlic(double dt);
    void alpha_flux_thincwlic_split(double dt,int steps);

    void update_boundary_faces();
    void update_properties_by_alpha();
    void set_boundary_velocity(SMACSolver solv);

    double calc_cfl();


    void build_vof_poisson_Ap(const double *p);
    void build_vof_poisson_invdiag();
    void make_poisson_rhs();
    void subtract_cell_mean(double *p, int Nx, int Ny);

    void set_boundary_star();
    void set_boundary_pressure();
    void set_boundary_array(double *const q);
    void set_boundary_alpha(int Nx, int Ny);

    void get_vof_vstar_rhouu_upwind_consistent(SMACSolver solv);


    void fix_pressure();
    void shift_pressure_reference();

    /* == gpu memory related == */
    void cpuTogpu(StaggeredGrid h_grid);
    void gpuTocpu(StaggeredGrid h_grid);
    void solver_malloc();
    void solver_free();

    /* == for debugging ==*/
    void check_nan_all(const char* tag);

    inline void set_block_grid(int Nx, int Ny, int Nz){
        int BLOCK_X = 8;
        int BLOCK_Y = 8;
        int BLOCK_Z = 4;

        block_dim_ = dim3(BLOCK_X,BLOCK_Y,BLOCK_Z);

        grid_dim_ = dim3( (Nx+3 + block_dim_.x -1)/block_dim_.x,
               (Ny+3 + block_dim_.y-1)/block_dim_.y,(Nz+3+block_dim_.z-1)/block_dim_.z);

    }

    /* == for cub == */
    void* cub_temp_storage_;
    size_t cub_temp_storage_bytes_;
    double* d_r2_;
    double* d_dot_;

    /* == IBM related == */
    void make_cylinder_ibm(double xc, double yc, double zc, double r);
    void set_solid_cell();

    /* == pressure solver == */
    G_PressureSolverBase* pressure_solver_;
    

};



/* misc function*/
static __device__ __forceinline__ double sgn(double a){
    double result;
    if(a<0){
        result =-1.;
    }else{
        result =1.;
    }

    return result;
}
