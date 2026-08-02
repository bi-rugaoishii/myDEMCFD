#pragma once
#include "StaggeredGrid.h"

struct SMACSolver{
    StaggeredGrid grid_;
    double rho1_,rho0_;
    double mu1_,mu0_;
    double t_now_;
    double dt_=0.001;
    double inv_dt_ = 1./dt_;
    double gx_,gy_,gz_; //gravity



    /* == functions == */

    void set_calc_properties(double originx, double originy, double originz, double sizex, double sizey, double sizez, int Nx, int Ny, int Nz);

    void set_face_type();

    void solve_poisson();
    void solve_vof_poisson();
    void solve_vof_poisson_pcg();

    void correct_velocity();
    void correct_vof_velocity();
    void check_divergence();

    void transport_alpha();
    void transport_alpha_thinc_x();
    void transport_alpha_thinc_y();

    void set_cell_type();
    void set_face_internal_direction();

    void clear_alpha_flux();
    void compute_mass_flux_from_alpha_flux();
    double calc_alpha_vol() ;
    void alpha_flux_upwind();
    void alpha_flux_thinc_x();
    void alpha_flux_thinc_y();
    void alpha_flux_thincwlic_x();
    void alpha_flux_thincwlic_y();

    void update_properties_by_alpha();
    void update_properties_by_alpha_initial();

    void solver_malloc();
    void solver_free();


    double calc_cfl();
    void build_vof_poisson_Ap(const double *p);
    void build_vof_poisson_invdiag();
    void make_poisson_rhs();
    void subtract_cell_mean(double *p);
    void set_gravity(double gx, double gy, double gz);
    void set_rhos(double rho0, double rho1);
    void set_mus(double mu0, double mu1);

    void set_boundary_star();
    void set_boundary_velocity();
    void set_boundary_pressure();
    void set_boundary_array(double *const q);
    void set_boundary_alpha();
    void set_boundary_neumann(MyArray<double,3>& alpha);

    void get_vof_ustar_rhouu_upwind_consistent();


    void fix_pressure();
    void shift_pressure_reference();


    /* == for debugging ==*/
    void check_nan_all(const char* tag);

    void set_sphere(double center_x, double center_y, double center_z, double r, double v_ini);
    void set_sphere_sub_voxel();
    void initialize_zalesak_disk();
    void set_sphere_zalesak();
    void initialize_disk();
    void set_zalesak_rotation_velocity();
    void check_pressure_jump_by_radius();

    void set_initial_x_velocity(double ini_v);
};


