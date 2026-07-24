#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "hardCodedParameters.h"
#include "StaggeredGrid.h"
#include "G_StaggeredGrid.h"
#include "misc.h"
#include "SMACSolver.h"
#include "pressure_solver/G_PressureSolverBase.h"
#include "pressure_solver/G_PCGSolver.h"
#include "pressure_solver/G_GMGSolver.h"
#include "G_SMACSolver.h"
#include "CFDTime.h"
#include "omp.h"
#include "FileInOut.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#define GPU_ON 1


/*
   ============================================================
   メイン
   ============================================================
   */



int main(){
    setvbuf(stdout,NULL,_IOLBF,0);
    setvbuf(stderr,NULL,_IONBF,0);
    printf("Starting myCFD\n\n");

    if (GPU_ON==1){
        printf("GPU IS ON!!!!\n");
    }
    const char* outdir ="results";

    int Nx=128;
    int Ny=128;
    int Nz=1;


    /*
    int Nx=96;
    int Ny=32;
    int Nz=96;
    */

    double rho = 1.;
    double rho_w = 1000.;
    //double rho_w = 1.;
    double rho_g = 1.;

    double u_lid = 0.;
    double nu = 0.01;


    double mu_w = 1.0e-3;
    double mu_g = 1.8e-5;
    //double mu_w = nu*rho_w;
    //double mu_g = nu*rho_g;


    double sizex=0.584;
    double sizey=0.584;
    double sizez=0.584;

    /*
    double sizex=0.04;
    double sizey=0.016;
    double sizez=0.04;
    */


    double dx=sizex/(double)Nx;
    double dy=sizey/(double)Ny;
    double dz=sizez/(double)Nz;
    //double sigma = 0.072;
    //double sigma = 0.00072;
    double sigma = 1e-16;
    //int endSteps = 10000;
    //int outStepsFreq=100;

    double Re = u_lid*sizex/nu;
    double dt=0.001;

    FileInOut fileIO;
    fileIO.solver_output_init(outdir);

    /*=== initialize === */
    SMACSolver solv;

    /* == set properties ==*/
    solv.set_calc_properties(rho, dt,u_lid, nu, sizex, sizey, sizez, Nx, Ny, Nz);


    solv.set_gravity(0., -9.81, 0.);
   // solv.set_gravity(0., 0,0.);
    solv.set_rhos(rho_g,rho_w);
    solv.set_mus(mu_g,mu_w);
   
    /* == set boundary id numbers == */
    int num_bc_id = 3;
    solv.grid_.set_num_bc_id(num_bc_id);


    solv.solver_malloc();

    double wallvel=0.000;

    solv.set_face_type();
    solv.set_cell_type();
    solv.set_face_internal_direction();

    solv.grid_.set_boundary_id(1,1,Ny+1); // dir bid index
    solv.grid_.set_boundary_id(2,2,Nz+1);
    solv.grid_.set_boundary_id(2,2,1);

    solv.grid_.bc_.set_boundary_velocity(1, wallvel, 0., 0.);
    solv.grid_.bc_.set_bctype(1, BC_NOSLIP);
    solv.grid_.bc_.set_bctype(2, BC_NOSLIP);

    solv.grid_.sigma_(0) =sigma; // temporal implementation

    solv.grid_.get_cell_coord();
    //solv.grid_.place_vof(0.,0.2,0.,0.5,1.0);
    solv.grid_.place_vof(0.,0.1461,0.,0.4,0.,1,1.0);
    //solv.grid_.place_solid(0.292,0.316,0.,0.048,0.,1.0,1);
    //solv.grid_.place_vof(0.4,0.5,0.4,0.5,0.,1.0,1.0);





    /*for zalesak test*/
    //solv.initialize_zalesak_disk();

    //solv.set_sphere();
    //solv.set_sphere_sub_voxel();
   //solv.grid_.place_vof(0.,1.0,0.,0.0015,0.,1.0,1.0);

   /*
    solv.set_sphere_zalesak();
    solv.set_zalesak_rotation_velocity();
    */

    //solv.set_boundary_neumann(solv.grid_.p_);
    //solv.set_boundary_neumann(solv.grid_.alpha_);
    solv.update_properties_by_alpha_initial();




    /* === calc time measurement === */
    double h_start, h_end;
    float ms;
    h_start = omp_get_wtime();

    double cfl_thresh = 0.4;
    double cfl_alpha_thresh = 0.2;
    int alpha_substeps = (int)ceil(cfl_thresh/cfl_alpha_thresh);
    Time_mode mode=VARIBALE_TIME_STEP;
    double outfreqtime = 0.05;
    double endTime = 20.0;
    double max_dt = 1e-2;
    double initial_dt = 1e-5;

    /* == set cfd time related parameters ==*/
    /*
    double cfl_thresh = 0.4;
    double cfl_alpha_thresh = 0.2;
    int alpha_substeps = (int)ceil(cfl_thresh/cfl_alpha_thresh);
    Time_mode mode=VARIBALE_TIME_STEP;
    double outfreqtime = 0.05;
    double endTime = 10.0;
    double max_dt = 1e-2;
    double initial_dt = 1e-5;
    */

    CFDTime cfdtime(initial_dt,max_dt,outfreqtime,endTime,cfl_thresh,mode);

    /* output initial data */
    fileIO.output_vti_binary_cellData(solv.grid_,0.);

    printf("output initial data done \n");

    /* == gpu initialization == */
    G_SMACSolver g_solv;
    g_solv.set_calc_properties(rho, dt,u_lid, nu, sizex, sizey,sizez, Nx, Ny, Nz);

    if(GPU_ON ==1){

        
        g_solv.grid_.bc_.num_boundary_id_ = solv.grid_.bc_.num_boundary_id_;


        printf("Allocating memory for gpu\n");
        g_solv.solver_malloc();
        printf("Allocated!\n");

        


        printf("Copying data to gpu\n");
        g_solv.cpuTogpu(solv.grid_);


        /* === need to make good function for this ==*/
        g_solv.rho0_ = solv.rho0_;
        g_solv.rho1_ = solv.rho1_;
        g_solv.mu0_ = solv.mu0_;
        g_solv.mu1_ = solv.mu1_;

        printf("Copying data done\n");

        g_solv.set_block_grid(Nx,Ny,Nz);

        g_solv.make_cylinder_ibm(0.3,0.,0.5,0.1);
    }


    G_PCGSolver pcgSolver;
    pcgSolver.copyData(g_solv);
    g_solv.pressure_solver_ = &pcgSolver;

    int num_levels = 4;
    G_GMGSolver gmgSolver;
    gmgSolver.initialize(g_solv,num_levels);
    gmgSolver.copyData(g_solv);
    pcgSolver.set_gmg(gmgSolver);

    /* ============================= 
       ======== main loop ==========
       ============================= */


    int cur_step = 0;
    if (GPU_ON==1){
        while(cfdtime.current_time_ < cfdtime.end_time_-EPS){

            /* == setup time == */
            double cfl=g_solv.calc_cfl();
            cfdtime.updateTime(cfl);

            solv.dt_ = cfdtime.dt_;
            solv.inv_dt_ = 1./cfdtime.dt_;

            g_solv.dt_ = cfdtime.dt_;
            g_solv.inv_dt_ = 1./cfdtime.dt_;

            printf("dt = %3.2e, current time = %f\n",cfdtime.dt_,cfdtime.current_time_);

            /* == transport alpha == */
            g_solv.clear_alpha_flux_accum();


            double sub_dt = cfdtime.dt_/(double)alpha_substeps;
            for (int substeps=0 ; substeps<alpha_substeps; substeps++){ 
                printf("alpha subcycle %d/%d\n", substeps+1,alpha_substeps);
                g_solv.clear_alpha_flux();
                g_solv.alpha_flux_thincwlic_split(sub_dt,cur_step);

                //g_solv.alpha_flux_thincwlic(sub_dt);
                //g_solv.transport_alpha(); //used for unsplit thinc

                g_solv.alpha_flux_accum();
            }
            /* == transport alpha done == */

            g_solv.update_properties_by_alpha();
            g_solv.update_boundary_faces();

            g_solv.calc_surface_tension();

            g_solv.compute_mass_flux_from_alpha_flux(solv);
            g_solv.get_vof_vstar_rhouu_upwind_consistent(solv);

            /* == needs boundary condition in general ==*/
            /* == we are skipping it since we assume wall stationary in normal direction ==*/


            printf("starting poisson\n");
            g_solv.solve_poisson();

            g_solv.correct_vof_velocity(solv);

            /* == needs boundary condition in general ==*/
            /* == we are skipping it since we assume stationary in normal direction ==*/


            if(cfdtime.isOutStep_){
                g_solv.gpuTocpu(solv.grid_);
                //solv.check_pressure_jump_by_radius();
                fileIO.output_vti_binary_cellData(solv.grid_,cfdtime.current_steps_);
                h_end = omp_get_wtime();
                ms = (float)(h_end-h_start);

                int step= cfdtime.current_steps_;
                double current_time= cfdtime.current_time_;
                printf("Output step %d, dt = %f s, current time: %f s, GPU time: %f s\n\n", step,cfdtime.dt_,current_time,ms);
                cfdtime.isOutStep_=false;
            }else{
                h_end = omp_get_wtime();
                ms = (float)(h_end-h_start);

                int step= cfdtime.current_steps_;
                double current_time= cfdtime.current_time_;
                printf("step %d,next dt = %f s, current time: %f s, GPU time: %f s\n\n", step,cfdtime.dt_,current_time,ms);
            }
            printf("\n");

            cur_step ++;
        }

    }

  solv.solver_free();
  if(GPU_ON==1){
      g_solv.solver_free();
      gmgSolver.free_levels();
    }

    printf("my CFD Done!!!\n");


    return 0;
}
