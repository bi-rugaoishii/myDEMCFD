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
#include "config/ConfigLoader.h"
#include "config/SimulationConfig.h"
#include "SimulationSetup.h"
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



/*
   ============================================================
   メイン
   ============================================================
   */



int main(int argc, char** argv){
    setvbuf(stdout,NULL,_IOLBF,0);
    setvbuf(stderr,NULL,_IONBF,0);
    printf("Starting myCFD\n\n");

    const std::string config_filename = argc >=2 ? argv[1] : "config.json";

    SimulationConfig config;

    try {
        config = load_simulation_config(config_filename);
    }catch (const std::exception& error) {
        fprintf(stderr,"Failed to load configuration:\n%s\n", error.what());
        return EXIT_FAILURE;
    }


    printf("Configuration file: %s\n", config_filename.c_str());

    config.validate();

    const bool use_gpu = config.hardware.use_gpu;

    if (use_gpu){
        printf("GPU IS ON!!!!\n");
    }else{
        fprintf(
                stderr,
                "CPU calculation loop is not implemented "
                "in this main function.\n");

        return EXIT_FAILURE;
    }

    const char* outdir = config.output.directory.c_str();

    const int Nx = config.grid.Nx;
    const int Ny = config.grid.Ny;
    const int Nz = config.grid.Nz;

    const double originx =
        config.grid.origin_x_;

    const double originy =
        config.grid.origin_y_;

    const double originz =
        config.grid.origin_z_;

    const double sizex =
        config.grid.size_x;

    const double sizey =
        config.grid.size_y;

    const double sizez =
        config.grid.size_z;


    const double rho_g =
        config.fluid.phase0.density;

    const double rho_w =
        config.fluid.phase1.density;

    const double mu_g =
        config.fluid.phase0.viscosity;

    const double mu_w =
        config.fluid.phase1.viscosity;

    const double sigma =
        config.fluid.surface_tension;








    FileInOut fileIO;
    fileIO.solver_output_init(outdir);

    /*=== initialize === */
    SMACSolver solv;

    /* == set properties ==*/
    solv.set_calc_properties(originx, originy, originz, sizex, sizey, sizez, Nx, Ny, Nz);

    solv.set_gravity(config.fluid.gravity.x,config.fluid.gravity.y,config.fluid.gravity.z);
    solv.set_rhos(rho_g,rho_w);
    solv.set_mus(mu_g,mu_w);

    /* == set boundary id numbers == */
    int num_bc_id = config.get_num_boundary_ids();
    solv.grid_.set_num_bc_id(num_bc_id);


    solv.solver_malloc();


    solv.set_face_type();
    solv.set_cell_type();
    solv.set_face_internal_direction();


    /* check if its pure neumann */
    bool isPureNeumann=config.is_pure_neumann();




    solv.grid_.sigma_(0) =sigma; // temporal implementation

    solv.grid_.get_cell_coord();


    SimulationSetup::apply_boundary_conditions(solv,config);
    SimulationSetup::apply_initial_conditions(solv,config);




    solv.update_properties_by_alpha_initial();




    /* === calc time measurement === */
    double h_start, h_end;
    float ms;
    h_start = omp_get_wtime();

    double cfl_thresh = config.time.cfl;
    int alpha_substeps = config.get_alpha_substeps();
    Time_mode mode=VARIBALE_TIME_STEP;
    double outfreqtime = config.time.interval;
    double endTime = config.time.end_time;
    double max_dt = config.time.maximum_dt;
    double initial_dt = config.time.initial_dt;


    CFDTime cfdtime(initial_dt,max_dt,outfreqtime,endTime,cfl_thresh,mode);


    /* == gpu initialization == */
    G_SMACSolver g_solv;
    g_solv.set_calc_properties(originx, originy, originz, sizex, sizey,sizez, Nx, Ny, Nz);

    if(use_gpu){


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



        SimulationSetup::apply_initial_conditions_device(g_solv,config);

        cudaDeviceSynchronize();
        printf("Applying solid cell\n");
        g_solv.set_solid_cell();
        printf("Applying solid cell done\n");
    }


    G_PCGSolver pcgSolver;
    pcgSolver.copyData(g_solv);
    g_solv.pressure_solver_ = &pcgSolver;

    int num_levels = config.pressure_solver.gmg_levels;
    G_GMGSolver gmgSolver;
    gmgSolver.initialize(g_solv,num_levels);
    gmgSolver.copyData(g_solv);
    pcgSolver.set_gmg(gmgSolver);


    /* choose solver according to the boundary condition*/
    if(isPureNeumann){
        pcgSolver.set_solver(PURENEUMANN_GMG_PCG);
        printf("\n\n PURENEUMANN\n\n");
    }else{
        pcgSolver.set_solver(GMG_PCG);
        //pcgSolver.set_solver(STANDARD_PCG);
    }

    /* ============================= 
       ======== main loop ==========
       ============================= */

    /* output initial data */
    g_solv.gpuTocpu(solv.grid_);
    fileIO.output_vti_binary_cellData(solv.grid_,0.);

    printf("output initial data done \n");


    int cur_step = 0;
    if (use_gpu){
        while(cfdtime.current_time_ < cfdtime.end_time_-EPS){

            if (cur_step == 0){

                /* initial step requires high tolerance*/

                g_solv.pressure_solver_->tol_ = 1e-11;
            }else{

                g_solv.pressure_solver_->tol_ = config.pressure_solver.tol;
            }
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
            g_solv.compute_mass_flux_from_alpha_flux(solv);

            g_solv.update_boundary_faces();


            g_solv.calc_surface_tension();


            g_solv.get_vof_vstar_rhouu_consistent(solv);

            g_solv.update_vstar_boundary();


            printf("starting poisson\n");
            g_solv.solve_poisson();

            g_solv.correct_vof_velocity(solv);
            g_solv.update_boundary_ghost(solv);
            g_solv.make_face_gradp();



            if(cfdtime.isOutStep_){

                /* this is only for output in order to interpolate velocity to cell centers*/

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
    if(use_gpu){
        g_solv.solver_free();
        gmgSolver.free_levels();
    }

    printf("my CFD Done!!!\n");


    return 0;
}
