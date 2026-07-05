#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "hardCodedParameters.h"
#include "StaggeredGrid.h"
#include "G_StaggeredGrid.h"
#include "misc.h"
#include "SMACSolver.h"
#include "pressure_solver/G_PressureSolverBase.h"
#include "pressure_solver/G_PCGSolver.h"
#include "G_SMACSolver.h"
#include "CFDTime.h"
#include "omp.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#define GPU_ON 1


int solver_output_init(const char* dir){
    struct stat st;

    /* 既に存在するか確認 */
    if(stat(dir,&st)==0)
    {
        if(S_ISDIR(st.st_mode))
        {
            printf("Output dir exists: %s\n",dir);
            return 0;
        }
    }

    /* 無ければ作成 */
    if(mkdir(dir,0755)==0)
    {
        printf("Created output dir: %s\n",dir);
        return 0;
    }

    if(errno==EEXIST)
        return 0;

    printf("ERROR: cannot create dir %s\n",dir);
    return -1;
}

void output_vti(const StaggeredGrid& grid, int step, const char* folderName){
    char filename[256];
    sprintf(filename, "%s/result_%06d.vti", folderName, step);

    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Cannot open VTI file: %s\n", filename);
        abort();
    }

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;

    double dx = grid.dx_;
    double dy = grid.dy_;
    double dz = grid.dz_;

    const MyArray<double,3>& p     = grid.p_;
    const MyArray<double,3>& alpha = grid.alpha_;
    const MyArray<double,3>& vx    = grid.f_vx_;
    const MyArray<double,3>& vy    = grid.f_vy_;
    const MyArray<double,3>& vz    = grid.f_vz_;

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");

    fprintf(fp,
        "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" "
        "Origin=\"%.15e %.15e %.15e\" "
        "Spacing=\"%.15e %.15e %.15e\">\n",
        Nx - 1, Ny - 1, Nz - 1,
        0.5 * dx, 0.5 * dy, 0.5 * dz,
        dx, dy, dz);

    fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n", Nx - 1, Ny - 1, Nz - 1);
    fprintf(fp, "      <PointData Scalars=\"alpha\" Vectors=\"velocity\">\n");

    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"pressure\" format=\"ascii\">\n");
    for (int iz = 1; iz <= Nz; iz++) {
        for (int iy = 1; iy <= Ny; iy++) {
            for (int ix = 1; ix <= Nx; ix++) {
                fprintf(fp, "%.15e\n", p(ix, iy, iz));
            }
        }
    }
    fprintf(fp, "        </DataArray>\n");

    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"alpha\" format=\"ascii\">\n");
    for (int iz = 1; iz <= Nz; iz++) {
        for (int iy = 1; iy <= Ny; iy++) {
            for (int ix = 1; ix <= Nx; ix++) {
                fprintf(fp, "%.15e\n", alpha(ix, iy, iz));
            }
        }
    }
    fprintf(fp, "        </DataArray>\n");

    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int iz = 1; iz <= Nz; iz++) {
        for (int iy = 1; iy <= Ny; iy++) {
            for (int ix = 1; ix <= Nx; ix++) {
                double ux = 0.5 * (vx(ix - 1, iy, iz) + vx(ix, iy, iz));
                double uy = 0.5 * (vy(ix, iy - 1, iz) + vy(ix, iy, iz));
                double uz = 0.5 * (vz(ix, iy, iz - 1) + vz(ix, iy, iz));

                fprintf(fp, "%.15e %.15e %.15e\n", ux, uy, uz);
            }
        }
    }
    fprintf(fp, "        </DataArray>\n");

    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"divergence\" format=\"ascii\">\n");
    for (int iz = 1; iz <= Nz; iz++) {
        for (int iy = 1; iy <= Ny; iy++) {
            for (int ix = 1; ix <= Nx; ix++) {
                double div =
                    (vx(ix, iy, iz) - vx(ix - 1, iy, iz)) * grid.inv_dx_
                  + (vy(ix, iy, iz) - vy(ix, iy - 1, iz)) * grid.inv_dy_
                  + (vz(ix, iy, iz) - vz(ix, iy, iz - 1)) * grid.inv_dz_;

                fprintf(fp, "%.15e\n", div);
            }
        }
    }
    fprintf(fp, "        </DataArray>\n");

    fprintf(fp, "      </PointData>\n");
    fprintf(fp, "      <CellData></CellData>\n");
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </ImageData>\n");
    fprintf(fp, "</VTKFile>\n");

    fclose(fp);
    printf("VTI output: %s\n", filename);
}


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

    int Nx=50;
    int Ny=50;
    int Nz=50;

    double rho = 1.;
    double rho_w = 1000.;
   // double rho_w = 1.;
    double rho_g = 1.;

    double u_lid = 0.;
    double nu = 0.01;


    double mu_w = 1.0e-3;
    double mu_g = 1.8e-5;
    //double mu_w = nu*rho_w;
    //double mu_g = nu*rho_g;

    double sizex=1.;
    double sizey=1.;
    double sizez=1.;
    double dx=sizex/(double)Nx;
    double dy=sizey/(double)Ny;
    double dz=sizez/(double)Nz;
    //double sigma = 0.072;
    double sigma = 1e-16;
    //int endSteps = 10000;
    //int outStepsFreq=100;

    double Re = u_lid*sizex/nu;
    double dt=0.001;

    solver_output_init(outdir);

    /*=== initialize === */
    SMACSolver solv;

    /* == set properties ==*/
    solv.set_calc_properties(rho, dt,u_lid, nu, sizex, sizey, sizez, Nx, Ny, Nz);


    solv.set_gravity(0., -9.81, 0.);
    //solv.set_gravity(0., 0,0.);
    solv.set_rhos(rho_g,rho_w);
    solv.set_mus(mu_g,mu_w);
   
    /* == set boundary id numbers == */
    int num_bc_id = 2;
    solv.grid_.set_num_bc_id(num_bc_id);


    solv.solver_malloc();

    solv.grid_.set_boundary_velocity();

    solv.set_face_type();

    solv.set_cell_type();

    solv.set_face_internal_direction();

    solv.grid_.sigma_(0) =sigma; // temporal implementation

    solv.grid_.get_cell_coord();
    //solv.grid_.place_vof(0.,0.2,0.,0.5,1.0);
    solv.grid_.place_vof(0.,0.6,0.,0.5,0.,0.6,1.0);
    //solv.grid_.place_vof(0.,1.0,0.,0.5,1.0);

    /*for surface tension test*/
   // solv.initialize_disk();


    /*for zalesak test*/
    /*
    solv.initialize_zalesak_disk();
    */

    //solv.set_sphere();
    //solv.set_zalesak_rotation_velocity();

    solv.set_boundary_neumann(solv.grid_.p_);
    solv.set_boundary_neumann(solv.grid_.alpha_);
    solv.update_properties_by_alpha_initial();




    /* === calc time measurement === */
    double h_start, h_end;
    float ms;
    h_start = omp_get_wtime();

    /* == set cfd time related parameters ==*/
    double cfl_thresh = 0.4;
    double cfl_alpha_thresh = 0.2;
    int alpha_substeps = (int)ceil(cfl_thresh/cfl_alpha_thresh);
    Time_mode mode=VARIBALE_TIME_STEP;
    double outfreqtime = 0.05;
    double endTime = 10.0;
    double max_dt = 5e-3;
    double initial_dt = 1e-4;

    CFDTime cfdtime(initial_dt,max_dt,outfreqtime,endTime,cfl_thresh,mode);

    /* output initial data */
    output_vti(solv.grid_,0.,outdir);
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
    }


    G_PCGSolver pcgSolver;
    pcgSolver.copyData(g_solv);
    g_solv.pressure_solver_ = &pcgSolver;


    /* ============================= 
       ======== main loop ==========
       ============================= */


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
                g_solv.alpha_flux_thincwlic(sub_dt);
                g_solv.transport_alpha();
                g_solv.alpha_flux_accum();
            }
            /* == transport alpha done == */

            g_solv.update_properties_by_alpha();
            g_solv.update_boundary_faces();

            g_solv.compute_mass_flux_from_alpha_flux(solv);
            g_solv.get_vof_vstar_rhouu_upwind_consistent(solv);

            /* == needs boundary condition in general ==*/
            /* == we are skipping it since we assume stationary in normal direction ==*/


            printf("starting poisson\n");
            g_solv.solve_poisson();

            g_solv.correct_vof_velocity(solv);

            /* == needs boundary condition in general ==*/
            /* == we are skipping it since we assume stationary in normal direction ==*/


            if(cfdtime.isOutStep_){
                g_solv.gpuTocpu(solv.grid_);
                printf("total alpha = %f\n",solv.calc_alpha_vol());
                solv.check_divergence();
                //solv.check_pressure_jump_by_radius();
                output_vti(solv.grid_,cfdtime.current_steps_,outdir);
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
        }
    }

  solv.solver_free();
  if(GPU_ON==1){
      g_solv.solver_free();
      //gmgSolver.levels_.free();
    }

    printf("my CFD Done!!!\n");


    return 0;
}
