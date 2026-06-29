#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "hardCodedParameters.h"
#include "StaggeredGrid.h"
#include "G_StaggeredGrid.h"
#include "SMACSolver.h"
#include "G_SMACSolver.h"
#include "pressure_solver/G_PressureSolverBase.h"
#include "CFDTime.h"
#include "omp.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#define GPU_ON 1



int solver_output_init(const char* dir)
{
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

void output_vtk(const StaggeredGrid& grid, int step,const char* folderName)
{
    char filename[256];

    sprintf(filename, "%s/result_%06d.vtk",folderName, step);

    FILE* fp = fopen(filename, "w");

    if (fp == NULL) {
        printf("Cannot open VTK file: %s\n", filename);
        abort();
    }

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;

    double dx = grid.dx_;
    double dy = grid.dy_;

    int pitch_p  = Nx + 2;
    int pitch_vx = Nx + 3;
    int pitch_vy = Nx + 2;

    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "SMAC  result step %d\n", step);
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_POINTS\n");

    /*
       セル中心データとして出力する。
       点の位置はセル中心：
       x = (j - 0.5) dx
       y = (i - 0.5) dy
       */
    fprintf(fp, "DIMENSIONS %d %d %d\n", Nx, Ny, 1);
    fprintf(fp, "ORIGIN %.15e %.15e %.15e\n", 0.5 * dx, 0.5 * dy, 0.0);
    fprintf(fp, "SPACING %.15e %.15e %.15e\n", dx, dy, 1.0);

    fprintf(fp, "POINT_DATA %d\n", Nx * Ny);

    /* pressure */
    fprintf(fp, "SCALARS pressure double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");

    for (int i = 1; i < Ny + 1; i++) {
        for (int j = 1; j < Nx + 1; j++) {
            double p = grid.p_[i * pitch_p + j];
            fprintf(fp, "%.15e\n", p);
        }
    }

    /* alpha*/
    fprintf(fp, "SCALARS alpha double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");

    for (int i = 1; i < Ny + 1; i++) {
        for (int j = 1; j < Nx + 1; j++) {
            double alpha = grid.alpha_[i * pitch_p + j];
            fprintf(fp, "%.15e\n", alpha);
        }
    }


    /* velocity */
    fprintf(fp, "VECTORS velocity double\n");

    for (int i = 1; i < Ny + 1; i++) {
        for (int j = 1; j < Nx + 1; j++) {

            double ux =
                0.5 *
                (
                 grid.vx_[i * pitch_vx + j - 1]
                 + grid.vx_[i * pitch_vx + j]
                );

            double uy =
                0.5 *
                (
                 grid.vy_[(i - 1) * pitch_vy + j]
                 + grid.vy_[i * pitch_vy + j]
                );

            fprintf(fp, "%.15e %.15e %.15e\n", ux, uy, 0.0);
        }
    }

    /* divergence */
    fprintf(fp, "SCALARS divergence double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");

    for (int i = 1; i < Ny + 1; i++) {
        for (int j = 1; j < Nx + 1; j++) {

            double div =
                (
                 grid.vx_[i * pitch_vx + j]
                 - grid.vx_[i * pitch_vx + j - 1]
                ) * grid.inv_dx_
                + (
                        grid.vy_[i * pitch_vy + j]
                        - grid.vy_[(i - 1) * pitch_vy + j]
                  ) * grid.inv_dy_;

            fprintf(fp, "%.15e\n", div);
        }
    }

    fclose(fp);

    printf("VTK output: %s\n", filename);
}


inline void print_array(double *array, int Nx, int Ny){
    for (int i=0; i<Ny; i++){
        for (int j=0; j<Nx; j++){
            printf("%3.2e ",array[i*Nx+j]);
        }
        printf("\n");
    }
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

    int Nx=64;
    int Ny=64;
    int Nz=64;

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
    //solv.set_gravity(0., 0.);
    solv.set_rhos(rho_g,rho_w);
    solv.set_mus(mu_g,mu_w);

    solv.solver_malloc();

    solv.grid_.sigma_[0] =sigma; // temporal implementation

    solv.grid_.get_cell_coord();
    //solv.grid_.place_vof(0.,0.2,0.,0.5,1.0);
    solv.grid_.place_vof(0.,0.6,0.,1.5,0.,0.6,1.0);
    //solv.grid_.place_vof(0.,1.0,0.,0.5,1.0);

    /*for surface tension test*/
   // solv.initialize_disk();

    /*for zalesak test*/
    /*
    solv.initialize_zalesak_disk();
    solv.set_zalesak_rotation_velocity();
    */

    solv.set_boundary_velocity();
    solv.set_boundary_pressure();
    solv.set_boundary_alpha();
    solv.update_properties_by_alpha_initial();


    //solv.fix_pressure();
    solv.shift_pressure_reference();

    /* output initial data */
    output_vtk(solv.grid_,0.,outdir);

    G_SMACSolver g_solv;
    g_solv.set_calc_properties(rho, dt,u_lid, nu, sizex, sizey, sizez,Nx, Ny,Nz);

    if(GPU_ON ==1){

        printf("Allocating memory for gpu\n");
        g_solv.solver_malloc();
        printf("Allocated!\n");

        printf("Copying data to gpu\n");
        g_solv.cpuTogpu(solv.grid_);
        printf("Copying data done\n");

        g_solv.set_block_grid(Nx,Ny);
    }

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
    double endTime = 2.0;
    double max_dt = 1e-2;

    CFDTime cfdtime(dt,max_dt,outfreqtime,endTime,cfl_thresh,mode);




    /* ============================= 
       ======== main loop ==========
       ============================= */


    solv.solver_free();

    if(GPU_ON==1){
        g_solv.solver_free();
        //gmgSolver.levels_.free();
    }

    printf("my CFD Done!!!\n");


    return 0;
}
