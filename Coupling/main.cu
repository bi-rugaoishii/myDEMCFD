#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// CFD includes//
#include "../CFD/hardCodedParameters.h"
#include "../CFD/StaggeredGrid.h"
#include "../CFD/G_StaggeredGrid.h"
#include "../CFD/misc.h"
#include "../CFD/SMACSolver.h"
#include "../CFD/config/ConfigLoader.h"
#include "../CFD/config/SimulationConfig.h"
#include "../CFD/SimulationSetup.h"
#include "../CFD/pressure_solver/G_PressureSolverBase.h"
#include "../CFD/pressure_solver/G_PCGSolver.h"
#include "../CFD/pressure_solver/G_GMGSolver.h"
#include "../CFD/G_SMACSolver.h"
#include "../CFD/CFDTime.h"
#include "omp.h"
#include "../CFD/FileInOut.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

// DEM includes//
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../DEM/hardCodedParameters.h"
#include <math.h>
#include <chrono>
#include "../DEM/ParticleSystem.h"
#include "../DEM/BoundingBox.h"
#include "../DEM/device_dem.h"
#include "../DEM/cpu_dem.h"
#include "../DEM/output.h"
#include "../DEM/TriangleMesh.h"
#include "../DEM/BVH.h"
#include "../DEM/solver_output.h"
#include "../DEM/Vec3.h"
#include "../DEM/settings_loader.h"
#define OUTPUT 1
#define NONDIM 0

// CFDDEM includes//
#include "G_CFDDEMCoupling.h"

int main(int argc, char** argv){ 
    setvbuf(stdout,NULL,_IOLBF,0);
    setvbuf(stderr,NULL,_IONBF,0);
    printf("OMP threads: %d\n", omp_get_max_threads());

    /* ===== dem initializations ==== */
    ParticleSys<HostMemory> ps;
    ParticleSys<HostMemory> tmpPs;


    /* copy particle system for later swap */
    // ParticleSystem tmpPs;

    BoundingBox box;

    /* ===  readfiles ========= */
    printf("reading demSettings.json ....\n");

    cJSON *jsonSettings = load_json_file("demSettings.json");

    if (!jsonSettings) {
        printf("reading demSettings.json failed !!!!\n");
        return 1;
    }

    printf("jsonSettings loaded successfully!\n");


    cJSON *particleTypes = get_json_object(jsonSettings, "particleTypes");
    cJSON *particle0     = get_json_object(particleTypes, "particle0");
    cJSON *json_inlet    = get_json_object(jsonSettings, "inlet");

    double r       = get_json_double(particle0, "radius");
    double res     = get_json_double(particle0, "CoR");
    double density = get_json_double(particle0, "density");
    double E       = get_json_double(particle0, "YoungModulus");
    double mu      = get_json_double(particle0, "mu");

    double m = density * 3.14 * r * r * r * 4. / 3.;

    cJSON *json_others = get_json_object(jsonSettings, "others");

    int isGPUon   = get_json_int(json_others, "gpuOn");
    int isBruteOn = get_json_int(json_others, "bruteOn");

    if(isBruteOn ==1){
        printf("BRUTE FORCE MODE!!!!USED ONLY FOR DEBUGGING PURPOSE!!!\n");
    }


    cJSON *json_walls = cJSON_GetObjectItem(jsonSettings,"walls");


    printf("allocating memory\n");
    ps.N = cJSON_GetObjectItem(json_inlet,"numParticle")->valueint;
    printf("N= %d\n",ps.N);
    tmpPs.N = ps.N;
    ps.mu = mu;

    allocate(&ps);
    allocate(&tmpPs);


    /* read triangles */
    printf("\n Loading Triangles\n");

    //const char* trianglesDir = "geometry/box.stl";
    cJSON *json_filepaths = cJSON_GetObjectItemCaseSensitive(json_walls,"filepaths");

    TriangleMesh mesh;
    load_ascii_stl_double(json_filepaths,&mesh);
    printf("Loading Triangles done!\n");




    printf("initalizing particles\n");
    initializeParticles(&ps,json_inlet,r,m,E,res);
    initializeTmpParticles(&tmpPs,json_inlet,r,m,E,res); 
    printf("initalizing particles done\n");
    printf("eta const[0] = %f\n",ps.etaconst[0]);



    /* give gravity */

    cJSON *json_gravity = cJSON_GetObjectItem(jsonSettings,"gravity");

    ps.g[0] = cJSON_GetObjectItem(json_gravity,"x")->valuedouble;
    ps.g[1] = cJSON_GetObjectItem(json_gravity,"y")->valuedouble;
    ps.g[2] = cJSON_GetObjectItem(json_gravity,"z")->valuedouble;
    printf("g=%f %f %f\n", ps.g[0],ps.g[1],ps.g[2]);

    /* set time step */
    double timestepFactor = cJSON_GetObjectItem(json_others,"dtFactor")->valuedouble;
    double dt = 2.*PI*sqrt(m/ps.k[0])/timestepFactor; 
    printf("dt = %e\n",dt);
    double out_time = cJSON_GetObjectItem(json_others,"outputTiming")->valuedouble;
    double end_time = cJSON_GetObjectItem(json_others,"endTime")->valuedouble;
    int outStep = (int)floor(out_time/dt);
    dt = out_time/(double)outStep; // chooses closest dt such that closest to initial set dt and is multiple of out_time
    printf("Outstep = %d,dt = %e\n",outStep,dt);

    ps.dt=dt;

    double initial_demdt = dt;


    const char *inlet_type = cJSON_GetObjectItem(json_inlet,"inputMode")->valuestring;
    cJSON *json_inlet_type = cJSON_GetObjectItem(json_inlet,inlet_type);

    printf("\nInitializing the Bounding Box\n");
    initialize_BoundingBox(&ps, &box, &mesh,json_inlet_type,isGPUon);
    printf("Initializing the Bounding Box Done!!\n");

    printf("Bounding box min (x,y,z)= %f %f %f\n",box.minx ,box.miny, box.minz);
    printf("Bounding box max (x,y,z)= %f %f %f\n",box.maxx ,box.maxy, box.maxz);

    printf("\nUpdating triangle list\n");
    update_tList(&box, &mesh);
    printf("Updating triangle list done!\n");



    printf("\nUpdating neighbor list\n");
    update_neighborlist(&ps,&tmpPs,&box);
    printf("Updating neighbor list done!\n");



    /* initialization for file output */

    #if OUTPUT
    const char* outdir = "results";
    solver_output_init(outdir);

    int numWrite=2;
    for (int i=0; i<numWrite; i++){
        write_header_text(outdir,0,&ps,i);
    }
    #endif

    write_initialPos_csv(outdir,&ps);

    /* non dimensionalize */
    #if NONDIM
    printf("nondimensionalizing ...\n");
    nondimensionalize(&ps,&box,&mesh);
    // nondimensionalize(&tmpPs,&box,&mesh);
    printf("nondimensionalizing done \n");
    printf("\n");
    printf("g after nondim is %f %f %f \n",ps.g[0],ps.g[1],ps.g[2]);
    printf("m[0]:%f r[0]:%f dt:%f \n",ps.m[0],ps.r[0],ps.dt);
    printf("\n");
    #endif

    /* create BVH */
    printf("\nCreating BVH\n");

    BVH bvh;

    initializeBVH(&bvh, mesh.nTri,isGPUon);
    buildBVH(&bvh, &mesh);
    printf("BVH built\n");


    printf("\nCreating wall neighborlist\n");
    update_neighborlist_wall(&ps,&mesh,&bvh,box.skinR);
    printf("created wall neighborlist\n");


    /* === Memory Allocation for gpu === */

    ParticleSys<DeviceMemory> d_ps;
    //ParticleSys<DeviceMemory> d_psTmp;
    d_ps.N = ps.N;
    d_ps.dt = ps.dt;
    d_ps.mu = ps.mu;
    #if USE_GPU
    if (isGPUon == 1){
        allocate(&d_ps);
        //   allocate(&d_psTmp);
        printf("copying memory to device\n");
        copyToDevice(&ps,&d_ps);
        //  copyToDevice(&tmpPs,&d_psTmp);
        copyToDeviceBox(&box,&ps);
        copyToDeviceBVH(&bvh,mesh.nTri);
        deviceMallocCopyTriangleMesh(&mesh);
        printf("copying memory to device done\n");
    }
    #endif




    int blockSize =0;
    int gridSize =0;
    if(isGPUon == 1){
        blockSize = 256;
        gridSize = (ps.N + blockSize - 1) / blockSize;
        printf("grid=%d, block=%d\n", gridSize, blockSize);

        check_g_kernel<<<1, 1>>>(d_ps.d_self,mesh.d_meshPtr);
        cudaDeviceSynchronize();
    }

    /* == dem initialization done ==*/
    /* == dem initialization done ==*/
    /* == dem initialization done ==*/

    /* ======= CFD initialization === */
    /* ======= CFD initialization === */
    /* ======= CFD initialization === */

    printf("Starting myCFD\n\n");

    const std::string config_filename = argc >=2 ? argv[1] : "cfdSettings.json";

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

    //const char* outdir_cfd = config.output.directory.c_str();

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
    g_solv.set_calc_properties(originx, originy, originz, sizex, sizey, sizez, Nx, Ny, Nz);

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
        //pcgSolver.set_solver(PURENEUMANN_STANDARD_PCG);
        printf("\n\n PURENEUMANN\n\n");
    }else{
        pcgSolver.set_solver(GMG_PCG);
        //pcgSolver.set_solver(STANDARD_PCG);
    }


    /* output initial data */
    g_solv.gpuTocpu(solv.grid_);

    printf("output initial data done \n");


    /* == CFD initialization done ==*/
    /* == CFD initialization done ==*/
    /* == CFD initialization done ==*/

    /* ==== CFDDEM coupling ==== */

    G_CFDDEMCoupling cfddem;
    cfddem.block_dim_ = g_solv.block_dim_;
    cfddem.grid_dim_ = g_solv.grid_dim_;

    /* == void fraction has to be initialized as 1 ==*/
    cfddem.initialize_void_fractions(g_solv.grid_);
    cfddem.set_particle_volume_to_cell(g_solv.grid_,d_ps,gridSize, blockSize);
    cfddem.gaussian_filter_particle_volume(g_solv.grid_);
    cfddem.calc_void_fraction(g_solv.grid_);
    cfddem.sync_initial_void_fraction(g_solv.grid_);
    cfddem.update_boundary_ghost_void_fraction(g_solv.grid_);

    /* ==== CFDDEM Coupling initialization done === */


    /* === calculate one step of CFD to get pressure gradient === */
    {
        int cur_step=0;

        /* initial step requires high tolerance*/
        g_solv.pressure_solver_->tol_ = 1e-11;

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
        g_solv.init_void_fraction_vof_two_way();

        double sub_dt=cfdtime.dt_/(double)alpha_substeps;
        for(int substeps=0;substeps<alpha_substeps;substeps++){
            printf("alpha subcycle %d/%d\n",substeps+1,alpha_substeps);
            g_solv.clear_alpha_flux();
            g_solv.alpha_flux_thincwlic_split_two_way(sub_dt,cur_step);
            g_solv.alpha_flux_accum();
        }/* == transport alpha done == */

        g_solv.finalize_alpha_two_way();
        g_solv.update_properties_by_alpha();
        g_solv.compute_mass_flux_from_alpha_flux_two_way(solv);

        cfddem.update_poisson_beta_two_way(g_solv.grid_);
        g_solv.update_boundary_faces();


        g_solv.calc_surface_tension();


        g_solv.get_vof_vstar_rhouu_consistent_two_way(solv);

        g_solv.update_vstar_boundary();


        printf("starting poisson\n");
        g_solv.solve_poisson();

        g_solv.correct_vof_velocity(solv);
        g_solv.update_boundary_ghost(solv);
        g_solv.make_face_gradp();

        g_solv.gpuTocpu(solv.grid_);
        //solv.check_pressure_jump_by_radius();
        fileIO.output_vti_binary_cellData(solv.grid_,0.0);

    }


    /* === calc time measurement === */
    double h_start, h_end;
    float ms;
    h_start = omp_get_wtime();

    /* ====== main dem routine ===== */

    printf("starting \n");

    /* == set initial tolerance ==*/
    g_solv.pressure_solver_->tol_ = config.pressure_solver.tol;

    /* ========== GPU ============= */
    if(isGPUon ==1){

        int cur_step = 0;
        while(cfdtime.current_time_ < cfdtime.end_time_-EPS){


            /* ==starting new step== */
            printf("====================================\n");


            /* == calculate cfd timestep == */
            double cfl=g_solv.calc_cfl();
            cfdtime.updateTime(cfl);

            solv.dt_ = cfdtime.dt_;
            solv.inv_dt_ = 1./cfdtime.dt_;

            g_solv.dt_ = cfdtime.dt_;
            g_solv.inv_dt_ = 1./cfdtime.dt_;

            printf("dt = %3.2e, current time = %f\n",cfdtime.dt_,cfdtime.current_time_);

            /* == for two way == */
            cudaMemset(g_solv.grid_.f_coupling_impulse_x_.data_, 0, sizeof(double)*g_solv.grid_.f_coupling_impulse_x_.size_);
            cudaMemset(g_solv.grid_.f_coupling_impulse_y_.data_, 0, sizeof(double)*g_solv.grid_.f_coupling_impulse_y_.size_);
            cudaMemset(g_solv.grid_.f_coupling_impulse_z_.data_, 0, sizeof(double)*g_solv.grid_.f_coupling_impulse_z_.size_);

            /* == calculate dem timestep == */

            int numDemSubSteps = 0;

            if(cfdtime.dt_ < initial_demdt){
                printf("CFD time step is smaller than dem timestep!\n");
                numDemSubSteps = 1;
                d_ps.dt = cfdtime.dt_;

                k_update_demdt<<<1,1>>>(d_ps.d_self, d_ps.dt);
            }else{
                numDemSubSteps = (int)ceil(cfdtime.dt_/initial_demdt);
                d_ps.dt = cfdtime.dt_/(double)numDemSubSteps;

                k_update_demdt<<<1,1>>>(d_ps.d_self, d_ps.dt);
            }

            printf("dem dt = %3.2e\n",d_ps.dt);


            for (int demStep = 0; demStep < numDemSubSteps; demStep++){

                cfddem.interpolate_fluid_to_particle(g_solv.grid_, d_ps, gridSize, blockSize);

                /* GPU */
                if(isBruteOn==1){
                    device_dem_naive(&d_ps,&box,&mesh,&bvh,gridSize, blockSize);

                }else{

                    //device_dem_verlet_verlet_cfd(&d_ps, &box, &mesh,&bvh,gridSize, blockSize);
                    device_dem_verlet_verlet_cfd_two_way(&d_ps, &box, &mesh,&bvh,g_solv.grid_,gridSize, blockSize);
                }


            }



            /* == dem steps done == */
            /* == starting cfd steps == */
            std::swap(g_solv.grid_.void_fraction_.data_, g_solv.grid_.void_fraction_old_.data_);
            k_swap_voidfraction<<<1,1>>>(g_solv.grid_.d_ptr_);

            cfddem.set_particle_volume_to_cell(g_solv.grid_,d_ps,gridSize, blockSize);
            cfddem.gaussian_filter_particle_volume(g_solv.grid_);
            cfddem.calc_void_fraction(g_solv.grid_);
            cfddem.update_boundary_ghost_void_fraction(g_solv.grid_);


            /* == transport alpha == */
            g_solv.clear_alpha_flux_accum();
            g_solv.init_void_fraction_vof_two_way();


            double sub_dt = cfdtime.dt_/(double)alpha_substeps;
            for (int substeps=0 ; substeps<alpha_substeps; substeps++){ 
                printf("alpha subcycle %d/%d\n", substeps+1,alpha_substeps);
                g_solv.clear_alpha_flux();
                g_solv.alpha_flux_thincwlic_split_two_way(sub_dt,cur_step);


                g_solv.alpha_flux_accum();
            }
            /* == transport alpha done == */

            g_solv.finalize_alpha_two_way();
            g_solv.update_properties_by_alpha();
            g_solv.compute_mass_flux_from_alpha_flux_two_way(solv);

            cfddem.update_poisson_beta_two_way(g_solv.grid_);
            g_solv.update_boundary_faces();


            g_solv.calc_surface_tension();


            g_solv.get_vof_vstar_rhouu_consistent_two_way(solv);

            g_solv.update_vstar_boundary();


            printf("starting poisson\n");
            g_solv.solve_poisson();

            g_solv.correct_vof_velocity(solv);
            g_solv.update_boundary_ghost(solv);
            g_solv.make_face_gradp();



            if(cfdtime.isOutStep_){

                copyFromDevice(&d_ps,&ps);
                write_frame_bin(outdir,cfdtime.current_steps_,&ps,1.0);
                for (int i=0; i<numWrite; i++){
                    write_single_text(outdir,cfdtime.current_steps_,&ps,i);
                }

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
                printf("step %d,next dt = %f s, current time: %f s, GPU time: %f s\n", step,cfdtime.dt_,current_time,ms);
            }
            printf("====================================\n");

            cur_step ++;

        }
    }

    /* === free memories === */

    printf("deallocating memories\n");
    deallocate(&ps);
    deallocate(&tmpPs);
#if USE_GPU
    if(isGPUon==1){
        deallocate(&d_ps);
    }
#endif

    solv.solver_free();
    if(use_gpu){
        g_solv.solver_free();
        gmgSolver.free_levels();
    }
    cJSON_Delete(jsonSettings);

    free_TriangleMesh(&mesh, isGPUon);
    free_BoundingBox(&box, isGPUon);
    free_BVH(&bvh, isGPUon);

#if USE_GPU
    if(isGPUon==1){
        cudaDeviceReset();
    }
#endif

    return 0;
}
