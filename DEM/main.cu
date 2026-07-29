#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "hardCodedParameters.h"
#include <math.h>
#include <chrono>
#include "ParticleSystem.h"
#include "BoundingBox.h"
#include "device_dem.h"
#include "cpu_dem.h"
#include "omp.h"
#include "output.h"
#include "TriangleMesh.h"
#include "BVH.h"
#include "solver_output.h"
#include "Vec3.h"
#include "settings_loader.h"
#define OUTPUT 1
#define NONDIM 0





/*
============================================================
メイン
============================================================
*/
int main(){
    setvbuf(stdout,NULL,_IOLBF,0);
    setvbuf(stderr,NULL,_IONBF,0);
    printf("OMP threads: %d\n", omp_get_max_threads());

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

    /*
       for (int i=0; i<mesh.nTri; i++){
       printf("%d %d %d %f %f %f %f %f %f\n",bvh.left[i],bvh.right[i],bvh.tri[i],bvh.minx[i],bvh.maxx[i],bvh.miny[i],bvh.maxy[i],bvh.minz[i],bvh.maxz[i]);
       }
     */

    printf("\nCreating wall neighborlist\n");
    update_neighborlist_wall(&ps,&mesh,&bvh,box.skinR);
    //update_neighborlist_wall_nobvh(&ps,&mesh,&box,box.skinR);
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
    
    
    
    int steps = (int)(end_time/dt);

    #if USE_GPU
    int blockSize =0;
    int gridSize =0;
    cudaEvent_t d_start, d_stop, d_now;
    double h_start, h_end;
    float ms;
    if(isGPUon == 1){
        blockSize = 256;
        gridSize = (ps.N + blockSize - 1) / blockSize;
        printf("grid=%d, block=%d\n", gridSize, blockSize);

        check_g_kernel<<<1, 1>>>(d_ps.d_self,mesh.d_meshPtr);
        cudaDeviceSynchronize();
        cudaEventCreate(&d_start);
        cudaEventCreate(&d_stop);
        cudaEventCreate(&d_now);
        cudaEventRecord(d_start);
    }else{
        h_start = omp_get_wtime();
    }
    #endif



    /* ====== main dem routine ===== */

    printf("starting \n");

    /* ========== GPU ============= */
    if(isGPUon ==1){
        for (int step = 1; step <= steps; step++){
            #if USE_GPU
            /* GPU */
            if(isBruteOn==1){
                device_dem_naive(&d_ps,&box,&mesh,&bvh,gridSize, blockSize);

            }else{

                //device_dem(&d_ps, &box, gridSize, blockSize);
                //device_dem_triangles(&d_ps, &box, &mesh,gridSize, blockSize);
                //device_dem_verlet_triangles(&d_ps, &box, &mesh,gridSize, blockSize);
                device_dem_verlet_verlet(&d_ps, &box, &mesh,&bvh,gridSize, blockSize);
                //device_dem_verlet_verlet_withSort(&d_ps,&d_psTmp, &box, &mesh,&bvh,gridSize, blockSize);
            }


            #if OUTPUT
            if (step % outStep == 0)
            {
                cudaDeviceSynchronize();
                copyFromDevice(&d_ps,&ps);
                #if NONDIM
                write_frame_bin(outdir,step,&ps,ps.length_factor);
                for (int i=0; i<numWrite; i++){
                    write_single_text(outdir,step,&ps,i);
                }
                #else
                write_frame_bin(outdir,step,&ps,1.0);
                for (int i=0; i<numWrite; i++){
                    write_single_text(outdir,step,&ps,i);
                }
                #endif

                cudaEventRecord(d_now);
                cudaEventSynchronize(d_now);

                cudaEventElapsedTime(&ms, d_start, d_now);
                printf("Output step %d, current time: %f, GPU time: %f s\n", step, (step)*dt,ms/1000.0f);
            }
        }
    }else{
        /* ============= CPU ============== */
        #endif
        for (int step = 1; step <= steps; step++){
            /* CPU */
            if(isBruteOn==1){
                cpu_dem_naive_triangle(&ps, &box, &mesh);
            }else{
                // cpu_dem_nosort_triangle(&ps, &tmpPs, &box,&mesh);
                //cpu_dem_sort(&ps, &tmpPs, &box, step);
                //cpu_dem_sort_triangles(&ps, &tmpPs, &box,&mesh, step);
                //cpu_dem_verlet_triangles(&ps, &tmpPs, &box,&mesh, step);
                // cpu_dem_verlet_BVH(&ps, &tmpPs, &box,&mesh, &bvh,step);
                cpu_dem_verlet_verlet(&ps,&tmpPs, &box,&mesh, &bvh,step);

            }

            checkOoB(&ps,&tmpPs,&box);
            #if OUTPUT
            if (step % outStep == 0){
                //  writeParticlesVTKBinary(&ps, step);
                #if NONDIM
                write_frame_bin(outdir,step,&ps,ps.length_factor);
                for (int i=0; i<numWrite; i++){
                    write_single_text(outdir,step,&ps,i);
                }
                #else
                write_frame_bin(outdir,step,&ps,1.0);
                for (int i=0; i<numWrite; i++){
                    write_single_text(outdir,step,&ps,i);
                }
                #endif
                h_end = omp_get_wtime();
                ms = (float)(h_end-h_start);
                printf("Output step %d,current time: %f s, CPU time: %f s\n", step,(step)*dt,ms);
            }
            #endif
            #endif
        }
    }

#if USE_GPU
    if(isGPUon == 1){
        cudaEventRecord(d_stop);
        cudaEventSynchronize(d_stop);
        cudaEventElapsedTime(&ms,d_start, d_stop);
        printf("GPU time: %f s\n",ms/1000.);
    }else{
        h_end = omp_get_wtime();
        ms = (float)(h_end-h_start);
        printf("CPU time: %f s\n",ms);
    }
#endif


    /* === free memories === */

#if USE_GPU
    if(isGPUon==1){
        cudaDeviceReset();
    }
#endif
    printf("deallocating memories\n");
    deallocate(&ps);
    deallocate(&tmpPs);
#if USE_GPU
    if(isGPUon==1){
        deallocate(&d_ps);
    }
#endif

    cJSON_Delete(jsonSettings);
    free_TriangleMesh(&mesh, isGPUon);
    free_BoundingBox(&box, isGPUon);
    free_BVH(&bvh, isGPUon);
    return 0;
}
