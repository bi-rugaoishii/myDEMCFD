#include "G_PCGSolver.h"

static __global__ void k_build_vof_poisson_Ap(G_StaggeredGrid grid_,const double *p, int Nx, int Ny){
    int i = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int j = blockIdx.x*blockDim.x + threadIdx.x+1;

    double* const Ap = grid_.pcg_Ap_;

    double inv_dx2 = grid_.inv_dx2_;
    double inv_dy2 = grid_.inv_dy2_;


    double* const f_bx = grid_.f_bx_;
    double* const f_by = grid_.f_by_;


    int pitch_vx=Nx+3;
    int pitch_vy=Nx+2;
    int pitch_c=Nx+2;

    int ind=i*pitch_c+j;
    if (i >=Ny+1 || j >= Nx+1) return;

    double p_c = p[ind];
    double p_xp = p[i*pitch_c+j+1];
    double p_xm = p[i*pitch_c+j-1];
    double p_yp = p[(i+1)*pitch_c+j];
    double p_ym = p[(i-1)*pitch_c+j];

    double b_xp = f_bx[i*pitch_vx+j];
    double b_xm = f_bx[i*pitch_vx+j-1];
    double b_yp = f_by[i*pitch_vy+j];
    double b_ym = f_by[(i-1)*pitch_vy+j];

    Ap[ind] = (b_xp*(p_c-p_xp) + b_xm*(p_c-p_xm))*inv_dx2
        + (b_yp*(p_c-p_yp) + b_ym*(p_c-p_ym))*inv_dy2;
}

static __global__ void k_build_vof_poisson_invdiag(G_StaggeredGrid grid_,int Nx, int Ny){

    double inv_dx2 = grid_.inv_dx2_;
    double inv_dy2 = grid_.inv_dy2_;

    double* const f_bx = grid_.f_bx_;
    double* const f_by = grid_.f_by_;

    double* const invdiag = grid_.pcg_invDiag_;

    int pitch_vx=Nx+3;
    int pitch_vy=Nx+2;
    int pitch_c=Nx+2;

    int i = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int j = blockIdx.x*blockDim.x + threadIdx.x+1;

    if (i >=Ny+1 || j >= Nx+1) return;
    int ind=i*pitch_c+j;


    double b_xp = f_bx[i*pitch_vx+j];
    double b_xm = f_bx[i*pitch_vx+j-1];
    double b_yp = f_by[i*pitch_vy+j];
    double b_ym = f_by[(i-1)*pitch_vy+j];

    double diag = (b_xp+b_xm)*inv_dx2 + (b_yp+b_ym)*inv_dy2;

    invdiag[ind] = 1.0/diag;
}

static __global__ void k_get_r(G_StaggeredGrid grid_,double* const mean_b, double* const tmp, int Nx, int Ny){
    int i = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int j = blockIdx.x*blockDim.x + threadIdx.x+1;

    double* const r = grid_.pcg_r_;
    const double* const Ap = grid_.pcg_Ap_;
    const double* const rhs = grid_.rhs_;

    int pitch = Nx + 2;
    int pitch_tmp = Nx;
    if (i >=Ny+1 || j >= Nx+1) return;
    int ind = (i)*pitch+j;

    double b = rhs[ind] - mean_b[0]; 
    r[ind] = b-Ap[ind];

    int cind = (i-1)*pitch_tmp+j-1;
    tmp[cind] = b*b;
}

static __global__ void k_make_pAp(G_StaggeredGrid grid_,int Nx,int Ny){
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;

    MyArray<double,2>&  dir     = grid_.wew_pcg_dir_;
    MyArray<double,2>&  Ap      = grid_.wew_pcg_Ap_;
    MyArray<double,2>&  pcg_tmp = grid_.wew_pcg_tmp_;

    MyArray<double,2>&  f_bx = grid_.wew_f_bx_;
    MyArray<double,2>&  f_by = grid_.wew_f_by_;



    if (ix >= Ap.sizex_-1 || iy >= Ap.sizey_-1) return;


    double inv_dx2 = grid_.inv_dx2_;
    double inv_dy2 = grid_.inv_dy2_;


    double q_c = dir(ix,iy);

    // Neumann boundary: ghost value = center value
    double q_xp = (ix == Ap.sizex_-2) ? q_c : dir(ix+1,iy);
    double q_xm = (ix == 1 ) ? q_c : dir(ix-1,iy);
    double q_yp = (iy == Ap.sizey_-2) ? q_c : dir(ix,iy+1);
    double q_ym = (iy == 1 ) ? q_c : dir(ix,iy-1);

    double b_xp = f_bx(ix,iy);
    double b_xm = f_bx(ix-1,iy);
    double b_yp = f_by(ix,iy);
    double b_ym = f_by(ix,iy-1);

    double Ap_val =
        (b_xp * (q_c - q_xp) + b_xm * (q_c - q_xm)) * inv_dx2
      + (b_yp * (q_c - q_yp) + b_ym * (q_c - q_ym)) * inv_dy2;

    Ap(ix,iy) = Ap_val;

    pcg_tmp(ix-1,iy-1)= dir(ix,iy)* Ap_val;
}

static __global__ void k_update_p_r_z_make_rz_tmp(G_StaggeredGrid grid,const double* scalars,int Nx,int Ny){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;

    if(ix>=Nx+1||iy>=Ny+1)return;

    int pitch_c=Nx+2;
    int pitch_tmp=Nx;

    int ind=ix+iy*pitch_c;
    int tid=(ix-1)+(iy-1)*pitch_tmp;

    double alpha=scalars[RZ_OLD]/scalars[PAP];

    double dir=grid.pcg_dir_[ind];
    double Ap=grid.pcg_Ap_[ind];

    double r_new=grid.pcg_r_[ind]-alpha*Ap;
    grid.p_[ind]+=alpha*dir;
    grid.pcg_r_[ind]=r_new;

    double z_new=grid.pcg_invDiag_[ind]*r_new;
    grid.pcg_z_[ind]=z_new;
    grid.pcg_tmp_[tid]=r_new*z_new;
}

static __global__ void k_get_r2_to_tmp(G_StaggeredGrid grid_,int Nx, int Ny){
    int i = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int j = blockIdx.x*blockDim.x + threadIdx.x+1;

    double* const r = grid_.pcg_r_;
    double* const tmp = grid_.pcg_tmp_;
    int pitch_c=Nx+2;
    int pitch_tmp=Nx;

    int ind=i*pitch_c+j;
    int indc=(i-1)*pitch_tmp+(j-1);

    if (i >=Ny+1 || j >= Nx+1) return;
    tmp[indc] = r[ind]*r[ind];
}

static __global__ void k_update_dir(G_StaggeredGrid grid_,double *const pcg_scalars_){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;

    MyArray<double,3> dir = grid_.pcg_dir_;
    MyArray<double,3> z = grid_.pcg_z_;


    if (iy >=dir.sizey_-1 || ix >= dir.sizex_-1 || iz >= dir.sizez_-1) return;
    double beta =pcg_scalars_[RZ_NEW]/pcg_scalars_[RZ_OLD] ; 
    dir(ix,iy,iz) = z(ix,iy,iz)+beta*dir(ix,iy,iz);
}

static __global__ void k_swap_rz(double *s){
    s[RZ_OLD] = s[RZ_NEW];
}



/* =================================================*/
/* ================ host functions == */
/* =================================================*/
G_PCGSolver::G_PCGSolver(){}

G_PCGSolver::~G_PCGSolver(){}

void G_PCGSolver::solve(G_SMACSolver& solv){

#if PCG_PROFILE
    PCGProfile prof;
    reset_pcg_profile(prof);

    cudaEvent_t ev_total_start;
    cudaEvent_t ev_total_stop;
    cudaEvent_t ev_start;
    cudaEvent_t ev_stop;

    cudaEventCreate(&ev_total_start);
    cudaEventCreate(&ev_total_stop);
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_total_start,0);
#endif

    G_StaggeredGrid& grid_=solv.grid_;

    int Nx = solv.grid_.Nx_;
    int Ny = solv.grid_.Ny_;

    int checkResidualFreq = 10;



    double* const r       = grid_.pcg_r_;
    double* const z       = grid_.pcg_z_;
    double* const dir     = grid_.pcg_dir_;
    double* const invdiag = grid_.pcg_invDiag_;

    int max_iter = 100000;
    double tol = 1e-6;
    double inv_dt_ = solv.inv_dt_;

#if PCG_PROFILE
    pcg_prof_start(ev_start);
#endif

    pres_k_make_poisson_rhs<<<grid_dim_,block_dim_>>>(grid_,inv_dt_);

    int block_size=256;
    int n=max(Nx,Ny);
    int grid_size_boundary=(n+block_size-1)/block_size;

    int grid_size_all=((Nx+2)*(Ny+2)+block_size-1)/block_size;

    pres_k_shift_pressure_reference<<<grid_dim_,block_dim_>>>(grid_,Nx,Ny);
    pres_k_fix_pressure_reference<<<1,1>>>(grid_,Nx,Ny);
    pres_k_set_boundary_array<<<grid_size_boundary,block_size>>>(grid_.p_,Nx, Ny);

    k_build_vof_poisson_invdiag<<<grid_dim_,block_dim_>>>(grid_,Nx,Ny);
    k_build_vof_poisson_Ap<<<grid_dim_,block_dim_>>>(grid_,grid_.p_,Nx,Ny);

    /* create mean_b*/
    pres_k_copy_to_tmp<<<grid_dim_,block_dim_>>>(grid_.rhs_,grid_.pcg_tmp_,Nx,Ny);
    cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_,d_dot_,Nx*Ny);

    /* mean_b is in d_dot_ */
    pres_k_divide<<<1,1>>>(d_dot_,(double)(Nx*Ny));

    /* initial residual; r = b- Ap */
    k_get_r<<<grid_dim_,block_dim_>>>(grid_,d_dot_,grid_.pcg_tmp_,Nx,Ny);

    cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_,d_dot_,Nx*Ny);
    /* norm_b2 is in d_dot_ */

    double norm_b;
    cudaMemcpy(&norm_b,d_dot_,sizeof(double),cudaMemcpyDeviceToHost);
    if (norm_b < 1.0e-16) {
        norm_b = 1.0;
    }
    norm_b=sqrt(norm_b);

    Base::subtract_cell_mean(grid_,r,Nx,Ny);

    /* z = M^{-1} r */
    pres_k_mult_elementwise_array<<<grid_size_all,block_size>>>(invdiag,r,z,(Nx+2)*(Ny+2));


    Base::subtract_cell_mean(grid_,z,Nx,Ny);
    cudaMemcpy(dir, z, (Nx+2)*(Ny+2) * sizeof(double), cudaMemcpyDeviceToDevice);

    pres_k_set_boundary_array<<<grid_size_boundary,block_size>>>(dir,Nx, Ny);

    /* rz_old = r dot z */
    pres_k_mult_elementwise_array_to_tmp<<<grid_dim_,block_dim_>>>(r,z,grid_.pcg_tmp_,Nx,Ny);
    double *const rz_old=&d_pcg_scalars_[RZ_OLD]; 

    cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_,rz_old,Nx*Ny);

#if PCG_PROFILE
    pcg_prof_stop(ev_start,ev_stop,prof.init_ms);
#endif

    int iter = 0;

    /* == start main loop ==*/
    for(iter=0;iter<max_iter;++iter){

#if PCG_PROFILE
        pcg_prof_start(ev_start);
#endif
        k_make_pAp<<<grid_dim_,block_dim_>>>(grid_,Nx,Ny);
#if PCG_PROFILE
        pcg_prof_stop(ev_start,ev_stop,prof.Ap_kernel_ms);
#endif

#if PCG_PROFILE
        pcg_prof_start(ev_start);
#endif
        double* const pAp=&d_pcg_scalars_[PAP];
        cub::DeviceReduce::Sum(cub_temp_storage_,cub_temp_storage_bytes_,grid_.pcg_tmp_,pAp,Nx*Ny);
#if PCG_PROFILE
        pcg_prof_stop(ev_start,ev_stop,prof.pAp_reduce_ms);
#endif


#if PCG_PROFILE
        pcg_prof_start(ev_start);
#endif
        k_update_p_r_z_make_rz_tmp<<<grid_dim_,block_dim_>>>(grid_,d_pcg_scalars_,Nx,Ny);
#if PCG_PROFILE
        pcg_prof_stop(ev_start,ev_stop,prof.update_prz_ms);
#endif

#if PCG_PROFILE
        pcg_prof_start(ev_start);
#endif
        double* const rz_new=&d_pcg_scalars_[RZ_NEW];
        cub::DeviceReduce::Sum(cub_temp_storage_,cub_temp_storage_bytes_,grid_.pcg_tmp_,rz_new,Nx*Ny);
#if PCG_PROFILE
        pcg_prof_stop(ev_start,ev_stop,prof.rz_reduce_ms);
#endif


        if(iter%checkResidualFreq==0){
#if PCG_PROFILE
            pcg_prof_start(ev_start);
#endif

            k_get_r2_to_tmp<<<grid_dim_,block_dim_>>>(grid_,Nx,Ny);
            cub::DeviceReduce::Sum(cub_temp_storage_,cub_temp_storage_bytes_,grid_.pcg_tmp_,d_dot_,Nx*Ny);

            double r2;
            cudaMemcpy(&r2,d_dot_,sizeof(double),cudaMemcpyDeviceToHost);
            double rel_res=sqrt(r2)/norm_b;

#if PCG_PROFILE
            pcg_prof_stop(ev_start,ev_stop,prof.residual_check_ms);
#endif

            if(rel_res<tol){
                printf("norm r = %3.2e, norm b = %3.2e, rel_res = %3.2e \n",sqrt(r2),norm_b,rel_res);
                iter++;
                break;
            }
        }


#if PCG_PROFILE
        pcg_prof_start(ev_start);
#endif
        k_update_dir<<<grid_dim_,block_dim_>>>(grid_,d_pcg_scalars_,Nx,Ny);
#if PCG_PROFILE
        pcg_prof_stop(ev_start,ev_stop,prof.update_dir_ms);
#endif

#if PCG_PROFILE
        pcg_prof_start(ev_start);
#endif
        k_swap_rz<<<1,1>>>(d_pcg_scalars_);
#if PCG_PROFILE
        pcg_prof_stop(ev_start,ev_stop,prof.swap_rz_ms);
#endif
    }


#if PCG_PROFILE
    pcg_prof_start(ev_start);
#endif

    pres_k_shift_pressure_reference<<<grid_dim_,block_dim_>>>(grid_,Nx,Ny);
    pres_k_fix_pressure_reference<<<1,1>>>(grid_,Nx,Ny);

#if PCG_PROFILE
    pcg_prof_stop(ev_start,ev_stop,prof.final_ms);
#endif

#if PCG_PROFILE
    cudaEventRecord(ev_total_stop,0);
    cudaEventSynchronize(ev_total_stop);
    cudaEventElapsedTime(&prof.total_ms,ev_total_start,ev_total_stop);

    prof.iter=iter;

    float loop_ms=prof.Ap_kernel_ms
        +prof.pAp_reduce_ms
        +prof.update_prz_ms
        +prof.rz_reduce_ms
        +prof.residual_check_ms
        +prof.update_dir_ms
        +prof.swap_rz_ms;

    printf("=== PCG profile ===\n");
    printf("PCG iter              = %d\n",prof.iter);
    printf("PCG total             = %.3f ms\n",prof.total_ms);
    printf("init                  = %.3f ms\n",prof.init_ms);
    printf("Ap kernel             = %.3f ms\n",prof.Ap_kernel_ms);
    printf("pAp reduction         = %.3f ms\n",prof.pAp_reduce_ms);
    printf("update p/r/z          = %.3f ms\n",prof.update_prz_ms);
    printf("rz reduction          = %.3f ms\n",prof.rz_reduce_ms);
    printf("residual check        = %.3f ms\n",prof.residual_check_ms);
    printf("update dir            = %.3f ms\n",prof.update_dir_ms);
    printf("swap rz               = %.3f ms\n",prof.swap_rz_ms);
    printf("final                 = %.3f ms\n",prof.final_ms);

    if(prof.iter>0){
        printf("--- per iter ---\n");
        printf("avg total/iter        = %.6f ms\n",prof.total_ms/(float)prof.iter);
        printf("avg Ap kernel         = %.6f ms\n",prof.Ap_kernel_ms/(float)prof.iter);
        printf("avg pAp reduction     = %.6f ms\n",prof.pAp_reduce_ms/(float)prof.iter);
        printf("avg update p/r/z      = %.6f ms\n",prof.update_prz_ms/(float)prof.iter);
        printf("avg rz reduction      = %.6f ms\n",prof.rz_reduce_ms/(float)prof.iter);
        printf("avg residual check    = %.6f ms\n",prof.residual_check_ms/(float)prof.iter);
        printf("avg update dir        = %.6f ms\n",prof.update_dir_ms/(float)prof.iter);
        printf("avg swap rz           = %.6f ms\n",prof.swap_rz_ms/(float)prof.iter);
    }

    if(loop_ms>0.0f){
        printf("--- loop ratio ---\n");
        printf("Ap kernel ratio       = %.1f %%\n",100.0f*prof.Ap_kernel_ms/loop_ms);
        printf("pAp reduction ratio   = %.1f %%\n",100.0f*prof.pAp_reduce_ms/loop_ms);
        printf("update p/r/z ratio    = %.1f %%\n",100.0f*prof.update_prz_ms/loop_ms);
        printf("rz reduction ratio    = %.1f %%\n",100.0f*prof.rz_reduce_ms/loop_ms);
        printf("residual ratio        = %.1f %%\n",100.0f*prof.residual_check_ms/loop_ms);
        printf("update dir ratio      = %.1f %%\n",100.0f*prof.update_dir_ms/loop_ms);
        printf("swap rz ratio         = %.1f %%\n",100.0f*prof.swap_rz_ms/loop_ms);
    }

    printf("===================\n");

    cudaEventDestroy(ev_total_start);
    cudaEventDestroy(ev_total_stop);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
#endif
    printf("PCG iter = %d\n", iter);
}

