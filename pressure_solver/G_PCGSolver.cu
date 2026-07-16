#include "G_PCGSolver.h"
#include "G_GMGSolver.h"

static __global__ void k_get_r2_to_tmp(G_StaggeredGrid grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy= blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;

    MyArray<double,3> r = grid.pcg_r_;
    MyArray<double,3> tmp = grid.pcg_tmp_;


    if (ix >= Nx+1 || iy >=Ny+1 || iz >= Nz+1) return;
    tmp(ix-1,iy-1,iz-1) = r(ix,iy,iz)*r(ix,iy,iz);
}

static __global__ void k_get_r(G_StaggeredGrid grid,double* const mean_b, MyArray<double,3> tmp){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1; //+1 for ghost cell
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;


    MyArray<double,3> r = grid.pcg_r_;
    MyArray<double,3> Ap = grid.pcg_Ap_;
    MyArray<double,3> rhs = grid.rhs_;

    if (iy >=Ny+1 || ix >= Nx+1 || iz >= Nz+1) return;

    double b = rhs(ix,iy,iz)- mean_b[0]; 
    r(ix,iy,iz) = b-Ap(ix,iy,iz);

    tmp(ix-1,iy-1,iz-1) = b*b;
}
static __global__ void k_update_p_r(G_StaggeredGrid grid,const double* scalars,int rz_old_id){
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;

    if (ix >= Nx + 1 || iy >= Ny + 1 || iz >= Nz+1) return;


    double alpha = scalars[rz_old_id] / scalars[PAP];

    double dir = grid.pcg_dir_(ix,iy,iz);
    double Ap  = grid.pcg_Ap_(ix,iy,iz);

    grid.p_delta_(ix,iy,iz)     += alpha * dir;
    grid.pcg_r_(ix,iy,iz) -= alpha * Ap;
}

static __global__ void k_make_pAp(G_StaggeredGrid grid_){
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z * blockDim.z + threadIdx.z + 1;

    MyArray<double,3>&  dir     = grid_.pcg_dir_;
    MyArray<double,3>&  Ap      = grid_.pcg_Ap_;
    MyArray<double,3>&  pcg_tmp = grid_.pcg_tmp_;

    MyArray<double,3> Axp = grid_.Axp_;
    MyArray<double,3> Axm = grid_.Axm_;
    MyArray<double,3> Ayp = grid_.Ayp_;
    MyArray<double,3> Aym = grid_.Aym_;
    MyArray<double,3> Azp = grid_.Azp_;
    MyArray<double,3> Azm = grid_.Azm_;
    MyArray<double,3> Adiag = grid_.Adiag_;

    if (ix >= Ap.sizex_-1 || iy >= Ap.sizey_-1 || iz >= Ap.sizez_ -1) return;




    double q_c = dir(ix,iy,iz);

    double q_xp = dir(ix+1,iy,iz);
    double q_xm = dir(ix-1,iy,iz);
    double q_yp = dir(ix,iy+1,iz);
    double q_ym = dir(ix,iy-1,iz);
    double q_zp = dir(ix,iy,iz+1);
    double q_zm = dir(ix,iy,iz-1);

    double Ap_val= Adiag(ix,iy,iz)*q_c-(Axp(ix,iy,iz)*q_xp+Axm(ix,iy,iz)*q_xm+Ayp(ix,iy,iz)*q_yp+Aym(ix,iy,iz)*q_ym+Azp(ix,iy,iz)*q_zp+Azm(ix,iy,iz)*q_zm);


    Ap(ix,iy,iz) = Ap_val;

    pcg_tmp(ix-1,iy-1,iz-1)= dir(ix,iy,iz)* Ap_val;
}
static __global__ void k_build_vof_poisson_A(G_StaggeredGrid grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell

    MyArray<double,3> Axp = grid.Axp_;
    MyArray<double,3> Axm = grid.Axm_;
    MyArray<double,3> Ayp = grid.Ayp_;
    MyArray<double,3> Aym = grid.Aym_;
    MyArray<double,3> Azp = grid.Azp_;
    MyArray<double,3> Azm = grid.Azm_;
    MyArray<double,3> Adiag = grid.Adiag_;
    MyArray<double,3> invAdiag = grid.invAdiag_;

    double inv_dx2 = grid.inv_dx2_;
    double inv_dy2 = grid.inv_dy2_;
    double inv_dz2 = grid.inv_dz2_;


    MyArray<double,3> f_bx = grid.f_bx_;
    MyArray<double,3> f_by = grid.f_by_;
    MyArray<double,3> f_bz = grid.f_bz_;
    MyArray<unsigned char,3> ct=grid.celltype_;

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;

    if (ix>= Nx+1 ||iy >=Ny+1 || iz >= Nz+1) return;

    if (ct(ix,iy,iz) != C_INTERIOR){
        Axp(ix,iy,iz) = 0.;
        Axm(ix,iy,iz) = 0.;
        Ayp(ix,iy,iz) = 0.;
        Aym(ix,iy,iz) = 0.;
        Azp(ix,iy,iz) = 0.;
        Azm(ix,iy,iz) = 0.;
        Adiag(ix,iy,iz) = 0.;
        invAdiag(ix,iy,iz) = 0.;
        return;
    }

    Axp(ix,iy,iz) = f_bx(ix+1,iy,iz)*inv_dx2;
    Axm(ix,iy,iz) = f_bx(ix,iy,iz)*inv_dx2;
    Ayp(ix,iy,iz) = f_by(ix,iy+1,iz)*inv_dy2;
    Aym(ix,iy,iz) = f_by(ix,iy,iz)*inv_dy2;
    Azp(ix,iy,iz) = f_bz(ix,iy,iz+1)*inv_dz2;
    Azm(ix,iy,iz) = f_bz(ix,iy,iz)*inv_dz2;
    Adiag(ix,iy,iz) = (Axp(ix,iy,iz)+Axm(ix,iy,iz)+Ayp(ix,iy,iz)+Aym(ix,iy,iz)+Azp(ix,iy,iz)+Azm(ix,iy,iz));
    invAdiag(ix,iy,iz) = 1./Adiag(ix,iy,iz);


}

static __global__ void k_build_vof_poisson_Ap(G_StaggeredGrid grid,MyArray<double,3> p){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;

    MyArray<double,3> Axp = grid.Axp_;
    MyArray<double,3> Axm = grid.Axm_;
    MyArray<double,3> Ayp = grid.Ayp_;
    MyArray<double,3> Aym = grid.Aym_;
    MyArray<double,3> Azp = grid.Azp_;
    MyArray<double,3> Azm = grid.Azm_;
    MyArray<double,3> Adiag = grid.Adiag_;
    MyArray<double,3> Ap = grid.pcg_Ap_;

    if (ix >= Nx+1 || iy >= Ny+1 || iz >= Nz+1) return;

    double p_c = p(ix,iy,iz);
    double p_xp = p(ix+1,iy,iz);
    double p_xm = p(ix-1,iy,iz);
    double p_yp = p(ix,iy+1,iz);
    double p_ym = p(ix,iy-1,iz);
    double p_zp = p(ix,iy,iz+1);
    double p_zm = p(ix,iy,iz-1);

    Ap(ix,iy,iz)= Adiag(ix,iy,iz)*p_c-(Axp(ix,iy,iz)*p_xp+Axm(ix,iy,iz)*p_xm+Ayp(ix,iy,iz)*p_yp+Aym(ix,iy,iz)*p_ym+Azp(ix,iy,iz)*p_zp+Azm(ix,iy,iz)*p_zm);
}


static __global__ void k_update_p_r_z_make_rz_tmp(G_StaggeredGrid grid,const double* scalars,int rz_id){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;

    if(ix>=Nx+1||iy>=Ny+1 || iz>=Nz+1)return;



    double alpha=scalars[rz_id]/scalars[PAP];

    double dir=grid.pcg_dir_(ix,iy,iz);
    double Ap=grid.pcg_Ap_(ix,iy,iz);

    double r_new=grid.pcg_r_(ix,iy,iz)-alpha*Ap;
    grid.p_delta_(ix,iy,iz)+=alpha*dir;
    grid.pcg_r_(ix,iy,iz)=r_new;

    double z_new=grid.invAdiag_(ix,iy,iz)*r_new;
    grid.pcg_z_(ix,iy,iz)=z_new;
    grid.pcg_tmp_(ix-1,iy-1,iz-1)=r_new*z_new;
}


static __global__ void k_update_dir(G_StaggeredGrid grid,double *const pcg_scalars_,int rz_old_id, int rz_new_id){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

    MyArray<double,3> dir = grid.pcg_dir_;
    MyArray<double,3> z = grid.pcg_z_;


    if (iy >=dir.sizey_-1 || ix >= dir.sizex_-1 || iz >= dir.sizez_-1) return;
    double beta =pcg_scalars_[rz_new_id]/pcg_scalars_[rz_old_id] ; 
    dir(ix,iy,iz) = z(ix,iy,iz)+beta*dir(ix,iy,iz);
}

/*
   static __global__ void k_swap_rz(double *s){
   s[RZ_OLD] = s[RZ_NEW];
   }
 */



/* =================================================*/
/* ================ host functions == */
/* =================================================*/
G_PCGSolver::G_PCGSolver(){}

G_PCGSolver::~G_PCGSolver(){}

void G_PCGSolver::solve_pcg(G_SMACSolver& solv){                                                       

    G_StaggeredGrid& grid_=solv.grid_;

    int Nx = solv.grid_.Nx_;
    int Ny = solv.grid_.Ny_;
    int Nz = solv.grid_.Nz_;

    int checkResidualFreq = 10;



    MyArray<double,3> r       = grid_.pcg_r_;
    MyArray<double,3> z       = grid_.pcg_z_;
    MyArray<double,3> dir     = grid_.pcg_dir_;
    MyArray<double,3> invdiag     = grid_.invAdiag_;

    int max_iter = 2000;
    double tol = 1e-6;
    double inv_dt_ = solv.inv_dt_;


    base_k_make_poisson_rhs<<<grid_dim_,block_dim_>>>(grid_,inv_dt_);


    k_build_vof_poisson_A<<<grid_dim_,block_dim_>>>(grid_);
    k_build_vof_poisson_Ap<<<grid_dim_,block_dim_>>>(grid_,grid_.p_delta_);

    /* create mean_b*/
    base_k_copy_to_tmp<<<grid_dim_,block_dim_>>>(grid_.rhs_,grid_.pcg_tmp_,Nx,Ny,Nz);
    cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_.data_,d_dot_,Nx*Ny*Nz);

    /* mean_b is in d_dot_ */
    base_k_divide<<<1,1>>>(d_dot_,(double)(Nx*Ny*Nz));

    /* initial residual; r = b- Ap */
    k_get_r<<<grid_dim_,block_dim_>>>(grid_,d_dot_,grid_.pcg_tmp_);

    cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_.data_,d_dot_,Nx*Ny*Nz);
    /* norm_b2 is in d_dot_ */

    double norm_b;
    cudaMemcpy(&norm_b,d_dot_,sizeof(double),cudaMemcpyDeviceToHost);
    if (norm_b < 1.0e-16) {
        norm_b = 1.0;
    }
    norm_b=sqrt(norm_b);

    Base::subtract_cell_mean(grid_,r);

    /* z = M^{-1} r */

    base_k_mult_elementwise_array<<<grid_dim_,block_dim_>>>(invdiag,r,z,(Nx+2),(Ny+2),(Nz+2));


    Base::subtract_cell_mean(grid_,z);
    cudaMemcpy(dir.data_, z.data_, (Nx+2)*(Ny+2)*(Nz+2) * sizeof(double), cudaMemcpyDeviceToDevice);


    /* rz_old = r dot z */
    int rz_old_id = RZ_BUF0;
    int rz_new_id = RZ_BUF1;
    base_k_mult_elementwise_array_to_tmp<<<grid_dim_,block_dim_>>>(r,z,grid_.pcg_tmp_,Nx,Ny,Nz);

    double *const rz_old=&d_pcg_scalars_[rz_old_id]; 

    cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_.data_,rz_old,Nx*Ny*Nz);


    int iter = 0;

    /* == start main loop ==*/
    for (iter = 0; iter < max_iter; ++iter){


        /*  Ap = A dir */
        k_make_pAp<<<grid_dim_,block_dim_>>>(grid_);

        double* const pAp = &d_pcg_scalars_[PAP];
        cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_.data_,pAp,Nx*Ny*Nz);

        /* == debug == */
        /*
           double h_pAp;
           cudaMemcpy(&h_pAp,&d_pcg_scalars_[PAP],sizeof(double),cudaMemcpyDeviceToHost);
           printf("pAp = %f\n",h_pAp);
         */




        /*
           p = p + alpha dir
           r = r - alpha Ap
         */

        k_update_p_r_z_make_rz_tmp<<<grid_dim_,block_dim_>>>(grid_,d_pcg_scalars_,rz_old_id);

        double* const rz_new = &d_pcg_scalars_[rz_new_id];
        cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_.data_,rz_new,Nx*Ny*Nz);

        if(iter%checkResidualFreq==0){
            /* get residual norm */

            k_get_r2_to_tmp<<<grid_dim_,block_dim_>>>(grid_);
            cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_.data_,d_dot_,Nx*Ny*Nz);

            double r2;
            cudaMemcpy(&r2,d_dot_,sizeof(double),cudaMemcpyDeviceToHost);
            double rel_res = sqrt(r2) / norm_b;

            if (rel_res < tol) {
                printf("norm r = %3.2e, norm b = %3.2e, rel_res = %3.2e \n",sqrt(r2),norm_b,rel_res);
                iter++;
                break;
            }
        }


        /*
           if (fabs(rz_old) < 1.0e-300) {
           break;
           }
         */

        k_update_dir<<<grid_dim_,block_dim_>>>(grid_,d_pcg_scalars_,rz_old_id,rz_new_id);



        int tmp_id=rz_new_id;
        rz_new_id = rz_old_id;
        rz_old_id = tmp_id;
    }

    printf("PCG iter = %d\n", iter);
}

void G_PCGSolver::set_gmg(G_GMGSolver & gmg){
    gmg_ = &gmg;
}

void G_PCGSolver::solve(G_SMACSolver& solv){

    G_StaggeredGrid& grid_=solv.grid_;

    int Nx = solv.grid_.Nx_;
    int Ny = solv.grid_.Ny_;
    int Nz = solv.grid_.Nz_;

    int checkResidualFreq = 5;



    MyArray<double,3> r       = grid_.pcg_r_;
    MyArray<double,3> z       = grid_.pcg_z_;
    MyArray<double,3> dir     = grid_.pcg_dir_;

    int max_iter = 2000;
    double tol = 1e-5;
    double inv_dt_ = solv.inv_dt_;
    double dt = solv.dt_;


    base_k_make_poisson_rhs<<<grid_dim_,block_dim_>>>(grid_,inv_dt_);


    k_build_vof_poisson_A<<<grid_dim_,block_dim_>>>(grid_);
    k_build_vof_poisson_Ap<<<grid_dim_,block_dim_>>>(grid_,grid_.p_delta_);

    /* create mean_b*/
    base_k_copy_to_tmp<<<grid_dim_,block_dim_>>>(grid_.rhs_,grid_.pcg_tmp_,Nx,Ny,Nz);
    cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_.data_,d_dot_,Nx*Ny*Nz);

    /* mean_b is in d_dot_ */
    base_k_divide<<<1,1>>>(d_dot_,(double)(Nx*Ny*Nz));

    /* initial residual; r = b- Ap */
    k_get_r<<<grid_dim_,block_dim_>>>(grid_,d_dot_,grid_.pcg_tmp_);

    cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_.data_,d_dot_,Nx*Ny*Nz);
    /* norm_b2 is in d_dot_ */

    double norm_b;
    cudaMemcpy(&norm_b,d_dot_,sizeof(double),cudaMemcpyDeviceToHost);
    if (norm_b < 1.0e-16) {
        norm_b = 1.0;
    }
    norm_b=sqrt(norm_b);

    Base::subtract_cell_mean(grid_,r);

    /* z = M^{-1} r */
    gmg_->create_coeffs(grid_);
    gmg_->v_cycle_as_preconditioner(grid_,grid_.pcg_z_,grid_.pcg_r_);
    z = grid_.pcg_z_;



    Base::subtract_cell_mean(grid_,z);
    cudaMemcpy(dir.data_, z.data_, (Nx+2)*(Ny+2)*(Nz+2) * sizeof(double), cudaMemcpyDeviceToDevice);


    /* rz_old = r dot z */
    int rz_old_id = RZ_BUF0;
    int rz_new_id = RZ_BUF1;
    base_k_mult_elementwise_array_to_tmp<<<grid_dim_,block_dim_>>>(r,z,grid_.pcg_tmp_,Nx,Ny,Nz);

    double *const rz_old=&d_pcg_scalars_[rz_old_id]; 

    cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_.data_,rz_old,Nx*Ny*Nz);


    int iter = 0;

    /* == start main loop ==*/
    for (iter = 0; iter < max_iter; ++iter){


        /*  Ap = A dir */
        k_make_pAp<<<grid_dim_,block_dim_>>>(grid_);

        double* const pAp = &d_pcg_scalars_[PAP];
        cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_.data_,pAp,Nx*Ny*Nz);

        /* == debug == */
        /*
           double h_pAp;
           cudaMemcpy(&h_pAp,&d_pcg_scalars_[PAP],sizeof(double),cudaMemcpyDeviceToHost);
           printf("pAp = %f\n",h_pAp);
         */




        /*
           p = p + alpha dir
           r = r - alpha Ap
         */

        k_update_p_r<<<grid_dim_,block_dim_>>>(grid_,d_pcg_scalars_,rz_old_id);
        Base::subtract_cell_mean(grid_, grid_.pcg_r_);

        gmg_->v_cycle_as_preconditioner(grid_,grid_.pcg_z_, grid_.pcg_r_);
        z = grid_.pcg_z_;

        /*
           rz_new = dot(r, z)
         */
        base_k_mult_elementwise_array_to_tmp<<<grid_dim_, block_dim_>>>(grid_.pcg_r_,grid_.pcg_z_,grid_.pcg_tmp_, Nx,Ny,Nz);


        double* const rz_new = &d_pcg_scalars_[rz_new_id];
        cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_.data_,rz_new,Nx*Ny*Nz);

        if(iter%checkResidualFreq==0){
            /* get max residual */

            //k_get_r2_to_tmp<<<grid_dim_,block_dim_>>>(grid_);
            base_k_copy_to_tmp<<<grid_dim_,block_dim_>>>(grid_.pcg_r_,grid_.pcg_tmp_,Nx,Ny,Nz);
            cub::DeviceReduce::Max(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_.data_,d_dot_,Nx*Ny*Nz);

            double max_divu;
            cudaMemcpy(&max_divu,d_dot_,sizeof(double),cudaMemcpyDeviceToHost);
            max_divu *= dt;
            //double rel_res = sqrt(r2) / norm_b;


            //printf("norm r = %3.2e, norm b = %3.2e, rel_res = %3.2e \n",sqrt(r2),norm_b,rel_res);
            if (max_divu < tol) {
                printf("max_divu = %3.2e, dt*max_divu = %3.2e \n",max_divu, dt*max_divu);
                iter++;
                break;
            }
        }


        /*
           if (fabs(rz_old) < 1.0e-300) {
           break;
           }
         */

        k_update_dir<<<grid_dim_,block_dim_>>>(grid_,d_pcg_scalars_,rz_old_id,rz_new_id);



        int tmp_id=rz_new_id;
        rz_new_id = rz_old_id;
        rz_old_id = tmp_id;
    }

    printf("PCG iter = %d\n", iter);
}
