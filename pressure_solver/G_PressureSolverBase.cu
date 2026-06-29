#include "G_PressureSolverBase.h"
/* == kernel function ==*/
__global__ void pres_k_divide(double* const num,const double div){
    num[0]/=div;
}

__global__ void pres_k_add_scalar_to_array(const double a, const double* const b,double* const q,int size){
    int t=blockIdx.x*blockDim.x+threadIdx.x;

    if (t>= size) return;
    q[t]+=a*b[0];
}

__global__ void pres_k_mult_elementwise_array(double* const b,double* const q,double* const result,int size){
    int t=blockIdx.x*blockDim.x+threadIdx.x;

    if (t>= size) return;
    result[t]=b[t]*q[t];
}

__global__ void pres_k_mult_elementwise_array_to_tmp(double* const b,double* const q,double* const result,int Nx, int Ny){
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int pitch = Nx + 2;
    int pitch_tmp = Nx;

    if (i >=Ny || j >= Nx) return;

    int original_id = (i+1)*pitch+j+1;
    int ind = (i)*pitch_tmp+j;

    result[ind]=b[original_id]*q[original_id];
}
__global__ void pres_k_fix_pressure_reference(G_StaggeredGrid grid_,int Nx, int Ny){
    int pitch = Nx + 2;

    int indref = 1*pitch+1;

    grid_.p_[indref] = 0.;
}

__global__ void pres_k_make_poisson_rhs(G_StaggeredGrid grid_,double inv_dt){
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;

    double inv_dx= grid_.inv_dx_;
    double inv_dy= grid_.inv_dy_;

    const MyArray<double,3>& f_bx = grid_.wew_f_bx_;
    const MyArray<double,3>& f_by = grid_.wew_f_by_;

    const MyArray<double,3>& f_sx = grid_.wew_f_sx_;
    const MyArray<double,3>& f_sy = grid_.wew_f_sy_;

    if (iy >=Ny+1 || ix >= Nx+1) return;
    const MyArray<double,3>& vx_star=grid_.wew_vx_star_;
    const MyArray<double,3>& vy_star=grid_.wew_vy_star_;
    MyArray<double,3>& rhs=grid_.wew_rhs_;



    // surface tension //
    double f_st = (f_bx(ix+1,iy)*f_sx(ix+1,iy)-f_bx(ix,iy)*f_sx(ix,iy))*inv_dx  
                + (f_by(ix,iy+1)*f_sy(ix,iy+1)-f_by(ix,iy)*f_sy(ix,iy))*inv_dy;

    rhs(ix,iy) = -1.*(inv_dt*((vx_star(ix+1,iy)-vx_star(ix,iy))*inv_dx+(vy_star(ix,iy+1)-vy_star(ix,iy))*inv_dy) + f_st);
}


__global__ void pres_k_shift_pressure_reference(G_StaggeredGrid grid_,int Nx, int Ny){
    int i = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int j = blockIdx.x*blockDim.x + threadIdx.x+1;
    int pitch = Nx + 2;

    if (i >=Ny+1 || j >= Nx+1) return;
    if (i ==1 && j == 1) return; // return if reference pressure point

    double pref = grid_.p_[1 * pitch + 1];

    grid_.p_[i * pitch + j] -= pref;
}

/* removes ghost cell */
__global__ void pres_k_copy_to_tmp(const double* const q,double* const tmp, int Nx, int Ny){
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int pitch = Nx + 2;
    int pitch_tmp = Nx;
    if (i >=Ny || j >= Nx) return;

    int original_id = (i+1)*pitch+j+1;
    int ind = (i)*pitch_tmp+j;

    tmp[ind] = q[original_id];
}

/* only for cells. not for faces*/
__global__ void pres_k_set_boundary_array(double* const q, int Nx, int Ny){
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int pitch=Nx+2;

    if(t<Nx){
        int j=t+1;

        q[j]=q[pitch+j];
        q[(Ny+1)*pitch+j]=q[Ny*pitch+j];
    }

    if(t<Ny){
        int i=t+1;

        q[i*pitch]=q[i*pitch+1];
        q[i*pitch+Nx+1]=q[i*pitch+Nx];
    }

    if(t==0){
        q[0]=q[pitch+1];
        q[Nx+1]=q[pitch+Nx];
        q[(Ny+1)*pitch]=q[Ny*pitch+1];
        q[(Ny+1)*pitch+Nx+1]=q[Ny*pitch+Nx];
    }
}

void G_PressureSolverBase::copyData(G_SMACSolver& solv){
    block_dim_ = solv.block_dim_;
    grid_dim_ = solv.grid_dim_;
    d_pcg_scalars_ = solv.d_pcg_scalars_;
    cub_temp_storage_=solv.cub_temp_storage_;
   cub_temp_storage_bytes_=solv.cub_temp_storage_bytes_;
    d_r2_=solv.d_r2_;
   d_dot_=solv.d_dot_;
}

void G_PressureSolverBase::subtract_cell_mean(G_StaggeredGrid& grid_,double *p, int Nx, int Ny){


    double *const sum = &d_pcg_scalars_[SCA_TMP];
    pres_k_copy_to_tmp<<<grid_dim_,block_dim_>>>(p,grid_.pcg_tmp_,Nx,Ny);
    cub::DeviceReduce::Sum(cub_temp_storage_,cub_temp_storage_bytes_,grid_.pcg_tmp_,sum,(Nx)*(Ny));

    double size_inv = 1./(double)(Nx*Ny);

    int block_size=256;
    int n=(Nx+2)*(Ny+2);
    int grid_size=(n+block_size-1)/block_size;

    pres_k_add_scalar_to_array<<<grid_size,block_size>>>(-1.*size_inv,sum,p,n);
    pres_k_set_boundary_array<<<grid_size,block_size>>>(p,Nx,Ny);
}

