#include "G_PressureSolverBase.h"

__global__ void base_k_mult_elementwise_array_to_tmp(MyArray<double,3> b,MyArray<double,3> q,MyArray<double,3> result,int Nx, int Ny, int Nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    if (ix >=Nx || iy >= Ny || iz >= Nz) return;


    result(ix,iy,iz)=b(ix+1,iy+1,iz+1)*q(ix+1,iy+1,iz+1);
}

__global__ void base_k_copy_to_tmp(MyArray<double,3> q,MyArray<double,3> tmp, int Nx, int Ny, int Nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    if (ix >=Nx || iy >= Ny || iz >= Nz) return;


    tmp(ix,iy,iz)= q(ix+1,iy+1,iz+1);
}

__global__ void base_k_mult_elementwise_array(MyArray<double,3> b,MyArray<double,3> q,MyArray<double,3> result,int Nx, int Ny, int Nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    if (ix >=Nx || iy >= Ny || iz >= Nz) return;
    result(ix,iy,iz)=b(ix,iy,iz)*q(ix,iy,iz);
}

__global__ void base_k_divide(double* const num,const double div){
    num[0]/=div;
}

__global__ void base_k_make_poisson_rhs(G_StaggeredGrid grid_,double inv_dt){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;
    int Nz=grid_.Nz_;

    double inv_dx= grid_.inv_dx_;
    double inv_dy= grid_.inv_dy_;
    double inv_dz= grid_.inv_dz_;

    const MyArray<double,3>& f_bx = grid_.f_bx_;
    const MyArray<double,3>& f_by = grid_.f_by_;
    const MyArray<double,3>& f_bz = grid_.f_bz_;

    const MyArray<double,3>& f_sx = grid_.f_sx_;
    const MyArray<double,3>& f_sy = grid_.f_sy_;
    const MyArray<double,3>& f_sz = grid_.f_sz_;


    if (iz>= Nz+1 ||  iy >=Ny+1 || ix >= Nx+1) return;
    const MyArray<double,3>& vx_star=grid_.f_vx_star_;
    const MyArray<double,3>& vy_star=grid_.f_vy_star_;
    const MyArray<double,3>& vz_star=grid_.f_vz_star_;
    MyArray<double,3>& rhs=grid_.rhs_;
    MyArray<unsigned char,3> ct=grid_.celltype_;

    if(ct(ix,iy,iz)!= C_INTERIOR){
        rhs(ix,iy,iz)=0.;
        return;

    }



    // surface tension //
    double f_st = (f_bx(ix+1,iy,iz)*f_sx(ix+1,iy,iz)-f_bx(ix,iy,iz)*f_sx(ix,iy,iz))*inv_dx  
                + (f_by(ix,iy+1,iz)*f_sy(ix,iy+1,iz)-f_by(ix,iy,iz)*f_sy(ix,iy,iz))*inv_dy
                + (f_bz(ix,iy,iz+1)*f_sz(ix,iy,iz+1)-f_bz(ix,iy,iz)*f_sz(ix,iy,iz))*inv_dz;


    double div= (vx_star(ix+1,iy,iz)-vx_star(ix,iy,iz))*inv_dx
        +(vy_star(ix,iy+1,iz)-vy_star(ix,iy,iz))*inv_dy
        +(vz_star(ix,iy,iz+1)-vz_star(ix,iy,iz))*inv_dz;

    // surface tension //

    rhs(ix,iy,iz) = -inv_dt*(div)-f_st;
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

__global__ void base_k_add_scalar_to_array(const double a, const double* const b, MyArray<double,3> q){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    int Nx = q.sizex_;
    int Ny = q.sizey_;
    int Nz = q.sizez_;

    if (ix >=Nx|| iy >=Ny || iz >= Nz) return;
    q(ix,iy,iz)+=a*b[0];
}

void G_PressureSolverBase::subtract_cell_mean(G_StaggeredGrid& grid_,MyArray<double,3> p){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;


    double *const sum = &d_pcg_scalars_[SCA_TMP];
    base_k_copy_to_tmp<<<grid_dim_,block_dim_>>>(p,grid_.pcg_tmp_,Nx,Ny,Nz);
    cub::DeviceReduce::Sum(cub_temp_storage_,cub_temp_storage_bytes_,grid_.pcg_tmp_.data_,sum,(Nx)*(Ny)*(Nz));

    double size_inv = 1./(double)(Nx*Ny*Nz);

    base_k_add_scalar_to_array<<<grid_dim_,block_dim_>>>(-1.*size_inv,sum,p);
}

