#include "G_PressureSolverBase.h"

__global__ void base_k_mult_elementwise_array_to_tmp(MyArray<double,3> b,MyArray<double,3> q,MyArray<double,3> result,int Nx, int Ny, int Nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    if (ix >=Nx || iy >= Ny || iz >= Nz) return;


    result(ix,iy,iz)=b(ix+1,iy+1,iz+1)*q(ix+1,iy+1,iz+1);
}

__global__ void base_k_copy_abs_to_tmp(MyArray<double,3> q,MyArray<double,3> tmp, int Nx, int Ny, int Nz){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    if (ix >=Nx || iy >= Ny || iz >= Nz) return;


    tmp(ix,iy,iz)= fabs(q(ix+1,iy+1,iz+1));
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

__global__ void base_k_make_poisson_rhs(G_StaggeredGrid* grid,double inv_dt){
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;

    if(ix>Nx || iy>Ny || iz>Nz) return;

    MyArray<double,3>& rhs=grid->rhs_;
    MyArray<unsigned char,3>& ct=grid->celltype_;

    if(ct(ix,iy,iz)!=C_INTERIOR){
        rhs(ix,iy,iz)=0.0;
        return;
    }

    const MyArray<double,3>& vx_star=grid->f_vx_star_;
    const MyArray<double,3>& vy_star=grid->f_vy_star_;
    const MyArray<double,3>& vz_star=grid->f_vz_star_;

    const MyArray<double,3>& eps_new=grid->void_fraction_;
    const MyArray<double,3>& eps_old=grid->void_fraction_old_;

    const MyArray<double,3>& f_bx=grid->f_bx_;
    const MyArray<double,3>& f_by=grid->f_by_;
    const MyArray<double,3>& f_bz=grid->f_bz_;

    const MyArray<double,3>& f_sx=grid->f_sx_;
    const MyArray<double,3>& f_sy=grid->f_sy_;
    const MyArray<double,3>& f_sz=grid->f_sz_;

    const double inv_dx=grid->inv_dx_;
    const double inv_dy=grid->inv_dy_;
    const double inv_dz=grid->inv_dz_;

    /* epsilon at faces */
    const double eps_xp=0.5*(eps_new(ix,iy,iz)+eps_new(ix+1,iy,iz));
    const double eps_xm=0.5*(eps_new(ix-1,iy,iz)+eps_new(ix,iy,iz));

    const double eps_yp=0.5*(eps_new(ix,iy,iz)+eps_new(ix,iy+1,iz));
    const double eps_ym=0.5*(eps_new(ix,iy-1,iz)+eps_new(ix,iy,iz));

    const double eps_zp=0.5*(eps_new(ix,iy,iz)+eps_new(ix,iy,iz+1));
    const double eps_zm=0.5*(eps_new(ix,iy,iz-1)+eps_new(ix,iy,iz));

    /* div(epsilon*u_star) */
    const double div_eps=
        (eps_xp*vx_star(ix+1,iy,iz)-eps_xm*vx_star(ix,iy,iz))*inv_dx+
        (eps_yp*vy_star(ix,iy+1,iz)-eps_ym*vy_star(ix,iy,iz))*inv_dy+
        (eps_zp*vz_star(ix,iy,iz+1)-eps_zm*vz_star(ix,iy,iz))*inv_dz;


    /* d epsilon / dt */
    const double deps=eps_new(ix,iy,iz)-eps_old(ix,iy,iz);


    /* div(epsilon/rho * surface tension) */
    const double f_st=
        (f_bx(ix+1,iy,iz)*f_sx(ix+1,iy,iz)-f_bx(ix,iy,iz)*f_sx(ix,iy,iz))*inv_dx+
        (f_by(ix,iy+1,iz)*f_sy(ix,iy+1,iz)-f_by(ix,iy,iz)*f_sy(ix,iy,iz))*inv_dy+
        (f_bz(ix,iy,iz+1)*f_sz(ix,iy,iz+1)-f_bz(ix,iy,iz)*f_sz(ix,iy,iz))*inv_dz;

    rhs(ix,iy,iz)=
        -inv_dt*div_eps
        -deps*inv_dt*inv_dt
        -f_st;

}void G_PressureSolverBase::copyData(G_SMACSolver& solv){
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

void G_PressureSolverBase::subtract_cell_mean(G_StaggeredGrid& grid,MyArray<double,3> p){
    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;


    double *const sum = &d_pcg_scalars_[SCA_TMP];
    base_k_copy_to_tmp<<<grid_dim_,block_dim_>>>(p,grid.pcg_tmp_,Nx,Ny,Nz);
    cub::DeviceReduce::Sum(cub_temp_storage_,cub_temp_storage_bytes_,grid.pcg_tmp_.data_,sum,(Nx)*(Ny)*(Nz));

    double size_inv = 1./(double)(Nx*Ny*Nz);

    base_k_add_scalar_to_array<<<grid_dim_,block_dim_>>>(-1.*size_inv,sum,p);
}

