#include "G_CFDDEMCoupling.h"
#include <cmath>
__device__ __forceinline__ int d_mirror_cell_index(int id,int n){
    if(id < 1) return 1-id;
    if(id > n) return 2*n+1-id;
    return id;
}

__device__ __forceinline__ int d_fold_index(int id, int lower, int upper){
    if(id < lower) return lower;
    if(id > upper) return upper;
    return id;
}

__global__ void k_gaussian_filter_x(G_StaggeredGrid* grid,MyArray<double,3> src,MyArray<double,3> dst){
    const int ix = blockIdx.x*blockDim.x+threadIdx.x+1;
    const int iy = blockIdx.y*blockDim.y+threadIdx.y+1;
    const int iz = blockIdx.z*blockDim.z+threadIdx.z+1;
    if(ix > grid->Nx_ || iy > grid->Ny_ || iz > grid->Nz_) return;

    const int im2 = d_mirror_cell_index(ix-2,grid->Nx_);
    const int im1 = d_mirror_cell_index(ix-1,grid->Nx_);
    const int ip1 = d_mirror_cell_index(ix+1,grid->Nx_);
    const int ip2 = d_mirror_cell_index(ix+2,grid->Nx_);

    dst(ix,iy,iz)=(src(im2,iy,iz)+4.0*src(im1,iy,iz)+6.0*src(ix,iy,iz)+4.0*src(ip1,iy,iz)+src(ip2,iy,iz))/16.0;
}

__global__ void k_gaussian_filter_y(G_StaggeredGrid* grid,MyArray<double,3> src,MyArray<double,3> dst){
    const int ix = blockIdx.x*blockDim.x+threadIdx.x+1;
    const int iy = blockIdx.y*blockDim.y+threadIdx.y+1;
    const int iz = blockIdx.z*blockDim.z+threadIdx.z+1;
    if(ix > grid->Nx_ || iy > grid->Ny_ || iz > grid->Nz_) return;

    const int jm2 = d_mirror_cell_index(iy-2,grid->Ny_);
    const int jm1 = d_mirror_cell_index(iy-1,grid->Ny_);
    const int jp1 = d_mirror_cell_index(iy+1,grid->Ny_);
    const int jp2 = d_mirror_cell_index(iy+2,grid->Ny_);

    dst(ix,iy,iz)=(src(ix,jm2,iz)+4.0*src(ix,jm1,iz)+6.0*src(ix,iy,iz)+4.0*src(ix,jp1,iz)+src(ix,jp2,iz))/16.0;
}

__global__ void k_gaussian_filter_z(G_StaggeredGrid* grid,MyArray<double,3> src,MyArray<double,3> dst){
    const int ix = blockIdx.x*blockDim.x+threadIdx.x+1;
    const int iy = blockIdx.y*blockDim.y+threadIdx.y+1;
    const int iz = blockIdx.z*blockDim.z+threadIdx.z+1;
    if(ix > grid->Nx_ || iy > grid->Ny_ || iz > grid->Nz_) return;

    const int km2 = d_mirror_cell_index(iz-2,grid->Nz_);
    const int km1 = d_mirror_cell_index(iz-1,grid->Nz_);
    const int kp1 = d_mirror_cell_index(iz+1,grid->Nz_);
    const int kp2 = d_mirror_cell_index(iz+2,grid->Nz_);

    dst(ix,iy,iz)=(src(ix,iy,km2)+4.0*src(ix,iy,km1)+6.0*src(ix,iy,iz)+4.0*src(ix,iy,kp1)+src(ix,iy,kp2))/16.0;
}


__device__ __forceinline__ void d_atomic_add_trilinear_folded(MyArray<double,3> field, const TrilinearStencil& stencil, double value, int imin, int imax, int jmin, int jmax, int kmin, int kmax){
    const double wx[2] = {1.0-stencil.tx,stencil.tx};
    const double wy[2] = {1.0-stencil.ty,stencil.ty};
    const double wz[2] = {1.0-stencil.tz,stencil.tz};

    #pragma unroll
    for(int dk=0;dk<2;dk++){
        #pragma unroll
        for(int dj=0;dj<2;dj++){
            #pragma unroll
            for(int di=0;di<2;di++){
                const int ix = d_fold_index(stencil.i0+di,imin,imax);
                const int iy = d_fold_index(stencil.j0+dj,jmin,jmax);
                const int iz = d_fold_index(stencil.k0+dk,kmin,kmax);
                const double weight = wx[di]*wy[dj]*wz[dk];
                atomicAdd(&field(ix,iy,iz),value*weight);
            }
        }
    }
}

static __device__ __forceinline__ void d_save_stencil_base(ParticleSys<DeviceMemory>* ps, int pid, const TrilinearStencil& vx_stencil, const TrilinearStencil& vy_stencil, const TrilinearStencil& vz_stencil){

    ps->cfd_vx_x0_[pid] = vx_stencil.i0;
    ps->cfd_vx_y0_[pid] = vx_stencil.j0;
    ps->cfd_vx_z0_[pid] = vx_stencil.k0;

    ps->cfd_vy_x0_[pid] = vy_stencil.i0;
    ps->cfd_vy_y0_[pid] = vy_stencil.j0;
    ps->cfd_vy_z0_[pid] = vy_stencil.k0;

    ps->cfd_vz_x0_[pid] = vz_stencil.i0;
    ps->cfd_vz_y0_[pid] = vz_stencil.j0;
    ps->cfd_vz_z0_[pid] = vz_stencil.k0;
}

static __device__ __forceinline__ bool d_is_inside_cfd(G_StaggeredGrid *grid, const double xp,const double yp, const double zp){



    const int Nx = grid->Nx_;
    const int Ny = grid->Ny_;
    const int Nz = grid->Nz_;


    const double originx = grid->origin_x_;
    const double originy = grid->origin_y_;
    const double originz = grid->origin_z_;

    const double inv_dx = grid->inv_dx_;
    const double inv_dy = grid->inv_dy_;
    const double inv_dz = grid->inv_dz_;

    const double sx = (xp - originx) * inv_dx;
    const double sy = (yp - originy) * inv_dy;
    const double sz = (zp - originz) * inv_dz;

    const int ix_raw = static_cast<int>(floor(sx));
    const int iy_raw = static_cast<int>(floor(sy));
    const int iz_raw = static_cast<int>(floor(sz));


    if(ix_raw < 0 || ix_raw >= Nx ||
       iy_raw < 0 || iy_raw >= Ny ||
       iz_raw < 0 || iz_raw >= Nz){

        return false;
    }

    return true;

}

static __device__ __forceinline__ void d_get_cfd_cell_index(G_StaggeredGrid *grid, ParticleSys<DeviceMemory>* ps, int i){


    int bi=i*DIM;

    const int Nx = grid->Nx_;
    const int Ny = grid->Ny_;
    const int Nz = grid->Nz_;

    const double xp = ps->x[bi+0];
    const double yp = ps->x[bi+1];
    const double zp = ps->x[bi+2];

    const double originx = grid->origin_x_;
    const double originy = grid->origin_y_;
    const double originz = grid->origin_z_;

    const double inv_dx = grid->inv_dx_;
    const double inv_dy = grid->inv_dy_;
    const double inv_dz = grid->inv_dz_;

    const double sx = (xp - originx) * inv_dx;
    const double sy = (yp - originy) * inv_dy;
    const double sz = (zp - originz) * inv_dz;

    const int ix_raw = static_cast<int>(floor(sx));
    const int iy_raw = static_cast<int>(floor(sy));
    const int iz_raw = static_cast<int>(floor(sz));


    if(ix_raw < 0 || ix_raw >= Nx ||
       iy_raw < 0 || iy_raw >= Ny ||
       iz_raw < 0 || iz_raw >= Nz){

        ps->cfd_is_in_CFD_[i]= false;
        return;
    }

    // Real CFD cells start from index 1.
    ps->cfd_cellid_x_[i] = ix_raw + 1;
    ps->cfd_cellid_y_[i] = iy_raw + 1;
    ps->cfd_cellid_z_[i] = iz_raw + 1;

    ps->cfd_is_in_CFD_[i] = true;

}

static __global__ void k_get_cfd_cell_index(G_StaggeredGrid *grid, ParticleSys<DeviceMemory>* ps){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps->N || ps->isActive[i]!=1) return;

    d_get_cfd_cell_index(grid,ps,i);

}

__global__ void k_update_demdt(ParticleSys<DeviceMemory>* ps, double dt){
    ps->dt = dt;
}

__global__ void k_deposit_particle_volume(G_StaggeredGrid* grid, ParticleSys<DeviceMemory>* p){

    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i >= p->N || p->isActive[i]!=1) return;

    int bi=i*DIM;

    const double xp = p->x[bi+0];
    const double yp = p->x[bi+1];
    const double zp = p->x[bi+2];

    if(!d_is_inside_cfd(grid, xp, yp, zp)) return;

    const TrilinearStencil stencil =
        d_get_stencil<1,1,1>(grid,p,i);

    d_atomic_add_trilinear_folded(
        grid->particle_volume_,
        stencil,
        p->vol[i],
        1,
        grid->Nx_,
        1,
        grid->Ny_,
        1,
        grid->Nz_
    );
}

__global__ void k_swap_voidfraction(G_StaggeredGrid* grid){
    double* tmp = grid->void_fraction_.data_;
    grid->void_fraction_.data_ = grid->void_fraction_old_.data_;
    grid->void_fraction_old_.data_ = tmp;

}
__global__ void k_calculate_void_fraction(G_StaggeredGrid* grid){
    const int ix = blockIdx.x*blockDim.x+threadIdx.x+1;
    const int iy = blockIdx.y*blockDim.y+threadIdx.y+1;
    const int iz = blockIdx.z*blockDim.z+threadIdx.z+1;
    if(ix > grid->Nx_ || iy > grid->Ny_ || iz > grid->Nz_) return;

    const double inv_volume = grid->inv_dx_*grid->inv_dy_*grid->inv_dz_;
    const double solid_fraction = grid->particle_volume_tmp_(ix,iy,iz)*inv_volume;
    grid->void_fraction_(ix,iy,iz)=1.0-solid_fraction;

    /*debug*/
    if(solid_fraction>0.7){
        printf("solid_fraction maybe too large\n");

    }
}

__global__ void k_calculate_void_half(G_StaggeredGrid* grid){
    const int ix = blockIdx.x*blockDim.x+threadIdx.x+1;
    const int iy = blockIdx.y*blockDim.y+threadIdx.y+1;
    const int iz = blockIdx.z*blockDim.z+threadIdx.z+1;
    if(ix > grid->Nx_ || iy > grid->Ny_ || iz > grid->Nz_) return;

    grid->void_fraction_half_(ix,iy,iz)= 0.5*(grid->void_fraction_(ix,iy,iz)+grid->void_fraction_old_(ix,iy,iz));
}

__global__ void k_interpolate_fluid_to_particle(G_StaggeredGrid* grid, ParticleSys<DeviceMemory>* ps){
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;

    if (pid >= ps->N || ps->isActive[pid]!=1) return;

    int bi=pid*DIM;

    const double xp = ps->x[bi+0];
    const double yp = ps->x[bi+1];
    const double zp = ps->x[bi+2];

    if(!d_is_inside_cfd(grid, xp, yp, zp)){
        ps->cfd_gradp_x_[pid] = 0.0;
        ps->cfd_gradp_y_[pid] = 0.0;
        ps->cfd_gradp_z_[pid] = 0.0;
        ps->cfd_vy_[pid] = 0.0;
        ps->cfd_vz_[pid] = 0.0;
        ps->cfd_vx_[pid] = 0.0;
        ps->cfd_vy_[pid] = 0.0;
        ps->cfd_vz_[pid] = 0.0;
        ps->cfd_rho_[pid] = 0.0;
        ps->cfd_mu_[pid] = 0.0;
        return;
    }

    // Cell-centered field: offset = (0.5, 0.5, 0.5)
    const TrilinearStencil cell_stencil = d_get_stencil<1,1,1>(grid, ps, pid);

    // Vx field: offset = (0.0, 0.5, 0.5)
    const TrilinearStencil vx_stencil = d_get_stencil<0,1,1>(grid, ps, pid);

    // Vy field: offset = (0.5, 0.0, 0.5)
    const TrilinearStencil vy_stencil = d_get_stencil<1,0,1>(grid, ps, pid);

    // Vz field: offset = (0.5, 0.5, 0.0)
    const TrilinearStencil vz_stencil = d_get_stencil<1,1,0>(grid, ps, pid);

    const MyArray<double,3> vx = grid->f_vx_;
    const MyArray<double,3> vy = grid->f_vy_;
    const MyArray<double,3> vz = grid->f_vz_;

    const MyArray<double,3> gradp_x = grid->f_gradp_x_;
    const MyArray<double,3> gradp_y = grid->f_gradp_y_;
    const MyArray<double,3> gradp_z = grid->f_gradp_z_;

    const MyArray<double,3> rho = grid->rho_;
    const MyArray<double,3> mu = grid->mu_;

    const double fluid_vx = d_interpolate_trilinear(vx, vx_stencil);
    const double fluid_vy = d_interpolate_trilinear(vy, vy_stencil);
    const double fluid_vz = d_interpolate_trilinear(vz, vz_stencil);

    const double fluid_gradp_x = d_interpolate_trilinear(gradp_x, vx_stencil);
    const double fluid_gradp_y = d_interpolate_trilinear(gradp_y, vy_stencil);
    const double fluid_gradp_z = d_interpolate_trilinear(gradp_z, vz_stencil);

    const double fluid_rho = d_interpolate_trilinear(rho, cell_stencil);
    const double fluid_mu = d_interpolate_trilinear(mu, cell_stencil);

    d_save_stencil_base(ps, pid, vx_stencil, vy_stencil, vz_stencil);

    ps->cfd_vx_[pid] = fluid_vx;
    ps->cfd_vy_[pid] = fluid_vy;
    ps->cfd_vz_[pid] = fluid_vz;

    ps->cfd_gradp_x_[pid] = fluid_gradp_x;
    ps->cfd_gradp_y_[pid] = fluid_gradp_y;
    ps->cfd_gradp_z_[pid] = fluid_gradp_z;

    ps->cfd_rho_[pid] = fluid_rho;
    ps->cfd_mu_[pid] = fluid_mu;

}

__device__ __forceinline__ void d_extend_void_fraction_solid(MyArray<double,3> eps,MyArray<unsigned char,3> celltype,int ix,int iy,int iz){
    if(celltype(ix,iy,iz)!=C_SOLID) return;

    if(celltype(ix-1,iy,iz)==C_INTERIOR){
        eps(ix,iy,iz)=eps(ix-1,iy,iz);
    }else if(celltype(ix+1,iy,iz)==C_INTERIOR){
        eps(ix,iy,iz)=eps(ix+1,iy,iz);
    }else if(celltype(ix,iy-1,iz)==C_INTERIOR){
        eps(ix,iy,iz)=eps(ix,iy-1,iz);
    }else if(celltype(ix,iy+1,iz)==C_INTERIOR){
        eps(ix,iy,iz)=eps(ix,iy+1,iz);
    }else if(celltype(ix,iy,iz-1)==C_INTERIOR){
        eps(ix,iy,iz)=eps(ix,iy,iz-1);
    }else if(celltype(ix,iy,iz+1)==C_INTERIOR){
        eps(ix,iy,iz)=eps(ix,iy,iz+1);
    }else{
        eps(ix,iy,iz)=1.0;
    }
}

static __global__ void k_update_cell_solid_void_fraction(G_StaggeredGrid* grid){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;

    d_extend_void_fraction_solid(grid->void_fraction_,grid->celltype_,ix,iy,iz);
    d_extend_void_fraction_solid(grid->void_fraction_old_,grid->celltype_,ix,iy,iz);
    d_extend_void_fraction_solid(grid->void_fraction_half_,grid->celltype_,ix,iy,iz);
}

static __global__ void k_update_cell_ghost_void_fraction(G_StaggeredGrid* grid){
    int ix=blockIdx.x*blockDim.x+threadIdx.x;
    int iy=blockIdx.y*blockDim.y+threadIdx.y;
    int iz=blockIdx.z*blockDim.z+threadIdx.z;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;

    if(ix>=Nx+2 || iy>=Ny+2 || iz>=Nz+2) return;

    bool is_ghost=ix==0 || ix==Nx+1 || iy==0 || iy==Ny+1 || iz==0 || iz==Nz+1;
    if(!is_ghost) return;

    int src_ix=ix==0?1:ix==Nx+1?Nx:ix;
    int src_iy=iy==0?1:iy==Ny+1?Ny:iy;
    int src_iz=iz==0?1:iz==Nz+1?Nz:iz;

    grid->void_fraction_(ix,iy,iz)=grid->void_fraction_(src_ix,src_iy,src_iz);
    grid->void_fraction_old_(ix,iy,iz)=grid->void_fraction_old_(src_ix,src_iy,src_iz);
    grid->void_fraction_half_(ix,iy,iz)=grid->void_fraction_half_(src_ix,src_iy,src_iz);
}

void G_CFDDEMCoupling::update_boundary_ghost_void_fraction(G_StaggeredGrid& grid){
    k_update_cell_solid_void_fraction<<<grid_dim_,block_dim_>>>(grid.d_ptr_);
    k_update_cell_ghost_void_fraction<<<grid_dim_,block_dim_>>>(grid.d_ptr_);
}

void G_CFDDEMCoupling::set_particle_volume_to_cell(G_StaggeredGrid& grid, ParticleSys<DeviceMemory>& ps, int gridSize, int blockSize){

    cudaMemset(grid.particle_volume_.data_,0,sizeof(double)*grid.particle_volume_.size_);

    k_deposit_particle_volume<<<gridSize, blockSize>>>(grid.d_ptr_, ps.d_self);

}

void G_CFDDEMCoupling::get_index_of_Cell(G_StaggeredGrid& grid, ParticleSys<DeviceMemory>& ps, int gridSize, int blockSize){

    k_get_cfd_cell_index<<<gridSize, blockSize>>>(grid.d_ptr_, ps.d_self);

}

void G_CFDDEMCoupling::interpolate_fluid_to_particle(G_StaggeredGrid& grid, ParticleSys<DeviceMemory>& ps, int gridSize, int blockSize){

    k_interpolate_fluid_to_particle<<<gridSize, blockSize>>>(grid.d_ptr_, ps.d_self);

}

void G_CFDDEMCoupling::gaussian_filter_particle_volume(G_StaggeredGrid& grid){

    k_gaussian_filter_x<<<grid_dim_, block_dim_>>>(grid.d_ptr_,grid.particle_volume_,grid.particle_volume_tmp_);
    k_gaussian_filter_y<<<grid_dim_, block_dim_>>>(grid.d_ptr_,grid.particle_volume_tmp_,grid.particle_volume_);
    k_gaussian_filter_z<<<grid_dim_, block_dim_>>>(grid.d_ptr_,grid.particle_volume_,grid.particle_volume_tmp_);
}


void G_CFDDEMCoupling::calc_void_fraction(G_StaggeredGrid& grid){
    k_calculate_void_fraction<<<grid_dim_, block_dim_>>>(grid.d_ptr_);
    k_calculate_void_half<<<grid_dim_, block_dim_>>>(grid.d_ptr_);
}

static __global__ void k_update_poisson_beta_two_way(G_StaggeredGrid* grid){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;

    MyArray<double,3> eps=grid->void_fraction_;

    MyArray<double,3> f_inv_rhox=grid->f_inv_rhox_;
    MyArray<double,3> f_inv_rhoy=grid->f_inv_rhoy_;
    MyArray<double,3> f_inv_rhoz=grid->f_inv_rhoz_;

    MyArray<double,3> f_bx=grid->f_bx_;
    MyArray<double,3> f_by=grid->f_by_;
    MyArray<double,3> f_bz=grid->f_bz_;

    /* x-face: ix=1..Nx+1, iy=1..Ny, iz=1..Nz */
    if(ix<=Nx+1 && iy<=Ny && iz<=Nz){
        double epsf=0.5*(eps(ix-1,iy,iz)+eps(ix,iy,iz));
        f_bx(ix,iy,iz)=epsf*f_inv_rhox(ix,iy,iz);
    }

    /* y-face: ix=1..Nx, iy=1..Ny+1, iz=1..Nz */
    if(ix<=Nx && iy<=Ny+1 && iz<=Nz){
        double epsf=0.5*(eps(ix,iy-1,iz)+eps(ix,iy,iz));
        f_by(ix,iy,iz)=epsf*f_inv_rhoy(ix,iy,iz);
    }

    /* z-face: ix=1..Nx, iy=1..Ny, iz=1..Nz+1 */
    if(ix<=Nx && iy<=Ny && iz<=Nz+1){
        double epsf=0.5*(eps(ix,iy,iz-1)+eps(ix,iy,iz));
        f_bz(ix,iy,iz)=epsf*f_inv_rhoz(ix,iy,iz);
    }
}

void G_CFDDEMCoupling::update_poisson_beta_two_way(G_StaggeredGrid& grid){
    k_update_poisson_beta_two_way<<<grid_dim_, block_dim_>>>(grid.d_ptr_);
}

static __global__ void k_init_void_fraction_one(G_StaggeredGrid* grid){
    int ix=blockIdx.x*blockDim.x+threadIdx.x;
    int iy=blockIdx.y*blockDim.y+threadIdx.y;
    int iz=blockIdx.z*blockDim.z+threadIdx.z;

    if(ix>=grid->void_fraction_.sizex_ ||
       iy>=grid->void_fraction_.sizey_ ||
       iz>=grid->void_fraction_.sizez_) return;

    grid->void_fraction_(ix,iy,iz)=1.0;
    grid->void_fraction_old_(ix,iy,iz)=1.0;
    grid->void_fraction_half_(ix,iy,iz)=1.0;
    grid->void_fraction_vof_(ix,iy,iz)=1.0;
    grid->void_fraction_vof_new_(ix,iy,iz)=1.0;
}

void G_CFDDEMCoupling::initialize_void_fractions(G_StaggeredGrid& grid){
    k_init_void_fraction_one<<<grid_dim_, block_dim_>>>(grid.d_ptr_);
}
