#include "G_SMACSolver.h"
#include "G_BoundaryFunctions.h"
#include "G_StaggeredGrid.h"
#include "PCG_Scalars.h"
#include "hardCodedParameters.h"
#include "MyArray.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <algorithm>
#include <cub/cub.cuh>



/* ================================= */
/* ==== surface tension related ==== */
/* ================================= */
/*
static __global__ void k_calc_alpha_s(G_StaggeredGrid grid){
    MyArray<double,3> a = grid.alpha_;
    MyArray<double,3> a_s = grid.alpha_s_;
    MyArray<unsigned char,3> ct = grid.celltype_;

    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;

    if (ix > Nx || iy > Ny || iz > Nz) {
        return;
    }

    if (ct(ix,iy,iz) != C_INTERIOR){
        return;
    }

    double sum = 0.0;
    double wsum = 0.0;

    for (int kz=-1; kz<=1; kz++){
        for (int ky=-1; ky<=1; ky++){
            for (int kx=-1; kx<=1; kx++){
                int i = ix + kx;
                int j = iy + ky;
                int k = iz + kz;

                if(ct(i,j,k) != C_INTERIOR){
                    continue;
                }

                int n = abs(kx) + abs(ky) + abs(kz);

                double w;

                if(n==0){ //center
                    w=8.0;
                }else if(n==1){ //face
                    w=4.0;
                }else if(n==2){ //edge
                    w=2.0;
                }else{ //corner
                    w=1.0;
                }

                sum += w*a(i,j,k);
                wsum += w;
            }
        }
    }

    a_s(ix,iy,iz) = sum/wsum;

    if(a_s(ix,iy,iz) > 1.0 - 1e-6){
        a_s(ix,iy,iz) = 1.;
    }

    if(a_s(ix,iy,iz) < 1e-6){
        a_s(ix,iy,iz) = 0.;
    }
}
*/

static __global__  void k_calc_interface_normal(G_StaggeredGrid* grid){

    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;


    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if (ix > Nx|| iy > Ny || iz > Nz) return;


    MyArray<double,3> a_s = grid->alpha_;
    MyArray<double,3> nx = grid->nx_;
    MyArray<double,3> ny = grid->ny_;
    MyArray<double,3> nz = grid->nz_;

    MyArray<double,3> grad_alpha_mag = grid->p_tmp_;

    MyArray<unsigned char,3> ct = grid->celltype_;

    if (ct(ix,iy,iz) != C_INTERIOR) return;


    double inv_2dx = grid->inv_2dx_;
    double inv_2dy = grid->inv_2dy_;
    double inv_2dz = grid->inv_2dz_;

    double inv_dx = grid->inv_dx_;
    double inv_dy = grid->inv_dy_;
    double inv_dz = grid->inv_dz_;

    bool is_xp_interior = ct(ix+1,iy,iz) == C_INTERIOR;
    bool is_xm_interior = ct(ix-1,iy,iz) == C_INTERIOR;

    bool is_yp_interior = ct(ix,iy+1,iz) == C_INTERIOR;
    bool is_ym_interior = ct(ix,iy-1,iz) == C_INTERIOR;

    bool is_zp_interior = ct(ix,iy,iz+1) == C_INTERIOR;
    bool is_zm_interior = ct(ix,iy,iz-1) == C_INTERIOR;

    double grad_ax = 0.0;

    if(is_xp_interior && is_xm_interior){
        grad_ax = (a_s(ix+1,iy,iz) - a_s(ix-1,iy,iz))*inv_2dx;
    }else if(is_xp_interior && !is_xm_interior){
        grad_ax = (a_s(ix+1,iy,iz) - a_s(ix,iy,iz))*inv_dx;
    }else if(!is_xp_interior && is_xm_interior){
        grad_ax = (a_s(ix,iy,iz) - a_s(ix-1,iy,iz))*inv_dx;
    }else{
        grad_ax = 0.0;
    }

    double grad_ay = 0.;

    if(is_yp_interior && is_ym_interior){
        grad_ay = (a_s(ix,iy+1,iz) - a_s(ix,iy-1,iz))*inv_2dy;
    }else if(is_yp_interior && !is_ym_interior){
        grad_ay = (a_s(ix,iy+1,iz) - a_s(ix,iy,iz))*inv_dy;
    }else if(!is_yp_interior && is_ym_interior){
        grad_ay = (a_s(ix,iy,iz) - a_s(ix,iy-1,iz))*inv_dy;
    }else{
        grad_ay = 0.0;
    }

    double grad_az = 0.;

    if(is_zp_interior && is_zm_interior){
        grad_az = (a_s(ix,iy,iz+1) - a_s(ix,iy,iz-1))*inv_2dz;
    }else if(is_zp_interior && !is_zm_interior){
        grad_az = (a_s(ix,iy,iz+1) - a_s(ix,iy,iz))*inv_dz;
    }else if(!is_zp_interior && is_zm_interior){
        grad_az = (a_s(ix,iy,iz) - a_s(ix,iy,iz-1))*inv_dz;
    }else{
        grad_az = 0.0;
    }

    double mag2 = grad_ax*grad_ax+grad_ay*grad_ay+grad_az*grad_az;
    double mag = sqrt(mag2);

    grad_alpha_mag(ix,iy,iz) = mag;

    if (mag2>1e-12){
        double inv_norm_grad = 1./(mag);
        nx(ix,iy,iz) = grad_ax*inv_norm_grad;
        ny(ix,iy,iz) = grad_ay*inv_norm_grad;
        nz(ix,iy,iz) = grad_az*inv_norm_grad;
    }else{
        nx(ix,iy,iz) = 0.;
        ny(ix,iy,iz) = 0.;
        nz(ix,iy,iz) = 0.;
    }

}

static __global__ void k_calc_curvature(G_StaggeredGrid* grid){

    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if (ix > Nx|| iy > Ny || iz > Nz) return;


    MyArray<double,3> nx = grid->nx_;
    MyArray<double,3> ny = grid->ny_;
    MyArray<double,3> nz = grid->nz_;
    MyArray<double,3> kappa = grid->kappa_;

    MyArray<double,3> a = grid->alpha_;

    MyArray<unsigned char,3> ct = grid->celltype_;

    if(a(ix,iy,iz)>0.99 || a(ix,iy,iz)<0.01){
        kappa(ix,iy,iz) = 0.;
        return;
    }

    MyArray<double,3> grad_alpha_mag = grid->p_tmp_;


    double inv_2dx = grid->inv_2dx_;
    double inv_2dy = grid->inv_2dy_;
    double inv_2dz = grid->inv_2dz_;

    double inv_dx = grid->inv_dx_;
    double inv_dy = grid->inv_dy_;
    double inv_dz = grid->inv_dz_;

    double mag_thresh = 1e-6;
    bool is_xp_interior = ct(ix+1,iy,iz) == C_INTERIOR && grad_alpha_mag(ix+1,iy,iz) > mag_thresh;
    bool is_xm_interior = ct(ix-1,iy,iz) == C_INTERIOR && grad_alpha_mag(ix-1,iy,iz) > mag_thresh;

    bool is_yp_interior = ct(ix,iy+1,iz) == C_INTERIOR && grad_alpha_mag(ix,iy+1,iz) > mag_thresh;
    bool is_ym_interior = ct(ix,iy-1,iz) == C_INTERIOR && grad_alpha_mag(ix,iy-1,iz) > mag_thresh;

    bool is_zp_interior = ct(ix,iy,iz+1) == C_INTERIOR && grad_alpha_mag(ix,iy,iz+1) > mag_thresh;

    bool is_zm_interior = ct(ix,iy,iz-1) == C_INTERIOR && grad_alpha_mag(ix,iy,iz-1) > mag_thresh;





    /* == kappa = -div(n) == */

    double dnxdx = 0.0; 

    if(is_xp_interior && is_xm_interior){
        dnxdx = (nx(ix+1,iy,iz) - nx(ix-1,iy,iz))*inv_2dx;
    }else if(is_xp_interior && !is_xm_interior){
        dnxdx = (nx(ix+1,iy,iz) - nx(ix,iy,iz))*inv_dx;
    }else if(!is_xp_interior && is_xm_interior){
        dnxdx = (nx(ix,iy,iz) - nx(ix-1,iy,iz))*inv_dx;
    }else{
        dnxdx = 0.0;
    }

    double dnydy = 0.;

    if(is_yp_interior && is_ym_interior){
        dnydy = (ny(ix,iy+1,iz) - ny(ix,iy-1,iz))*inv_2dy;
    }else if(is_yp_interior && !is_ym_interior){
        dnydy = (ny(ix,iy+1,iz) - ny(ix,iy,iz))*inv_dy;
    }else if(!is_yp_interior && is_ym_interior){
        dnydy = (ny(ix,iy,iz) - ny(ix,iy-1,iz))*inv_dy;
    }else{
        dnydy = 0.0;
    }

    double dnzdz = 0.;

    if(is_zp_interior && is_zm_interior){
        dnzdz = (nz(ix,iy,iz+1) - nz(ix,iy,iz-1))*inv_2dz;
    }else if(is_zp_interior && !is_zm_interior){
        dnzdz = (nz(ix,iy,iz+1) - nz(ix,iy,iz))*inv_dz;
    }else if(!is_zp_interior && is_zm_interior){
        dnzdz = (nz(ix,iy,iz) - nz(ix,iy,iz-1))*inv_dz;
    }else{
        dnzdz = 0.0;
    }



    kappa(ix,iy,iz) = -1.0 * (dnxdx + dnydy + dnzdz);

    /* debug */
    /*
       if(kappa(ix,iy,iz)>1e3){
       printf("%f\n",kappa(ix,iy,iz));
       }
     */



    //kappa(ix,iy,iz) = 2./0.3;

}

static __global__ void k_calc_surface_tension_face(G_StaggeredGrid* grid){

    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z * blockDim.z + threadIdx.z + 1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;


    double sigma = grid->sigma_(0);
    MyArray<double,3> f_sx = grid->f_sx_;
    MyArray<double,3> f_sy = grid->f_sy_;
    MyArray<double,3> f_sz = grid->f_sz_;
    MyArray<double,3> kappa = grid->kappa_;
    MyArray<double,3> a_s = grid->alpha_;
    MyArray<unsigned char,3> celltype = grid->celltype_;

    double inv_dx = grid->inv_dx_;
    double inv_dy = grid->inv_dy_;
    double inv_dz = grid->inv_dz_;

    if (ix > Nx || iy > Ny || iz > Nz|| ix < 2 || iy < 1|| iz <1){
        /* do nothing */

    }else{
        /* == x-faces == */

        if(celltype(ix,iy,iz)== C_INTERIOR && celltype(ix-1,iy,iz) == C_INTERIOR){
            bool is_xp_valid = a_s(ix,iy,iz) > 1e-2 && a_s(ix,iy,iz) < 1-1e-2;
            bool is_xm_valid = a_s(ix-1,iy,iz) > 1e-2 && a_s(ix-1,iy,iz) < 1-1e-2;

            double kappa_x;
            if (is_xp_valid && is_xm_valid){
                kappa_x = 0.5*(kappa(ix,iy,iz)+kappa(ix-1,iy,iz));
            }else if(is_xp_valid && !is_xm_valid){
                kappa_x = kappa(ix,iy,iz);
            }else if(!is_xp_valid && is_xm_valid){
                kappa_x = kappa(ix-1,iy,iz);
            }else{
                kappa_x = 0.;
            }

            f_sx(ix,iy,iz) = sigma*kappa_x*(a_s(ix,iy,iz)-a_s(ix-1,iy,iz))*inv_dx;
        }else{
            f_sx(ix,iy,iz)=0.0;
        }

        /* debug*/
        /*
           MyArray<double,3> f_bx = grid->f_bx_;
           MyArray<double,3> f_by = grid->f_by_;
           MyArray<double,3> f_bz = grid->f_bz_;
           double a_sig_x = f_sx(ix,iy,iz)*f_bx(ix,iy,iz);
           if(a_sig_x>1e3){
           printf("a_sig_x %f\n",a_sig_x);
           }
         */

    }

    if (ix > Nx || iy > Ny || iz > Nz|| ix < 1 || iy < 2|| iz <1){
        /* do nothing */

    }else{
        /* == y-faces == */
        if(celltype(ix,iy,iz)== C_INTERIOR && celltype(ix,iy-1,iz) == C_INTERIOR){


            bool is_yp_valid = a_s(ix,iy,iz) > 1e-2 && a_s(ix,iy,iz) < 1-1e-2;
            bool is_ym_valid = a_s(ix,iy-1,iz) > 1e-2 && a_s(ix,iy-1,iz) < 1-1e-2;

            double kappa_y;
            if (is_yp_valid && is_ym_valid){
                kappa_y = 0.5*(kappa(ix,iy,iz)+kappa(ix,iy-1,iz));
            }else if(is_yp_valid && !is_ym_valid){
                kappa_y = kappa(ix,iy,iz);
            }else if(!is_yp_valid && is_ym_valid){
                kappa_y = kappa(ix,iy-1,iz);
            }else{
                kappa_y = 0.;
            }

            f_sy(ix,iy,iz) = sigma*kappa_y*(a_s(ix,iy,iz)-a_s(ix,iy-1,iz))*inv_dy;
        }else{
            f_sy(ix,iy,iz)=0.0;
        }

        /* debug*/
        /*
           MyArray<double,3> f_bx = grid->f_bx_;
           MyArray<double,3> f_by = grid->f_by_;
           MyArray<double,3> f_bz = grid->f_bz_;
           double a_sig_y = f_sy(ix,iy,iz)*f_by(ix,iy,iz);
           if(a_sig_y>1e3){
           printf("a_sig_y %f\n",a_sig_y);
           }
         */
    }

    if (ix > Nx || iy > Ny || iz > Nz|| ix < 1 || iy < 1|| iz <2){
        /* do nothing */

    }else{
        /* == z-faces == */
        if(celltype(ix,iy,iz)== C_INTERIOR && celltype(ix,iy,iz-1) == C_INTERIOR){
            bool is_zp_valid = a_s(ix,iy,iz) > 1e-2 && a_s(ix,iy,iz) < 1-1e-2;
            bool is_zm_valid = a_s(ix,iy,iz-1) > 1e-2 && a_s(ix,iy,iz-1) < 1-1e-2;

            double kappa_z;
            if (is_zp_valid && is_zm_valid){
                kappa_z = 0.5*(kappa(ix,iy,iz)+kappa(ix,iy,iz-1));
            }else if(is_zp_valid && !is_zm_valid){
                kappa_z = kappa(ix,iy,iz);
            }else if(!is_zp_valid && is_zm_valid){
                kappa_z = kappa(ix,iy,iz-1);
            }else{
                kappa_z = 0.;
            }

            f_sz(ix,iy,iz) = sigma*kappa_z*(a_s(ix,iy,iz)-a_s(ix,iy,iz-1))*inv_dz;
        }else{
            f_sz(ix,iy,iz)=0.0;
        }

        /* debug*/
        /*
           MyArray<double,3> f_bx = grid->f_bx_;
           MyArray<double,3> f_by = grid->f_by_;
           MyArray<double,3> f_bz = grid->f_bz_;
           double a_sig_z = f_sz(ix,iy,iz)*f_bz(ix,iy,iz);
           if(a_sig_z>1e3){
           printf("a_sig_z %f\n",a_sig_z);
           }
         */
    }


}



void G_SMACSolver::calc_surface_tension(){

    cudaMemset(grid_.f_sx_.data_, 0, grid_.f_sx_.size_ * sizeof(double));
    cudaMemset(grid_.f_sy_.data_, 0, grid_.f_sy_.size_ * sizeof(double));
    cudaMemset(grid_.f_sz_.data_, 0, grid_.f_sz_.size_ * sizeof(double));

    //k_calc_alpha_s<<<grid_dim_,block_dim_>>>(grid_);
    k_calc_interface_normal<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
    k_calc_curvature<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
    k_calc_surface_tension_face<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
}

/* =============================
   ======== set properties =====
   ============================*/

void G_SMACSolver::set_calc_properties(double sizex, double sizey,double sizez, int Nx, int Ny, int Nz){
    grid_.sizex_=sizex;
    grid_.sizey_=sizey;
    grid_.sizez_=sizez;
    grid_.Nx_=Nx;
    grid_.Ny_=Ny;
    grid_.Nz_=Nz;

    grid_.dx_ = sizex/(double)Nx;
    grid_.dy_ = sizey/(double)Ny;
    grid_.dz_ = sizez/(double)Nz;
    grid_.inv_dx_ = 1./grid_.dx_;
    grid_.inv_dy_ = 1./grid_.dy_;
    grid_.inv_dz_ = 1./grid_.dz_;
    grid_.inv_2dx_ = 1./(2.*grid_.dx_);
    grid_.inv_2dy_ = 1./(2.*grid_.dy_);
    grid_.inv_2dz_ = 1./(2.*grid_.dz_);
    grid_.inv_dx2_ = 1./(grid_.dx_*grid_.dx_);
    grid_.inv_dy2_ = 1./(grid_.dy_*grid_.dy_);
    grid_.inv_dz2_ = 1./(grid_.dz_*grid_.dz_);

}

/* ========================= */
/* ===== alpha related ===== */
/* ========================= */

static  __device__ __forceinline__ double integrate_thinc(double a, double b, double gamma,double xi){
    double result = 0.5*(b-a)+gamma/(2.*BETA)*(log(cosh(BETA*(b-xi)))-log(cosh(BETA*(a-xi))));

    return result;
}


static __device__ __forceinline__ double find_xi0_analytic(double alpha, double gamma){
    double A=exp(BETA*gamma*(2.*alpha-1.));

    double result = 1./(2.*BETA)*log((exp(BETA)-A)/(A-exp(-BETA)));

    return result;
}


static __global__ void k_alpha_flux_accum(G_StaggeredGrid* grid){
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int ix = blockIdx.x*blockDim.x + threadIdx.x;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;


    MyArray<double,3>& Fz_accum  = grid->f_Fz_accum_;
    MyArray<double,3>& Fz  = grid->f_Fz_;

    MyArray<double,3>& Fx_accum  = grid->f_Fx_accum_;
    MyArray<double,3>& Fx  = grid->f_Fx_;
    MyArray<double,3>& Fy_accum  = grid->f_Fy_accum_;
    MyArray<double,3>& Fy  = grid->f_Fy_;


    if(iy>=Ny+2 || ix>= Nx+3 || iz>=Nz+2){
        /* do nothing */
    }else{
        Fx_accum(ix,iy,iz) += Fx(ix,iy,iz);
    }

    if(iy>=Ny+3 || ix>= Nx+2 || iz>=Nz+2){
        /* do nothing */
    }else{
        Fy_accum(ix,iy,iz) += Fy(ix,iy,iz);
    }

    if(iy>=Ny+2 || ix>= Nx+2 || iz>=Nz+3){
        /* do nothing */
    }else{
        Fz_accum(ix,iy,iz) += Fz(ix,iy,iz);
    }
}


static __global__ void k_alpha_flux_thincwlic_x(G_StaggeredGrid* grid, double dt){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+2;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    MyArray<double,3>& a  = grid->alpha_;
    MyArray<double,3>& vx = grid->f_vx_;
    MyArray<double,3>& Fx = grid->f_Fx_;
    MyArray<unsigned char,3>& f_xtype = grid->f_xtype_;
    MyArray<unsigned char,3>& celltype = grid->celltype_;

    double inv_dx  = grid->inv_dx_;
    double inv_dy  = grid->inv_dy_;
    double inv_dz  = grid->inv_dz_;
    double inv_2dx = grid->inv_2dx_;
    double inv_2dy = grid->inv_2dy_;
    double inv_2dz = grid->inv_2dz_;

    double dtbydx = dt*inv_dx;

    // x-face index:
    // ix = 2..Nx, iy = 1..Ny, iz = 1..Nz
    if(ix > Nx || iy > Ny || iz > Nz){
        return;
    }

    // x ghost  face
    if (f_xtype(ix,iy,iz)==F_GHOST){
        Fx(ix,iy,iz) = 0.0;
        return ;
    }



    double vxf = vx(ix,iy,iz);

    int donorInd = vxf > 0.0 ? ix-1 : ix;


    double axf = a(donorInd,iy,iz);

    unsigned char ctyped = celltype(donorInd,iy,iz);
    unsigned char ctypep = celltype(donorInd+1,iy,iz);
    unsigned char ctypem = celltype(donorInd-1,iy,iz);


    /* check if donor cell is near boundary in flow direction*/
    /* if then give upwind*/
    if(ctyped != C_INTERIOR){
        Fx(ix,iy,iz) = 0.;
        return;
    }

    if (f_xtype(ix,iy,iz)==F_BOUNDARY){
        int int_id = grid->f_xinternal_id_(ix,iy,iz);
        int bid = grid->f_xbcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);

        if(bcType == BC_OUTLET){
            Fx(ix,iy,iz) = vxf*axf*dtbydx;
            return;
        }else{
            Fx(ix,iy,iz)=0.;
            return;
        }
    }

    if (ctypep != C_INTERIOR || ctypem != C_INTERIOR ) {
        Fx(ix,iy,iz) = vxf*axf*dtbydx;
        return;
    }


    double gamma_x = a(donorInd+1,iy,iz) - a(donorInd-1,iy,iz);

    if(axf < EPS || axf > 1.0-EPS || fabs(gamma_x) < 1e-6){
        Fx(ix,iy,iz) = vxf*axf*dtbydx;
        return;
    }

    double nx = -gamma_x*inv_2dx;

    /* == check cell interior== */
    bool is_yp_interior = celltype(donorInd,iy+1,iz) == C_INTERIOR;
    bool is_ym_interior = celltype(donorInd,iy-1,iz) == C_INTERIOR;


    bool is_zp_interior = celltype(donorInd,iy,iz+1) == C_INTERIOR;
    bool is_zm_interior = celltype(donorInd,iy,iz-1) == C_INTERIOR;


    double ny = 0.;

    if (is_yp_interior && is_ym_interior){
        ny =  -(a(donorInd,iy+1,iz) - a(donorInd,iy-1,iz))*inv_2dy;
    }else if (!is_yp_interior && is_ym_interior){
        ny =  -(a(donorInd,iy,iz) - a(donorInd,iy-1,iz))*inv_dy;
    }else if (is_yp_interior && !is_ym_interior){
        ny =  -(a(donorInd,iy+1,iz) - a(donorInd,iy,iz))*inv_dy;
    }else{
        ny = 0.;
    }

    double nz = 0.;

    if (is_zp_interior && is_zm_interior){
        nz =  -(a(donorInd,iy,iz+1) - a(donorInd,iy,iz-1))*inv_2dz;
    }else if (!is_zp_interior && is_zm_interior){
        nz =  -(a(donorInd,iy,iz) - a(donorInd,iy,iz-1))*inv_dz;
    }else if (is_zp_interior && !is_zm_interior){
        nz =  -(a(donorInd,iy,iz+1) - a(donorInd,iy,iz))*inv_dz;
    }else{
        nz = 0.;
    }



    double s_sq = nx*nx + ny*ny + nz*nz;
    double s = sqrt(s_sq);
    double inv_s = 1.0/(s+EPS);

    double nx_abs = fabs(nx);
    double theta = acos(nx_abs*inv_s);
    double awlic_coeff = 2.*theta/M_PI;

    double wx = 1.0-awlic_coeff*awlic_coeff;

    /*
       double nx_abs = fabs(nx);
       double ny_abs = fabs(ny);
       double nz_abs = fabs(nz);

       double s = nx_abs + ny_abs + nz_abs + EPS;
       double inv_s = 1.0/s;

       double wx = nx_abs*inv_s;
     */

    double gamma = sgn(gamma_x);

    double xi0 = find_xi0_analytic(a(donorInd,iy,iz), gamma);
    double lambda = vxf*dtbydx;

    double Fx_thinc;

    if(vxf > 0.0){
        Fx_thinc = integrate_thinc(1.0-lambda, 1.0, gamma, xi0);
    }else{
        Fx_thinc = -integrate_thinc(0.0, -lambda, gamma, xi0);
    }

    double Fx_upwind = lambda*axf;


    Fx(ix,iy,iz) = wx*Fx_thinc + (1.-wx)*Fx_upwind;
}

static __global__ void k_alpha_flux_thincwlic_y(G_StaggeredGrid* grid, double dt){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+2;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    MyArray<double,3>& a  = grid->alpha_;
    MyArray<double,3>& vy = grid->f_vy_;
    MyArray<double,3>& Fy = grid->f_Fy_;
    MyArray<unsigned char,3>& f_ytype = grid->f_ytype_;
    MyArray<unsigned char,3>& celltype = grid->celltype_;

    double inv_dx  = grid->inv_dx_;
    double inv_dy  = grid->inv_dy_;
    double inv_dz  = grid->inv_dz_;
    double inv_2dx = grid->inv_2dx_;
    double inv_2dy = grid->inv_2dy_;
    double inv_2dz = grid->inv_2dz_;

    double dtbydy = dt*inv_dy;

    // y-face index:
    // ix = 1..Nx, iy = 2..Ny, iz = 1..Nz

    if(ix > Nx || iy > Ny || iz > Nz){
        return;
    }


    // y ghost  face
    if (f_ytype(ix,iy,iz)==F_GHOST){
        Fy(ix,iy,iz) = 0.0;
        return ;
    }

    double vyf = vy(ix,iy,iz);

    int donorInd = vyf > 0.0 ? iy-1 : iy;


    double ayf = a(ix,donorInd,iz);


    unsigned char ctyped = celltype(ix,donorInd,iz);
    unsigned char ctypep = celltype(ix,donorInd+1,iz);
    unsigned char ctypem = celltype(ix,donorInd-1,iz);

    // upwind near boundary

    if(ctyped != C_INTERIOR){
        Fy(ix,iy,iz) = 0.;
        return;
    }

    if (f_ytype(ix,iy,iz)==F_BOUNDARY){
        int int_id = grid->f_yinternal_id_(ix,iy,iz);
        int bid = grid->f_ybcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);

        if(bcType == BC_OUTLET){
            Fy(ix,iy,iz) = vyf*ayf*dtbydy;
            return;
        }else{
            Fy(ix,iy,iz)=0.;
            return;
        }
    }

    if (ctypep != C_INTERIOR || ctypem != C_INTERIOR ) {
        Fy(ix,iy,iz) = vyf*ayf*dtbydy;
        return;
    }


    double gamma_y = a(ix,donorInd+1,iz) - a(ix,donorInd-1,iz);

    if(ayf < EPS || ayf > 1.0-EPS || fabs(gamma_y) < 1e-6){
        Fy(ix,iy,iz) = vyf*ayf*dtbydy;
        return;
    }

    double ny = -gamma_y*inv_2dy;


    /* == check cell interior== */
    bool is_xp_interior = celltype(ix+1,donorInd,iz) == C_INTERIOR;
    bool is_xm_interior = celltype(ix-1,donorInd,iz) == C_INTERIOR;


    bool is_zp_interior = celltype(ix,donorInd,iz+1) == C_INTERIOR;
    bool is_zm_interior = celltype(ix,donorInd,iz-1) == C_INTERIOR;


    double nx = 0.;

    if (is_xp_interior && is_xm_interior){
        nx =  -(a(ix+1,donorInd,iz) - a(ix-1,donorInd,iz))*inv_2dx;
    }else if (!is_xp_interior && is_xm_interior){
        nx =  -(a(ix,donorInd,iz) - a(ix-1,donorInd,iz))*inv_dx;
    }else if (is_xp_interior && !is_xm_interior){
        nx =  -(a(ix+1,donorInd,iz) - a(ix,donorInd,iz))*inv_dx;
    }else{
        nx = 0.;
    }

    double nz = 0.;

    if (is_zp_interior && is_zm_interior){
        nz =  -(a(ix,donorInd,iz+1) - a(ix,donorInd,iz-1))*inv_2dz;
    }else if (!is_zp_interior && is_zm_interior){
        nz =  -(a(ix,donorInd,iz) - a(ix,donorInd,iz-1))*inv_dz;
    }else if (is_zp_interior && !is_zm_interior){
        nz =  -(a(ix,donorInd,iz+1) - a(ix,donorInd,iz))*inv_dz;
    }else{
        nz = 0.;
    }



    double nx_abs = fabs(nx);
    double ny_abs = fabs(ny);
    double nz_abs = fabs(nz);

    /*
       double s = nx_abs + ny_abs + nz_abs + EPS;
       double inv_s = 1.0/s;

       double wy = ny_abs*inv_s;
     */

    double s_sq = nx*nx + ny*ny + nz*nz;
    double s = sqrt(s_sq);
    double inv_s = 1.0/(s+EPS);

    double theta = acos(ny_abs*inv_s);
    double awlic_coeff = 2.*theta/M_PI;

    double wy = 1.0-awlic_coeff*awlic_coeff;

    double gamma = sgn(gamma_y);

    double xi0 = find_xi0_analytic(a(ix,donorInd,iz), gamma);
    double lambda = vyf*dtbydy;

    double Fy_thinc;

    if(vyf > 0.0){
        Fy_thinc = integrate_thinc(1.0-lambda, 1.0, gamma, xi0);
    }else{
        Fy_thinc = -integrate_thinc(0.0, -lambda, gamma, xi0);
    }

    double Fy_upwind = lambda*ayf;


    Fy(ix,iy,iz) = wy*Fy_thinc + (1.-wy)*Fy_upwind;
}

static __global__ void k_alpha_flux_thincwlic_z(G_StaggeredGrid* grid, double dt){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+2;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    MyArray<double,3>& a  = grid->alpha_;
    MyArray<double,3>& vz = grid->f_vz_;
    MyArray<double,3>& Fz = grid->f_Fz_;
    MyArray<unsigned char,3>& f_ztype = grid->f_ztype_;
    MyArray<unsigned char,3>& celltype = grid->celltype_;

    double inv_dx  = grid->inv_dx_;
    double inv_dy  = grid->inv_dy_;
    double inv_dz  = grid->inv_dz_;
    double inv_2dx = grid->inv_2dx_;
    double inv_2dy = grid->inv_2dy_;
    double inv_2dz = grid->inv_2dz_;

    double dtbydz = dt*inv_dz;

    // z-face index:
    // ix = 1..Nx, iy = 1..Ny, iz = 2..Nz
    if(ix > Nx || iy > Ny || iz > Nz){
        return;
    }

    // z ghost  face
    if (f_ztype(ix,iy,iz)==F_GHOST || f_ztype(ix,iy,iz)==F_BOUNDARY){
        Fz(ix,iy,iz) = 0.0;
        return ;
    }

    double vzf = vz(ix,iy,iz);

    int donorInd = vzf > 0.0 ? iz-1 : iz;


    double azf = a(ix,iy,donorInd);

    unsigned char ctyped = celltype(ix,iy,donorInd);
    unsigned char ctypep = celltype(ix,iy,donorInd+1);
    unsigned char ctypem = celltype(ix,iy,donorInd-1);


    /* check if donor cell is near boundary in flow direction*/
    /* if then give upwind*/
    if(ctyped != C_INTERIOR){
        Fz(ix,iy,iz) = 0.;
        return;
    }

    if (f_ztype(ix,iy,iz)==F_BOUNDARY){
        int int_id = grid->f_zinternal_id_(ix,iy,iz);
        int bid = grid->f_zbcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);

        if(bcType == BC_OUTLET){
            Fz(ix,iy,iz) = vzf*azf*dtbydz;
            return;
        }else{
            Fz(ix,iy,iz)=0.;
            return;
        }
    }


    if (ctypep != C_INTERIOR || ctypem != C_INTERIOR) {
        Fz(ix,iy,iz) = vzf*azf*dtbydz;
        return;
    }

    double gamma_z = a(ix,iy,donorInd+1) - a(ix,iy,donorInd-1);

    if(azf < EPS || azf > 1.0-EPS || fabs(gamma_z) < 1e-6){
        Fz(ix,iy,iz) = vzf*azf*dtbydz;
        return;
    }

    double nz = -gamma_z*inv_2dz;

    /* == check cell interior== */
    bool is_xp_interior = celltype(ix+1,iy,donorInd) == C_INTERIOR;
    bool is_xm_interior = celltype(ix-1,iy,donorInd) == C_INTERIOR;

    bool is_yp_interior = celltype(ix,iy+1,donorInd) == C_INTERIOR;
    bool is_ym_interior = celltype(ix,iy-1,donorInd) == C_INTERIOR;




    double nx = 0.;

    if (is_xp_interior && is_xm_interior){
        nx =  -(a(ix+1,iy,donorInd) - a(ix-1,iy,donorInd))*inv_2dx;
    }else if (!is_xp_interior && is_xm_interior){
        nx =  -(a(ix,iy,donorInd) - a(ix-1,iy,donorInd))*inv_dx;
    }else if (is_xp_interior && !is_xm_interior){
        nx =  -(a(ix+1,iy,donorInd) - a(ix,iy,donorInd))*inv_dx;
    }else{
        nx = 0.;
    }

    double ny = 0.;

    if (is_yp_interior && is_ym_interior){
        ny =  -(a(ix,iy+1,donorInd) - a(ix,iy-1,donorInd))*inv_2dy;
    }else if (!is_yp_interior && is_ym_interior){
        ny =  -(a(ix,iy,donorInd) - a(ix,iy-1,donorInd))*inv_dy;
    }else if (is_yp_interior && !is_ym_interior){
        ny =  -(a(ix,iy+1,donorInd) - a(ix,iy,donorInd))*inv_dy;
    }else{
        ny = 0.;
    }

    double nx_abs = fabs(nx);
    double ny_abs = fabs(ny);
    double nz_abs = fabs(nz);

    double s_sq = nx*nx + ny*ny + nz*nz;
    double s = sqrt(s_sq);
    double inv_s = 1.0/(s+EPS);

    double theta = acos(nz_abs*inv_s);
    double awlic_coeff = 2.*theta/M_PI;

    double wz = 1.0-awlic_coeff*awlic_coeff;

    /*
       double s = nx_abs + ny_abs + nz_abs + EPS;
       double inv_s = 1.0/s;

       double wz = nz_abs*inv_s;
     */

    double gamma = sgn(gamma_z);

    double xi0 = find_xi0_analytic(a(ix,iy,donorInd), gamma);
    double lambda = vzf*dtbydz;

    double Fz_thinc;

    if(vzf > 0.0){
        Fz_thinc = integrate_thinc(1.0-lambda, 1.0, gamma, xi0);
    }else{
        Fz_thinc = -integrate_thinc(0.0, -lambda, gamma, xi0);
    }

    double Fz_upwind = lambda*azf;


    Fz(ix,iy,iz) = wz*Fz_thinc + (1.-wz)*Fz_upwind;
}

static __global__ void k_set_correct_coeff(G_StaggeredGrid* grid){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

    MyArray<double,3> a = grid->alpha_;
    MyArray<unsigned char,3>& coeff = grid->correct_coeff_;




    if (iy >=a.sizey_-1 || ix >= a.sizex_-1 || iz>= a.sizez_-1) return;

    MyArray<unsigned char,3> ct= grid->celltype_;
    if (ct(ix,iy,iz) != C_INTERIOR){
        return;
    }

    if(a(ix,iy,iz)>0.5){
        coeff(ix,iy,iz)=1;
    }else{
        coeff(ix,iy,iz)=0;
    }
}

static __global__ void k_transport_alpha_x(G_StaggeredGrid* grid, double dt){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

    MyArray<double,3>& a = grid->alpha_;
    MyArray<double,3>& a_new = grid->alpha_new_;
    const MyArray<double,3>& Fx = grid->f_Fx_;




    if (iy >=a.sizey_-1 || ix >= a.sizex_-1 || iz>= a.sizez_-1) return;

    MyArray<unsigned char,3> ct= grid->celltype_;
    if (ct(ix,iy,iz) != C_INTERIOR){
        a_new(ix,iy,iz)=0.0;
        return;
    }

    double flux = 0.;

    /* == x direction == */
    flux += Fx(ix+1,iy,iz) - Fx(ix,iy,iz);


    /* == correction == */
    double up = d_get_vx_xface(grid,ix+1,iy,iz);
    double um = d_get_vx_xface(grid,ix,iy,iz);
    double inv_dx = grid->inv_dx_;
    double coeff = (double)grid->correct_coeff_(ix,iy,iz);

    flux -= coeff*(up-um)*dt*inv_dx;


    a_new(ix,iy,iz) = a(ix,iy,iz) - flux;

    /* clipping */
    /*
       if (a_new(ix,iy,iz)<EPS){
       a_new(ix,iy,iz)=0.;
       }else if (a_new(ix,iy,iz)>1.){
       a_new(ix,iy,iz)=1.;
       }
     */

}

static __global__ void k_transport_alpha_y(G_StaggeredGrid* grid, double dt){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

    MyArray<double,3>& a = grid->alpha_;
    MyArray<double,3>& a_new = grid->alpha_new_;
    const MyArray<double,3>& Fy = grid->f_Fy_;



    if (iy >=a.sizey_-1 || ix >= a.sizex_-1 || iz>= a.sizez_-1) return;

    MyArray<unsigned char,3> ct= grid->celltype_;
    if (ct(ix,iy,iz) != C_INTERIOR){
        a_new(ix,iy,iz)=0.0;
        return;
    }


    double flux = 0.;

    /* == y direction == */
    flux += Fy(ix,iy+1,iz) - Fy(ix,iy,iz);


    /* == correction == */
    double up = d_get_vy_yface(grid,ix,iy+1,iz);
    double um = d_get_vy_yface(grid,ix,iy,iz);
    double inv_dy = grid->inv_dy_;
    double coeff = (double)grid->correct_coeff_(ix,iy,iz);

    flux -= coeff*(up-um)*dt*inv_dy;

    a_new(ix,iy,iz) = a(ix,iy,iz) - flux;

    /* clipping */
    /*
       if (a_new(ix,iy,iz)<EPS){
       a_new(ix,iy,iz)=0.;
       }else if (a_new(ix,iy,iz)>1.){
       a_new(ix,iy,iz)=1.;
       }
     */

}

static __global__ void k_transport_alpha_z(G_StaggeredGrid* grid, double dt){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

    MyArray<double,3>& a = grid->alpha_;
    MyArray<double,3>& a_new = grid->alpha_new_;
    const MyArray<double,3>& Fz = grid->f_Fz_;



    if (iy >=a.sizey_-1 || ix >= a.sizex_-1 || iz>= a.sizez_-1) return;

    MyArray<unsigned char,3> ct= grid->celltype_;
    if (ct(ix,iy,iz) != C_INTERIOR){
        a_new(ix,iy,iz)=0.0;
        return;
    }

    double flux = 0.;

    /* == z direction == */
    flux += Fz(ix,iy,iz+1) - Fz(ix,iy,iz);


    /* == correction == */
    double up = d_get_vz_zface(grid,ix,iy,iz+1);
    double um = d_get_vz_zface(grid,ix,iy,iz);
    double inv_dz = grid->inv_dz_;
    double coeff = (double)grid->correct_coeff_(ix,iy,iz);

    flux -= coeff*(up-um)*dt*inv_dz;


    a_new(ix,iy,iz) = a(ix,iy,iz) - flux;

    /* clipping */
    /*
       if (a_new(ix,iy,iz)<EPS){
       a_new(ix,iy,iz)=0.;
       }else if (a_new(ix,iy,iz)>1.){
       a_new(ix,iy,iz)=1.;
       }
     */

}

/*
   static __global__ void k_calc_alpha_with_solidfrac(G_StaggeredGrid* grid){
   int iz = blockIdx.z*blockDim.z + threadIdx.z+1;
   int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
   int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

   MyArray<double,3>& a = grid->alpha_;
   MyArray<double,3>& asf = grid->alpha_with_solidfrac_;
   MyArray<double,3>& sf = grid->ibm_solid_fraction_;



   if (iy >=a.sizey_-1 || ix >= a.sizex_-1 || iz>= a.sizez_-1) return;

   asf(ix,iy,iz)=a(ix,iy,iz)*(1.-sf(ix,iy,iz));

   }
 */


static __global__ void k_clip_alpha(G_StaggeredGrid* grid){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

    MyArray<double,3>& a = grid->alpha_;



    if (iy >=a.sizey_-1 || ix >= a.sizex_-1 || iz>= a.sizez_-1) return;

    if (a(ix,iy,iz)<EPS){
        a(ix,iy,iz)=0.;
    }else if (a(ix,iy,iz)>1.){
        a(ix,iy,iz)=1.;
    }

}

static __global__ void k_transport_alpha(G_StaggeredGrid grid_){
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

    MyArray<double,3>& a = grid_.alpha_;
    MyArray<double,3>& a_new = grid_.alpha_new_;
    const MyArray<double,3>& Fx = grid_.f_Fx_;
    const MyArray<double,3>& Fy = grid_.f_Fy_;
    const MyArray<double,3>& Fz = grid_.f_Fz_;

    MyArray<unsigned char,3>ct = grid_.celltype_;


    if (iy >=a.sizey_-1 || ix >= a.sizex_-1 || iz>= a.sizez_-1) return;

    if (ct(ix,iy,iz) != C_INTERIOR){
        a_new(ix,iy,iz)=0.0;
        return;
    }

    double flux = 0.;

    /* == x direction == */
    flux += Fx(ix+1,iy,iz) - Fx(ix,iy,iz);

    /* == y direction == */
    flux += Fy(ix,iy+1,iz) - Fy(ix,iy,iz);

    /* == z direction == */
    flux += Fz(ix,iy,iz+1) - Fz(ix,iy,iz);

    a_new(ix,iy,iz) = a(ix,iy,iz) - flux;

    /* clipping */
    if (a_new(ix,iy,iz)<EPS){
        a_new(ix,iy,iz)=0.;
    }else if (a_new(ix,iy,iz)>1.){
        a_new(ix,iy,iz)=1.;
    }

}

void G_SMACSolver::transport_alpha(){
    k_transport_alpha<<<grid_dim_, block_dim_>>>(grid_);

    /* == swap == */
    std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);


}


static __global__ void k_update_cell_properties_by_alpha(G_StaggeredGrid* grid,double rho0, double rho1, double mu0, double mu1){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;


    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;

    MyArray<double,3> a = grid->alpha_;
    MyArray<double,3> rho = grid->rho_;
    MyArray<double,3> inv_rho = grid->inv_rho_;
    MyArray<double,3> mu = grid->mu_;

    MyArray<unsigned char,3>& celltype = grid->celltype_;

    if (ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    unsigned char ctype = celltype(ix,iy,iz);

    if(ctype == C_INTERIOR || ctype == C_NEAR_BOUNDARY){

        /* == update rho == */
        double alpha = a(ix,iy,iz);
        rho(ix,iy,iz)= (1.-alpha)*rho0+alpha*rho1;

        /* == update inv_rho == */
        inv_rho(ix,iy,iz)= 1./rho(ix,iy,iz);

        /* == update mu == */
        mu(ix,iy,iz) = (1.-alpha)*mu0+alpha*mu1;
    }
}

static __global__ void k_update_x_face_properties_by_alpha(G_StaggeredGrid* grid,double rho0, double rho1){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;


    if (ix >=Nx+3 || iy >= Ny+2 || iz >= Nz+2) return;


    MyArray<double,3> rho = grid->rho_;
    MyArray<double,3> mu = grid->mu_;

    /* == update inv rho at face== */

    /* == x faces == */
    MyArray<double,3>  f_bx = grid->f_bx_;
    unsigned char f_xtype= grid->f_xtype_(ix,iy,iz);

    if(f_xtype!=F_INTERIOR){
        f_bx(ix,iy,iz)=0.;
        return;
    }



    /* == update rho at face== */
    MyArray<double,3>  f_rhox = grid->f_rhox_;
    f_rhox(ix,iy,iz) = 0.5*(rho(ix,iy,iz)+rho(ix-1,iy,iz));

    //f_bx[ind] = (inv_rho[ind1]+inv_rho[ind0])*0.5;
    f_bx(ix,iy,iz) = 1./f_rhox(ix,iy,iz);

    /* == update mu at face== */
    MyArray<double,3>  f_mux = grid->f_mux_;
    f_mux(ix,iy,iz) = 0.5*(mu(ix,iy,iz)+mu(ix-1,iy,iz));
}


static __global__ void k_update_y_face_properties_by_alpha(G_StaggeredGrid* grid,double rho0, double rho1){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;


    if (ix >=Nx+2 || iy >= Ny+3 || iz >= Nz+2) return;


    MyArray<double,3> rho = grid->rho_;
    MyArray<double,3> mu = grid->mu_;


    /* == update inv rho at face== */
    MyArray<double,3>  f_by = grid->f_by_;

    unsigned char f_ytype= grid->f_ytype_(ix,iy,iz);
    if(f_ytype!=F_INTERIOR){
        f_by(ix,iy,iz)=0.;
        return;
    }


    /* == update rho at face== */
    MyArray<double,3>  f_rhoy = grid->f_rhoy_;
    f_rhoy(ix,iy,iz) = 0.5*(rho(ix,iy,iz)+rho(ix,iy-1,iz));

    f_by(ix,iy,iz) = 1./f_rhoy(ix,iy,iz);

    /* == update mu at face== */
    MyArray<double,3>  f_muy = grid->f_muy_;
    f_muy(ix,iy,iz) = 0.5*(mu(ix,iy,iz)+mu(ix,iy-1,iz));
}

static __global__ void k_update_z_face_properties_by_alpha(G_StaggeredGrid* grid,double rho0, double rho1){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;


    if (ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+3) return;


    MyArray<double,3> rho = grid->rho_;
    MyArray<double,3> mu = grid->mu_;

    /* == update inv rho at face== */
    /* == x faces == */
    MyArray<double,3>  f_bz = grid->f_bz_;
    unsigned char f_ztype= grid->f_ztype_(ix,iy,iz);

    if(f_ztype!=F_INTERIOR){
        f_bz(ix,iy,iz)=0.;
        return;
    }



    /* == update rho at face== */
    MyArray<double,3>  f_rhoz = grid->f_rhoz_;
    f_rhoz(ix,iy,iz) = 0.5*(rho(ix,iy,iz)+rho(ix,iy,iz-1));

    f_bz(ix,iy,iz) = 1./f_rhoz(ix,iy,iz);

    /* == update mu at face== */
    MyArray<double,3>  f_muz = grid->f_muz_;
    f_muz(ix,iy,iz) = 0.5*(mu(ix,iy,iz)+mu(ix,iy,iz-1));
}

__global__ void k_swap_rho(G_StaggeredGrid* grid){
    double* tmp;
    tmp =grid->rho_.data_; 
    grid->rho_.data_=grid->rho_old_.data_; 
    grid->rho_old_.data_=tmp; 
}

void G_SMACSolver::update_properties_by_alpha(){

    const double rho1 = rho1_;
    const double rho0 = rho0_;

    const double mu1 = mu1_;
    const double mu0 = mu0_;

    std::swap(grid_.rho_old_.data_,grid_.rho_.data_);
    cudaDeviceSynchronize();
    /* debug */
    k_swap_rho<<<1,1>>>(grid_.d_ptr_);

    k_update_cell_properties_by_alpha<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,rho0, rho1, mu0, mu1);
    k_update_x_face_properties_by_alpha<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,rho0, rho1);
    k_update_y_face_properties_by_alpha<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,rho0, rho1);
    k_update_z_face_properties_by_alpha<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,rho0, rho1);

}

__global__ void k_swap_alpha(G_StaggeredGrid *grid){
    double* tmp = grid->alpha_new_.data_;
    grid->alpha_new_.data_ = grid->alpha_.data_;
    grid->alpha_.data_ = tmp;
}

void G_SMACSolver::alpha_flux_thincwlic_split(double dt,int steps){


    k_set_correct_coeff<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);



    if (steps%3 == 0){
        k_alpha_flux_thincwlic_x<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_x<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);




        k_alpha_flux_thincwlic_y<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_y<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);



        k_alpha_flux_thincwlic_z<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_z<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);


    }else if(steps%3 == 1){

        k_alpha_flux_thincwlic_y<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_y<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);



        k_alpha_flux_thincwlic_z<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_z<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);

        k_alpha_flux_thincwlic_x<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_x<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);


    }else{


        k_alpha_flux_thincwlic_z<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_z<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);

        k_alpha_flux_thincwlic_x<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_x<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);


        k_alpha_flux_thincwlic_y<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_y<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);


    }


    k_clip_alpha<<<grid_dim_, block_dim_>>>(grid_.d_ptr_);

    cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_, grid_.alpha_.data_, d_r2_,grid_.alpha_.size_);
    double sum;
    cudaMemcpy(&sum,d_r2_,sizeof(double),cudaMemcpyDeviceToHost);
    printf("total alpha = %.7e\n",sum);

    /* debug */

    /*
       k_calc_alpha_with_solidfrac<<<grid_dim_, block_dim_>>>(grid_.d_ptr_);
       cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_, grid_.alpha_with_solidfrac_.data_, d_r2_,grid_.alpha_.size_);
       cudaMemcpy(&sum,d_r2_,sizeof(double),cudaMemcpyDeviceToHost);
       printf("total alpha solid frac= %.7e\n",sum);
     */



}

void G_SMACSolver::alpha_flux_thincwlic(double dt){

    //k_alpha_flux_upwind<<<grid_dim_,block_dim_>>>(grid_, dt);
    k_alpha_flux_thincwlic_x<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
    k_alpha_flux_thincwlic_y<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
    k_alpha_flux_thincwlic_z<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
}

void G_SMACSolver::alpha_flux_accum(){
    k_alpha_flux_accum<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
}

void G_SMACSolver::clear_alpha_flux_accum(){
    MyArray<double,3> &Fx_accum = grid_.f_Fx_accum_;
    MyArray<double,3> &Fy_accum = grid_.f_Fy_accum_;
    MyArray<double,3> &Fz_accum = grid_.f_Fz_accum_;

    cudaMemset(Fx_accum.data_, 0, sizeof(double) * Fx_accum.size_);
    cudaMemset(Fy_accum.data_, 0, sizeof(double) * Fy_accum.size_);
    cudaMemset(Fz_accum.data_, 0, sizeof(double) * Fz_accum.size_);
}

void G_SMACSolver::clear_alpha_flux(){
    MyArray<double,3> &Fx = grid_.f_Fx_;
    MyArray<double,3> &Fy = grid_.f_Fy_;
    MyArray<double,3> &Fz = grid_.f_Fz_;

    cudaMemset(Fx.data_, 0, sizeof(double) * Fx.size_);
    cudaMemset(Fy.data_, 0, sizeof(double) * Fy.size_);
    cudaMemset(Fz.data_, 0, sizeof(double) * Fz.size_);
}

static __global__ void k_compute_mass_flux_from_alpha_flux(SMACSolver solv,G_StaggeredGrid* grid){
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int ix = blockIdx.x*blockDim.x + threadIdx.x;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    MyArray<double,3> Fx  = grid->f_Fx_accum_;
    MyArray<double,3> Fy  = grid->f_Fy_accum_;
    MyArray<double,3> Fz  = grid->f_Fz_accum_;
    MyArray<double,3> mfx = grid->f_mfx_;
    MyArray<double,3> mfy = grid->f_mfy_;
    MyArray<double,3> mfz = grid->f_mfz_;

    double inv_dt = solv.inv_dt_;
    double dx = grid->dx_;
    double dy = grid->dy_;
    double dz = grid->dz_;

    double rho0 = solv.rho0_;
    double drho = solv.rho1_ - solv.rho0_;

    double dxbydt = dx*inv_dt;
    double dybydt = dy*inv_dt;
    double dzbydt = dz*inv_dt;

    if(iy>=Ny+2 || ix>= Nx+3 || iz >= Nz+2){
        /* do nothing */
    }else{
        double q = d_get_vx_xface(grid,ix,iy,iz);                    // u_f
        double alpha_q = Fx(ix,iy,iz)* dxbydt; // u_f * alpha_f
        mfx(ix,iy,iz) = rho0 * q + drho * alpha_q; // rho*u

        /*
           if(grid->f_xtype_(ix,iy,iz) == F_INTERIOR){
           double q = vx(ix,iy,iz);                    // u_f
           double alpha_q = Fx(ix,iy,iz)* dxbydt; // u_f * alpha_f
           mfx(ix,iy,iz) = rho0 * q + drho * alpha_q; // rho*u
           }else{
           mfx(ix,iy,iz) = 0.0; // rho*u
           }
         */

    }

    if(iy>=Ny+3 || ix>= Nx+2 || iz>= Nz+2){
        /* do nothing */
    }else{

        double q = d_get_vy_yface(grid,ix,iy,iz);                    // u_f
        double alpha_q = Fy(ix,iy,iz)* dybydt; // u_f * alpha_f
        mfy(ix,iy,iz) = rho0 * q + drho * alpha_q; // rho*u

        /*
           if(grid->f_ytype_(ix,iy,iz) == F_INTERIOR){
           double q = vy(ix,iy,iz);                    // u_f
           double alpha_q = Fy(ix,iy,iz)* dybydt; // u_f * alpha_f
           mfy(ix,iy,iz) = rho0 * q + drho * alpha_q; // rho*u
           }else{
           mfy(ix,iy,iz) = 0.0; // rho*u
           }
         */
    }

    if(iy>=Ny+2 || ix>= Nx+2 || iz>= Nz+3){
        /* do nothing */
    }else{
        double q = d_get_vz_zface(grid,ix,iy,iz);                    // u_f
        double alpha_q = Fz(ix,iy,iz)* dzbydt; // u_f * alpha_f
        mfz(ix,iy,iz) = rho0 * q + drho * alpha_q; // rho*u
        /*
           if(grid->f_ztype_(ix,iy,iz) == F_INTERIOR){
           double q = vz(ix,iy,iz);                    // u_f
           double alpha_q = Fz(ix,iy,iz)* dzbydt; // u_f * alpha_f
           mfz(ix,iy,iz) = rho0 * q + drho * alpha_q; // rho*u
           }else{
           mfz(ix,iy,iz) = 0.0; // rho*u
           }
         */
    }
}

void G_SMACSolver::compute_mass_flux_from_alpha_flux(SMACSolver solv){
    k_compute_mass_flux_from_alpha_flux<<<grid_dim_,block_dim_>>>(solv,grid_.d_ptr_);

}

/* === boundary condition related == */
static __global__ void k_update_x_face_boundary_properties(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;


    if (ix >=Nx+3 || iy >= Ny+2 || iz >= Nz+2) return;

    MyArray<double,3> f_rhox = grid->f_rhox_;
    MyArray<double,3> f_mux = grid->f_mux_;
    MyArray<double,3>  f_bx = grid->f_bx_;
    //MyArray<double,3> mfx = grid->f_mfx_;

    /* pressure is assumed to be zero at the boundary*/

    if(grid->f_xtype_(ix,iy,iz) == F_BOUNDARY){
        int int_id = grid->f_xinternal_id_(ix,iy,iz);
        int bid = grid->f_xbcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);

        f_rhox(ix,iy,iz) = grid->rho_(ix+int_id,iy,iz);
        f_mux(ix,iy,iz) = grid->mu_(ix+int_id,iy,iz);

        if(bcType == BC_OUTLET){
            int int_id_shift = grid->f_xinternal_id_(ix,iy,iz);
            double b_beta = 0.;

            if(int_id_shift < 0){
                b_beta = 2.0*grid->f_bx_(ix-1,iy,iz);
                //mfx(ix,iy,iz) = mfx(ix-1,iy,iz);
            }else{
                b_beta = 2.0*grid->f_bx_(ix+1,iy,iz);
                //mfx(ix,iy,iz) = mfx(ix+1,iy,iz);
            }

            f_bx(ix,iy,iz)=b_beta;

        }else{
            f_bx(ix,iy,iz)=0.;
            return;
        }
    };

}

static __global__ void k_update_y_face_boundary_properties(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;


    if (ix >=Nx+2 || iy >= Ny+3 || iz >= Nz+2) return;

    MyArray<double,3> f_rhoy = grid->f_rhoy_;
    MyArray<double,3> f_muy = grid->f_muy_;
    MyArray<double,3>  f_by = grid->f_by_;
    //MyArray<double,3> mfy = grid->f_mfy_;

    if(grid->f_ytype_(ix,iy,iz) == F_BOUNDARY){
        int int_id = grid->f_yinternal_id_(ix,iy,iz);
        f_rhoy(ix,iy,iz) = grid->rho_(ix,iy+int_id,iz);
        f_muy(ix,iy,iz) = grid->mu_(ix,iy+int_id,iz);

        int bid = grid->f_ybcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);

        /* pressure is assumed to be zero at the boundary*/

        if(bcType == BC_OUTLET){
            double b_beta = 0.;

            if(int_id < 0){
                b_beta = 2.0*grid->f_by_(ix,iy-1,iz);
                //mfy(ix,iy,iz) = mfy(ix,iy-1,iz);
            }else{
                b_beta = 2.0*grid->f_by_(ix,iy+1,iz);
                //mfy(ix,iy,iz) = mfy(ix,iy+1,iz);
            }

            f_by(ix,iy,iz)=b_beta;

        }else{
            f_by(ix,iy,iz)=0.;
            return;
        }
    }

}

static __global__ void k_update_z_face_boundary_properties(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;


    if (ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+3) return;

    MyArray<double,3> f_rhoz = grid->f_rhoz_;
    MyArray<double,3> f_muz = grid->f_muz_;
    MyArray<double,3>  f_bz = grid->f_bz_;
    //MyArray<double,3> mfz = grid->f_mfz_;

    if(grid->f_ztype_(ix,iy,iz) == F_BOUNDARY){
        int int_id = grid->f_zinternal_id_(ix,iy,iz);
        f_rhoz(ix,iy,iz) = grid->rho_(ix,iy,iz+int_id);
        f_muz(ix,iy,iz) = grid->mu_(ix,iy,iz+int_id);

        int bid = grid->f_zbcid_(ix,iy,iz);
        unsigned char bcType = grid->bc_.bcType_(bid);

        /* pressure is assumed to be zero at the boundary*/

        if(bcType == BC_OUTLET){
            double b_beta = 0.;

            if(int_id < 0){
                b_beta = 2.0*grid->f_bz_(ix,iy,iz-1);
                //mfz(ix,iy,iz) = mfz(ix,iy,iz-1);
            }else{
                b_beta = 2.0*grid->f_bz_(ix,iy,iz+1);
                //mfz(ix,iy,iz) = mfz(ix,iy,iz+1);
            }

            f_bz(ix,iy,iz)=b_beta;

        }else{
            f_bz(ix,iy,iz)=0.;
            return;
        }
    }

}


void G_SMACSolver::update_boundary_faces(){

    k_update_x_face_boundary_properties<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
    k_update_y_face_boundary_properties<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
    k_update_z_face_boundary_properties<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);

}

