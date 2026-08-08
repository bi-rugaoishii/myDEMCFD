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
    if (f_ztype(ix,iy,iz)==F_GHOST){
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

    if (a(ix,iy,iz)<1e-16){
        a(ix,iy,iz)=0.;
    }else if (a(ix,iy,iz)>1.-1e-16){
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

static __global__ void k_update_cell_boundary_properties_by_alpha(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;


    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;

    MyArray<double,3> rho = grid->rho_;
    MyArray<double,3> mu = grid->mu_;

    if (ix >=Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;


    if(ix == 0){

        /* == update rho == */
        rho(ix,iy,iz)= rho(ix+1,iy,iz);


        /* == update mu == */
        mu(ix,iy,iz) = mu(ix+1,iy,iz);
    }

    if(ix == Nx+1){
        /* == update rho == */
        rho(ix,iy,iz)= rho(ix-1,iy,iz);


        /* == update mu == */
        mu(ix,iy,iz) = mu(ix-1,iy,iz);
    }

    if(iy == 0){
        /* == update rho == */
        rho(ix,iy,iz)= rho(ix,iy+1,iz);


        /* == update mu == */
        mu(ix,iy,iz) = mu(ix,iy+1,iz);
    }

    if(iy == Ny+1){
        /* == update rho == */
        rho(ix,iy,iz)= rho(ix,iy-1,iz);


        /* == update mu == */
        mu(ix,iy,iz) = mu(ix,iy-1,iz);
    }

    if(iz == 0){
        /* == update rho == */
        rho(ix,iy,iz)= rho(ix,iy,iz+1);


        /* == update mu == */
        mu(ix,iy,iz) = mu(ix,iy,iz+1);
    }

    if(iz == Nz+1){
        /* == update rho == */
        rho(ix,iy,iz)= rho(ix,iy,iz-1);


        /* == update mu == */
        mu(ix,iy,iz) = mu(ix,iy,iz-1);
    }

    MyArray<unsigned char,3>& celltype = grid->celltype_;

    if(celltype(ix,iy,iz) == C_SOLID){

        if(celltype(ix-1,iy,iz) == C_INTERIOR){
            rho(ix,iy,iz) = rho(ix-1,iy,iz);
            mu(ix,iy,iz) = mu(ix-1,iy,iz);
        }else if(celltype(ix+1,iy,iz) == C_INTERIOR){
            rho(ix,iy,iz) = rho(ix+1,iy,iz);
            mu(ix,iy,iz) = mu(ix+1,iy,iz);
        }else if(celltype(ix,iy-1,iz) == C_INTERIOR){
            rho(ix,iy,iz) = rho(ix,iy-1,iz);
            mu(ix,iy,iz) = mu(ix,iy-1,iz);
        }else if(celltype(ix,iy+1,iz) == C_INTERIOR){
            rho(ix,iy,iz) = rho(ix,iy+1,iz);
            mu(ix,iy,iz) = mu(ix,iy+1,iz);
        }else if(celltype(ix,iy,iz-1) == C_INTERIOR){
            rho(ix,iy,iz) = rho(ix,iy,iz-1);
            mu(ix,iy,iz) = mu(ix,iy,iz-1);
        }else if(celltype(ix,iy,iz+1) == C_INTERIOR){
            rho(ix,iy,iz) = rho(ix,iy,iz+1);
            mu(ix,iy,iz) = mu(ix,iy,iz+1);
        }else{
            rho(ix,iy,iz)=0.;
            mu(ix,iy,iz)=0.;
        }

    }
}

static __global__ void k_update_cell_ghost_properties(G_StaggeredGrid* grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;

    int Nx = grid->Nx_;
    int Ny = grid->Ny_;
    int Nz = grid->Nz_;

    if(ix >= Nx+2 || iy >= Ny+2 || iz >= Nz+2) return;

    bool is_ghost =
        ix == 0 || ix == Nx+1 ||
        iy == 0 || iy == Ny+1 ||
        iz == 0 || iz == Nz+1;

    if(!is_ghost) return;

    MyArray<double,3> rho = grid->rho_;
    MyArray<double,3> mu = grid->mu_;

    int src_ix = ix == 0 ? 1 : ix == Nx+1 ? Nx : ix;
    int src_iy = iy == 0 ? 1 : iy == Ny+1 ? Ny : iy;
    int src_iz = iz == 0 ? 1 : iz == Nz+1 ? Nz : iz;

    rho(ix,iy,iz) = rho(src_ix,src_iy,src_iz);
    mu(ix,iy,iz) = mu(src_ix,src_iy,src_iz);
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
    MyArray<double,3>  f_inv_rhox = grid->f_inv_rhox_;
    unsigned char f_xtype= grid->f_xtype_(ix,iy,iz);

    if(f_xtype!=F_INTERIOR){
        f_bx(ix,iy,iz)=0.;
        f_inv_rhox(ix,iy,iz)=0.;
        return;
    }



    /* == update rho at face== */
    MyArray<double,3>  f_rhox = grid->f_rhox_;
    f_rhox(ix,iy,iz) = 0.5*(rho(ix,iy,iz)+rho(ix-1,iy,iz));

    //f_bx[ind] = (inv_rho[ind1]+inv_rho[ind0])*0.5;
    f_bx(ix,iy,iz) = 1./f_rhox(ix,iy,iz);
    f_inv_rhox(ix,iy,iz)=f_bx(ix,iy,iz);

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
    MyArray<double,3>  f_inv_rhoy = grid->f_inv_rhoy_;

    unsigned char f_ytype= grid->f_ytype_(ix,iy,iz);
    if(f_ytype!=F_INTERIOR){
        f_by(ix,iy,iz)=0.;
        f_inv_rhoy(ix,iy,iz)=0.;
        return;
    }


    /* == update rho at face== */
    MyArray<double,3>  f_rhoy = grid->f_rhoy_;
    f_rhoy(ix,iy,iz) = 0.5*(rho(ix,iy,iz)+rho(ix,iy-1,iz));

    f_by(ix,iy,iz) = 1./f_rhoy(ix,iy,iz);
    f_inv_rhoy(ix,iy,iz)=f_by(ix,iy,iz);

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
    MyArray<double,3>  f_inv_rhoz = grid->f_inv_rhoz_;
    unsigned char f_ztype= grid->f_ztype_(ix,iy,iz);

    if(f_ztype!=F_INTERIOR){
        f_bz(ix,iy,iz)=0.;
        f_inv_rhoz(ix,iy,iz) = 0.0;
        return;
    }



    /* == update rho at face== */
    MyArray<double,3>  f_rhoz = grid->f_rhoz_;
    f_rhoz(ix,iy,iz) = 0.5*(rho(ix,iy,iz)+rho(ix,iy,iz-1));

    f_bz(ix,iy,iz) = 1./f_rhoz(ix,iy,iz);
    f_inv_rhoz(ix,iy,iz)=f_bz(ix,iy,iz);

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
    k_update_cell_boundary_properties_by_alpha<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
    k_update_cell_ghost_properties<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);


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

/* ====== for two cfd coupling ======= */

__device__ __forceinline__ double d_get_void_fraction_half_xface(G_StaggeredGrid* grid,int ix,int iy,int iz){
    MyArray<double,3>& eps=grid->void_fraction_half_;
    return 0.5*(eps(ix-1,iy,iz)+eps(ix,iy,iz));
}

__device__ __forceinline__ double d_get_void_fraction_half_yface(G_StaggeredGrid* grid,int ix,int iy,int iz){
    MyArray<double,3>& eps=grid->void_fraction_half_;
    return 0.5*(eps(ix,iy-1,iz)+eps(ix,iy,iz));
}

__device__ __forceinline__ double d_get_void_fraction_half_zface(G_StaggeredGrid* grid,int ix,int iy,int iz){
    MyArray<double,3>& eps=grid->void_fraction_half_;
    return 0.5*(eps(ix,iy,iz-1)+eps(ix,iy,iz));
}
__device__ __forceinline__ double d_get_void_fraction_t(const G_StaggeredGrid* grid,int ix,int iy,int iz,double theta){
    const double eps_old=grid->void_fraction_old_(ix,iy,iz);
    const double eps_new=grid->void_fraction_(ix,iy,iz);
    return eps_old+theta*(eps_new-eps_old);
}

__device__ __forceinline__ double d_get_void_fraction_xface_t(const G_StaggeredGrid* grid,int ix,int iy,int iz,double theta){
    return 0.5*(d_get_void_fraction_t(grid,ix-1,iy,iz,theta)+d_get_void_fraction_t(grid,ix,iy,iz,theta));
}

__device__ __forceinline__ double d_get_void_fraction_yface_t(const G_StaggeredGrid* grid,int ix,int iy,int iz,double theta){
    return 0.5*(d_get_void_fraction_t(grid,ix,iy-1,iz,theta)+d_get_void_fraction_t(grid,ix,iy,iz,theta));
}

__device__ __forceinline__ double d_get_void_fraction_zface_t(const G_StaggeredGrid* grid,int ix,int iy,int iz,double theta){
    return 0.5*(d_get_void_fraction_t(grid,ix,iy,iz-1,theta)+d_get_void_fraction_t(grid,ix,iy,iz,theta));
}


static __global__ void k_alpha_flux_thincwlic_x_two_way(G_StaggeredGrid* grid,double dt){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+2;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;

    MyArray<double,3>& a=grid->alpha_;
    MyArray<double,3>& vx=grid->f_vx_;
    MyArray<double,3>& Fx=grid->f_Fx_;
    MyArray<unsigned char,3>& f_xtype=grid->f_xtype_;
    MyArray<unsigned char,3>& celltype=grid->celltype_;

    double inv_dx=grid->inv_dx_;
    double inv_dy=grid->inv_dy_;
    double inv_dz=grid->inv_dz_;
    double inv_2dx=grid->inv_2dx_;
    double inv_2dy=grid->inv_2dy_;
    double inv_2dz=grid->inv_2dz_;
    double dtbydx=dt*inv_dx;

    if(ix>Nx || iy>Ny || iz>Nz) return;

    if(f_xtype(ix,iy,iz)==F_GHOST){
        Fx(ix,iy,iz)=0.0;
        return;
    }

    double vxf=vx(ix,iy,iz);
    int donorInd=vxf>0.0?ix-1:ix;
    double axf=a(donorInd,iy,iz);
    double epsf=d_get_void_fraction_half_xface(grid,ix,iy,iz);

    unsigned char ctyped=celltype(donorInd,iy,iz);
    unsigned char ctypep=celltype(donorInd+1,iy,iz);
    unsigned char ctypem=celltype(donorInd-1,iy,iz);

    if(ctyped!=C_INTERIOR){
        Fx(ix,iy,iz)=0.0;
        return;
    }

    if(f_xtype(ix,iy,iz)==F_BOUNDARY){
        int bid=grid->f_xbcid_(ix,iy,iz);
        unsigned char bcType=grid->bc_.bcType_(bid);
        if(bcType==BC_OUTLET){
            Fx(ix,iy,iz)=epsf*vxf*axf*dtbydx;
            return;
        }else{
            Fx(ix,iy,iz)=0.0;
            return;
        }
    }


    if(ctypep!=C_INTERIOR || ctypem!=C_INTERIOR){
        Fx(ix,iy,iz)=epsf*vxf*axf*dtbydx;
        return;
    }

    double gamma_x=a(donorInd+1,iy,iz)-a(donorInd-1,iy,iz);

    if(axf<EPS || axf>1.0-EPS || fabs(gamma_x)<1e-6){
        Fx(ix,iy,iz)=epsf*vxf*axf*dtbydx;
        return;
    }

    double nx=-gamma_x*inv_2dx;

    bool is_yp_interior=celltype(donorInd,iy+1,iz)==C_INTERIOR;
    bool is_ym_interior=celltype(donorInd,iy-1,iz)==C_INTERIOR;
    bool is_zp_interior=celltype(donorInd,iy,iz+1)==C_INTERIOR;
    bool is_zm_interior=celltype(donorInd,iy,iz-1)==C_INTERIOR;

    double ny=0.0;
    if(is_yp_interior && is_ym_interior){
        ny=-(a(donorInd,iy+1,iz)-a(donorInd,iy-1,iz))*inv_2dy;
    }else if(!is_yp_interior && is_ym_interior){
        ny=-(a(donorInd,iy,iz)-a(donorInd,iy-1,iz))*inv_dy;
    }else if(is_yp_interior && !is_ym_interior){
        ny=-(a(donorInd,iy+1,iz)-a(donorInd,iy,iz))*inv_dy;
    }

    double nz=0.0;
    if(is_zp_interior && is_zm_interior){
        nz=-(a(donorInd,iy,iz+1)-a(donorInd,iy,iz-1))*inv_2dz;
    }else if(!is_zp_interior && is_zm_interior){
        nz=-(a(donorInd,iy,iz)-a(donorInd,iy,iz-1))*inv_dz;
    }else if(is_zp_interior && !is_zm_interior){
        nz=-(a(donorInd,iy,iz+1)-a(donorInd,iy,iz))*inv_dz;
    }

    double s_sq=nx*nx+ny*ny+nz*nz;
    double s=sqrt(s_sq);
    double inv_s=1.0/(s+EPS);
    double theta=acos(fabs(nx)*inv_s);
    double awlic_coeff=2.0*theta/M_PI;
    double wx=1.0-awlic_coeff*awlic_coeff;
    double gamma=sgn(gamma_x);
    double xi0=find_xi0_analytic(a(donorInd,iy,iz),gamma);
    double lambda=vxf*dtbydx;
    double Fx_thinc;

    if(vxf>0.0){
        Fx_thinc=integrate_thinc(1.0-lambda,1.0,gamma,xi0);
    }else{
        Fx_thinc=-integrate_thinc(0.0,-lambda,gamma,xi0);
    }

    double Fx_upwind=lambda*axf;
    Fx(ix,iy,iz)=epsf*(wx*Fx_thinc+(1.0-wx)*Fx_upwind);
}

static __global__ void k_transport_alpha_x_two_way(G_StaggeredGrid* grid,double dt){
    const int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    const int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    const int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    MyArray<double,3>& a=grid->alpha_;
    MyArray<double,3>& a_new=grid->alpha_new_;
    MyArray<double,3>& eps=grid->void_fraction_vof_;
    MyArray<double,3>& eps_new=grid->void_fraction_vof_new_;
    const MyArray<double,3>& Fx=grid->f_Fx_;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;

    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR){
        a_new(ix,iy,iz)=0.0;
        eps_new(ix,iy,iz)=eps(ix,iy,iz);
        return;
    }

    const double eps_xp=d_get_void_fraction_half_xface(grid,ix+1,iy,iz);
    const double eps_xm=d_get_void_fraction_half_xface(grid,ix,iy,iz);
    const double up=d_get_vx_xface(grid,ix+1,iy,iz);
    const double um=d_get_vx_xface(grid,ix,iy,iz);
    const double Qp=eps_xp*up*dt*grid->inv_dx_;
    const double Qm=eps_xm*um*dt*grid->inv_dx_;
    const double liquid=eps(ix,iy,iz)*a(ix,iy,iz);
    const double liquid_new=liquid-(Fx(ix+1,iy,iz)-Fx(ix,iy,iz));
    const double eps_next=eps(ix,iy,iz)-(Qp-Qm);

    eps_new(ix,iy,iz)=eps_next;
    a_new(ix,iy,iz)=liquid_new/eps_next;
}
static __global__ void k_alpha_flux_thincwlic_y_two_way(G_StaggeredGrid* grid,double dt){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+2;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;

    MyArray<double,3>& a=grid->alpha_;
    MyArray<double,3>& vy=grid->f_vy_;
    MyArray<double,3>& Fy=grid->f_Fy_;
    MyArray<unsigned char,3>& f_ytype=grid->f_ytype_;
    MyArray<unsigned char,3>& celltype=grid->celltype_;

    double inv_dx=grid->inv_dx_;
    double inv_dy=grid->inv_dy_;
    double inv_dz=grid->inv_dz_;
    double inv_2dx=grid->inv_2dx_;
    double inv_2dy=grid->inv_2dy_;
    double inv_2dz=grid->inv_2dz_;
    double dtbydy=dt*inv_dy;

    if(ix>Nx || iy>Ny || iz>Nz) return;

    if(f_ytype(ix,iy,iz)==F_GHOST){
        Fy(ix,iy,iz)=0.0;
        return;
    }

    double vyf=vy(ix,iy,iz);
    int donorInd=vyf>0.0?iy-1:iy;
    double ayf=a(ix,donorInd,iz);
    double epsf=d_get_void_fraction_half_yface(grid,ix,iy,iz);

    unsigned char ctyped=celltype(ix,donorInd,iz);
    unsigned char ctypep=celltype(ix,donorInd+1,iz);
    unsigned char ctypem=celltype(ix,donorInd-1,iz);

    if(ctyped!=C_INTERIOR){
        Fy(ix,iy,iz)=0.0;
        return;
    }

    if(f_ytype(ix,iy,iz)==F_BOUNDARY){
        int bid=grid->f_ybcid_(ix,iy,iz);
        unsigned char bcType=grid->bc_.bcType_(bid);
        if(bcType==BC_OUTLET){
            Fy(ix,iy,iz)=epsf*vyf*ayf*dtbydy;
            return;
        }else{
            Fy(ix,iy,iz)=0.0;
            return;
        }
    }

    if(ctypep!=C_INTERIOR || ctypem!=C_INTERIOR){
        Fy(ix,iy,iz)=epsf*vyf*ayf*dtbydy;
        return;
    }


    double gamma_y=a(ix,donorInd+1,iz)-a(ix,donorInd-1,iz);

    if(ayf<EPS || ayf>1.0-EPS || fabs(gamma_y)<1e-6){
        Fy(ix,iy,iz)=epsf*vyf*ayf*dtbydy;
        return;
    }

    double ny=-gamma_y*inv_2dy;

    bool is_xp_interior=celltype(ix+1,donorInd,iz)==C_INTERIOR;
    bool is_xm_interior=celltype(ix-1,donorInd,iz)==C_INTERIOR;
    bool is_zp_interior=celltype(ix,donorInd,iz+1)==C_INTERIOR;
    bool is_zm_interior=celltype(ix,donorInd,iz-1)==C_INTERIOR;

    double nx=0.0;
    if(is_xp_interior && is_xm_interior){
        nx=-(a(ix+1,donorInd,iz)-a(ix-1,donorInd,iz))*inv_2dx;
    }else if(!is_xp_interior && is_xm_interior){
        nx=-(a(ix,donorInd,iz)-a(ix-1,donorInd,iz))*inv_dx;
    }else if(is_xp_interior && !is_xm_interior){
        nx=-(a(ix+1,donorInd,iz)-a(ix,donorInd,iz))*inv_dx;
    }

    double nz=0.0;
    if(is_zp_interior && is_zm_interior){
        nz=-(a(ix,donorInd,iz+1)-a(ix,donorInd,iz-1))*inv_2dz;
    }else if(!is_zp_interior && is_zm_interior){
        nz=-(a(ix,donorInd,iz)-a(ix,donorInd,iz-1))*inv_dz;
    }else if(is_zp_interior && !is_zm_interior){
        nz=-(a(ix,donorInd,iz+1)-a(ix,donorInd,iz))*inv_dz;
    }

    double s_sq=nx*nx+ny*ny+nz*nz;
    double s=sqrt(s_sq);
    double inv_s=1.0/(s+EPS);
    double theta=acos(fabs(ny)*inv_s);
    double awlic_coeff=2.0*theta/M_PI;
    double wy=1.0-awlic_coeff*awlic_coeff;
    double gamma=sgn(gamma_y);
    double xi0=find_xi0_analytic(a(ix,donorInd,iz),gamma);
    double lambda=vyf*dtbydy;
    double Fy_thinc;

    if(vyf>0.0){
        Fy_thinc=integrate_thinc(1.0-lambda,1.0,gamma,xi0);
    }else{
        Fy_thinc=-integrate_thinc(0.0,-lambda,gamma,xi0);
    }

    double Fy_upwind=lambda*ayf;
    Fy(ix,iy,iz)=epsf*(wy*Fy_thinc+(1.0-wy)*Fy_upwind);
}

static __global__ void k_transport_alpha_y_two_way(G_StaggeredGrid* grid,double dt){
    const int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    const int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    const int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    MyArray<double,3>& a=grid->alpha_;
    MyArray<double,3>& a_new=grid->alpha_new_;
    MyArray<double,3>& eps=grid->void_fraction_vof_;
    MyArray<double,3>& eps_new=grid->void_fraction_vof_new_;
    const MyArray<double,3>& Fy=grid->f_Fy_;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;

    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR){
        a_new(ix,iy,iz)=0.0;
        eps_new(ix,iy,iz)=eps(ix,iy,iz);
        return;
    }

    const double eps_yp=d_get_void_fraction_half_yface(grid,ix,iy+1,iz);
    const double eps_ym=d_get_void_fraction_half_yface(grid,ix,iy,iz);
    const double up=d_get_vy_yface(grid,ix,iy+1,iz);
    const double um=d_get_vy_yface(grid,ix,iy,iz);
    const double Qp=eps_yp*up*dt*grid->inv_dy_;
    const double Qm=eps_ym*um*dt*grid->inv_dy_;
    const double liquid=eps(ix,iy,iz)*a(ix,iy,iz);
    const double liquid_new=liquid-(Fy(ix,iy+1,iz)-Fy(ix,iy,iz));
    const double eps_next=eps(ix,iy,iz)-(Qp-Qm);

    eps_new(ix,iy,iz)=eps_next;
    a_new(ix,iy,iz)=liquid_new/eps_next;
}

static __global__ void k_alpha_flux_thincwlic_z_two_way(G_StaggeredGrid* grid,double dt){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+2;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;

    MyArray<double,3>& a=grid->alpha_;
    MyArray<double,3>& vz=grid->f_vz_;
    MyArray<double,3>& Fz=grid->f_Fz_;
    MyArray<unsigned char,3>& f_ztype=grid->f_ztype_;
    MyArray<unsigned char,3>& celltype=grid->celltype_;

    double inv_dx=grid->inv_dx_;
    double inv_dy=grid->inv_dy_;
    double inv_dz=grid->inv_dz_;
    double inv_2dx=grid->inv_2dx_;
    double inv_2dy=grid->inv_2dy_;
    double inv_2dz=grid->inv_2dz_;
    double dtbydz=dt*inv_dz;

    if(ix>Nx || iy>Ny || iz>Nz) return;

    if(f_ztype(ix,iy,iz)==F_GHOST){
        Fz(ix,iy,iz)=0.0;
        return;
    }

    double vzf=vz(ix,iy,iz);
    int donorInd=vzf>0.0?iz-1:iz;
    double azf=a(ix,iy,donorInd);
    double epsf=d_get_void_fraction_half_zface(grid,ix,iy,iz);

    unsigned char ctyped=celltype(ix,iy,donorInd);
    unsigned char ctypep=celltype(ix,iy,donorInd+1);
    unsigned char ctypem=celltype(ix,iy,donorInd-1);

    if(ctyped!=C_INTERIOR){
        Fz(ix,iy,iz)=0.0;
        return;
    }

    if(f_ztype(ix,iy,iz)==F_BOUNDARY){
        int bid=grid->f_zbcid_(ix,iy,iz);
        unsigned char bcType=grid->bc_.bcType_(bid);
        if(bcType==BC_OUTLET){
            Fz(ix,iy,iz)=epsf*vzf*azf*dtbydz;
            return;
        }else{
            Fz(ix,iy,iz)=0.0;
            return;
        }
    }

    if(ctypep!=C_INTERIOR || ctypem!=C_INTERIOR){
        Fz(ix,iy,iz)=epsf*vzf*azf*dtbydz;
        return;
    }


    double gamma_z=a(ix,iy,donorInd+1)-a(ix,iy,donorInd-1);

    if(azf<EPS || azf>1.0-EPS || fabs(gamma_z)<1e-6){
        Fz(ix,iy,iz)=epsf*vzf*azf*dtbydz;
        return;
    }

    double nz=-gamma_z*inv_2dz;

    bool is_xp_interior=celltype(ix+1,iy,donorInd)==C_INTERIOR;
    bool is_xm_interior=celltype(ix-1,iy,donorInd)==C_INTERIOR;
    bool is_yp_interior=celltype(ix,iy+1,donorInd)==C_INTERIOR;
    bool is_ym_interior=celltype(ix,iy-1,donorInd)==C_INTERIOR;

    double nx=0.0;
    if(is_xp_interior && is_xm_interior){
        nx=-(a(ix+1,iy,donorInd)-a(ix-1,iy,donorInd))*inv_2dx;
    }else if(!is_xp_interior && is_xm_interior){
        nx=-(a(ix,iy,donorInd)-a(ix-1,iy,donorInd))*inv_dx;
    }else if(is_xp_interior && !is_xm_interior){
        nx=-(a(ix+1,iy,donorInd)-a(ix,iy,donorInd))*inv_dx;
    }

    double ny=0.0;
    if(is_yp_interior && is_ym_interior){
        ny=-(a(ix,iy+1,donorInd)-a(ix,iy-1,donorInd))*inv_2dy;
    }else if(!is_yp_interior && is_ym_interior){
        ny=-(a(ix,iy,donorInd)-a(ix,iy-1,donorInd))*inv_dy;
    }else if(is_yp_interior && !is_ym_interior){
        ny=-(a(ix,iy+1,donorInd)-a(ix,iy,donorInd))*inv_dy;
    }

    double s_sq=nx*nx+ny*ny+nz*nz;
    double s=sqrt(s_sq);
    double inv_s=1.0/(s+EPS);
    double theta=acos(fabs(nz)*inv_s);
    double awlic_coeff=2.0*theta/M_PI;
    double wz=1.0-awlic_coeff*awlic_coeff;
    double gamma=sgn(gamma_z);
    double xi0=find_xi0_analytic(a(ix,iy,donorInd),gamma);
    double lambda=vzf*dtbydz;
    double Fz_thinc;

    if(vzf>0.0){
        Fz_thinc=integrate_thinc(1.0-lambda,1.0,gamma,xi0);
    }else{
        Fz_thinc=-integrate_thinc(0.0,-lambda,gamma,xi0);
    }

    double Fz_upwind=lambda*azf;
    Fz(ix,iy,iz)=epsf*(wz*Fz_thinc+(1.0-wz)*Fz_upwind);
}

static __global__ void k_transport_alpha_z_two_way(G_StaggeredGrid* grid,double dt){
    const int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    const int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    const int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    MyArray<double,3>& a=grid->alpha_;
    MyArray<double,3>& a_new=grid->alpha_new_;
    MyArray<double,3>& eps=grid->void_fraction_vof_;
    MyArray<double,3>& eps_new=grid->void_fraction_vof_new_;
    const MyArray<double,3>& Fz=grid->f_Fz_;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;

    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR){
        a_new(ix,iy,iz)=0.0;
        eps_new(ix,iy,iz)=eps(ix,iy,iz);
        return;
    }

    const double eps_zp=d_get_void_fraction_half_zface(grid,ix,iy,iz+1);
    const double eps_zm=d_get_void_fraction_half_zface(grid,ix,iy,iz);
    const double up=d_get_vz_zface(grid,ix,iy,iz+1);
    const double um=d_get_vz_zface(grid,ix,iy,iz);
    const double Qp=eps_zp*up*dt*grid->inv_dz_;
    const double Qm=eps_zm*um*dt*grid->inv_dz_;
    const double liquid=eps(ix,iy,iz)*a(ix,iy,iz);
    const double liquid_new=liquid-(Fz(ix,iy,iz+1)-Fz(ix,iy,iz));
    const double eps_next=eps(ix,iy,iz)-(Qp-Qm);

    eps_new(ix,iy,iz)=eps_next;
    a_new(ix,iy,iz)=liquid_new/eps_next;
}

static __global__ void k_init_void_fraction_vof_two_way(G_StaggeredGrid* grid){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;

    grid->void_fraction_vof_(ix,iy,iz)=grid->void_fraction_old_(ix,iy,iz);
}

static __global__ void k_get_eps_alpha_to_tmp_(G_StaggeredGrid* grid, MyArray<double,3> tmp){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;

    tmp(ix,iy,iz)=grid->alpha_(ix,iy,iz)*grid->void_fraction_(ix,iy,iz);
}


__global__ void k_swap_void_frac_vof(G_StaggeredGrid* grid){
    double* tmp;
    tmp =grid->void_fraction_vof_.data_; 
    grid->void_fraction_vof_.data_=grid->void_fraction_vof_new_.data_; 
    grid->void_fraction_vof_new_.data_=tmp; 
}

static __global__ void k_compute_mass_flux_from_alpha_flux_two_way(SMACSolver solv,G_StaggeredGrid* grid){
    int iz=blockIdx.z*blockDim.z+threadIdx.z;
    int iy=blockIdx.y*blockDim.y+threadIdx.y;
    int ix=blockIdx.x*blockDim.x+threadIdx.x;

    int Nx=grid->Nx_;
    int Ny=grid->Ny_;
    int Nz=grid->Nz_;

    MyArray<double,3> Fx=grid->f_Fx_accum_;
    MyArray<double,3> Fy=grid->f_Fy_accum_;
    MyArray<double,3> Fz=grid->f_Fz_accum_;
    MyArray<double,3> mfx=grid->f_mfx_;
    MyArray<double,3> mfy=grid->f_mfy_;
    MyArray<double,3> mfz=grid->f_mfz_;

    double inv_dt=solv.inv_dt_;
    double dx=grid->dx_;
    double dy=grid->dy_;
    double dz=grid->dz_;

    double rho0=solv.rho0_;
    double drho=solv.rho1_-solv.rho0_;

    double dxbydt=dx*inv_dt;
    double dybydt=dy*inv_dt;
    double dzbydt=dz*inv_dt;

    /* x mass flux: epsilon*rho*u */
    if(iy<Ny+2 && ix<Nx+3 && iz<Nz+2){
        if(ix>=1 && ix<=Nx+1){
            double epsf;
            if(ix==1){
                epsf=grid->void_fraction_half_(1,iy,iz);
            }else if(ix==Nx+1){
                epsf=grid->void_fraction_half_(Nx,iy,iz);
            }else{
                epsf=d_get_void_fraction_half_xface(grid,ix,iy,iz);
            }

            double q=epsf*d_get_vx_xface(grid,ix,iy,iz);
            double alpha_q=Fx(ix,iy,iz)*dxbydt;
            mfx(ix,iy,iz)=rho0*q+drho*alpha_q;
        }else{
            mfx(ix,iy,iz)=0.0;
        }
    }

    /* y mass flux: epsilon*rho*v */
    if(iy<Ny+3 && ix<Nx+2 && iz<Nz+2){
        if(iy>=1 && iy<=Ny+1){
            double epsf;
            if(iy==1){
                epsf=grid->void_fraction_half_(ix,1,iz);
            }else if(iy==Ny+1){
                epsf=grid->void_fraction_half_(ix,Ny,iz);
            }else{
                epsf=d_get_void_fraction_half_yface(grid,ix,iy,iz);
            }

            double q=epsf*d_get_vy_yface(grid,ix,iy,iz);
            double alpha_q=Fy(ix,iy,iz)*dybydt;
            mfy(ix,iy,iz)=rho0*q+drho*alpha_q;
        }else{
            mfy(ix,iy,iz)=0.0;
        }
    }

    /* z mass flux: epsilon*rho*w */
    if(iy<Ny+2 && ix<Nx+2 && iz<Nz+3){
        if(iz>=1 && iz<=Nz+1){
            double epsf;
            if(iz==1){
                epsf=grid->void_fraction_half_(ix,iy,1);
            }else if(iz==Nz+1){
                epsf=grid->void_fraction_half_(ix,iy,Nz);
            }else{
                epsf=d_get_void_fraction_half_zface(grid,ix,iy,iz);
            }

            double q=epsf*d_get_vz_zface(grid,ix,iy,iz);
            double alpha_q=Fz(ix,iy,iz)*dzbydt;
            mfz(ix,iy,iz)=rho0*q+drho*alpha_q;
        }else{
            mfz(ix,iy,iz)=0.0;
        }
    }
}

void G_SMACSolver::init_void_fraction_vof_two_way(){
    k_init_void_fraction_vof_two_way<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
}

void G_SMACSolver::alpha_flux_thincwlic_split_two_way(double dt,int steps){





    if (steps%3 == 0){
        k_alpha_flux_thincwlic_x_two_way<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_x_two_way<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);

        std::swap(grid_.void_fraction_vof_.data_,grid_.void_fraction_vof_new_.data_);
        k_swap_void_frac_vof<<<1,1>>>(grid_.d_ptr_);



        k_alpha_flux_thincwlic_y_two_way<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_y_two_way<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);

        std::swap(grid_.void_fraction_vof_.data_,grid_.void_fraction_vof_new_.data_);
        k_swap_void_frac_vof<<<1,1>>>(grid_.d_ptr_);


        k_alpha_flux_thincwlic_z_two_way<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_z_two_way<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);

        std::swap(grid_.void_fraction_vof_.data_,grid_.void_fraction_vof_new_.data_);
        k_swap_void_frac_vof<<<1,1>>>(grid_.d_ptr_);

    }else if(steps%3 == 1){

        k_alpha_flux_thincwlic_y_two_way<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_y_two_way<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);

        std::swap(grid_.void_fraction_vof_.data_,grid_.void_fraction_vof_new_.data_);
        k_swap_void_frac_vof<<<1,1>>>(grid_.d_ptr_);



        k_alpha_flux_thincwlic_z_two_way<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_z_two_way<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);

        std::swap(grid_.void_fraction_vof_.data_,grid_.void_fraction_vof_new_.data_);
        k_swap_void_frac_vof<<<1,1>>>(grid_.d_ptr_);

        k_alpha_flux_thincwlic_x_two_way<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_x_two_way<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);

        std::swap(grid_.void_fraction_vof_.data_,grid_.void_fraction_vof_new_.data_);
        k_swap_void_frac_vof<<<1,1>>>(grid_.d_ptr_);


    }else{


        k_alpha_flux_thincwlic_z_two_way<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_z_two_way<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);

        std::swap(grid_.void_fraction_vof_.data_,grid_.void_fraction_vof_new_.data_);
        k_swap_void_frac_vof<<<1,1>>>(grid_.d_ptr_);

        k_alpha_flux_thincwlic_x_two_way<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_x_two_way<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);

        std::swap(grid_.void_fraction_vof_.data_,grid_.void_fraction_vof_new_.data_);
        k_swap_void_frac_vof<<<1,1>>>(grid_.d_ptr_);

        k_alpha_flux_thincwlic_y_two_way<<<grid_dim_,block_dim_ >>>(grid_.d_ptr_, dt);
        k_transport_alpha_y_two_way<<<grid_dim_, block_dim_>>>(grid_.d_ptr_,dt);
        /* == swap == */
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);

        std::swap(grid_.void_fraction_vof_.data_,grid_.void_fraction_vof_new_.data_);
        k_swap_void_frac_vof<<<1,1>>>(grid_.d_ptr_);

    }


    /* debug */
    cub::DeviceReduce::Max(
            cub_temp_storage_,
            cub_temp_storage_bytes_,
            grid_.alpha_.data_,
            d_r2_,
            grid_.alpha_.size_);
    double max_alpha = 0.0;
    cudaMemcpy(&max_alpha,d_r2_,sizeof(double),cudaMemcpyDeviceToHost);
    printf("max_alpha before redistribution = %f\n",max_alpha);


}

void G_SMACSolver::compute_mass_flux_from_alpha_flux_two_way(SMACSolver solv){
    k_compute_mass_flux_from_alpha_flux_two_way<<<grid_dim_,block_dim_>>>(solv,grid_.d_ptr_);

}

/* ============================================================ */
/* === Conservative bounded liquid redistribution for CFD-DEM === */
/* ============================================================ */

static __device__ __forceinline__ bool d_is_non_excess_two_way(
    G_StaggeredGrid* grid,int ix,int iy,int iz,double tol){

    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR) return false;

    double q=grid->alpha_(ix,iy,iz);
    double eps=grid->void_fraction_(ix,iy,iz);

    return q<=eps+tol;
}

static __device__ __forceinline__ int d_count_non_excess_neighbors_two_way(
    G_StaggeredGrid* grid,int ix,int iy,int iz,double tol){

    int n=0;

    n+=d_is_non_excess_two_way(grid,ix-1,iy,iz,tol);
    n+=d_is_non_excess_two_way(grid,ix+1,iy,iz,tol);
    n+=d_is_non_excess_two_way(grid,ix,iy-1,iz,tol);
    n+=d_is_non_excess_two_way(grid,ix,iy+1,iz,tol);
    n+=d_is_non_excess_two_way(grid,ix,iy,iz-1,tol);
    n+=d_is_non_excess_two_way(grid,ix,iy,iz+1,tol);

    return n;
}

static __global__ void k_prepare_liquid_redistribution_two_way(G_StaggeredGrid* grid){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;
    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR) return;

    grid->alpha_(ix,iy,iz)*=grid->void_fraction_vof_(ix,iy,iz);
}

static __global__ void k_make_liquid_redistribution_flux_x_two_way(
    G_StaggeredGrid* grid,double tol){

    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>=grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;

    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR) return;
    if(grid->celltype_(ix+1,iy,iz)!=C_INTERIOR) return;

    double qL=grid->alpha_(ix,iy,iz);
    double qR=grid->alpha_(ix+1,iy,iz);

    double epsL=grid->void_fraction_(ix,iy,iz);
    double epsR=grid->void_fraction_(ix+1,iy,iz);

    double excessL=fmax(qL-epsL,0.0);
    double excessR=fmax(qR-epsR,0.0);

    double flux=0.0;

    if(excessL>tol && excessR<=tol){
        int n=d_count_non_excess_neighbors_two_way(grid,ix,iy,iz,tol);
        if(n>0) flux=excessL/(double)n;
    }else if(excessR>tol && excessL<=tol){
        int n=d_count_non_excess_neighbors_two_way(grid,ix+1,iy,iz,tol);
        if(n>0) flux=-excessR/(double)n;
    }

    grid->f_Fx_(ix+1,iy,iz)=flux;
}

static __global__ void k_make_liquid_redistribution_flux_y_two_way(
    G_StaggeredGrid* grid,double tol){

    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>=grid->Ny_ || iz>grid->Nz_) return;

    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR) return;
    if(grid->celltype_(ix,iy+1,iz)!=C_INTERIOR) return;

    double qL=grid->alpha_(ix,iy,iz);
    double qR=grid->alpha_(ix,iy+1,iz);

    double epsL=grid->void_fraction_(ix,iy,iz);
    double epsR=grid->void_fraction_(ix,iy+1,iz);

    double excessL=fmax(qL-epsL,0.0);
    double excessR=fmax(qR-epsR,0.0);

    double flux=0.0;

    if(excessL>tol && excessR<=tol){
        int n=d_count_non_excess_neighbors_two_way(grid,ix,iy,iz,tol);
        if(n>0) flux=excessL/(double)n;
    }else if(excessR>tol && excessL<=tol){
        int n=d_count_non_excess_neighbors_two_way(grid,ix,iy+1,iz,tol);
        if(n>0) flux=-excessR/(double)n;
    }

    grid->f_Fy_(ix,iy+1,iz)=flux;
}

static __global__ void k_make_liquid_redistribution_flux_z_two_way(
    G_StaggeredGrid* grid,double tol){

    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>=grid->Nz_) return;

    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR) return;
    if(grid->celltype_(ix,iy,iz+1)!=C_INTERIOR) return;

    double qL=grid->alpha_(ix,iy,iz);
    double qR=grid->alpha_(ix,iy,iz+1);

    double epsL=grid->void_fraction_(ix,iy,iz);
    double epsR=grid->void_fraction_(ix,iy,iz+1);

    double excessL=fmax(qL-epsL,0.0);
    double excessR=fmax(qR-epsR,0.0);

    double flux=0.0;

    if(excessL>tol && excessR<=tol){
        int n=d_count_non_excess_neighbors_two_way(grid,ix,iy,iz,tol);
        if(n>0) flux=excessL/(double)n;
    }else if(excessR>tol && excessL<=tol){
        int n=d_count_non_excess_neighbors_two_way(grid,ix,iy,iz+1,tol);
        if(n>0) flux=-excessR/(double)n;
    }

    grid->f_Fz_(ix,iy,iz+1)=flux;
}

static __global__ void k_update_liquid_redistribution_two_way(G_StaggeredGrid* grid){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;
    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR) return;

    double divF=
        grid->f_Fx_(ix+1,iy,iz)-grid->f_Fx_(ix,iy,iz)
       +grid->f_Fy_(ix,iy+1,iz)-grid->f_Fy_(ix,iy,iz)
       +grid->f_Fz_(ix,iy,iz+1)-grid->f_Fz_(ix,iy,iz);

    grid->alpha_new_(ix,iy,iz)=grid->alpha_(ix,iy,iz)-divF;
}


static __global__ void k_accum_liquid_redistribution_flux_two_way(G_StaggeredGrid* grid){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;
    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR) return;

    grid->f_Fx_accum_(ix+1,iy,iz)+=grid->f_Fx_(ix+1,iy,iz);
    grid->f_Fy_accum_(ix,iy+1,iz)+=grid->f_Fy_(ix,iy+1,iz);
    grid->f_Fz_accum_(ix,iy,iz+1)+=grid->f_Fz_(ix,iy,iz+1);
}

static __global__ void k_finalize_alpha_from_liquid_two_way(G_StaggeredGrid* grid){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;
    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR) return;

    const double q=grid->alpha_(ix,iy,iz);
    const double eps=grid->void_fraction_(ix,iy,iz);


    grid->alpha_(ix,iy,iz)=q/eps;
}

static __global__ void k_get_liquid_excess_two_way(
    G_StaggeredGrid* grid,MyArray<double,3> tmp){

    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;

    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR){
        tmp(ix,iy,iz)=0.0;
        return;
    }

    double q=grid->alpha_(ix,iy,iz);
    double eps=grid->void_fraction_(ix,iy,iz);

    tmp(ix,iy,iz)=fmax(q-eps,0.0);
}

static __global__ void k_get_liquid_to_tmp_two_way(G_StaggeredGrid* grid,MyArray<double,3> tmp){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;

    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR){
        tmp(ix,iy,iz)=0.0;
        return;
    }

    tmp(ix,iy,iz)=grid->alpha_(ix,iy,iz);
}

static __global__ void k_snapshot_liquid_two_way(G_StaggeredGrid* grid){
    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;
    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR) return;

    grid->alpha_new_(ix,iy,iz)=grid->alpha_(ix,iy,iz);
}

static __device__ __forceinline__ bool d_has_liquid_neighbor_snapshot_two_way(
    G_StaggeredGrid* grid,int ix,int iy,int iz,double alpha_min){

    const int dx[6]={-1,1,0,0,0,0};
    const int dy[6]={0,0,-1,1,0,0};
    const int dz[6]={0,0,0,0,-1,1};

    for(int n=0;n<6;n++){
        int jx=ix+dx[n];
        int jy=iy+dy[n];
        int jz=iz+dz[n];

        if(grid->celltype_(jx,jy,jz)!=C_INTERIOR) continue;

        double q=grid->alpha_new_(jx,jy,jz);
        double eps=grid->void_fraction_(jx,jy,jz);

        if(eps<=0.0) continue;

        if(q/eps>alpha_min) return true;
    }

    return false;
}

static __global__ void k_get_interface_capacity_two_way(
    G_StaggeredGrid* grid,MyArray<double,3> tmp,double alpha_min,double alpha_max){

    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;

    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR){
        tmp(ix,iy,iz)=0.0;
        return;
    }

    double q=grid->alpha_new_(ix,iy,iz);
    double eps=grid->void_fraction_(ix,iy,iz);

    if(eps<=0.0){
        tmp(ix,iy,iz)=0.0;
        return;
    }

    double alpha=q/eps;

    if(alpha>alpha_min && alpha<alpha_max){
        tmp(ix,iy,iz)=fmax(eps-q,0.0);
    }else{
        tmp(ix,iy,iz)=0.0;
    }
}

static __global__ void k_get_adjacent_gas_capacity_two_way(
    G_StaggeredGrid* grid,MyArray<double,3> tmp,double alpha_min){

    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;

    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR){
        tmp(ix,iy,iz)=0.0;
        return;
    }

    double q=grid->alpha_new_(ix,iy,iz);
    double eps=grid->void_fraction_(ix,iy,iz);

    if(eps<=0.0){
        tmp(ix,iy,iz)=0.0;
        return;
    }

    double alpha=q/eps;

    if(alpha<=alpha_min &&
       d_has_liquid_neighbor_snapshot_two_way(grid,ix,iy,iz,alpha_min)){
        tmp(ix,iy,iz)=fmax(eps-q,0.0);
    }else{
        tmp(ix,iy,iz)=0.0;
    }
}

static __global__ void k_priority_interface_cleanup_two_way(
    G_StaggeredGrid* grid,double lambda_interface,double lambda_gas,
    double alpha_min,double alpha_max){

    int ix=blockIdx.x*blockDim.x+threadIdx.x+1;
    int iy=blockIdx.y*blockDim.y+threadIdx.y+1;
    int iz=blockIdx.z*blockDim.z+threadIdx.z+1;

    if(ix>grid->Nx_ || iy>grid->Ny_ || iz>grid->Nz_) return;
    if(grid->celltype_(ix,iy,iz)!=C_INTERIOR) return;

    double q=grid->alpha_new_(ix,iy,iz);
    double eps=grid->void_fraction_(ix,iy,iz);

    if(eps<=0.0) return;

    /* Remove all remaining excess from donor cells. */
    if(q>eps){
        grid->alpha_(ix,iy,iz)=eps;
        return;
    }

    double alpha=q/eps;
    double capacity=eps-q;

    /* First priority: existing interface cells. */
    if(alpha>alpha_min && alpha<alpha_max){
        grid->alpha_(ix,iy,iz)=q+lambda_interface*capacity;
        return;
    }

    /* Second priority: gas cells adjacent to liquid. */
    if(alpha<=alpha_min &&
       d_has_liquid_neighbor_snapshot_two_way(grid,ix,iy,iz,alpha_min)){
        grid->alpha_(ix,iy,iz)=q+lambda_gas*capacity;
        return;
    }

    grid->alpha_(ix,iy,iz)=q;
}

void G_SMACSolver::finalize_alpha_two_way(){
    constexpr int max_iter=16;
    constexpr double tol=1e-12;
    constexpr double alpha_interface_min=0.01;
    constexpr double alpha_interface_max=0.99;

    k_prepare_liquid_redistribution_two_way<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);

    /* Check liquid volume before redistribution. */
    cudaMemset(grid_.array_tmp_.data_,0,sizeof(double)*grid_.array_tmp_.size_);
    k_get_liquid_to_tmp_two_way<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,grid_.array_tmp_);

    cub::DeviceReduce::Sum(
        cub_temp_storage_,
        cub_temp_storage_bytes_,
        grid_.array_tmp_.data_,
        d_r2_,
        grid_.array_tmp_.size_);

    double total_before=0.0;
    cudaMemcpy(&total_before,d_r2_,sizeof(double),cudaMemcpyDeviceToHost);

    double total_excess=0.0;
    int iter=0;

    /* Local conservative redistribution. */
    for(iter=0;iter<max_iter;iter++){
        cudaMemset(grid_.array_tmp_.data_,0,sizeof(double)*grid_.array_tmp_.size_);
        k_get_liquid_excess_two_way<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,grid_.array_tmp_);

        cub::DeviceReduce::Sum(
            cub_temp_storage_,
            cub_temp_storage_bytes_,
            grid_.array_tmp_.data_,
            d_r2_,
            grid_.array_tmp_.size_);

        cudaMemcpy(&total_excess,d_r2_,sizeof(double),cudaMemcpyDeviceToHost);

        if(total_excess<=tol) break;

        clear_alpha_flux();

        k_make_liquid_redistribution_flux_x_two_way<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,tol);
        k_make_liquid_redistribution_flux_y_two_way<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,tol);
        k_make_liquid_redistribution_flux_z_two_way<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,tol);

        k_accum_liquid_redistribution_flux_two_way<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);

        k_update_liquid_redistribution_two_way<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);
        std::swap(grid_.alpha_.data_,grid_.alpha_new_.data_);
        k_swap_alpha<<<1,1>>>(grid_.d_ptr_);
    }

    /* Recompute excess after the last local iteration. */
    cudaMemset(grid_.array_tmp_.data_,0,sizeof(double)*grid_.array_tmp_.size_);
    k_get_liquid_excess_two_way<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,grid_.array_tmp_);

    cub::DeviceReduce::Sum(
        cub_temp_storage_,
        cub_temp_storage_bytes_,
        grid_.array_tmp_.data_,
        d_r2_,
        grid_.array_tmp_.size_);

    cudaMemcpy(&total_excess,d_r2_,sizeof(double),cudaMemcpyDeviceToHost);

    double total_capacity=0.0;
    double lambda=0.0;

    double total_interface_capacity=0.0;
    double total_gas_capacity=0.0;
    double lambda_interface=0.0;
    double lambda_gas=0.0;
    bool cleanup=false;

    /* Direct cleanup of the remaining excess.
     * Priority:
     *   1. existing interface cells
     *   2. gas cells adjacent to liquid
     */
    if(total_excess>tol){

        /* Freeze the state used for receiver classification. */
        k_snapshot_liquid_two_way<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);

        /* First priority: existing interface cells. */
        cudaMemset(grid_.array_tmp_.data_,0,sizeof(double)*grid_.array_tmp_.size_);

        k_get_interface_capacity_two_way<<<grid_dim_,block_dim_>>>(
                grid_.d_ptr_,
                grid_.array_tmp_,
                alpha_interface_min,
                alpha_interface_max);

        cub::DeviceReduce::Sum(
                cub_temp_storage_,
                cub_temp_storage_bytes_,
                grid_.array_tmp_.data_,
                d_r2_,
                grid_.array_tmp_.size_);

        cudaMemcpy(
                &total_interface_capacity,
                d_r2_,
                sizeof(double),
                cudaMemcpyDeviceToHost);

        if(total_interface_capacity>=total_excess && total_interface_capacity>0.0){

            /* Existing interface cells can absorb all excess. */
            lambda_interface=total_excess/total_interface_capacity;
            lambda_gas=0.0;
            cleanup=true;

        }else{

            /* Existing interface cells are filled first. */
            double remaining=total_excess-total_interface_capacity;

            lambda_interface=total_interface_capacity>0.0 ? 1.0 : 0.0;

            /* Second priority: gas cells adjacent to liquid. */
            cudaMemset(grid_.array_tmp_.data_,0,sizeof(double)*grid_.array_tmp_.size_);

            k_get_adjacent_gas_capacity_two_way<<<grid_dim_,block_dim_>>>(
                    grid_.d_ptr_,
                    grid_.array_tmp_,
                    alpha_interface_min);

            cub::DeviceReduce::Sum(
                    cub_temp_storage_,
                    cub_temp_storage_bytes_,
                    grid_.array_tmp_.data_,
                    d_r2_,
                    grid_.array_tmp_.size_);

            cudaMemcpy(
                    &total_gas_capacity,
                    d_r2_,
                    sizeof(double),
                    cudaMemcpyDeviceToHost);

            if(total_gas_capacity>=remaining && total_gas_capacity>0.0){
                lambda_gas=remaining/total_gas_capacity;
                cleanup=true;
            }else{
                printf(
                        "WARNING: insufficient cleanup capacity: "
                        "excess=%.7e interface=%.7e gas=%.7e\n",
                        total_excess,
                        total_interface_capacity,
                        total_gas_capacity);
            }
        }

        if(cleanup){
            k_priority_interface_cleanup_two_way<<<grid_dim_,block_dim_>>>(
                    grid_.d_ptr_,
                    lambda_interface,
                    lambda_gas,
                    alpha_interface_min,
                    alpha_interface_max);
        }
    }
   
    /* Check remaining excess after interface cleanup. */
    double final_excess=0.0;

    cudaMemset(grid_.array_tmp_.data_,0,sizeof(double)*grid_.array_tmp_.size_);
    k_get_liquid_excess_two_way<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,grid_.array_tmp_);

    cub::DeviceReduce::Sum(
            cub_temp_storage_,
            cub_temp_storage_bytes_,
            grid_.array_tmp_.data_,
            d_r2_,
            grid_.array_tmp_.size_);

    cudaMemcpy(&final_excess,d_r2_,sizeof(double),cudaMemcpyDeviceToHost);

    /* Convert q back to alpha. */
    k_finalize_alpha_from_liquid_two_way<<<grid_dim_,block_dim_>>>(grid_.d_ptr_);

    k_clip_alpha<<<grid_dim_, block_dim_>>>(grid_.d_ptr_);

    /* Check total liquid volume after redistribution. */
    cudaMemset(grid_.array_tmp_.data_,0,sizeof(double)*grid_.array_tmp_.size_);
    k_get_eps_alpha_to_tmp_<<<grid_dim_,block_dim_>>>(grid_.d_ptr_,grid_.array_tmp_);

    cub::DeviceReduce::Sum(
            cub_temp_storage_,
            cub_temp_storage_bytes_,
            grid_.array_tmp_.data_,
            d_r2_,
            grid_.array_tmp_.size_);

    double total_after=0.0;
    cudaMemcpy(&total_after,d_r2_,sizeof(double),cudaMemcpyDeviceToHost);

    printf(
            "liquid redistribution: iter=%d local_excess=%.7e final_excess=%.7e\n "
            "capacity=%.7e lambda=%.7e before=%.7e after=%.7e diff=%.3e\n\n",
            iter,total_excess,final_excess,total_capacity,lambda,
            total_before,total_after,total_after-total_before);
}
