#include "SMACSolver.h"
#include "hardCodedParameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>

static void check_array_finite(
    const char* name,
    const double* a,
    int size
)
{
    double amin = a[0];
    double amax = a[0];

    for (int k = 0; k < size; k++) {
        if (!isfinite(a[k])) {
            printf("NaN/Inf detected in %s at k=%d, value=%e\n",
                   name, k, a[k]);
            abort();
        }

        if (a[k] < amin) amin = a[k];
        if (a[k] > amax) amax = a[k];
    }

    printf("%s min=%e max=%e\n", name, amin, amax);
}

void SMACSolver::check_nan_all(const char* tag)
{
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;

    printf("==== check_nan_all: %s ====\n", tag);

    check_array_finite("alpha",   grid_.alpha_,   (Nx+2)*(Ny+2));
    check_array_finite("rho",     grid_.rho_,     (Nx+2)*(Ny+2));
    check_array_finite("inv_rho", grid_.inv_rho_, (Nx+2)*(Ny+2));
    check_array_finite("mu",      grid_.mu_,      (Nx+2)*(Ny+2));

    check_array_finite("vx",      grid_.vx_,      (Nx+3)*(Ny+2));
    check_array_finite("vy",      grid_.vy_,      (Nx+2)*(Ny+3));
    check_array_finite("vx_star", grid_.vx_star_, (Nx+3)*(Ny+2));
    check_array_finite("vy_star", grid_.vy_star_, (Nx+2)*(Ny+3));

    check_array_finite("p",       grid_.p_,       (Nx+2)*(Ny+2));
    check_array_finite("rhs",     grid_.rhs_,     (Nx+2)*(Ny+2));
}

/* == function for thinc == */
static double integrate_thinc(double a, double b, double gamma,double xi){
    double result = 0.5*(b-a)+gamma/(2.*BETA)*(log(cosh(BETA*(b-xi)))-log(cosh(BETA*(a-xi))));

    return result;
}


static double find_xi0_analytic(double alpha, double gamma){
    double A=exp(BETA*gamma*(2.*alpha-1.));

    double result = 1./(2.*BETA)*log((exp(BETA)-A)/(A-exp(-BETA)));

    return result;
}


/* == misc functions == */
static double max_abs_double(double *array, int size){

    if (array == NULL || size <= 0){
        printf("Null Array at max_double function!!!\n");
        abort();
        return 0.0;
    }

    double max =fabs(array[0]);

    for(int i=1; i<size; i++){
        double value = fabs(array[i]);
        max= max<value? value:max;
    }

    return max;
}

static inline double sgn(double a){
    double result;
    if(a<0){
        result =-1.;
    }else{
        result =1.;
    }

    return result;
}

/* ================
 * Solver Class 
 * ===================
 */


void SMACSolver::alpha_flux_upwind(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    double* a = grid_.alpha_;

    double* vx = grid_.vx_;
    double* vy = grid_.vy_;

    double* Fx = grid_.Fx_;
    double* Fy = grid_.Fy_;

    double div_max=0.0;
    double inv_dx = grid_.inv_dx_;
    double inv_dy = grid_.inv_dy_;
    double dt=dt_;

    int pitch_vx=Nx+3;
    int pitch_vy=Nx+2;
    int pitch_a=Nx+2;

    double dtbydx = dt* inv_dx;
    double dtbydy = dt* inv_dy;

    /* == vx ==*/
    for (int i=1; i<Ny+1; i++){
        for (int j=1; j<Nx; j++){

            int indf = i*pitch_vx+j;
            /* == x direction == */
            /* == upwind == */
            double vxf = vx[indf];
            double axf= vxf>0. ? a[i*pitch_a+j]:a[i*pitch_a+j+1];


            Fx[indf]= vxf*axf*dtbydx;
        }
    }

    /* == vy ==*/
    for (int i=1; i<Ny; i++){
        for (int j=1; j<Nx+1; j++){

            int indf = i*pitch_vy+j;
            /* == y direction == */
            /* == upwind == */

            double vyf = vy[indf];
            double ayf= vyf>0. ? a[i*pitch_a+j]:a[(i+1)*pitch_a+j];


            Fy[indf]= vyf*ayf*dtbydy;
        }
    }
}

void SMACSolver::update_properties_by_alpha_initial(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int pitch_c = Nx+2;

    const double rho1 = rho1_;
    const double rho0 = rho0_;

    const double mu1 = mu1_;
    const double mu0 = mu0_;

    double* a = grid_.alpha_;
    double* rho = grid_.rho_;
    double* rho_old = grid_.rho_old_;
    double* inv_rho = grid_.inv_rho_;
    double* mu = grid_.mu_;

    /* == update rho == */
    for (int i=0; i<Ny+2; i++){
        for (int j=0; j<Nx+2; j++){
            int ind = i*pitch_c+j;
            double alpha = a[ind];
            rho[ind] = (1.-alpha)*rho0+alpha*rho1;
        }
    }

    for (int i=0; i<Ny+2; i++){
        for (int j=0; j<Nx+2; j++){
            int ind = i*pitch_c+j;
            rho_old[ind] = rho[ind];
        }
    }

    /* == update inv_rho == */
    for (int i=0; i<Ny+2; i++){
        for (int j=0; j<Nx+2; j++){
            int ind = i*pitch_c+j;
            inv_rho[ind] = 1./rho[ind];
        }
    }

    /* == update mu == */
    for (int i=0; i<Ny+2; i++){
        for (int j=0; j<Nx+2; j++){
            int ind = i*pitch_c+j;
            double alpha = a[ind];
            mu[ind] = (1.-alpha)*mu0+alpha*mu1;
        }
    }

    /* == update inv rho at face== */
    /* == x faces == */
    int pitch_fx = Nx+3;
    double* const f_bx = grid_.f_bx_;

    for (int i=0; i<Ny+2; i++){
        for (int j=0; j<Nx+1; j++){
            int ind = i*pitch_fx+j;
            int ind1 = i*pitch_c+j;
            int ind0 = i*pitch_c+j+1;
            f_bx[ind] = (inv_rho[ind1]+inv_rho[ind0])*0.5;
        }
    }

    /* == y faces == */
    int pitch_fy = Nx+2;
    double* const f_by = grid_.f_by_;

    for (int i=0; i<Ny+1; i++){
        for (int j=0; j<Nx+2; j++){
            int ind = i*pitch_fy+j;
            int ind1 = i*pitch_c+j;
            int ind0 = (i+1)*pitch_c+j;
            f_by[ind] = (inv_rho[ind1]+inv_rho[ind0])*0.5;
        }
    }

    /* == update rho at face== */

    /* == x faces == */

    double* const f_rhox = grid_.f_rhox_;
    for (int i=0; i<Ny+2; i++){
        for (int j=0; j<Nx+1; j++){
            int ind = i*pitch_fx+j;
            int ind1 = i*pitch_c+j;
            int ind0 = i*pitch_c+j+1;
            f_rhox[ind] = 1./f_bx[ind];
        }
    }

    /* == y faces == */

    double* const f_rhoy = grid_.f_rhoy_;
    for (int i=0; i<Ny+1; i++){
        for (int j=0; j<Nx+2; j++){
            int ind = i*pitch_fy+j;
            int ind1 = i*pitch_c+j;
            int ind0 = (i+1)*pitch_c+j;
            f_rhoy[ind] = 1./f_by[ind];
        }
    }
}

void SMACSolver::update_properties_by_alpha(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int pitch_c = Nx+2;

    const double rho1 = rho1_;
    const double rho0 = rho0_;

    const double mu1 = mu1_;
    const double mu0 = mu0_;

    double* a = grid_.alpha_;
    double* rho = grid_.rho_;
    double* inv_rho = grid_.inv_rho_;
    double* mu = grid_.mu_;

    /* == update rho == */
    for (int i=0; i<Ny+2; i++){
        for (int j=0; j<Nx+2; j++){
            int ind = i*pitch_c+j;
            double alpha = a[ind];
            rho[ind] = (1.-alpha)*rho0+alpha*rho1;
        }
    }

    /* == update inv_rho == */
    for (int i=0; i<Ny+2; i++){
        for (int j=0; j<Nx+2; j++){
            int ind = i*pitch_c+j;
            inv_rho[ind] = 1./rho[ind];
        }
    }

    /* == update mu == */
    for (int i=0; i<Ny+2; i++){
        for (int j=0; j<Nx+2; j++){
            int ind = i*pitch_c+j;
            double alpha = a[ind];
            mu[ind] = (1.-alpha)*mu0+alpha*mu1;
        }
    }

    /* == update inv rho at face== */
    /* == x faces == */
    int pitch_fx = Nx+3;
    double* const f_bx = grid_.f_bx_;

    for (int i=0; i<Ny+2; i++){
        for (int j=0; j<Nx+1; j++){
            int ind = i*pitch_fx+j;
            int ind1 = i*pitch_c+j;
            int ind0 = i*pitch_c+j+1;
            f_bx[ind] = (inv_rho[ind1]+inv_rho[ind0])*0.5;
        }
    }

    /* == y faces == */
    int pitch_fy = Nx+2;
    double* const f_by = grid_.f_by_;

    for (int i=0; i<Ny+1; i++){
        for (int j=0; j<Nx+2; j++){
            int ind = i*pitch_fy+j;
            int ind1 = i*pitch_c+j;
            int ind0 = (i+1)*pitch_c+j;
            f_by[ind] = (inv_rho[ind1]+inv_rho[ind0])*0.5;
        }
    }

    /* == update rho at face== */

    /* == x faces == */

    double* const f_rhox = grid_.f_rhox_;
    for (int i=0; i<Ny+2; i++){
        for (int j=0; j<Nx+1; j++){
            int ind = i*pitch_fx+j;
            int ind1 = i*pitch_c+j;
            int ind0 = i*pitch_c+j+1;
            f_rhox[ind] = 1./f_bx[ind];
        }
    }

    /* == y faces == */

    double* const f_rhoy = grid_.f_rhoy_;
    for (int i=0; i<Ny+1; i++){
        for (int j=0; j<Nx+2; j++){
            int ind = i*pitch_fy+j;
            int ind1 = i*pitch_c+j;
            int ind0 = (i+1)*pitch_c+j;
            f_rhoy[ind] = 1./f_by[ind];
        }
    }
}

void SMACSolver::alpha_flux_thincwlic_x(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    double* a = grid_.alpha_;

    double* vx = grid_.vx_;

    double* Fx = grid_.Fx_;

    double inv_dx = grid_.inv_dx_;
    double inv_2dx = grid_.inv_2dx_;
    double inv_dy = grid_.inv_dy_;
    double inv_2dy = grid_.inv_2dy_;
    double dt=dt_;

    int pitch_vx=Nx+3;
    int pitch_vy=Nx+2;
    int pitch_a=Nx+2;

    double dtbydx = dt* inv_dx;

    for (int i=1; i<Ny+1; i++){
        for (int j=1; j<Nx; j++){

            /* right */
            int indf = i*pitch_vx+j;
            double vxf = vx[indf];

            int donorInd = vxf> 0. ? j:j+1;
            double axf=  a[i*pitch_a+donorInd];

            double gamma=(a[i*pitch_a+donorInd+1]-a[i*pitch_a+donorInd-1]);
            if (axf < EPS || axf > 1. -EPS || fabs(gamma)<1e-6){// check if the donor cell is an interface
                /* == upwind == */
                Fx[indf]= vxf*axf*dtbydx;
                continue;

            }
            

            /* == thinc/wlic == */
            double nx = -1.*gamma*inv_2dx;
            double ny= -1.*(a[(i+1)*pitch_a+donorInd]-a[(i-1)*pitch_a+donorInd])*inv_2dy;

            double nx_abs = fabs(nx);
            double ny_abs = fabs(ny);

            double s = nx_abs + ny_abs + EPS;
            double inv_s = 1./s;

            double wx = nx_abs*inv_s;
            double wy = 1. - wx;

            gamma = sgn(gamma);

            double xi0 = find_xi0_analytic(a[i*pitch_a+donorInd],gamma);
            double lambda = vxf*dtbydx;

            double Fx_thinc = vxf > 0. ? integrate_thinc(1.-lambda, 1., gamma, xi0):-integrate_thinc(0.0, -lambda, gamma, xi0);

                /* == upwind == */
            double  Fx_upwind = lambda*axf;

            Fx[i*pitch_vx+j] = wx*Fx_thinc + wy*Fx_upwind;


        }
    }
}

void SMACSolver::alpha_flux_thincwlic_y(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    double* a = grid_.alpha_;

    double* vy = grid_.vy_;

    double* Fy = grid_.Fy_;

    double inv_dy = grid_.inv_dy_;
    double inv_2dy = grid_.inv_2dy_;
    double inv_dx = grid_.inv_dx_;
    double inv_2dx = grid_.inv_2dx_;
    double dt=dt_;

    int pitch_vx=Nx+3;
    int pitch_vy=Nx+2;
    int pitch_a=Nx+2;

    double dtbydy = dt* inv_dy;

    for (int i=1; i<Ny; i++){
        for (int j=1; j<Nx+1; j++){

            int indf = i*pitch_vy+j;
            double vyf = vy[indf];
            int donorInd = vyf> 0. ? i:i+1;
            double ayf=  a[donorInd*pitch_a+j];

            double gamma=(a[(donorInd+1)*pitch_a+j]-a[(donorInd-1)*pitch_a+j]);
            if (ayf < EPS || ayf > 1. -EPS || fabs(gamma)<1e-6){// check if the donor cell is an interface
                /* == upwind == */
                Fy[indf]= vyf*ayf*dtbydy;
                continue;
            }

            /* == thinc/wlic == */
            double nx = -1.*(a[(donorInd*pitch_a+j+1)]-a[(donorInd*pitch_a+j-1)])*inv_2dx;
            double ny= -1.*gamma*inv_2dy;

            double nx_abs = fabs(nx);
            double ny_abs = fabs(ny);

            double s = nx_abs + ny_abs + EPS;
            double inv_s = 1./s;

            double wx = nx_abs*inv_s;
            double wy = 1. - wx;

            /* == thinc == */
            gamma = sgn(gamma);

            double xi0 = find_xi0_analytic(a[donorInd*pitch_a+j],gamma);
            double lambda = vyf*dtbydy;

            double Fy_thinc = vyf > 0. ? integrate_thinc(1.-lambda, 1., gamma, xi0):-integrate_thinc(0.0, -lambda, gamma, xi0);

            double Fy_upwind = lambda*ayf;

            Fy[i*pitch_vy+j] = wx*Fy_upwind + wy*Fy_thinc;

        }
    }
}

void SMACSolver::alpha_flux_thinc_x(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    double* a = grid_.alpha_;

    double* vx = grid_.vx_;

    double* Fx = grid_.Fx_;

    double div_max=0.0;
    double inv_dx = grid_.inv_dx_;
    double dt=dt_;

    int pitch_vx=Nx+3;
    int pitch_vy=Nx+2;
    int pitch_a=Nx+2;

    double dtbydx = dt* inv_dx;

    for (int i=1; i<Ny+1; i++){
        for (int j=1; j<Nx; j++){

            /* right */
            int indf = i*pitch_vx+j;
            double vxf = vx[indf];

            int donorInd = vxf> 0. ? j:j+1;
            double axf=  a[i*pitch_a+donorInd];

            double diffa=(a[i*pitch_a+j+1]-a[i*pitch_a+j]);
            if (axf < EPS || axf > 1. -EPS || fabs(diffa)<1e-6){// check if the donor cell is an interface
                /* == upwind == */
                Fx[indf]= vxf*axf*dtbydx;
                continue;

            }else{
                if (donorInd <2 || donorInd > Nx-1){ /* uses upwind for adjacent boundary cell */
                    /* == upwind == */
                    Fx[indf]= vxf*axf*dtbydx;
                    continue;
                }
            }

            /* == thinc == */
            double gamma = a[i*pitch_a+donorInd+1] - a[i*pitch_a+donorInd-1];
            gamma = sgn(gamma);

            double xi0 = find_xi0_analytic(a[i*pitch_a+donorInd],gamma);
            double lambda = vxf*dtbydx;

            Fx[i*pitch_vx+j] = vxf > 0. ? integrate_thinc(1.-lambda, 1., gamma, xi0):-integrate_thinc(0.0, -lambda, gamma, xi0);


        }
    }
}


void SMACSolver::alpha_flux_thinc_y(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    double* a = grid_.alpha_;

    double* vy = grid_.vy_;

    double* Fy = grid_.Fy_;

    double div_max=0.0;
    double inv_dy = grid_.inv_dy_;
    double dt=dt_;

    int pitch_vx=Nx+3;
    int pitch_vy=Nx+2;
    int pitch_a=Nx+2;

    double dtbydy = dt* inv_dy;

    for (int i=1; i<Ny; i++){
        for (int j=1; j<Nx+1; j++){

            int indf = i*pitch_vy+j;
            double vyf = vy[indf];
            int donorInd = vyf> 0. ? i:i+1;
            double ayf=  a[donorInd*pitch_a+j];

            double diffa=(a[(i+1)*pitch_a+j]-a[i*pitch_a+j]);
            if (ayf < EPS || ayf > 1. -EPS || fabs(diffa)<1e-6){// check if the donor cell is an interface
                /* == upwind == */
                Fy[indf]= vyf*ayf*dtbydy;
                continue;

            }else{
                if (donorInd <2 || donorInd > Ny-1){ /* uses upwind for adjacent boundary cell */
                    /* == upwind == */
                    Fy[indf]= vyf*ayf*dtbydy;
                    continue;
                }
            }

            /* == thinc == */
            double gamma = a[(donorInd+1)*pitch_a+j] - a[(donorInd-1)*pitch_a+j];
            gamma = sgn(gamma);

            double xi0 = find_xi0_analytic(a[donorInd*pitch_a+j],gamma);
            double lambda = vyf*dtbydy;

            Fy[i*pitch_vy+j] = vyf > 0. ? integrate_thinc(1.-lambda, 1., gamma, xi0):-integrate_thinc(0.0, -lambda, gamma, xi0);


        }
    }
}

void SMACSolver::transport_alpha(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    double* a = grid_.alpha_;
    double* a_new = grid_.alpha_new_;
    double* const Fx = grid_.Fx_;
    double* const Fy = grid_.Fy_;

    double div_max=0.0;

    int pitch_Fx=Nx+3;
    int pitch_Fy=Nx+2;
    int pitch_a=Nx+2;

    for (int i=1; i<Ny+1; i++){
        for (int j=1; j<Nx+1; j++){
            double flux = 0.;

            /* == x direction == */
            flux += Fx[i*pitch_Fx+j] - Fx[i*pitch_Fx+j-1];

            /* == y direction == */
            flux += Fy[i*pitch_Fy+j] - Fy[(i-1)*pitch_Fy+j];

            a_new[i*pitch_a+j] = a[i*pitch_a+j] - flux;

            /* clipping */
            if (a_new[i*pitch_a+j]<EPS){
                a_new[i*pitch_a+j]=0.;
            }else if (a_new[i*pitch_a+j]>1.){
                a_new[i*pitch_a+j]=1.;
            }

        }
    }
    /* == swap == */
    grid_.alpha_=a_new;
    grid_.alpha_new_=a;
}

void SMACSolver::check_divergence(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    double* p = grid_.p_;
    double* vx = grid_.vx_;
    double* vy = grid_.vy_;

    double div_max=0.0;
    double inv_dx = grid_.inv_dx_;
    double inv_dy = grid_.inv_dy_;

    for (int i=1; i<Ny+1; i++){
        for (int j=1; j<Nx+1; j++){
            int pitch_vx = Nx+3;
            int pitch_vy = Nx+2;

            double div = (vx[i*pitch_vx+j+1] - vx[i*pitch_vx+j])*inv_dx
                + (vy[(i+1)*pitch_vy+j]-vy[(i)*pitch_vy+j])*inv_dy;

            div = fabs(div);
            div_max = div_max<div? div:div_max; 
        }
    }

    printf("div_max = %e , div_max*dt = %e\n", div_max, div_max*dt_);
}

void SMACSolver::correct_vof_velocity(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    double* p = grid_.p_;
    double* vx = grid_.vx_;
    double* vy = grid_.vy_;
    double* vx_star = grid_.vx_star_;
    double* vy_star = grid_.vy_star_;
    double* f_bx = grid_.f_bx_;
    double* f_by = grid_.f_by_;

    double inv_dx = grid_.inv_dx_;
    double inv_dy = grid_.inv_dy_;

    double dtInvdx= dt_*inv_dx;

    /* == fix vx == */
    for (int i=1; i<Ny+1; i++){
        for (int j=1; j<Nx; j++){
            int ind_vx = i*(Nx+3)+j;
            vx[ind_vx] = vx_star[ind_vx] - f_bx[ind_vx]*dtInvdx*(p[(i)*(Nx+2)+j+1]-p[i*(Nx+2)+j]);
        }
    }

    double dtInvdy = dt_*inv_dy;
    /* == fix vy == */
    for (int i=1; i<Ny; i++){
        for (int j=1; j<Nx+1; j++){
            int ind_vy = i*(Nx+2)+j;
            vy[ind_vy] = vy_star[ind_vy] - f_by[ind_vy]*dtInvdy*(p[(i+1)*(Nx+2)+j]-p[i*(Nx+2)+j]);
        }
    }

}
void SMACSolver::correct_velocity(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    double* p = grid_.p_;
    double* vx = grid_.vx_;
    double* vy = grid_.vy_;
    double* vx_star = grid_.vx_star_;
    double* vy_star = grid_.vy_star_;

    double inv_dx = grid_.inv_dx_;
    double inv_dy = grid_.inv_dy_;

    double dtInvdxByRho = dt_*inv_dx/rho_;

    /* == fix vx == */
    for (int i=1; i<Ny+1; i++){
        for (int j=1; j<Nx; j++){
            int ind_vx = i*(Nx+3)+j;
            vx[ind_vx] = vx_star[ind_vx] - dtInvdxByRho*(p[(i)*(Nx+2)+j+1]-p[i*(Nx+2)+j]);
        }
    }

    double dtInvdyByRho = dt_*inv_dy/rho_;
    /* == fix vy == */
    for (int i=1; i<Ny; i++){
        for (int j=1; j<Nx+1; j++){
            int ind_vy = i*(Nx+2)+j;
            vy[ind_vy] = vy_star[ind_vy] - dtInvdyByRho*(p[(i+1)*(Nx+2)+j]-p[i*(Nx+2)+j]);
        }
    }

}

void SMACSolver::set_calc_properties(double rho, double dt,double u_lid, double nu, double sizex, double sizey, int Nx, int Ny, int Nz){
    rho_=rho;
    u_lid_=u_lid;
    nu_=nu;
    grid_.sizex_=sizex;
    grid_.sizey_=sizey;
    grid_.sizez_=sizez;
    grid_.Nx_=Nx;
    grid_.Ny_=Ny;
    grid_.Nz_=Nz;
    dt_=dt;
    inv_dt_=1./dt;

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

    grid_.v_b_1_ = 2.*u_lid;
    grid_.v_b_2_ = 0.;
}

void SMACSolver::compute_mass_flux_from_alpha_flux(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;

    int pitch_vx = Nx + 3;
    int pitch_vy = Nx + 2;

    double* vx  = grid_.vx_;
    double* vy  = grid_.vy_;
    double* Fx  = grid_.Fx_;
    double* Fy  = grid_.Fy_;
    double* mfx = grid_.f_mfx_;
    double* mfy = grid_.f_mfy_;

    double dt = dt_;
    double inv_dt = inv_dt_;
    double dx = grid_.dx_;
    double dy = grid_.dy_;

    double drho = rho1_ - rho0_;

    double dxbydt = dx*inv_dt;
    double dybydt = dy*inv_dt;

    for (int i = 0; i < Ny + 2; i++) {
        for (int j = 0; j < Nx + 3; j++) {
            int ind = i * pitch_vx + j;

            double q = vx[ind];                    // u_f
            double alpha_q = Fx[ind] * dxbydt; // u_f * alpha_f

            mfx[ind] = rho0_ * q + drho * alpha_q; // rho*u
        }
    }

    for (int i = 0; i < Ny + 3; i++) {
        for (int j = 0; j < Nx + 2; j++) {
            int ind = i * pitch_vy + j;

            double q = vy[ind];                    // v_f
            double alpha_q = Fy[ind] * dybydt; // v_f * alpha_f

            mfy[ind] = rho0_ * q + drho * alpha_q; // rho*v
        }
    }
}

void SMACSolver::clear_alpha_flux()
{
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;

    int pitch_vx = Nx + 3;
    int pitch_vy = Nx + 2;

    std::memset(grid_.Fx_, 0, sizeof(double) * pitch_vx * (Ny + 2));
    std::memset(grid_.Fy_, 0, sizeof(double) * pitch_vy * (Ny + 3));
}

void SMACSolver::build_vof_poisson_Ap(const double *p){
    double* const Ap = grid_.pcg_Ap_;
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;

    double inv_dx2 = grid_.inv_dx2_;
    double inv_dy2 = grid_.inv_dy2_;


    double* const f_bx = grid_.f_bx_;
    double* const f_by = grid_.f_by_;


    int pitch_vx=Nx+3;
    int pitch_vy=Nx+2;
    int pitch_c=Nx+2;

    for (int i=1; i<Ny+1;i++){
        for (int j=1; j<Nx+1;j++){
            int ind=i*pitch_c+j;

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
    }

}

void SMACSolver::build_vof_poisson_invdiag(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;

    double inv_dx2 = grid_.inv_dx2_;
    double inv_dy2 = grid_.inv_dy2_;

    double* const f_bx = grid_.f_bx_;
    double* const f_by = grid_.f_by_;

    double* const invdiag = grid_.pcg_invDiag_;

    int pitch_vx=Nx+3;
    int pitch_vy=Nx+2;
    int pitch_c=Nx+2;

    for (int i=1; i<Ny+1;i++){
        for (int j=1; j<Nx+1;j++){
            int ind=i*pitch_c+j;


            double b_xp = f_bx[i*pitch_vx+j];
            double b_xm = f_bx[i*pitch_vx+j-1];
            double b_yp = f_by[i*pitch_vy+j];
            double b_ym = f_by[(i-1)*pitch_vy+j];

            double diag = (b_xp+b_xm)*inv_dx2 + (b_yp+b_ym)*inv_dy2;

            invdiag[ind] = 1.0/diag;
        }
    }

}

void SMACSolver::subtract_cell_mean(double *p)
{
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int pitch = Nx + 2;


    double sum = 0.0;
    int count = 0;

    for (int i = 1; i <= Ny; ++i) {
        for (int j = 1; j <= Nx; ++j) {
            sum += p[i*pitch + j];
            count++;
        }
    }

    double mean = sum / (double)count;

    for (int i = 1; i <= Ny; ++i) {
        for (int j = 1; j <= Nx; ++j) {
            p[i*pitch + j] -= mean;
        }
    }

    this->set_boundary_array(p);
}

void SMACSolver::solve_vof_poisson_pcg(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;

    int pitch_c =Nx+2;


    double inv_dy = grid_.inv_dy_;
    double inv_2dx = grid_.inv_2dx_;
    double inv_2dy = grid_.inv_2dy_;
    double inv_dx2 = grid_.inv_dx2_;
    double inv_dy2 = grid_.inv_dy2_;

    double* const p       = grid_.p_;
    double* const rhs     = grid_.rhs_;

    double* const r       = grid_.pcg_r_;
    double* const z       = grid_.pcg_z_;
    double* const dir     = grid_.pcg_dir_;
    double* const Ap      = grid_.pcg_Ap_;
    double* const invdiag = grid_.pcg_invDiag_;

    int max_iter = 100000;
    double tol = 1e-6;

    this->make_poisson_rhs();

    this->shift_pressure_reference();

    this->build_vof_poisson_invdiag();
    this->build_vof_poisson_Ap(p);

    double mean_b = 0.0;
    int count = 0;

    for (int i=1; i<Ny+1; i++){
        for (int j=1; j<Nx+1; j++){
            int ind = i*pitch_c + j;

            mean_b += -rhs[ind];
            count++;
        }
    }

    mean_b /= (double)count;

    double rz_old =0.0;
    double norm_b2=0.0;
    /* initial residual; r = b- Ap */
    for (int i=1; i<Ny+1; i++){
        for (int j=1; j<Nx+1; j++){
            int ind = i*pitch_c + j;
            double b = -rhs[ind] - mean_b;

            r[ind] = b-Ap[ind];
            norm_b2 += b*b;
        }
    }


    this->subtract_cell_mean(r);

    /* z = M^{-1} r */

    for (int i = 1; i <= Ny; ++i) {
        for (int j = 1; j <= Nx; ++j) {
            int ind = i*pitch_c + j;
            z[ind] = invdiag[ind] * r[ind];
        }
    }

    this->subtract_cell_mean(z);

    for (int i = 1; i <= Ny; ++i) {
        for (int j = 1; j <= Nx; ++j) {
            int id = i*pitch_c + j;
            dir[id] = z[id];
        }
    }

    this->set_boundary_array(dir);

    /* rz_old = r dot z */
    rz_old = 0.0;

    for (int i = 1; i <= Ny; ++i) {
        for (int j = 1; j <= Nx; ++j) {
            int id = i*pitch_c + j;
            rz_old += r[id] * z[id];
        }
    }

    if (norm_b2 < 1.0e-16) {
        norm_b2 = 1.0;
    }

    double norm_b = sqrt(norm_b2);

    int iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {

        /*  Ap = A dir */
        this->set_boundary_array(dir);
        this->build_vof_poisson_Ap(dir);

        double pAp = 0.0;

        for (int i = 1; i <= Ny; ++i) {
            for (int j = 1; j <= Nx; ++j) {
                int id = i*pitch_c + j;
                pAp += dir[id] * Ap[id];
            }
        }

        if (fabs(pAp) < 1.0e-300) {
            break;
        }

        double alpha = rz_old / pAp;

        /*
           p = p + alpha dir
           r = r - alpha Ap
           */
        //double r2 = 0.0;

        for (int i = 1; i <= Ny; ++i) {
            for (int j = 1; j <= Nx; ++j) {

                int id = i*pitch_c + j;

                p[id] += alpha * dir[id];
                r[id] -= alpha * Ap[id];

         //       r2 += r[id] * r[id];
            }
        }

        this->subtract_cell_mean(r);

        /* get residual norm */
        double r2 = 0.0;

        for (int i = 1; i <= Ny; ++i) {
            for (int j = 1; j <= Nx; ++j) {
                int id = i*pitch_c + j;
                r2 += r[id] * r[id];
            }
        }

        double rel_res = sqrt(r2) / norm_b;

        if (rel_res < tol) {
            iter++;
            break;
        }

        /*
           z = M^{-1} r
           */
        for (int i = 1; i <= Ny; ++i) {
            for (int j = 1; j <= Nx; ++j) {
                int id = i*pitch_c + j;
                z[id] = invdiag[id] * r[id];
            }
        }

        this->subtract_cell_mean(z);

        double rz_new = 0.0;

        for (int i = 1; i <= Ny; ++i) {
            for (int j = 1; j <= Nx; ++j) {
                int id = i*pitch_c + j;
                rz_new += r[id] * z[id];
            }
        }

        if (fabs(rz_old) < 1.0e-300) {
            break;
        }

        double beta = rz_new / rz_old;

        /*
           dir = z + beta dir
           */
        for (int i = 1; i <= Ny; ++i) {
            for (int j = 1; j <= Nx; ++j) {
                int id = i*pitch_c + j;
                dir[id] = z[id] + beta * dir[id];
            }
        }

        this->subtract_cell_mean(dir);

        rz_old = rz_new;
    }

    //this->set_boundary_pressure();
    this->shift_pressure_reference();

    printf("PCG iter = %d\n", iter);
}

void SMACSolver::get_vof_ustar_rhouu_upwind_consistent(){
    double inv_dx = grid_.inv_dx_;
    double inv_dy = grid_.inv_dy_;
    double inv_2dx = grid_.inv_2dx_;
    double inv_2dy = grid_.inv_2dy_;
    double inv_dx2 = grid_.inv_dx2_;
    double inv_dy2 = grid_.inv_dy2_;
    double dt= dt_;

    double gx = gx_;
    double gy = gy_;

    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;

    double* const mfx = grid_.f_mfx_;
    double* const mfy = grid_.f_mfy_;

    double* const rho = grid_.rho_;
    double* const mu = grid_.mu_;

    double* const vx = grid_.vx_;
    double* const vy = grid_.vy_;

    double* const vx_star = grid_.vx_star_;
    double* const vy_star = grid_.vy_star_;


    int pitch_vx=Nx+3;
    int pitch_vy=Nx+2;
    int pitch_c=Nx+2;

    /* == vx == */
    for (int i=1; i<Ny+1;i++){
        for (int j=1; j<Nx;j++){
            double vx_21 =vx[i*(pitch_vx)+j+1];
            double vx_11 =vx[i*(pitch_vx)+j];
            double vx_11_2 =2.*vx_11;
            double vx_01 =vx[i*(pitch_vx)+j-1];
            double vx_12 =vx[(i+1)*(pitch_vx)+j];
            double vx_10 =vx[(i-1)*(pitch_vx)+j];
            double vx_02 =vx[(i-1)*(pitch_vx)+j+1];

            double vy_21 = vy[i*(pitch_vy)+j+1];
            double vy_11 = vy[i*(pitch_vy)+j];
            double vy_11_2 = 2.*vy_11;
            double vy_01 = vy[(i)*(pitch_vy)+j-1];
            double vy_10 = vy[(i-1)*(pitch_vy)+j];
            double vy_12 = vy[(i+1)*(pitch_vy)+j];
            double vy_20 = vy[(i-1)*(pitch_vy)+j+1];

            /* === vx === */
            double tmp_vx = 0.;
            /* viscous */
            tmp_vx +=  (vx_21-vx_11_2+vx_01)*inv_dx2;
            tmp_vx +=  (vx_12-vx_11_2+vx_10)*inv_dy2;

            /* calculate face nu */
            double f_mu= (mu[(i)*(pitch_c)+j]+mu[i*pitch_c+j+1])*0.5;
            tmp_vx *= f_mu;

            /* convection */
            double vx_xp= 0.5*(mfx[i*pitch_vx+j]+mfx[i*pitch_vx+j+1]);
            double vx_xm= 0.5*(mfx[i*pitch_vx+j-1]+mfx[i*pitch_vx+j]);


            /* == upwind == */
            double ux_xp= vx_xp > 0. ? vx[i*pitch_vx+j]: vx[i*pitch_vx+j+1];
            double ux_xm= vx_xm > 0. ? vx[i*pitch_vx+j-1]: vx[i*pitch_vx+j];

            double M_xp = vx_xp * ux_xp;
            double M_xm = vx_xm * ux_xm;

            tmp_vx -= (M_xp - M_xm)*inv_dx;

            /* == y direction == */
            double vy_yp= 0.5*(mfy[i*pitch_vy+j]+mfy[(i)*pitch_vy+j+1]);
            double vy_ym= 0.5*(mfy[(i-1)*pitch_vy+j]+mfy[(i-1)*pitch_vy+j+1]);


            /* == upwind == */
            double uy_yp= vy_yp > 0. ? vx[i*pitch_vx+j]: vx[(i+1)*pitch_vx+j];
            double uy_ym= vy_ym > 0. ? vx[(i-1)*pitch_vx+j]: vx[(i)*pitch_vx+j];

            double M_yp = vy_yp * uy_yp;
            double M_ym = vy_ym * uy_ym;

            tmp_vx -= (M_yp - M_ym)*inv_dy;

            double div_m = (vx_xp-vx_xm)*inv_dx + (vy_yp-vy_ym)*inv_dy;

            tmp_vx += vx_11*div_m;

            double f_inv_rho = 1./(0.5*(rho[i*pitch_c+j]+rho[i*pitch_c+j+1]));
            vx_star[i*(pitch_vx)+j]=vx_11+dt*(f_inv_rho*tmp_vx+gx);
        }
    }

    /* == vy == */
    for (int i=1; i<Ny;i++){
        for (int j=1; j<Nx+1;j++){
            double vx_11 =vx[i*(pitch_vx)+j];
            double vx_11_2 =2.*vx_11;
            double vx_01 =vx[i*(pitch_vx)+j-1];
            double vx_12 =vx[(i+1)*(pitch_vx)+j];
            double vx_02 =vx[(i+1)*(pitch_vx)+j-1];

            double vy_21 = vy[i*(pitch_vy)+j+1];
            double vy_11 = vy[i*(pitch_vy)+j];
            double vy_11_2 = 2.*vy_11;
            double vy_01 = vy[(i)*(pitch_vy)+j-1];
            double vy_10 = vy[(i-1)*(pitch_vy)+j];
            double vy_12 = vy[(i+1)*(pitch_vy)+j];

            double tmp_vy = 0.;
            /* viscous */

            tmp_vy +=  (vy_21-vy_11_2+vy_01)*inv_dx2;
            tmp_vy +=  (vy_12-vy_11_2+vy_10)*inv_dy2;

            /* calculate face nu */
            double f_mu = (mu[(i)*(pitch_c)+j]+mu[(i+1)*pitch_c+j])*0.5;
            tmp_vy *= f_mu;

            /* convection */
            double vy_yp = 0.5*(mfy[i*pitch_vy+j]+mfy[(i+1)*pitch_vy+j]);
            double vy_ym = 0.5*(mfy[i*pitch_vy+j]+mfy[(i-1)*pitch_vy+j]);

            /* == upwind == */
            double uy_yp= vy_yp > 0. ? vy[i*pitch_vy+j]: vy[(i+1)*pitch_vy+j];
            double uy_ym= vy_ym > 0. ? vy[(i-1)*pitch_vy+j]: vy[i*pitch_vy+j];

            double M_yp = vy_yp * uy_yp;
            double M_ym = vy_ym * uy_ym;

            tmp_vy -= (M_yp - M_ym)*inv_dy;

            /* == x direction == */
            double vx_xp = 0.5*(mfx[i*pitch_vx+j]+mfx[(i+1)*pitch_vx+j]);
            double vx_xm = 0.5*(mfx[i*pitch_vx+j-1]+mfx[(i+1)*pitch_vx+j-1]);

            /* == upwind == */
            double ux_xp= vx_xp > 0. ? vy[i*pitch_vy+j]: vy[(i)*pitch_vy+j+1];
            double ux_xm= vx_xm > 0. ? vy[(i)*pitch_vy+j-1]: vy[(i)*pitch_vy+j];

            double M_xp = vx_xp * ux_xp;
            double M_xm = vx_xm * ux_xm;

            tmp_vy -= (M_xp - M_xm)*inv_dx;

            double div_m = (vx_xp-vx_xm)*inv_dx + (vy_yp-vy_ym)*inv_dy;

            tmp_vy += vy_11*div_m;

            double f_inv_rho = 1./(0.5*(rho[i*pitch_c+j]+rho[(i+1)*pitch_c+j]));
            vy_star[i*(pitch_vy)+j]=vy_11+dt*(f_inv_rho*tmp_vy+gy);
        }
    }
}

void SMACSolver::get_vof_ustar_rhouu_upwind(){
    double inv_dx = grid_.inv_dx_;
    double inv_dy = grid_.inv_dy_;
    double inv_2dx = grid_.inv_2dx_;
    double inv_2dy = grid_.inv_2dy_;
    double inv_dx2 = grid_.inv_dx2_;
    double inv_dy2 = grid_.inv_dy2_;
    double dt= dt_;

    double gx = gx_;
    double gy = gy_;

    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;

    double* const rho = grid_.rho_;
    double* const mu = grid_.mu_;

    double* const vx = grid_.vx_;
    double* const vy = grid_.vy_;

    double* const vx_star = grid_.vx_star_;
    double* const vy_star = grid_.vy_star_;


    int pitch_vx=Nx+3;
    int pitch_vy=Nx+2;
    int pitch_c=Nx+2;

    /* == vx == */
    for (int i=1; i<Ny+1;i++){
        for (int j=1; j<Nx;j++){
            double vx_21 =vx[i*(pitch_vx)+j+1];
            double vx_11 =vx[i*(pitch_vx)+j];
            double vx_11_2 =2.*vx_11;
            double vx_01 =vx[i*(pitch_vx)+j-1];
            double vx_12 =vx[(i+1)*(pitch_vx)+j];
            double vx_10 =vx[(i-1)*(pitch_vx)+j];
            double vx_02 =vx[(i-1)*(pitch_vx)+j+1];

            double vy_21 = vy[i*(pitch_vy)+j+1];
            double vy_11 = vy[i*(pitch_vy)+j];
            double vy_11_2 = 2.*vy_11;
            double vy_01 = vy[(i)*(pitch_vy)+j-1];
            double vy_10 = vy[(i-1)*(pitch_vy)+j];
            double vy_12 = vy[(i+1)*(pitch_vy)+j];
            double vy_20 = vy[(i-1)*(pitch_vy)+j+1];

            /* === vx === */
            double tmp_vx = 0.;
            /* viscous */
            tmp_vx +=  (vx_21-vx_11_2+vx_01)*inv_dx2;
            tmp_vx +=  (vx_12-vx_11_2+vx_10)*inv_dy2;

            /* calculate face nu */
            double f_mu= (mu[(i)*(pitch_c)+j]+mu[i*pitch_c+j+1])*0.5;
            tmp_vx *= f_mu;

            /* convection */
            double vx_xp= 0.5*(vx_11+vx_21)*rho[i*pitch_c+j+1];
            double vx_xm= 0.5*(vx_01+vx_11)*rho[i*pitch_c+j];


            /* == upwind == */
            double ux_xp= vx_xp > 0. ? vx[i*pitch_vx+j]: vx[i*pitch_vx+j+1];
            double ux_xm= vx_xm > 0. ? vx[i*pitch_vx+j-1]: vx[i*pitch_vx+j];

            double M_xp = vx_xp * ux_xp;
            double M_xm = vx_xm * ux_xm;

            tmp_vx -= (M_xp - M_xm)*inv_dx;

            /* == y direction == */
            double rho_yp = 0.25*(rho[i*pitch_c+j]+rho[i*pitch_c+j+1]+rho[(i+1)*pitch_c+j+1]+rho[(i+1)*pitch_c+j]);
            double rho_ym = 0.25*(rho[i*pitch_c+j]+rho[i*pitch_c+j+1]+rho[(i-1)*pitch_c+j+1]+rho[(i-1)*pitch_c+j]);

            double vy_yp= 0.5*(vy_11+vy_21)*rho_yp;
            double vy_ym= 0.5*(vy_10+vy_20)*rho_ym;


            /* == upwind == */
            double uy_yp= vy_yp > 0. ? vx[i*pitch_vx+j]: vx[(i+1)*pitch_vx+j];
            double uy_ym= vy_ym > 0. ? vx[(i-1)*pitch_vx+j]: vx[(i)*pitch_vx+j];

            double M_yp = vy_yp * uy_yp;
            double M_ym = vy_ym * uy_ym;

            tmp_vx -= (M_yp - M_ym)*inv_dy;

            double f_inv_rho = 1./(0.5*(rho[i*pitch_c+j]+rho[i*pitch_c+j+1]));
            vx_star[i*(pitch_vx)+j]=vx_11+dt*(f_inv_rho*tmp_vx+gx);
        }
    }

    /* == vy == */
    for (int i=1; i<Ny;i++){
        for (int j=1; j<Nx+1;j++){
            double vx_11 =vx[i*(pitch_vx)+j];
            double vx_11_2 =2.*vx_11;
            double vx_01 =vx[i*(pitch_vx)+j-1];
            double vx_12 =vx[(i+1)*(pitch_vx)+j];
            double vx_02 =vx[(i+1)*(pitch_vx)+j-1];

            double vy_21 = vy[i*(pitch_vy)+j+1];
            double vy_11 = vy[i*(pitch_vy)+j];
            double vy_11_2 = 2.*vy_11;
            double vy_01 = vy[(i)*(pitch_vy)+j-1];
            double vy_10 = vy[(i-1)*(pitch_vy)+j];
            double vy_12 = vy[(i+1)*(pitch_vy)+j];

            double tmp_vy = 0.;
            /* viscous */

            tmp_vy +=  (vy_21-vy_11_2+vy_01)*inv_dx2;
            tmp_vy +=  (vy_12-vy_11_2+vy_10)*inv_dy2;

            /* calculate face nu */
            double f_mu = (mu[(i)*(pitch_c)+j]+mu[(i+1)*pitch_c+j])*0.5;
            tmp_vy *= f_mu;

            /* convection */
            double vy_yp = 0.5*(vy_11+vy_12)*rho[(i+1)*pitch_c+j];
            double vy_ym = 0.5*(vy_10+vy_11)*rho[(i)*pitch_c+j];

            /* == upwind == */
            double uy_yp= vy_yp > 0. ? vy[i*pitch_vy+j]: vy[(i+1)*pitch_vy+j];
            double uy_ym= vy_ym > 0. ? vy[(i-1)*pitch_vy+j]: vy[i*pitch_vy+j];

            double M_yp = vy_yp * uy_yp;
            double M_ym = vy_ym * uy_ym;

            tmp_vy -= (M_yp - M_ym)*inv_dy;

            /* == x direction == */
            double rho_xp = 0.25*(rho[i*pitch_c+j]+rho[(i+1)*pitch_c+j]+rho[(i+1)*pitch_c+j+1]+rho[(i)*pitch_c+j+1]);
            double rho_xm = 0.25*(rho[i*pitch_c+j]+rho[i*pitch_c+j-1]+rho[(i+1)*pitch_c+j-1]+rho[(i+1)*pitch_c+j]);

            double vx_xp = 0.5*(vx_12 + vx_11)*rho_xp;
            double vx_xm = 0.5*(vx_02 + vx_01)*rho_xm;

            /* == upwind == */
            double ux_xp= vx_xp > 0. ? vy[i*pitch_vy+j]: vy[(i)*pitch_vy+j+1];
            double ux_xm= vx_xm > 0. ? vy[(i)*pitch_vy+j-1]: vy[(i)*pitch_vy+j];

            double M_xp = vx_xp * ux_xp;
            double M_xm = vx_xm * ux_xm;

            tmp_vy -= (M_xp - M_xm)*inv_dx;

            double f_inv_rho = 1./(0.5*(rho[i*pitch_c+j]+rho[(i+1)*pitch_c+j]));
            vy_star[i*(pitch_vy)+j]=vy_11+dt*(f_inv_rho*tmp_vy+gy);
        }
    }
}

void SMACSolver::get_vof_ustar(){
    double inv_dx = grid_.inv_dx_;
    double inv_dy = grid_.inv_dy_;
    double inv_2dx = grid_.inv_2dx_;
    double inv_2dy = grid_.inv_2dy_;
    double inv_dx2 = grid_.inv_dx2_;
    double inv_dy2 = grid_.inv_dy2_;
    double dt= dt_;

    double gx = gx_;
    double gy = gy_;

    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;

    double* const mu = grid_.mu_;

    double* const vx = grid_.vx_;
    double* const vy = grid_.vy_;

    double* const vx_star = grid_.vx_star_;
    double* const vy_star = grid_.vy_star_;

    double* const f_bx = grid_.f_bx_;
    double* const f_by = grid_.f_by_;

    int pitch_vx=Nx+3;
    int pitch_vy=Nx+2;
    int pitch_c=Nx+2;

    /* == vx == */
    for (int i=1; i<Ny+1;i++){
        for (int j=1; j<Nx;j++){
            double vx_21 =vx[i*(pitch_vx)+j+1];
            double vx_11 =vx[i*(pitch_vx)+j];
            double vx_11_2 =2.*vx_11;
            double vx_01 =vx[i*(pitch_vx)+j-1];
            double vx_12 =vx[(i+1)*(pitch_vx)+j];
            double vx_10 =vx[(i-1)*(pitch_vx)+j];
            double vx_02 =vx[(i-1)*(pitch_vx)+j+1];

            double vy_21 = vy[i*(pitch_vy)+j+1];
            double vy_11 = vy[i*(pitch_vy)+j];
            double vy_11_2 = 2.*vy_11;
            double vy_01 = vy[(i)*(pitch_vy)+j-1];
            double vy_10 = vy[(i-1)*(pitch_vy)+j];
            double vy_12 = vy[(i+1)*(pitch_vy)+j];
            double vy_20 = vy[(i-1)*(pitch_vy)+j+1];

            /* === vx === */
            double tmp_vx = 0.;
            /* viscous */
            tmp_vx +=  (vx_21-vx_11_2+vx_01)*inv_dx2;
            tmp_vx +=  (vx_12-vx_11_2+vx_10)*inv_dy2;

            /* calculate face nu */
            double nu = (mu[(i)*(pitch_c)+j]+mu[i*pitch_c+j+1])*0.5*f_bx[i*pitch_vx+j];
            tmp_vx *= nu;

            /* convection */
            double vxe= 0.5*(vx_11+vx_21);
            double vxw= 0.5*(vx_01+vx_11);
            tmp_vx -= (vxe*vxe-vxw*vxw)*inv_dx;

            double vxn= 0.5*(vx_11+vx_12);
            double vyn= 0.5*(vy_11+vy_21);
            double vxs= 0.5*(vx_10+vx_11);
            double vys= 0.5*(vy_10+vy_20);

            tmp_vx -= (vxn*vyn-vxs*vys)*inv_dy;
            vx_star[i*(pitch_vx)+j]=vx_11+dt*(tmp_vx+gx);
        }
    }

    for (int i=1; i<Ny;i++){
        for (int j=1; j<Nx+1;j++){
            double vx_11 =vx[i*(pitch_vx)+j];
            double vx_11_2 =2.*vx_11;
            double vx_01 =vx[i*(pitch_vx)+j-1];
            double vx_12 =vx[(i+1)*(pitch_vx)+j];
            double vx_02 =vx[(i+1)*(pitch_vx)+j-1];

            double vy_21 = vy[i*(pitch_vy)+j+1];
            double vy_11 = vy[i*(pitch_vy)+j];
            double vy_11_2 = 2.*vy_11;
            double vy_01 = vy[(i)*(pitch_vy)+j-1];
            double vy_10 = vy[(i-1)*(pitch_vy)+j];
            double vy_12 = vy[(i+1)*(pitch_vy)+j];

            double tmp_vy = 0.;
            /* viscous */

            tmp_vy +=  (vy_21-vy_11_2+vy_01)*inv_dx2;
            tmp_vy +=  (vy_12-vy_11_2+vy_10)*inv_dy2;

            /* calculate face nu */
            double nu = (mu[(i)*(pitch_c)+j]+mu[(i+1)*pitch_c+j])*0.5*f_by[i*pitch_vy+j];
            tmp_vy *= nu;

            /* convection */
            double vyn = 0.5*(vy_11+vy_12);
            double vys = 0.5*(vy_10+vy_11);

            tmp_vy -= (vyn*vyn-vys*vys)*inv_dy;

            double vxe= 0.5*(vx_11+vx_12);
            double vye= 0.5*(vy_11+vy_21);
            double vxw = 0.5*(vx_01+vx_02);
            double vyw = 0.5*(vy_01+vy_11);

            tmp_vy -= (vxe*vye-vxw*vyw)*inv_dx;
            vy_star[i*(pitch_vy)+j]=vy_11+dt*(tmp_vy+gy);

        }
    }
}
void SMACSolver::get_ustar(){
    double inv_dx = grid_.inv_dx_;
    double inv_dy = grid_.inv_dy_;
    double inv_2dx = grid_.inv_2dx_;
    double inv_2dy = grid_.inv_2dy_;
    double inv_dx2 = grid_.inv_dx2_;
    double inv_dy2 = grid_.inv_dy2_;
    double dt= dt_;

    double gx = gx_;
    double gy = gy_;

    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;

    double* const vx = grid_.vx_;
    double* const vy = grid_.vy_;

    double* const vx_star = grid_.vx_star_;
    double* const vy_star = grid_.vy_star_;

    int pitch_vx=Nx+3;
    int pitch_vy=Nx+2;

    /* == vx == */
    for (int i=1; i<Ny+1;i++){
        for (int j=1; j<Nx;j++){
            double vx_21 =vx[i*(pitch_vx)+j+1];
            double vx_11 =vx[i*(pitch_vx)+j];
            double vx_11_2 =2.*vx_11;
            double vx_01 =vx[i*(pitch_vx)+j-1];
            double vx_12 =vx[(i+1)*(pitch_vx)+j];
            double vx_10 =vx[(i-1)*(pitch_vx)+j];
            double vx_02 =vx[(i-1)*(pitch_vx)+j+1];

            double vy_21 = vy[i*(pitch_vy)+j+1];
            double vy_11 = vy[i*(pitch_vy)+j];
            double vy_11_2 = 2.*vy_11;
            double vy_01 = vy[(i)*(pitch_vy)+j-1];
            double vy_10 = vy[(i-1)*(pitch_vy)+j];
            double vy_12 = vy[(i+1)*(pitch_vy)+j];
            double vy_20 = vy[(i-1)*(pitch_vy)+j+1];

            /* === vx === */
            double tmp_vx = 0.;
            /* viscous */
            tmp_vx +=  nu_*(vx_21-vx_11_2+vx_01)*inv_dx2;
            tmp_vx +=  nu_*(vx_12-vx_11_2+vx_10)*inv_dy2;

            /* convection */
            double vxe= 0.5*(vx_11+vx_21);
            double vxw= 0.5*(vx_01+vx_11);
            tmp_vx -= (vxe*vxe-vxw*vxw)*inv_dx;

            double vxn= 0.5*(vx_11+vx_12);
            double vyn= 0.5*(vy_11+vy_21);
            double vxs= 0.5*(vx_10+vx_11);
            double vys= 0.5*(vy_10+vy_20);

            tmp_vx -= (vxn*vyn-vxs*vys)*inv_dy;
            vx_star[i*(pitch_vx)+j]=vx_11+dt*(tmp_vx+gx);
        }
    }

    for (int i=1; i<Ny;i++){
        for (int j=1; j<Nx+1;j++){
            double vx_11 =vx[i*(pitch_vx)+j];
            double vx_11_2 =2.*vx_11;
            double vx_01 =vx[i*(pitch_vx)+j-1];
            double vx_12 =vx[(i+1)*(pitch_vx)+j];
            double vx_02 =vx[(i+1)*(pitch_vx)+j-1];

            double vy_21 = vy[i*(pitch_vy)+j+1];
            double vy_11 = vy[i*(pitch_vy)+j];
            double vy_11_2 = 2.*vy_11;
            double vy_01 = vy[(i)*(pitch_vy)+j-1];
            double vy_10 = vy[(i-1)*(pitch_vy)+j];
            double vy_12 = vy[(i+1)*(pitch_vy)+j];

            double tmp_vy = 0.;
            /* viscous */
            tmp_vy +=  nu_*(vy_21-vy_11_2+vy_01)*inv_dx2;
            tmp_vy +=  nu_*(vy_12-vy_11_2+vy_10)*inv_dy2;

            /* convection */
            double vyn = 0.5*(vy_11+vy_12);
            double vys = 0.5*(vy_10+vy_11);

            tmp_vy -= (vyn*vyn-vys*vys)*inv_dy;

            double vxe= 0.5*(vx_11+vx_12);
            double vye= 0.5*(vy_11+vy_21);
            double vxw = 0.5*(vx_01+vx_02);
            double vyw = 0.5*(vy_01+vy_11);

            tmp_vy -= (vxe*vye-vxw*vyw)*inv_dx;
            vy_star[i*(pitch_vy)+j]=vy_11+dt*(tmp_vy+gy);

        }
    }
}


void SMACSolver::solver_malloc(){
    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;
    int Nz=grid_.Nz_;

    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) grid_.wew_##name.data_ = (type*)calloc(sizex*sizey*sizez,sizeof(type));\
    grid_.wew_##name.sizex_= sizex;\
    grid_.wew_##name.sizey_= sizey;\
    grid_.wew_##name.sizez_= sizez;\
    grid_.wew_##name.size_ = sizex*sizey*sizez;
    #include "memberList/gridMembers.def"
    #undef MEMBER

    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) grid_.name = (type*)calloc(sizex*sizey*sizez,sizeof(type));
    #include "memberList/gridMembers.def"
    #undef MEMBER

}

void SMACSolver::solver_free(){
    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) free(grid_.name);
    #include "memberList/gridMembers.def"
    #undef MEMBER

    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) free(grid_.wew_##name.data_);
    #include "memberList/gridMembers.def"
    #undef MEMBER
}

void SMACSolver::set_gravity(double gx, double gy, double gz){
    gx_ = gx;
    gy_ = gy;
    gz_ = gz;
}

void SMACSolver::set_rhos(double rho0, double rho1){
    rho0_ = rho0;
    rho1_ = rho1;
}

void SMACSolver::set_mus(double mu0, double mu1){
    mu0_ = mu0;
    mu1_ = mu1;

}


void SMACSolver::set_boundary_neumann(MyArray<double,3>& alpha){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;


    for (int iz=0; iz<Nz+2; iz++){
        for (int iy=0; iy<Ny+2; iy++){
                alpha(0,iy,iz) =alpha(1,iy,iz);
                alpha(Nx+1,iy,iz) =alpha(Nx,iy,iz);
        }
    }

    for (int iz=0; iy<Nz+2; iz++){
        for (int ix=0; ix<Nx+2; ix++){
            alpha(ix,0,iz) =alpha(ix,1,iz);
            alpha(ix,Ny+1,iz) =alpha(ix,Ny,iz);
        }
    }

    for (int iy=0; iy<Ny+2; iy++){
        for (int ix=0; ix<Nx+2; ix++){
            alpha(ix,iy,0) =alpha(ix,iy,1);
            alpha(ix,iy,Nx+1) =alpha(ix,iy,Nx);
        }
    }
}

void SMACSolver::set_boundary_array(double* const q){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int pitch_c = Nx+2;

    for (int i=0; i<Nx+2; i++){
        q[(pitch_c)*(Ny+1)+i]=q[(pitch_c)*(Ny)+i];
    }


    for (int i=0; i<Nx+2; i++){
        q[i]=q[1*(pitch_c)+i];
    }


    for (int i=0; i<Ny+2; i++){
        q[i*(pitch_c)+Nx+1]=q[i*(pitch_c)+Nx];
    }

    for (int i=0; i<Ny+2; i++){
        q[i*(pitch_c)]=q[i*(pitch_c)+1];
    }

    q[0*pitch_c+0] = q[1*pitch_c+1];
    q[0*pitch_c+Nx+1] = q[1*pitch_c+Nx];
    q[(Ny+1)*pitch_c+0] = q[Ny*pitch_c+1];
    q[(Ny+1)*pitch_c+Nx+1] = q[Ny*pitch_c+Nx];
}

void SMACSolver::set_boundary_pressure(){
    double* const p = grid_.p_;

    set_boundary_array(p);
}

void SMACSolver::set_boundary_velocity(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    double v_b_1 = grid_.v_b_1_; 
    double v_b_2 = grid_.v_b_2_; 

    double* const vx = grid_.vx_;
    double* const vy = grid_.vy_;
    double* const p = grid_.p_;

    for (int i=0; i<Nx+3; i++){
        vx[(Nx+3)*(Ny+1)+i]=v_b_1-vx[(Nx+3)*(Ny)+i];
    }

    for (int i=0; i<Nx+3; i++){
        vx[i]=v_b_2-vx[(Nx+3)*(1)+i];
    }

    for (int i=0; i<Ny+3; i++){
        vy[(Nx+2)*i+Nx+1]=v_b_2-vy[(Nx+2)*i+Nx];
    }

    for (int i=0; i<Ny+3; i++){
        vy[(Nx+2)*i]=v_b_2-vy[(Nx+2)*i+1];
    }

}

void SMACSolver::set_boundary_star(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    double v_b_1 = grid_.v_b_1_; 
    double v_b_2 = grid_.v_b_2_; 

    double* const vx = grid_.vx_star_;
    double* const vy = grid_.vy_star_;


    for (int i=0; i<Nx+3; i++){
        vx[(Nx+3)*(Ny+1)+i]=v_b_1-vx[(Nx+3)*(Ny)+i];
    }

    for (int i=0; i<Nx+3; i++){
        vx[i]=v_b_2-vx[(Nx+3)*(1)+i];
    }

    for (int i=0; i<Ny+3; i++){
        vy[(Nx+2)*i+Nx+1]=v_b_2-vy[(Nx+2)*i+Nx];
    }

    for (int i=0; i<Ny+3; i++){
        vy[(Nx+2)*i]=v_b_2-vy[(Nx+2)*i+1];
    }
}

void SMACSolver::solve_poisson(){
    double *const vx_star=grid_.vx_star_;
    double *const vy_star=grid_.vy_star_;
    double *const rhs=grid_.rhs_;

    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;
    double rho = rho_;
    double inv_dt= inv_dt_;
    double inv_dx= grid_.inv_dx_;
    double inv_dy= grid_.inv_dy_;
    double rho_inv_dt = rho*inv_dt;
    double eta = 0.5*1./(grid_.inv_dx2_+grid_.inv_dy2_);

    for (int i=1; i<Ny+1; i++){
        for (int j=1; j<Nx+1; j++){
            rhs[i*(Nx+2)+j] = rho_inv_dt*((vx_star[i*(Nx+3)+j]-vx_star[(i)*(Nx+3)+j-1])*inv_dx+(vy_star[i*(Nx+2)+j]-vy_star[(i-1)*(Nx+2)+j])*inv_dy);
        }
    }

    double max_rhs = max_abs_double(rhs,(Nx+2)*(Ny+2));

    int max_iter = 10000;
    int res_check_freq = 20;

    /* == jacobi iteration loop == */
    int k=0;
    while(k<max_iter){
        for (int i=1; i<Ny+1; i++){
            for (int j=1; j<Nx+1; j++){
                double tmp_p =(grid_.p_[(i)*(Nx+2)+j+1]+grid_.p_[(i)*(Nx+2)+j-1])*grid_.inv_dx2_;
                tmp_p +=(grid_.p_[(i+1)*(Nx+2)+j]+grid_.p_[(i-1)*(Nx+2)+j])*grid_.inv_dy2_;
                grid_.p_tmp_[i*(Nx+2)+j] = eta*(tmp_p-rhs[i*(Nx+2)+j]);
            }
        }

        /* ===swap === */
        double *p_tmp_swap ;
        p_tmp_swap = grid_.p_;
        grid_.p_ = grid_.p_tmp_;
        grid_.p_tmp_= p_tmp_swap;

        this->fix_pressure();
        this->set_boundary_pressure();

        /* == calculate residue == */
        if(k%res_check_freq ==0){
            for (int i=1; i<Ny+1; i++){
                for (int j=1; j<Nx+1; j++){
                    if (i==1 && j==1){
                        grid_.residue_[i*(Nx+2)+j] = 0.;
                        continue ;
                    }
                    double tmp_2pij=2.*grid_.p_[(i)*(Nx+2)+j];
                    double tmp_res =(grid_.p_[(i)*(Nx+2)+j+1]-tmp_2pij+grid_.p_[(i)*(Nx+2)+j-1])*grid_.inv_dx2_;
                    tmp_res +=(grid_.p_[(i+1)*(Nx+2)+j]-tmp_2pij+grid_.p_[(i-1)*(Nx+2)+j])*grid_.inv_dy2_; //laplacian of pressure
                    grid_.residue_[i*(Nx+2)+j] = rhs[i*(Nx+2)+j]-tmp_res;
                }
            }

            double max_residue = max_abs_double(grid_.residue_,(Nx+2)*(Ny+2));
            double rel_residue = max_residue/(max_rhs+1e-16);

            if(rel_residue< 1e-8){
                printf("iteration = %d,max_residue = %3.2e, relative residue = %3.2e\n",k,max_residue, rel_residue);
                break;
            }
        }

        k+=1;
    }

}

void SMACSolver::make_poisson_rhs(){
    double *const vx_star=grid_.vx_star_;
    double *const vy_star=grid_.vy_star_;
    double *const rhs=grid_.rhs_;
    double *const inv_rho=grid_.inv_rho_;

    double *const f_bx=grid_.f_bx_;
    double *const f_by=grid_.f_by_;

    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;
    double rho = rho_;
    double inv_dt= inv_dt_;
    double inv_dx= grid_.inv_dx_;
    double inv_dy= grid_.inv_dy_;
    double inv_dx2= grid_.inv_dx2_;
    double inv_dy2= grid_.inv_dy2_;

    int pitch_c= Nx+2;
    int pitch_fx= Nx+3;
    int pitch_fy= Nx+2;

    for (int i=1; i<Ny+1; i++){
        for (int j=1; j<Nx+1; j++){
            rhs[i*(Nx+2)+j] = inv_dt*((vx_star[i*(Nx+3)+j]-vx_star[(i)*(Nx+3)+j-1])*inv_dx+(vy_star[i*(Nx+2)+j]-vy_star[(i-1)*(Nx+2)+j])*inv_dy);
        }
    }
}

void SMACSolver::solve_vof_poisson(){
    double *const vx_star=grid_.vx_star_;
    double *const vy_star=grid_.vy_star_;
    double *const rhs=grid_.rhs_;
    double *const inv_rho=grid_.inv_rho_;

    double *const f_bx=grid_.f_bx_;
    double *const f_by=grid_.f_by_;

    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;
    double rho = rho_;
    double inv_dt= inv_dt_;
    double inv_dx= grid_.inv_dx_;
    double inv_dy= grid_.inv_dy_;
    double inv_dx2= grid_.inv_dx2_;
    double inv_dy2= grid_.inv_dy2_;

    int pitch_c= Nx+2;
    int pitch_fx= Nx+3;
    int pitch_fy= Nx+2;

    for (int i=1; i<Ny+1; i++){
        for (int j=1; j<Nx+1; j++){
            rhs[i*(Nx+2)+j] = inv_dt*((vx_star[i*(Nx+3)+j]-vx_star[(i)*(Nx+3)+j-1])*inv_dx+(vy_star[i*(Nx+2)+j]-vy_star[(i-1)*(Nx+2)+j])*inv_dy);
        }
    }

    double max_rhs = max_abs_double(rhs,(Nx+2)*(Ny+2));

    int max_iter = 50000;
    int res_check_freq = 20;

    /* == jacobi iteration loop == */
    int k=0;
    while(k<max_iter){
        double *const p=grid_.p_;
        double *const p_tmp=grid_.p_tmp_;
        for (int i=1; i<Ny+1; i++){
            for (int j=1; j<Nx+1; j++){
                int ind21 = i*pitch_c+j+1;
                int ind12 = (i+1)*pitch_c+j;
                int ind11 = (i)*pitch_c+j;

                int ind01 = i*pitch_c+j-1;
                int ind10 = (i-1)*pitch_c+j;

                double a21 = f_bx[i*pitch_fx+j]*inv_dx2;
                double a01 = f_bx[i*pitch_fx+j-1]*inv_dx2;

                double a12 = f_by[i*pitch_fy+j]*inv_dy2;
                double a10 = f_by[(i-1)*pitch_fy+j]*inv_dy2;

                double ap = a21+a01+a12+a10;

                double tmp_p =a21*p[ind21]+a01*p[ind01]+a12*p[ind12]+a10*p[ind10];
                p_tmp[ind11] = (tmp_p-rhs[i*(Nx+2)+j])/ap;
            }
        }

        /* ===swap === */
        double *p_tmp_swap ;
        p_tmp_swap = grid_.p_;
        grid_.p_ = grid_.p_tmp_;
        grid_.p_tmp_= p_tmp_swap;

        this->shift_pressure_reference();

        double const *p_new = grid_.p_;
        /* == calculate residue == */
        if(k%res_check_freq ==0){
            for (int i=1; i<Ny+1; i++){
                for (int j=1; j<Nx+1; j++){
                    if (i==1 && j==1){
                        grid_.residue_[i*(Nx+2)+j] = 0.;
                        continue ;
                    }

                    int ind21 = i*pitch_c+j+1;
                    int ind12 = (i+1)*pitch_c+j;
                    int ind11 = (i)*pitch_c+j;

                    int ind01 = i*pitch_c+j-1;
                    int ind10 = (i-1)*pitch_c+j;

                    double a21 = f_bx[i*pitch_fx+j]*inv_dx2;
                    double a01 = f_bx[i*pitch_fx+j-1]*inv_dx2;

                    double a12 = f_by[i*pitch_fy+j]*inv_dy2;
                    double a10 = f_by[(i-1)*pitch_fy+j]*inv_dy2;

                    double ap = a21+a01+a12+a10;

                    double tmp_res =a21*p_new[ind21]+a01*p_new[ind01]+a12*p_new[ind12]+a10*p_new[ind10]-ap*p_new[ind11];

                    grid_.residue_[i*(Nx+2)+j] = rhs[i*(Nx+2)+j]-tmp_res;
                }
            }

            double max_residue = max_abs_double(grid_.residue_,(Nx+2)*(Ny+2));
            double rel_residue = max_residue/(max_rhs+1e-16);

            if(rel_residue< 1e-8){
                printf("iteration = %d,max_residue = %3.2e, relative residue = %3.2e\n",k,max_residue, rel_residue);
                break;
            }
        }

        k+=1;
    }

}

void SMACSolver::initialize_disk()
{
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int pitch = Nx + 2;

    double* a = grid_.alpha_;

    double dx = grid_.dx_;
    double dy = grid_.dy_;

    const double xc = 0.5;
    const double yc = 0.5;
    const double R  = 0.15;

    const int ns = 8;

    for (int i = 0; i < Ny + 2; i++) {
        for (int j = 0; j < Nx + 2; j++) {
            a[i*pitch + j] = 0.0;
        }
    }

    for (int i = 1; i <= Ny; i++) {
        for (int j = 1; j <= Nx; j++) {

            double x_left   = (j - 1) * dx;
            double y_bottom = (i - 1) * dy;

            double count = 0.0;

            for (int sy = 0; sy < ns; sy++) {
                for (int sx = 0; sx < ns; sx++) {

                    double x = x_left   + dx * ((double)sx + 0.5) / (double)ns;
                    double y = y_bottom + dy * ((double)sy + 0.5) / (double)ns;

                    double rx = x - xc;
                    double ry = y - yc;

                    if (rx*rx + ry*ry <= R*R) {
                        count += 1.0;
                    }
                }
            }

            a[i*pitch + j] = count / (double)(ns * ns);
        }
    }

    set_boundary_alpha();
}

void SMACSolver::check_pressure_jump_by_radius(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int pitch = Nx + 2;

    double* p = grid_.p_;
    double* a = grid_.alpha_;
    double* kappa = grid_.kappa_;   

    double dx = grid_.dx_;
    double dy = grid_.dy_;

    double h = dx;
    if(dy < h) h = dy;

    const double xc = 0.5;
    const double yc = 0.5;
    const double R  = 0.15;

    double margin = 2.0 * h;

    double sum_inside = 0.0;
    double sum_outside = 0.0;

    int count_inside = 0;
    int count_outside = 0;

    /*
       kappa statistics on interface cells
       */
    double sum_kappa = 0.0;
    double sum_kappa2 = 0.0;
    double kappa_min = 0.0;
    double kappa_max = 0.0;
    int count_kappa = 0;

    for(int i = 1; i <= Ny; i++){
        for(int j = 1; j <= Nx; j++){

            int id = i * pitch + j;

            double x = ((double)j - 0.5) * dx;
            double y = ((double)i - 0.5) * dy;

            double rx = x - xc;
            double ry = y - yc;

            double r = sqrt(rx*rx + ry*ry);

            /*
               Pressure inside / outside.
               Exclude interface neighborhood by using margin.
               */
            if(r < R - margin){
                sum_inside += p[id];
                count_inside++;
            }

            if(r > R + margin){
                sum_outside += p[id];
                count_outside++;
            }

            /*
               Curvature statistics.
               Use only interface cells.
               */
            if(a[id] > 0.01 && a[id] < 0.99){

                double kap = kappa[id];

                sum_kappa += kap;
                sum_kappa2 += kap * kap;

                if(count_kappa == 0){
                    kappa_min = kap;
                    kappa_max = kap;
                }else{
                    if(kap < kappa_min) kappa_min = kap;
                    if(kap > kappa_max) kappa_max = kap;
                }

                count_kappa++;
            }
        }
    }

    double p_inside_avg = 0.0;
    double p_outside_avg = 0.0;

    if(count_inside > 0){
        p_inside_avg = sum_inside / (double)count_inside;
    }

    if(count_outside > 0){
        p_outside_avg = sum_outside / (double)count_outside;
    }

    double dp_measured = p_inside_avg - p_outside_avg;

    double sigma = grid_.sigma_[0];

    double dp_theory = sigma / R;
    double kappa_theory = 1.0 / R;

    printf("p_inside_avg = %.10e\n", p_inside_avg);
    printf("p_outside_avg = %.10e\n", p_outside_avg);
    printf("dp_measured   = %.10e\n", dp_measured);
    printf("dp_theory     = %.10e\n", dp_theory);

    if(fabs(dp_theory) > 1.0e-300){
        printf("dp ratio      = %.10e\n", dp_measured / dp_theory);
    }else{
        printf("dp ratio      = undefined\n");
    }

    /*
       Print kappa statistics
       */
    if(count_kappa > 0){

        double kappa_avg = sum_kappa / (double)count_kappa;
        double kappa2_avg = sum_kappa2 / (double)count_kappa;

        double kappa_var = kappa2_avg - kappa_avg * kappa_avg;
        if(kappa_var < 0.0) kappa_var = 0.0;

        double kappa_std = sqrt(kappa_var);

        printf("kappa_count   = %d\n", count_kappa);
        printf("kappa_avg     = %.10e\n", kappa_avg);
        printf("kappa_min     = %.10e\n", kappa_min);
        printf("kappa_max     = %.10e\n", kappa_max);
        printf("kappa_std     = %.10e\n", kappa_std);
        printf("kappa_theory  = %.10e\n", kappa_theory);

        if(fabs(kappa_theory) > 1.0e-300){
            printf("kappa ratio   = %.10e\n", kappa_avg / kappa_theory);
        }else{
            printf("kappa ratio   = undefined\n");
        }

    }else{
        printf("kappa_count   = 0\n");
        printf("kappa statistics: no interface cells\n");
    }
}
void SMACSolver::initialize_zalesak_disk(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int pitch = Nx + 2;

    double* a = grid_.alpha_;

    double dx = grid_.dx_;
    double dy = grid_.dy_;

    const double xc = 0.5;
    const double yc = 0.75;
    const double R  = 0.15;

    const double slot_w = 0.05;
    const double slot_h = 0.25;

    const double x_slot_min = xc - 0.5 * slot_w;
    const double x_slot_max = xc + 0.5 * slot_w;

    const double y_slot_min = yc - R;
    const double y_slot_max = yc - R + slot_h;

    for (int i = 0; i < Ny + 2; i++) {
        for (int j = 0; j < Nx + 2; j++) {
            a[i*pitch + j] = 0.0;
        }
    }

    for (int i = 1; i <= Ny; i++) {
        for (int j = 1; j <= Nx; j++) {

            double x = (j - 0.5) * dx;
            double y = (i - 0.5) * dy;

            double rx = x - xc;
            double ry = y - yc;

            bool inside_disk = (rx*rx + ry*ry <= R*R);

            bool inside_slot =
                (x >= x_slot_min) &&
                (x <= x_slot_max) &&
                (y >= y_slot_min) &&
                (y <= y_slot_max);

            if (inside_disk && !inside_slot) {
                a[i*pitch + j] = 1.0;
            } else {
                a[i*pitch + j] = 0.0;
            }
        }
    }

    set_boundary_alpha();
}

double SMACSolver::calc_cfl(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    double dt= dt_;

    int pitch_c = Nx;
    int pitch_fx = Nx+3;
    int pitch_fy = Nx+2;

    double* const vx = grid_.vx_;
    double* const vy = grid_.vy_;
    double* const cfl = grid_.cfl_;
    double inv_dx = grid_.inv_dx_;
    double inv_dy = grid_.inv_dy_;

    double max_cfl = -1.;
    for (int iy=1; iy<Ny+1; iy++){
        for (int ix=1; ix<Nx+1; ix++){
            int indc = (ix-1)+(iy-1)*pitch_c;
            int ind_xp = ix+iy*pitch_fx;
            int ind_xm = ix-1+iy*pitch_fx;
            int ind_yp = ix+iy*pitch_fy;
            int ind_ym = ix+(iy-1)*pitch_fy;

            double vx_xp = fabs(vx[ind_xp]);
            double vx_xm = fabs(vx[ind_xm]);
            double vy_yp = fabs(vy[ind_yp]);
            double vy_ym = fabs(vy[ind_ym]);

            double max_vx= vx_xp>vx_xm ? vx_xp: vx_xm;
            double max_vy= vy_yp>vy_ym ? vy_yp: vy_ym;
            cfl[indc] = dt*(max_vx*inv_dx+max_vy*inv_dy);
            max_cfl = max_cfl > cfl[indc]? max_cfl : cfl[indc];
        }
    }

    printf("CFL = %3.2e\n", max_cfl);
    return max_cfl;
}

void SMACSolver::set_zalesak_rotation_velocity(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;

    double dx = grid_.dx_;
    double dy = grid_.dy_;

    double* vx = grid_.vx_;
    double* vy = grid_.vy_;

    int pitch_vx = Nx + 3;
    int pitch_vy = Nx + 2;

    const double omega = 2.0 * M_PI;
    const double xc = 0.5;
    const double yc = 0.5;

    // vx/u: vertical faces
    for (int i = 0; i < Ny + 2; i++) {
        for (int j = 0; j < Nx + 3; j++) {
            double y = (i - 0.5) * dy;
            vx[i*pitch_vx + j] = -omega * (y - yc);
        }
    }

    // vy/v: horizontal faces
    for (int i = 0; i < Ny + 3; i++) {
        for (int j = 0; j < Nx + 2; j++) {
            double x = (j - 0.5) * dx;
            vy[i*pitch_vy + j] = omega * (x - xc);
        }
    }
}
