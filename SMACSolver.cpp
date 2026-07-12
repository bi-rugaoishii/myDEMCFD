#include "SMACSolver.h"
#include "hardCodedParameters.h"
#include "Enums.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>


void SMACSolver::set_calc_properties(double rho, double dt,double u_lid, double nu, double sizex, double sizey,double sizez, int Nx, int Ny, int Nz){
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

}

void SMACSolver::set_face_internal_direction(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;

    MyArray<unsigned char,3>& ctype= grid_.celltype_;
    MyArray<unsigned char,3>& f_xtype = grid_.f_xtype_;
    MyArray<unsigned char,3>& f_ytype = grid_.f_ytype_;
    MyArray<unsigned char,3>& f_ztype = grid_.f_ztype_;

    MyArray<int,3>& f_xinternal_id = grid_.f_xinternal_id_;
    MyArray<int,3>& f_yinternal_id = grid_.f_yinternal_id_;
    MyArray<int,3>& f_zinternal_id = grid_.f_zinternal_id_;

    /* == get direction of internal faces == */
    /* == assuming no internal wall exists == */

    // valid x-faces
    for(int iz=1; iz<=Nz; iz++){
        for(int iy=1; iy<=Ny; iy++){
            for(int ix=1; ix<=Nx+1; ix++){
                if(f_xtype(ix,iy,iz)==F_BOUNDARY){
                    unsigned char ctypep = ctype(ix,iy,iz);
                    if(ctypep == C_INTERIOR ||ctypep==  C_NEAR_BOUNDARY){
                        f_xinternal_id(ix,iy,iz)=0;
                    }else{
                        f_xinternal_id(ix,iy,iz)=-1;
                    }
                }
            }
        }
    }

    // valid y-faces
    for(int iz=1; iz<=Nz; iz++){
        for(int iy=1; iy<=Ny+1; iy++){
            for(int ix=1; ix<=Nx; ix++){
                if(f_ytype(ix,iy,iz)==F_BOUNDARY){
                    unsigned char ctypep = ctype(ix,iy,iz);
                    if(ctypep == C_INTERIOR || ctypep== C_NEAR_BOUNDARY){
                        f_yinternal_id(ix,iy,iz)=0;
                    }else{
                        f_yinternal_id(ix,iy,iz)=-1;
                    }
                }
            }
        }
    }

    // valid z-faces
    for(int iz=1; iz<=Nz+1; iz++){
        for(int iy=1; iy<=Ny; iy++){
            for(int ix=1; ix<=Nx; ix++){
                if(f_ytype(ix,iy,iz)==F_BOUNDARY){
                    unsigned char ctypep = ctype(ix,iy,iz);
                    if(ctypep == C_INTERIOR || ctypep== C_NEAR_BOUNDARY){
                        f_zinternal_id(ix,iy,iz)=0;
                    }else{
                        f_zinternal_id(ix,iy,iz)=-1;
                    }
                }
            }
        }
    }

}

void SMACSolver::set_face_type(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;

    MyArray<unsigned char,3>& f_xtype = grid_.f_xtype_;
    MyArray<unsigned char,3>& f_ytype = grid_.f_ytype_;
    MyArray<unsigned char,3>& f_ztype = grid_.f_ztype_;

    MyArray<unsigned char,3>& f_xbcid = grid_.f_xbcid_;
    MyArray<unsigned char,3>& f_ybcid = grid_.f_ybcid_;
    MyArray<unsigned char,3>& f_zbcid = grid_.f_zbcid_;

    MyArray<int,3>& f_xinternal_id = grid_.f_xinternal_id_;
    MyArray<int,3>& f_yinternal_id = grid_.f_yinternal_id_;
    MyArray<int,3>& f_zinternal_id = grid_.f_zinternal_id_;

    // x-face: size = (Nx+3, Ny+2, Nz+2)
    for(int iz=0; iz<Nz+2; iz++){
        for(int iy=0; iy<Ny+2; iy++){
            for(int ix=0; ix<Nx+3; ix++){
                f_xtype(ix,iy,iz) = F_GHOST;
            }
        }
    }

    // y-face: size = (Nx+2, Ny+3, Nz+2)
    for(int iz=0; iz<Nz+2; iz++){
        for(int iy=0; iy<Ny+3; iy++){
            for(int ix=0; ix<Nx+2; ix++){
                f_ytype(ix,iy,iz) = F_GHOST;
            }
        }
    }

    // z-face: size = (Nx+2, Ny+2, Nz+3)
    for(int iz=0; iz<Nz+3; iz++){
        for(int iy=0; iy<Ny+2; iy++){
            for(int ix=0; ix<Nx+2; ix++){
                f_ztype(ix,iy,iz) = F_GHOST;
            }
        }
    }

    // valid x-faces
    for(int iz=1; iz<=Nz; iz++){
        for(int iy=1; iy<=Ny; iy++){
            for(int ix=1; ix<=Nx+1; ix++){
                f_xtype(ix,iy,iz) = F_INTERIOR;
            }
        }
    }

    // x-wall faces
    for(int iz=1; iz<=Nz; iz++){
        for(int iy=1; iy<=Ny; iy++){
            f_xtype(1,iy,iz) = F_BOUNDARY;
            f_xtype(Nx+1,iy,iz) = F_BOUNDARY;
        }
    }

    // valid y-faces
    for(int iz=1; iz<=Nz; iz++){
        for(int iy=1; iy<=Ny+1; iy++){
            for(int ix=1; ix<=Nx; ix++){
                f_ytype(ix,iy,iz) = F_INTERIOR;
            }
        }
    }

    // y-wall faces
    for(int iz=1; iz<=Nz; iz++){
        for(int ix=1; ix<=Nx; ix++){
            f_ytype(ix,1,iz) = F_BOUNDARY;
            f_ytype(ix,Ny+1,iz) = F_BOUNDARY;


        }
    }

    // valid z-faces
    for(int iz=1; iz<=Nz+1; iz++){
        for(int iy=1; iy<=Ny; iy++){
            for(int ix=1; ix<=Nx; ix++){
                f_ztype(ix,iy,iz) = F_INTERIOR;
            }
        }
    }

    // z-wall faces
    for(int iy=1; iy<=Ny; iy++){
        for(int ix=1; ix<=Nx; ix++){
            f_ztype(ix,iy,1) = F_BOUNDARY;
            f_ztype(ix,iy,Nz+1) = F_BOUNDARY;
        }
    }


}



void SMACSolver::set_cell_type(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;

    MyArray<unsigned char,3>& ctype= grid_.celltype_;
    MyArray<unsigned char,3>& fxtype= grid_.f_xtype_;
    MyArray<unsigned char,3>& fytype= grid_.f_ytype_;
    MyArray<unsigned char,3>& fztype= grid_.f_ztype_;

    /* initialize with ghost */
    for(int iz=0; iz<Nz+2; iz++){
        for(int iy=0; iy<Ny+2; iy++){
            for(int ix=0; ix<Nx+2; ix++){
                ctype(ix,iy,iz) = C_GHOST;
            }
        }
    }

    /* initialize check near boundary cell*/
    for(int iz=1; iz<Nz+1; iz++){
        for(int iy=1; iy<Ny+1; iy++){
            for(int ix=1; ix<Nx+1; ix++){
                unsigned char xp = fxtype(ix+1,iy,iz);
                unsigned char yp = fytype(ix,iy+1,iz);
                unsigned char zp = fztype(ix,iy,iz+1);

                unsigned char xm = fxtype(ix,iy,iz);
                unsigned char ym = fytype(ix,iy,iz);
                unsigned char zm = fztype(ix,iy,iz);

                if(xp != F_INTERIOR || yp != F_INTERIOR || zp != F_INTERIOR || 
                    xm != F_INTERIOR || ym != F_INTERIOR || zm != F_INTERIOR){
                    
                    //ctype(ix,iy,iz) = C_NEAR_BOUNDARY;//disabled temporarily
                    ctype(ix,iy,iz) = C_INTERIOR;
                }else{
                    ctype(ix,iy,iz) = C_INTERIOR;
                }
            }
        }
    }
}

void SMACSolver::check_divergence(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;

    MyArray<double,3> p = grid_.p_;
    MyArray<double,3> vx = grid_.f_vx_;
    MyArray<double,3> vy = grid_.f_vy_;
    MyArray<double,3> vz = grid_.f_vz_;

    double div_max=0.0;
    double inv_dx = grid_.inv_dx_;
    double inv_dy = grid_.inv_dy_;
    double inv_dz = grid_.inv_dz_;

    for (int iz=1; iz<Nz+1; iz++){
        for (int iy=1; iy<Ny+1; iy++){
            for (int ix=1; ix<Nx+1; ix++){

                double div = (vx(ix+1,iy,iz)-vx(ix,iy,iz))*inv_dx
                    + (vy(ix,iy+1,iz)-vy(ix,iy,iz))*inv_dy
                    + (vz(ix,iy,iz+1)-vz(ix,iy,iz))*inv_dz;

                div = fabs(div);
                div_max = div_max<div? div:div_max; 
            }
        }
    }

    printf("div_max = %e , div_max*dt = %e\n", div_max, div_max*dt_);
}


void SMACSolver::update_properties_by_alpha_initial(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;

    const double rho1 = rho1_;
    const double rho0 = rho0_;

    const double mu1 = mu1_;
    const double mu0 = mu0_;

    MyArray<double,3> a = grid_.alpha_;
    MyArray<double,3> rho = grid_.rho_;
    MyArray<double,3> rho_old = grid_.rho_old_;
    MyArray<double,3> inv_rho = grid_.inv_rho_;
    MyArray<double,3> mu = grid_.mu_;

    /* == update rho == */
    for (int iz=1; iz<Nz+1; iz++){
        for (int iy=1; iy<Ny+1; iy++){
            for (int ix=1; ix<Nx+1; ix++){
                double alpha = a(ix,iy,iz);
                rho(ix,iy,iz) = (1.-alpha)*rho0+alpha*rho1;
            }
        }
    }

    for (int iz=1; iz<Nz+1; iz++){
        for (int iy=1; iy<Ny+1; iy++){
            for (int ix=1; ix<Nx+1; ix++){
                rho_old(ix,iy,iz) = rho(ix,iy,iz);
            }
        }
    }

    /* == update inv_rho == */
    for (int iz=1; iz<Nz+1; iz++){
        for (int iy=1; iy<Ny+1; iy++){
            for (int ix=1; ix<Nx+1; ix++){
                inv_rho(ix,iy,iz) = 1./rho(ix,iy,iz);
            }
        }
    }

    /* == update mu == */
    for (int iz=1; iz<Nz+1; iz++){
        for (int iy=1; iy<Ny+1; iy++){
            for (int ix=1; ix<Nx+1; ix++){
                double alpha = a(ix,iy,iz);
                mu(ix,iy,iz) = (1.-alpha)*mu0+alpha*mu1;
            }
        }
    }

    /* == update inv rho at face== */
    /* == boundary condition is implemented here == */

    /* == x faces == */
    MyArray<double,3>& f_bx = grid_.f_bx_;

    for (int iz=0; iz<Nz+2; iz++){
        for (int iy=0; iy<Ny+2; iy++){
            for (int ix=0; ix<Nx+3; ix++){
                unsigned char face_type = grid_.f_xtype_(ix,iy,iz);
                switch(face_type){
                    case F_INTERIOR:
                        /* debug */
                        if (ix <= 0 || ix >= Nx+2) {
                            printf("bad x F_INTERIOR: ix=%d iy=%d iz=%d\n",ix,iy,iz);
                            abort();
                        }

                        f_bx(ix,iy,iz) = 2./(rho(ix,iy,iz)+rho(ix-1,iy,iz));
                        break;
                    case F_BOUNDARY:
                        f_bx(ix,iy,iz) = 0.;
                        break;
                    case F_GHOST:
                        f_bx(ix,iy,iz) = 0.;
                        break;
                }
            }
        }
    }

    /* == y faces == */
    MyArray<double,3>& f_by = grid_.f_by_;

    for (int iz=0; iz<Nz+2; iz++){
        for (int iy=0; iy<Ny+3; iy++){
            for (int ix=0; ix<Nx+2; ix++){
                unsigned char face_type = grid_.f_ytype_(ix,iy,iz);
                switch(face_type){
                    case F_INTERIOR:
                        if (iy <= 0 || iy >= Ny+2) {
                            printf("bad y F_INTERIOR: ix=%d iy=%d iz=%d\n",ix,iy,iz);
                            abort();
                        }
                        f_by(ix,iy,iz) = 2./(rho(ix,iy,iz)+rho(ix,iy-1,iz));
                        break;
                    case F_BOUNDARY:
                        f_by(ix,iy,iz) = 0.;
                        break;
                    case F_GHOST:
                        f_by(ix,iy,iz) = 0.;
                        break;
                }
            }
        }
    }

    /* == z faces == */
    MyArray<double,3>& f_bz = grid_.f_bz_;

    for (int iz=0; iz<Nz+3; iz++){
        for (int iy=0; iy<Ny+2; iy++){
            for (int ix=0; ix<Nx+2; ix++){
                unsigned char face_type = grid_.f_ztype_(ix,iy,iz);
                switch(face_type){
                    case F_INTERIOR:
                        /* debug */
                        if (iz <= 0 || iz >= Nz+2) {
                            printf("bad z F_INTERIOR: ix=%d iy=%d iz=%d\n",ix,iy,iz);
                            abort();
                        }

                        f_bz(ix,iy,iz) = 2./(rho(ix,iy,iz)+rho(ix,iy,iz-1));
                        break;
                    case F_BOUNDARY:
                        f_bz(ix,iy,iz) = 0.;
                        break;
                    case F_GHOST:
                        f_bz(ix,iy,iz) = 0.;
                        break;
                }
            }
        }
    }
}

void SMACSolver::solver_malloc(){
    int Nx=grid_.Nx_;
    int Ny=grid_.Ny_;
    int Nz=grid_.Nz_;

    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) grid_.name.data_ = (type*)calloc(sizex*sizey*sizez,sizeof(type));\
    grid_.name.sizex_= sizex;\
    grid_.name.sizey_= sizey;\
    grid_.name.sizez_= sizez;\
    grid_.name.size_ = sizex*sizey*sizez;
    #include "memberList/gridMembers.def"
    #undef MEMBER

    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) grid_.bc_.name.data_ = (type*)calloc(grid_.bc_.num_boundary_id_,sizeof(type));
    #include "memberList/boundaryConditionMembers.def"
    #undef MEMBER

}

void SMACSolver::solver_free(){
    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) free(grid_.name.data_);
    #include "memberList/gridMembers.def"
    #undef MEMBER

    #define MEMBER(type, name, sizex,sizey,sizez, isSAVE) free(grid_.bc_.name.data_);
    #include "memberList/boundaryConditionMembers.def"
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

    for (int iz=0; iz<Nz+2; iz++){
        for (int ix=0; ix<Nx+2; ix++){
            alpha(ix,0,iz) =alpha(ix,1,iz);
            alpha(ix,Ny+1,iz) =alpha(ix,Ny,iz);
        }
    }

    for (int iy=0; iy<Ny+2; iy++){
        for (int ix=0; ix<Nx+2; ix++){
            alpha(ix,iy,0) =alpha(ix,iy,1);
            alpha(ix,iy,Nz+1) =alpha(ix,iy,Nz);
        }
    }
}

void SMACSolver::set_sphere_sub_voxel()
{
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;

    MyArray<double,3> x = grid_.x_;
    MyArray<double,3> y = grid_.y_;
    MyArray<double,3> z = grid_.z_;
    MyArray<double,3> a = grid_.alpha_;

    double dx = grid_.dx_;
    double dy = grid_.dy_;
    double dz = grid_.dz_;

    double g_dy = grid_.dy_;

    double center_x = 0.02-0.5*g_dy;
    double center_y = 0.007-0.5*g_dy;
    double center_z = 0.02-0.5*g_dy;

    MyArray<double,3> f_vy = grid_.f_vy_;

    double v_ini = -2.25;


    double radius = 0.003;
    double radius2 = radius * radius;

    constexpr int Nsub = 4;
    constexpr int Nsample = Nsub * Nsub * Nsub;

    for (int iz = 1; iz <= Nz; iz++) {
        for (int iy = 1; iy <= Ny; iy++) {
            for (int ix = 1; ix <= Nx; ix++) {

                int inside_count = 0;

                for (int sz = 0; sz < Nsub; sz++) {
                    for (int sy = 0; sy < Nsub; sy++) {
                        for (int sx = 0; sx < Nsub; sx++) {

                            /*
                             * セル中心を基準に、セル内部へ
                             * 等間隔にサンプル点を置く。
                             */
                            double sample_x =
                                x(ix,iy,iz)
                                + (
                                    (sx + 0.5) / Nsub - 0.5
                                ) * dx;

                            double sample_y =
                                y(ix,iy,iz)
                                + (
                                    (sy + 0.5) / Nsub - 0.5
                                ) * dy;

                            double sample_z =
                                z(ix,iy,iz)
                                + (
                                    (sz + 0.5) / Nsub - 0.5
                                ) * dz;

                            double rx = sample_x - center_x;
                            double ry = sample_y - center_y;
                            double rz = sample_z - center_z;

                            if (rx * rx
                                + ry * ry
                                + rz * rz < radius2) {
                                inside_count++;
                                f_vy(ix,iy,iz) = v_ini;
                                f_vy(ix,iy+1,iz) = v_ini;
                            }
                        }
                    }
                }

                a(ix,iy,iz) =
                    static_cast<double>(inside_count)
                    / static_cast<double>(Nsample);
            }
        }
    }
}

void SMACSolver::set_sphere(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;


    MyArray<double,3> x = grid_.x_;
    MyArray<double,3> y = grid_.y_;
    MyArray<double,3> z = grid_.z_;
    MyArray<double,3> a = grid_.alpha_;

    double g_dy = grid_.dy_;

    MyArray<double,3> f_vy = grid_.f_vy_;
    

    double center_x = 0.02-0.5*g_dy;
    double center_y = 0.007-0.5*g_dy;
    double center_z = 0.02-0.5*g_dy;
    double r = 0.003;
    double rsq = r*r;

    double v_ini = -2.0;

    // Zalesak-like slot
    double slot_half_width = 0.00;       
    double slot_y_min = center_y;       
    double slot_y_max = center_y + r;  

    for(int iz=1; iz<Nz+1; iz++){
        for(int iy=1; iy<Ny+1; iy++){
            for(int ix=1; ix<Nx+1; ix++){

                double dx = x(ix,iy,iz) - center_x;
                double dy = y(ix,iy,iz) - center_y;
                double dz = z(ix,iy,iz) - center_z;

                bool inside_sphere = dx*dx + dy*dy + dz*dz < rsq;

                bool inside_slot =
                    fabs(dx) < slot_half_width &&
                    y(ix,iy,iz) > slot_y_min &&
                    y(ix,iy,iz) < slot_y_max;

                if (inside_sphere && !inside_slot) {
                    a(ix,iy,iz) = 1.0;
                    f_vy(ix,iy,iz) = v_ini;
                    f_vy(ix,iy+1,iz) = v_ini;

                } else if(inside_sphere && inside_slot) {
                    a(ix,iy,iz) = 0.0;
                }
            }

        }
    }
}



void SMACSolver::set_zalesak_rotation_velocity(){
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;

    double dx = grid_.dx_;
    double dy = grid_.dy_;

    MyArray<double,3>& vx = grid_.f_vx_;
    MyArray<double,3>& vy = grid_.f_vy_;
    MyArray<double,3>& vz = grid_.f_vz_;

    const double omega = 2.0 * M_PI;
    const double xc = 0.5;
    const double yc = 0.5;

    // vx/u: x-faces
    // size: ix = 0..Nx+2, iy = 0..Ny+1, iz = 0..Nz+1
    for(int iz=0; iz<Nz+2; iz++){
        for(int iy=0; iy<Ny+2; iy++){
            for(int ix=0; ix<Nx+3; ix++){
                double y = (iy - 0.5) * dy;
                vx(ix,iy,iz) = -omega * (y - yc);
            }
        }
    }

    // vy/v: y-faces
    // size: ix = 0..Nx+1, iy = 0..Ny+2, iz = 0..Nz+1
    for(int iz=0; iz<Nz+2; iz++){
        for(int iy=0; iy<Ny+3; iy++){
            for(int ix=0; ix<Nx+2; ix++){
                double x = (ix - 0.5) * dx;
                vy(ix,iy,iz) = omega * (x - xc);
            }
        }
    }

    // vz/w: z-faces
    // size: ix = 0..Nx+1, iy = 0..Ny+1, iz = 0..Nz+2
    for(int iz=0; iz<Nz+3; iz++){
        for(int iy=0; iy<Ny+2; iy++){
            for(int ix=0; ix<Nx+2; ix++){
                vz(ix,iy,iz) = 0.0;
            }
        }
    }
}


double SMACSolver::calc_alpha_vol() {
    int Nx = grid_.Nx_;
    int Ny = grid_.Ny_;
    int Nz = grid_.Nz_;

    MyArray<double,3>& a = grid_.alpha_;

    double sum = 0.0;

    for (int iz = 1; iz < Nz + 1; iz++) {
        for (int iy = 1; iy < Ny + 1; iy++) {
            for (int ix = 1; ix < Nx + 1; ix++) {
                sum += a(ix,iy,iz);
            }
        }
    }

    return sum * grid_.dx_ * grid_.dy_*grid_.dz_;
}
