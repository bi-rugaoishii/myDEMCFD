#include "G_GMGSolver.h"

void G_GMGSolver::free_levels(){
    for(int i=0; i<num_levels_; i++){
        levels_[i].free();
    }
}

void G_GMGSolver::initialize(G_SMACSolver &solv,int num_levels){

    printf("Initializing for GMG\n");
    num_levels_ = 0;

    if(num_levels > MAX_LEVELS){
        printf("!!!!!! levels = %d is over max levels = %d \n",num_levels,MAX_LEVELS);
    }
    G_StaggeredGrid& grid = solv.grid_;
    int Nx=grid.Nx_;
    int Ny=grid.Ny_;
    int Nz=grid.Nz_;

    G_Levels& cur_level = levels_[0];

    if(Nx/2 < min_size_){
        cur_level.is_x_coarse_ = false;
    }

    if(Ny/2 < min_size_){
        cur_level.is_y_coarse_ = false;
    }

    if(Nz/2 < min_size_){
        cur_level.is_z_coarse_ = false;
    }

    #define MEMBER(type, name, xshift,yshift,zshift, isSAVE) cur_level.name.sizex_= Nx/2 + xshift;\
    if(cur_level.is_x_coarse_ == false){\
        cur_level.name.sizex_ = Nx+xshift;\
    }\
    cur_level.name.sizey_= Ny/2 + yshift;\
    if(cur_level.is_y_coarse_ == false){\
        cur_level.name.sizey_ = Ny+yshift;\
    }\
    cur_level.name.sizez_= Nz/2 + zshift;\
    if(cur_level.is_z_coarse_ == false){\
        cur_level.name.sizez_ = Nz+zshift;\
    }\
    cur_level.name.size_ = cur_level.name.sizex_*cur_level.name.sizey_*cur_level.name.sizez_;\
    cudaMalloc((void**)&cur_level.name.data_,sizeof(double)*cur_level.name.size_);
    #include "../memberList/levelMembers.def"
    #undef MEMBER


    cur_level.inv_dx2_= cur_level.is_x_coarse_? grid.inv_dx2_*0.25:grid.inv_dx2_;
    cur_level.inv_dy2_= cur_level.is_y_coarse_? grid.inv_dy2_*0.25:grid.inv_dy2_;
    cur_level.inv_dz2_= cur_level.is_z_coarse_?  grid.inv_dz2_*0.25:grid.inv_dz2_;

    printf("q_ level %d : sizex=%d, sizey=%d, sizez=%d, size=%d\n",0,cur_level.q_.sizex_,cur_level.q_.sizey_,cur_level.q_.sizez_,cur_level.q_.size_);

    num_levels_ +=1;




    for (int i=1; i<num_levels; i++){
        G_Levels& cur_level = levels_[i];
        G_Levels& old_level = levels_[i-1];

        if((old_level.q_.sizex_-2)/2 < min_size_){
            cur_level.is_x_coarse_ = false;
        }

        if((old_level.q_.sizey_-2)/2 < min_size_){
            cur_level.is_y_coarse_ = false;
        }

        if((old_level.q_.sizez_-2)/2 < min_size_){
            cur_level.is_z_coarse_ = false;
        }

        #define MEMBER(type, name, xshift,yshift,zshift, isSAVE) cur_level.name.sizex_= (old_level.name.sizex_-xshift)/2 + xshift;\
        if(cur_level.is_x_coarse_ == false){\
            cur_level.name.sizex_ = old_level.name.sizex_;\
        }\
        cur_level.name.sizey_= (old_level.name.sizey_-yshift)/2 + yshift;\
        if(cur_level.is_y_coarse_ == false){\
            cur_level.name.sizey_ = old_level.name.sizey_;\
        }\
        cur_level.name.sizez_= (old_level.name.sizez_-zshift)/2 + zshift;\
        if(cur_level.is_z_coarse_ == false){\
            cur_level.name.sizez_ = old_level.name.sizez_;\
        }\
        cur_level.name.size_ = cur_level.name.sizex_*cur_level.name.sizey_*cur_level.name.sizez_;\
        cudaMalloc((void**)&cur_level.name.data_,sizeof(double)*cur_level.name.size_);
        #include "../memberList/levelMembers.def"
        #undef MEMBER

        cur_level.inv_dx2_= cur_level.is_x_coarse_? old_level.inv_dx2_*0.25:old_level.inv_dx2_;
        cur_level.inv_dy2_= cur_level.is_y_coarse_? old_level.inv_dy2_*0.25:old_level.inv_dy2_;
        cur_level.inv_dz2_= cur_level.is_z_coarse_? old_level.inv_dz2_*0.25:old_level.inv_dz2_;

        if(!cur_level.is_x_coarse_ &&  !cur_level.is_y_coarse_ && !cur_level.is_z_coarse_ ){
            break;
        }
        printf("q_ level %d : sizex=%d, sizey=%d, sizez=%d, size=%d\n",i,cur_level.q_.sizex_,cur_level.q_.sizey_,cur_level.q_.sizez_,cur_level.q_.size_);
        num_levels_ +=1;
    }



    printf("Initialization for GMG done!\n");
    printf("\n");
}


void G_GMGSolver::solve(G_SMACSolver& solv){};

static __global__ void k_get_res_levels(G_Levels levels){
   int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell
   int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
   int ix = blockIdx.x*blockDim.x + threadIdx.x+1;



    MyArray<double,3> &q = levels.q_;
    MyArray<double,3> &rhs = levels.rhs_;
    MyArray<double,3> &residue = levels.residue_;

    if (iy >=q.sizey_-1 || ix >= q.sizex_-1 ||iz >= q.sizez_-1) return;

    double Axp = levels.Axp_(ix,iy,iz);
    double Axm = levels.Axm_(ix,iy,iz);
    double Ayp = levels.Ayp_(ix,iy,iz);
    double Aym = levels.Aym_(ix,iy,iz);
    double Azp = levels.Azp_(ix,iy,iz);
    double Azm = levels.Azm_(ix,iy,iz);
    double Adiag = levels.Adiag_(ix,iy,iz);

    if (ix == 1 && iy == 1&& iz == 1){
        residue(ix, iy,iz) = 0.0;
        return;
    }


    double p_c = q(ix,iy,iz);
    double p_xp = q(ix+1,iy,iz);
    double p_xm = q(ix-1,iy,iz);
    double p_yp = q(ix,iy+1,iz);
    double p_ym = q(ix,iy-1,iz);
    double p_zp = q(ix,iy,iz+1);
    double p_zm = q(ix,iy,iz-1);


    double Ap = p_c*Adiag-(Axp*p_xp + Axm*p_xm 
            + Ayp*p_yp + Aym*p_ym + Azp*p_zp + Azm*p_zm);

    residue(ix,iy,iz) = rhs(ix,iy,iz)-Ap;
}


static __global__ void k_get_res_general(G_StaggeredGrid grid, MyArray<double,3> p, MyArray<double,3>rhs){
   int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

    int Nx=grid.Nx_;
    int Ny=grid.Ny_;
    int Nz=grid.Nz_;

    MyArray<double,3> &residue = grid.residue_;

    if (iy >=Ny+1 || ix >= Nx+1|| iz >=Nz+1) return;

    if (ix == 1 && iy == 1&& iz == 1){
        residue(ix, iy,iz) = 0.0;
        return;
    }

    double Axp = grid.Axp_(ix,iy,iz);
    double Axm = grid.Axm_(ix,iy,iz);
    double Ayp = grid.Ayp_(ix,iy,iz);
    double Aym = grid.Aym_(ix,iy,iz);
    double Azp = grid.Azp_(ix,iy,iz);
    double Azm = grid.Azm_(ix,iy,iz);
    double Adiag = grid.Adiag_(ix,iy,iz);

    double p_c = p(ix,iy,iz);
    double p_xp = p(ix+1,iy,iz);
    double p_xm = p(ix-1,iy,iz);
    double p_yp = p(ix,iy+1,iz);
    double p_ym = p(ix,iy-1,iz);
    double p_zp = p(ix,iy,iz+1);
    double p_zm = p(ix,iy,iz-1);


    double Ap = p_c*Adiag-(Axp*p_xp + Axm*p_xm 
            + Ayp*p_yp + Aym*p_ym + Azp*p_zp + Azm*p_zm);

    residue(ix,iy,iz) = rhs(ix,iy,iz)-Ap;
}


static __global__ void k_get_res(G_StaggeredGrid grid){
   int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;

    int Nx=grid.Nx_;
    int Ny=grid.Ny_;
    int Nz=grid.Nz_;

    const MyArray<double,3> &rhs = grid.rhs_;
    MyArray<double,3> &residue = grid.residue_;
    const MyArray<double,3>p= grid.p_;

    if (iy >=Ny+1 || ix >= Nx+1|| iz >=Nz+1) return;


    if (ix == 1 && iy == 1&& iz == 1){
        residue(ix, iy,iz) = 0.0;
        return;
    }

    double Axp = grid.Axp_(ix,iy,iz);
    double Axm = grid.Axm_(ix,iy,iz);
    double Ayp = grid.Ayp_(ix,iy,iz);
    double Aym = grid.Aym_(ix,iy,iz);
    double Azp = grid.Azp_(ix,iy,iz);
    double Azm = grid.Azm_(ix,iy,iz);
    double Adiag = grid.Adiag_(ix,iy,iz);

    double p_c = p(ix,iy,iz);
    double p_xp = p(ix+1,iy,iz);
    double p_xm = p(ix-1,iy,iz);
    double p_yp = p(ix,iy+1,iz);
    double p_ym = p(ix,iy-1,iz);
    double p_zp = p(ix,iy,iz+1);
    double p_zm = p(ix,iy,iz-1);


    double Ap = p_c*Adiag-(Axp*p_xp + Axm*p_xm 
            + Ayp*p_yp + Aym*p_ym + Azp*p_zp + Azm*p_zm);

    residue(ix,iy,iz) = rhs(ix,iy,iz)-Ap;
}

//static __global__ void k_get_sqr_to_tmp(const MyArray<double,2> q,MyArray<double,2> tmp){
//    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
//    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
//
//
//    if (iy >=q.sizey_-1 || ix >= q.sizex_-1) return;
//
//    double b = q(ix,iy); 
//
//    tmp(ix-1,iy-1) = b*b;
//}
//
//

/* == propagation == */

static __global__ void k_propagate_level0_to_array(G_Levels levels, MyArray<double,3> array){
    int ix = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z + 1;

    MyArray<double,3>& coarse_q = levels.q_;

    if (iy >=coarse_q.sizey_-1 || ix >= coarse_q.sizex_-1 || iz >= coarse_q.sizez_-1) return;

    bool is_x_coarse = levels.is_x_coarse_;
    bool is_y_coarse = levels.is_y_coarse_;
    bool is_z_coarse = levels.is_z_coarse_;

    int IX = is_x_coarse? 2*ix-1:ix;
    int IY = is_y_coarse? 2*iy-1:iy;
    int IZ = is_z_coarse? 2*iz-1:iz;

    int rangex = is_x_coarse ? 2 : 1;
    int rangey = is_y_coarse ? 2 : 1;
    int rangez = is_z_coarse ? 2 : 1;

    double prop = coarse_q(ix, iy,iz);

    for (int incz=0; incz<rangez; incz++){
        for (int incy=0; incy<rangey; incy++){
            for (int incx=0; incx<rangex; incx++){
                array(IX+incx,IY+incy,IZ+incz) += prop;
            }
        }
    }

}  


static __global__ void k_propagate_level0_to_grid(G_Levels levels, G_StaggeredGrid grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z + 1;

    MyArray<double,3>& fine_q = grid.p_;
    MyArray<double,3>& coarse_q = levels.q_;

    if (iy >=coarse_q.sizey_-1 || ix >= coarse_q.sizex_-1 || iz >= coarse_q.sizez_-1) return;




    bool is_x_coarse = levels.is_x_coarse_;
    bool is_y_coarse = levels.is_y_coarse_;
    bool is_z_coarse = levels.is_z_coarse_;

    int IX = is_x_coarse? 2*ix-1:ix;
    int IY = is_y_coarse? 2*iy-1:iy;
    int IZ = is_z_coarse? 2*iz-1:iz;

    int rangex = is_x_coarse ? 2 : 1;
    int rangey = is_y_coarse ? 2 : 1;
    int rangez = is_z_coarse ? 2 : 1;

    double prop = coarse_q(ix, iy,iz);

    for (int incz=0; incz<rangez; incz++){
        for (int incy=0; incy<rangey; incy++){
            for (int incx=0; incx<rangex; incx++){
                fine_q(IX+incx,IY+incy,IZ+incz) += prop;
            }
        }
    }
}  

static __global__ void k_propagate_level_to_level(G_Levels levels, G_Levels fine_levels){
    int ix = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z + 1;

    MyArray<double,3>& fine_q = fine_levels.q_;
    MyArray<double,3>& coarse_q = levels.q_;

    if (iy >=coarse_q.sizey_-1 || ix >= coarse_q.sizex_-1 || iz >= coarse_q.sizez_-1) return;


    bool is_x_coarse = levels.is_x_coarse_;
    bool is_y_coarse = levels.is_y_coarse_;
    bool is_z_coarse = levels.is_z_coarse_;

    int IX = is_x_coarse? 2*ix-1:ix;
    int IY = is_y_coarse? 2*iy-1:iy;
    int IZ = is_z_coarse? 2*iz-1:iz;

    int rangex = is_x_coarse ? 2 : 1;
    int rangey = is_y_coarse ? 2 : 1;
    int rangez = is_z_coarse ? 2 : 1;

    double prop = coarse_q(ix, iy,iz);

    for (int incz=0; incz<rangez; incz++){
        for (int incy=0; incy<rangey; incy++){
            for (int incx=0; incx<rangex; incx++){

                fine_q(IX+incx,IY+incy,IZ+incz) += prop;
            }
        }
    }

}  

//
///* =================================
//   restriction
//   ===============================*/
///*
//   static __global__ void k_restrict_array_to_level0(MyArray<double,2> array, G_Levels levels_){
//   int ix = blockIdx.x*blockDim.x + threadIdx.x + 1;
//   int iy = blockIdx.y*blockDim.y + threadIdx.y + 1;
//
//   MyArray<double,2>& coarse_rhs = levels_.rhs_[0];
//
//   if (iy >=coarse_rhs.sizey_-1 || ix >= coarse_rhs.sizex_-1) return;
//
//
//   int IX = 2*ix-1;
//   int IY = 2*iy-1;
//
//   if(ix==1 && iy==1){
//   coarse_rhs(1,1)=0.0;
//   }
//
//   coarse_rhs(ix,iy) = 0.25*(array(IX,IY)+array(IX,IY+1)+array(IX+1,IY)+array(IX+1,IY+1));
//   }  
// */
//
static __global__ void k_restrict_grid_to_level0(G_StaggeredGrid grid, G_Levels levels){
    int ix = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z + 1;

    MyArray<double,3>& residue = grid.residue_;
    MyArray<double,3>& coarse_rhs = levels.rhs_;

    if (iy >=coarse_rhs.sizey_-1 || ix >= coarse_rhs.sizex_-1 || iz >= coarse_rhs.sizez_-1) return;



    if(ix==1 && iy==1 && iz == 1){
        coarse_rhs(1,1,1)=0.0;
        return ;
    }
    bool is_x_coarse = levels.is_x_coarse_;
    bool is_y_coarse = levels.is_y_coarse_;
    bool is_z_coarse = levels.is_z_coarse_;

    int IX = is_x_coarse? 2*ix-1:ix;
    int IY = is_y_coarse? 2*iy-1:iy;
    int IZ = is_z_coarse? 2*iz-1:iz;

    int rangex = is_x_coarse ? 2 : 1;
    int rangey = is_y_coarse ? 2 : 1;
    int rangez = is_z_coarse ? 2 : 1;


    double count = 0.;
    double tmp = 0.;
    for (int incz=0; incz<rangez; incz++){
        for (int incy=0; incy<rangey; incy++){
            for (int incx=0; incx<rangex; incx++){
                tmp += residue(IX+incx,IY+incy,IZ+incz);
                count += 1.0;
            }
        }
    }


    coarse_rhs(ix,iy,iz) = tmp/count;

}  

static __global__ void k_restrict_level_to_level(G_Levels levels, G_Levels  next_level){
    int ix = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z + 1;

    MyArray<double,3>& residue = levels.residue_;
    MyArray<double,3>& coarse_rhs = next_level.rhs_;

    if (iy >=coarse_rhs.sizey_-1 || ix >= coarse_rhs.sizex_-1 || iz >= coarse_rhs.sizez_ -1) return;

    if(ix==1 && iy==1 && iz==1){
        coarse_rhs(1,1,1)=0.0;
        return;
    }

    bool is_x_coarse = next_level.is_x_coarse_;
    bool is_y_coarse = next_level.is_y_coarse_;
    bool is_z_coarse = next_level.is_z_coarse_;

    int IX = is_x_coarse? 2*ix-1:ix;
    int IY = is_y_coarse? 2*iy-1:iy;
    int IZ = is_z_coarse? 2*iz-1:iz;

    int rangex = is_x_coarse ? 2 : 1;
    int rangey = is_y_coarse ? 2 : 1;
    int rangez = is_z_coarse ? 2 : 1;


    double count = 0.;
    double tmp = 0.;
    for (int incz=0; incz<rangez; incz++){
        for (int incy=0; incy<rangey; incy++){
            for (int incx=0; incx<rangex; incx++){
                tmp += residue(IX+incx,IY+incy,IZ+incz);
                count += 1.0;
            }
        }
    }

    coarse_rhs(ix,iy,iz) = tmp/count;
}  

///* =================================
//   coarse grid coefficients
//   ===============================*/
//
static __global__ void k_create_level_coeffs(G_Levels levels, G_Levels fine_levels){
    int ix = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z + 1;

    MyArray<double,3>& f_bx= levels.f_bx_;
    MyArray<double,3>& p_f_bx= fine_levels.f_bx_;

    MyArray<double,3>& f_by= levels.f_by_;
    MyArray<double,3>& p_f_by= fine_levels.f_by_;

    MyArray<double,3>& f_bz= levels.f_bz_;
    MyArray<double,3>& p_f_bz= fine_levels.f_bz_;

    bool is_x_coarse = levels.is_x_coarse_;
    bool is_y_coarse = levels.is_y_coarse_;
    bool is_z_coarse = levels.is_z_coarse_;

    int IX = is_x_coarse? 2*ix-1:ix;
    int IY = is_y_coarse? 2*iy-1:iy;
    int IZ = is_z_coarse? 2*iz-1:iz;

    int rangex = is_x_coarse ? 2 : 1;
    int rangey = is_y_coarse ? 2 : 1;
    int rangez = is_z_coarse ? 2 : 1;



    if (ix <f_bx.sizex_-1 && iy < f_bx.sizey_-1 && iz < f_bx.sizez_-1){
        double count = 0.;
        double tmp = 0.;
        for (int incz=0; incz<rangez; incz++){
            for (int incy=0; incy<rangey; incy++){
                for (int incx=0; incx<rangex; incx++){
                    tmp += p_f_bx(IX+incx ,IY+incy, IZ+incz);
                    count += 1.0;
                }
            }
        }
        f_bx(ix,iy,iz) = tmp/count;
    }

    if (ix <f_by.sizex_-1 && iy < f_by.sizey_-1 && iz < f_by.sizez_-1){
        double count = 0.;
        double tmp = 0.;
        for (int incz=0; incz<rangez; incz++){
            for (int incy=0; incy<rangey; incy++){
                for (int incx=0; incx<rangex; incx++){
                    tmp += p_f_by(IX+incx ,IY+incy, IZ+incz);
                    count += 1.0;
                }
            }
        }
        f_by(ix,iy,iz) = tmp/count;
    }

    if (ix <f_bz.sizex_-1 && iy < f_bz.sizey_-1 && iz < f_bz.sizez_-1){
        double count = 0.;
        double tmp = 0.;
        for (int incz=0; incz<rangez; incz++){
            for (int incy=0; incy<rangey; incy++){
                for (int incx=0; incx<rangex; incx++){
                    tmp += p_f_bz(IX+incx ,IY+incy, IZ+incz);
                    count += 1.0;
                }
            }
        }
        f_bz(ix,iy,iz) = tmp/count;
    }

}


static __global__ void k_create_level0_coeffs(G_StaggeredGrid grid,G_Levels levels){
    int ix = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z + 1;


    MyArray<double,3>& f_bx= levels.f_bx_;
    MyArray<double,3>& p_f_bx= grid.f_bx_;

    MyArray<double,3>& f_by= levels.f_by_;
    MyArray<double,3>& p_f_by= grid.f_by_;

    MyArray<double,3>& f_bz= levels.f_bz_;
    MyArray<double,3>& p_f_bz= grid.f_bz_;

    bool is_x_coarse = levels.is_x_coarse_;
    bool is_y_coarse = levels.is_y_coarse_;
    bool is_z_coarse = levels.is_z_coarse_;

    int IX = is_x_coarse? 2*ix-1:ix;
    int IY = is_y_coarse? 2*iy-1:iy;
    int IZ = is_z_coarse? 2*iz-1:iz;

    int rangex = is_x_coarse ? 2 : 1;
    int rangey = is_y_coarse ? 2 : 1;
    int rangez = is_z_coarse ? 2 : 1;



    if (ix <f_bx.sizex_-1 && iy < f_bx.sizey_-1 && iz < f_bx.sizez_-1){
        double count = 0.;
        double tmp = 0.;
        for (int incz=0; incz<rangez; incz++){
            for (int incy=0; incy<rangey; incy++){
                for (int incx=0; incx<rangex; incx++){
                    tmp += p_f_bx(IX+incx ,IY+incy, IZ+incz);
                    count += 1.0;
                }
            }
        }
        f_bx(ix,iy,iz) = tmp/count;
    }

    if (ix <f_by.sizex_-1 && iy < f_by.sizey_-1 && iz < f_by.sizez_-1){
        double count = 0.;
        double tmp = 0.;
        for (int incz=0; incz<rangez; incz++){
            for (int incy=0; incy<rangey; incy++){
                for (int incx=0; incx<rangex; incx++){
                    tmp += p_f_by(IX+incx ,IY+incy, IZ+incz);
                    count += 1.0;
                }
            }
        }
        f_by(ix,iy,iz) = tmp/count;
    }

    if (ix <f_bz.sizex_-1 && iy < f_bz.sizey_-1 && iz < f_bz.sizez_-1){
        double count = 0.;
        double tmp = 0.;
        for (int incz=0; incz<rangez; incz++){
            for (int incy=0; incy<rangey; incy++){
                for (int incx=0; incx<rangex; incx++){
                    tmp += p_f_bz(IX+incx ,IY+incy, IZ+incz);
                    count += 1.0;
                }
            }
        }
        f_bz(ix,iy,iz) = tmp/count;
    }
    //printf("%d %d %d p_bx %e \n",ix,iy,iz,p_f_bx(IX,IY,IZ));
    //printf("%d %d %d f_bx %e f_by %e f_bz %e\n",ix,iy,iz,f_bx(ix,iy,iz),f_by(ix,iy,iz),f_bz(ix,iy,iz));
}

static __global__ void k_create_level_A(G_Levels levels){
    int ix = blockIdx.x*blockDim.x + threadIdx.x + 1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y + 1;
    int iz = blockIdx.z*blockDim.z + threadIdx.z + 1;

    MyArray<double,3> Axp = levels.Axp_;
    MyArray<double,3> Axm = levels.Axm_;
    MyArray<double,3> Ayp = levels.Ayp_;
    MyArray<double,3> Aym = levels.Aym_;
    MyArray<double,3> Azp = levels.Azp_;
    MyArray<double,3> Azm = levels.Azm_;
    MyArray<double,3> Adiag = levels.Adiag_;
    MyArray<double,3> invAdiag = levels.invAdiag_;

    double inv_dx2 = levels.inv_dx2_;
    double inv_dy2 = levels.inv_dy2_;
    double inv_dz2 = levels.inv_dz2_;


    MyArray<double,3> f_bx = levels.f_bx_;
    MyArray<double,3> f_by = levels.f_by_;
    MyArray<double,3> f_bz = levels.f_bz_;

    if (ix>= Axp.sizex_-1 ||iy >=Axp.sizey_-1 || iz >= Axp.sizez_-1) return;

    Axp(ix,iy,iz) = f_bx(ix+1,iy,iz)*inv_dx2;
    Axm(ix,iy,iz) = f_bx(ix,iy,iz)*inv_dx2;
    Ayp(ix,iy,iz) = f_by(ix,iy+1,iz)*inv_dy2;
    Aym(ix,iy,iz) = f_by(ix,iy,iz)*inv_dy2;
    Azp(ix,iy,iz) = f_bz(ix,iy,iz+1)*inv_dz2;
    Azm(ix,iy,iz) = f_bz(ix,iy,iz)*inv_dz2;
    Adiag(ix,iy,iz) = (Axp(ix,iy,iz)+Axm(ix,iy,iz)+Ayp(ix,iy,iz)+Aym(ix,iy,iz)+Azp(ix,iy,iz)+Azm(ix,iy,iz));
    invAdiag(ix,iy,iz) = 1./Adiag(ix,iy,iz);


}


/* =======================
   Jacobi Smoother
   ===================== */

static __global__ void k_jacobi_level_iteration(G_Levels levels){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell

    double omega=0.6;


    MyArray<double,3>& p=levels.q_;
    MyArray<double,3>& p_tmp=levels.q_tmp_;
    MyArray<double,3>& rhs=levels.rhs_;

    if (iy >=p.sizey_-1 || ix >=p.sizex_-1 || iz >=p.sizez_-1) return;

    double Axp = levels.Axp_(ix,iy,iz);
    double Axm = levels.Axm_(ix,iy,iz);
    double Ayp = levels.Ayp_(ix,iy,iz);
    double Aym = levels.Aym_(ix,iy,iz);
    double Azp = levels.Azp_(ix,iy,iz);
    double Azm = levels.Azm_(ix,iy,iz);
    double invAdiag = levels.invAdiag_(ix,iy,iz);

    double tmp_p =Axp*p(ix+1,iy,iz)+Axm*p(ix-1,iy,iz)+Ayp*p(ix,iy+1,iz)+Aym*p(ix,iy-1,iz)+Azp*p(ix,iy,iz+1)+Azm*p(ix,iy,iz-1);

    p_tmp(ix,iy,iz)=(1.0-omega)*p(ix,iy,iz)+omega*(tmp_p+rhs(ix,iy,iz))*invAdiag;
}

static __global__ void k_jacobi_iteration_general(G_StaggeredGrid grid,MyArray<double,3> p, MyArray<double,3> rhs){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell

    double omega=0.6;


    MyArray<double,3>& p_tmp=grid.p_tmp_;

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;

    if (iy >=Ny+1 || ix >= Nx+1 || iz >= Nz+1) return;

    double Axp = grid.Axp_(ix,iy,iz);
    double Axm = grid.Axm_(ix,iy,iz);
    double Ayp = grid.Ayp_(ix,iy,iz);
    double Aym = grid.Aym_(ix,iy,iz);
    double Azp = grid.Azp_(ix,iy,iz);
    double Azm = grid.Azm_(ix,iy,iz);
    double invAdiag = grid.invAdiag_(ix,iy,iz);

    double tmp_p =Axp*p(ix+1,iy,iz)+Axm*p(ix-1,iy,iz)+Ayp*p(ix,iy+1,iz)+Aym*p(ix,iy-1,iz)+Azp*p(ix,iy,iz+1)+Azm*p(ix,iy,iz-1);

    p_tmp(ix,iy,iz)=(1.0-omega)*p(ix,iy,iz)+omega*(tmp_p+rhs(ix,iy,iz))*invAdiag;

}


static __global__ void k_jacobi_iteration(G_StaggeredGrid grid){
    int ix = blockIdx.x*blockDim.x + threadIdx.x+1;
    int iy = blockIdx.y*blockDim.y + threadIdx.y+1; //+1 for ghost cell
    int iz = blockIdx.z*blockDim.z + threadIdx.z+1; //+1 for ghost cell

    int Nx=grid.Nx_;
    int Ny=grid.Ny_;
    int Nz=grid.Nz_;

    double omega=0.6;

    MyArray<double,3>& p=grid.p_;
    MyArray<double,3>& p_tmp=grid.p_tmp_;
    MyArray<double,3>& rhs=grid.rhs_;

    if (iy >=Ny+1 || ix >= Nx+1 || iz >= Nz+1) return;

    double Axp = grid.Axp_(ix,iy,iz);
    double Axm = grid.Axm_(ix,iy,iz);
    double Ayp = grid.Ayp_(ix,iy,iz);
    double Aym = grid.Aym_(ix,iy,iz);
    double Azp = grid.Azp_(ix,iy,iz);
    double Azm = grid.Azm_(ix,iy,iz);
    double invAdiag = grid.invAdiag_(ix,iy,iz);

    double tmp_p =Axp*p(ix+1,iy,iz)+Axm*p(ix-1,iy,iz)+Ayp*p(ix,iy+1,iz)+Aym*p(ix,iy-1,iz)+Azp*p(ix,iy,iz+1)+Azm*p(ix,iy,iz-1);
    p_tmp(ix,iy,iz)=(1.0-omega)*p(ix,iy,iz)+omega*(tmp_p+rhs(ix,iy,iz))*invAdiag;
}
//
//
///* ====================================
//   Boundary Conditions / Reference Fix
//   ============================= */
//
//static __global__ void k_set_boundary_and_fix_general(MyArray<double,2> q){
//    int ix = blockIdx.x*blockDim.x + threadIdx.x;
//    int iy = blockIdx.y*blockDim.y + threadIdx.y; //+1 for ghost cell
//
//    int sizex = q.sizex_;
//    int sizey = q.sizey_;
//
//    if (iy >= sizey|| ix >= sizex) return;
//
//    /*
//       Fix pressure reference at internal cell q(1,1)=0.
//
//       q(0,1), q(1,0), q(0,0) are ghost cells.
//       They are also set to 0 to keep Neumann consistency around
//       the fixed reference cell without inter-thread dependency.
//     */
//
//    if ((ix == 0 || ix == 1) && (iy == 0 || iy == 1)) {
//        q(ix,iy) = 0.0;
//        return;
//    }
//
//
//
//    if (ix == 0 && iy == sizey-1){
//        q(0, sizey-1) = q(1,sizey-2);
//        return;
//    }
//
//    if (ix == sizex-1 && iy == 0){
//        q(sizex-1,0) = q(sizex-2,1);
//        return;
//    }
//
//    if (ix == sizex-1 && iy == sizey-1){
//        q(sizex-1,sizey-1) = q(sizex-2,sizey-2);
//        return;
//    }
//
//    if (ix == 0){
//        q(ix,iy) = q(ix+1,iy);
//        return;
//    }
//
//    if (iy == 0){
//        q(ix,iy) = q(ix,iy+1);
//        return;
//    }
//
//    if (ix == sizex-1){
//        q(ix,iy) = q(ix-1,iy);
//        return;
//    }
//
//    if (iy == sizey-1){
//        q(ix,iy) = q(ix,iy-1);
//        return;
//    }
//}
//
//static __global__ void k_set_level_boundary_and_fix(G_Levels levels_, int cur_level){
//    int ix = blockIdx.x*blockDim.x + threadIdx.x;
//    int iy = blockIdx.y*blockDim.y + threadIdx.y; //+1 for ghost cell
//
//    MyArray<double,2>& q = levels_.q_[cur_level];
//    int sizex = q.sizex_;
//    int sizey = q.sizey_;
//
//    if (iy >= sizey|| ix >= sizex) return;
//
//    /*
//       Fix pressure reference at internal cell q(1,1)=0.
//
//       q(0,1), q(1,0), q(0,0) are ghost cells.
//       They are also set to 0 to keep Neumann consistency around
//       the fixed reference cell without inter-thread dependency.
//     */
//
//    if ((ix == 0 || ix == 1) && (iy == 0 || iy == 1)) {
//        q(ix,iy) = 0.0;
//        return;
//    }
//
//
//
//    if (ix == 0 && iy == sizey-1){
//        q(0, sizey-1) = q(1,sizey-2);
//        return;
//    }
//
//    if (ix == sizex-1 && iy == 0){
//        q(sizex-1,0) = q(sizex-2,1);
//        return;
//    }
//
//    if (ix == sizex-1 && iy == sizey-1){
//        q(sizex-1,sizey-1) = q(sizex-2,sizey-2);
//        return;
//    }
//
//    if (ix == 0){
//        q(ix,iy) = q(ix+1,iy);
//        return;
//    }
//
//    if (iy == 0){
//        q(ix,iy) = q(ix,iy+1);
//        return;
//    }
//
//    if (ix == sizex-1){
//        q(ix,iy) = q(ix-1,iy);
//        return;
//    }
//
//    if (iy == sizey-1){
//        q(ix,iy) = q(ix,iy-1);
//        return;
//    }
//}
//
//
//
//
void G_GMGSolver::coarse_zero_clear(){
    for (int i=0; i< num_levels_; i++){
        cudaMemset(levels_[i].q_.data_,0,sizeof(double)*levels_[i].q_.size_);
    }
}

void G_GMGSolver::create_coeffs(G_StaggeredGrid& grid){
    k_create_level0_coeffs<<<grid_dim_,block_dim_>>>(grid,levels_[0]);
    k_create_level_A<<<grid_dim_,block_dim_>>>(levels_[0]);

    for (int cur_level=1; cur_level < num_levels_; cur_level++){
        k_create_level_coeffs<<<grid_dim_,block_dim_>>>(levels_[cur_level],levels_[cur_level-1]);
        k_create_level_A<<<grid_dim_,block_dim_>>>(levels_[cur_level]);
    }

}


void G_GMGSolver::v_cycle_as_preconditioner(G_StaggeredGrid& grid, MyArray<double,3> &q,MyArray<double,3> &rhs){
    int num_iter_fine = num_iter_fine_;
    int num_iter_coarse = num_iter_coarse_;


    int num_levels = num_levels_;




    /* initialize */
    cudaMemset(q.data_,0,sizeof(double)*q.size_);
    this->coarse_zero_clear();


    /* initialize done */

    /* === finest level pre smoothing === */

    for (int iter = 0; iter < num_iter_fine; iter++){
        k_jacobi_iteration_general<<<grid_dim_,block_dim_>>>(grid,q,rhs);
        std::swap(grid.p_tmp_.data_,q.data_);

    }

    k_get_res_general<<<grid_dim_,block_dim_>>>(grid,q,rhs);

    k_restrict_grid_to_level0<<<grid_dim_,block_dim_>>>(grid,levels_[0]);

    for (int cur_level = 0; cur_level < num_levels_-1; cur_level++){
        for (int iter = 0; iter < num_iter_fine; iter++){
            k_jacobi_level_iteration<<<grid_dim_,block_dim_>>>(levels_[cur_level]);
            std::swap(levels_[cur_level].q_.data_,levels_[cur_level].q_tmp_.data_);
        }

        k_get_res_levels<<<grid_dim_,block_dim_>>>(levels_[cur_level]);
        k_restrict_level_to_level<<<grid_dim_,block_dim_>>>(levels_[cur_level],levels_[cur_level+1]);
    }

    for (int iter = 0; iter < num_iter_coarse; iter++){
        int cur_level = num_levels-1;

        k_jacobi_level_iteration<<<grid_dim_,block_dim_>>>(levels_[cur_level]);
        std::swap(levels_[cur_level].q_.data_,levels_[cur_level].q_tmp_.data_);

    }




    /* == propagation == */

    k_propagate_level_to_level<<<grid_dim_,block_dim_>>>(levels_[num_levels-1],levels_[num_levels-2]);

    for (int cur_level = num_levels-2; cur_level >= 0 ; cur_level--){
        for (int iter = 0; iter < num_iter_fine; iter++){
            k_jacobi_level_iteration<<<grid_dim_,block_dim_>>>(levels_[cur_level]);
            std::swap(levels_[cur_level].q_.data_,levels_[cur_level].q_tmp_.data_);
        }

        if (cur_level !=0){
            k_propagate_level_to_level<<<grid_dim_,block_dim_>>>(levels_[cur_level],levels_[cur_level-1]);
        }else{
            k_propagate_level0_to_array<<<grid_dim_,block_dim_>>>(levels_[cur_level],q);
        }
    }


    /* == post smoothing == */
    for (int iter = 0; iter < num_iter_fine; iter++){
        k_jacobi_iteration_general<<<grid_dim_,block_dim_>>>(grid,q,rhs);
        std::swap(grid.p_tmp_.data_,q.data_);
    }
}


void G_GMGSolver::v_cycle(G_StaggeredGrid& grid){
    int num_iter_fine = num_iter_fine_;
    int num_iter_coarse = num_iter_coarse_;


    int num_levels = num_levels_;




    /* === finest level pre smoothing === */

    for (int iter = 0; iter < num_iter_fine; iter++){
        k_jacobi_iteration<<<grid_dim_,block_dim_>>>(grid);
        std::swap(grid.p_tmp_.data_,grid.p_.data_);
    }

    k_get_res<<<grid_dim_,block_dim_>>>(grid);

    k_restrict_grid_to_level0<<<grid_dim_,block_dim_>>>(grid,levels_[0]);

    /* == mid coarse level smoothing == */
    for (int cur_level = 0; cur_level < num_levels_-1; cur_level++){
        for (int iter = 0; iter < num_iter_fine; iter++){
            k_jacobi_level_iteration<<<grid_dim_,block_dim_>>>(levels_[cur_level]);
            std::swap(levels_[cur_level].q_.data_,levels_[cur_level].q_tmp_.data_);
        }

        k_get_res_levels<<<grid_dim_,block_dim_>>>(levels_[cur_level]);
        k_restrict_level_to_level<<<grid_dim_,block_dim_>>>(levels_[cur_level],levels_[cur_level+1]);
    }

    /* == top coarse level smoothing == */
    for (int iter = 0; iter < num_iter_coarse; iter++){
        int cur_level = num_levels-1;

        k_jacobi_level_iteration<<<grid_dim_,block_dim_>>>(levels_[cur_level]);
        std::swap(levels_[cur_level].q_.data_,levels_[cur_level].q_tmp_.data_);
    }




    /* == propagation == */

    k_propagate_level_to_level<<<grid_dim_,block_dim_>>>(levels_[num_levels-1],levels_[num_levels-2]);

    for (int cur_level = num_levels-2; cur_level >= 0 ; cur_level--){
        for (int iter = 0; iter < num_iter_fine; iter++){
            k_jacobi_level_iteration<<<grid_dim_,block_dim_>>>(levels_[cur_level]);
            std::swap(levels_[cur_level].q_.data_,levels_[cur_level].q_tmp_.data_);
        }

        if (cur_level !=0){
            k_propagate_level_to_level<<<grid_dim_,block_dim_>>>(levels_[cur_level],levels_[cur_level-1]);
        }else{
            k_propagate_level0_to_grid<<<grid_dim_,block_dim_>>>(levels_[cur_level],grid);

        }
    }


    /* == post smoothing == */
    for (int iter = 0; iter < num_iter_fine; iter++){
        k_jacobi_iteration<<<grid_dim_,block_dim_>>>(grid);
        std::swap(grid.p_tmp_.data_,grid.p_.data_);
    }
}


//void G_GMGSolver::solve(G_SMACSolver& solv){
//    G_StaggeredGrid& grid_=solv.grid_;
//    double inv_dt_ = solv.inv_dt_;
//
//    double tol = 1e-6;
//    int max_v_cycle_iter = 1000;
//
//    int Nx = grid_.Nx_;
//    int Ny = grid_.Ny_;
//
//    int block_size=256;
//    int n=max(Nx,Ny);
//    int grid_size_boundary=(n+block_size-1)/block_size;
//
//    pres_k_shift_pressure_reference<<<grid_dim_,block_dim_>>>(grid_,Nx,Ny);
//    pres_k_fix_pressure_reference<<<1,1>>>(grid_,Nx,Ny);
//    pres_k_set_boundary_array<<<grid_size_boundary,block_size>>>(grid_.p_,Nx, Ny);
//
//    // rhs = -div(u_star) (negated like a PCG method)
//    pres_k_make_poisson_rhs<<<grid_dim_,block_dim_>>>(grid_,inv_dt_);
//
//
//    /* get mean_b */
//    pres_k_copy_to_tmp<<<grid_dim_,block_dim_>>>(grid_.rhs_,grid_.pcg_tmp_,Nx,Ny);
//    cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_,d_dot_,Nx*Ny);
//
//    /* mean_b is in d_dot_ */
//    int n_all=grid_.wew_rhs_.size_;
//    MyArray<double,2>& rhs = grid_.wew_rhs_;
//    int grid_size_all=(n_all+block_size-1)/block_size;
//
//    /* subtract mean_b */
//    double size_inv = 1./(double)(Nx*Ny);
//    pres_k_add_scalar_to_array<<<grid_size_all,block_size>>>(-1.*size_inv,d_dot_,rhs.data_,n_all);
//
//    /* get norm_b */
//    k_get_sqr_to_tmp<<<grid_dim_,block_dim_>>>(grid_.wew_rhs_,grid_.wew_pcg_tmp_);
//    cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_,d_dot_,Nx*Ny);
//
//    /* sum b2 is in d_dot_ */
//    double norm_b;
//    cudaMemcpy(&norm_b,d_dot_,sizeof(double),cudaMemcpyDeviceToHost);
//    norm_b=sqrt(norm_b);
//
//    if (norm_b < 1.0e-16){
//        norm_b = 1.0;
//    }
//
//    k_create_level0_coeffs<<<grid_dim_,block_dim_>>>(grid_,levels_);
//
//    for (int cur_level=1; cur_level < levels_.num_levels_; cur_level++){
//        k_create_level_coeffs<<<grid_dim_, block_dim_>>>(levels_,cur_level);
//    }
//
//
//    for(int v_cycle_iter =0; v_cycle_iter< max_v_cycle_iter; ++v_cycle_iter){
//
//        this->coarse_zero_clear();
//
//        this->v_cycle(grid_);
//
//
//        k_get_res<<<grid_dim_,block_dim_>>>(grid_);
//        k_get_sqr_to_tmp<<<grid_dim_,block_dim_>>>(grid_.wew_residue_,grid_.wew_pcg_tmp_);
//        cub::DeviceReduce::Sum(cub_temp_storage_, cub_temp_storage_bytes_,grid_.pcg_tmp_,d_dot_,Nx*Ny);
//        double r2;
//        cudaMemcpy(&r2,d_dot_,sizeof(double),cudaMemcpyDeviceToHost);
//        double rel_res = sqrt(r2) / norm_b;
//        if (rel_res < tol) {
//            printf("norm r = %3.2e, norm b = %3.2e, rel_res = %3.2e \n",sqrt(r2),norm_b,rel_res);
//            printf("v-cycle iter=%d\n",v_cycle_iter);
//            break;
//        }
//    }
//
//}
