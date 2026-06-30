#include "G_PressureSolverBase.h"

void G_PressureSolverBase::copyData(G_SMACSolver& solv){
    block_dim_ = solv.block_dim_;
    grid_dim_ = solv.grid_dim_;
    d_pcg_scalars_ = solv.d_pcg_scalars_;
    cub_temp_storage_=solv.cub_temp_storage_;
   cub_temp_storage_bytes_=solv.cub_temp_storage_bytes_;
    d_r2_=solv.d_r2_;
   d_dot_=solv.d_dot_;
}
