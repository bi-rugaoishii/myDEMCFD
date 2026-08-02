#pragma once
#include "../CFD/G_StaggeredGrid.h"
#include "../DEM/ParticleSystem.h"

struct DEMCFDCoupling{

    void get_index_of_Cell(G_StaggeredGrid grid, ParticleSys<DeviceMemory> ps);

}
