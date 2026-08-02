#pragma once
#include "../CFD/G_StaggeredGrid.h"
#include "../DEM/ParticleSystem.h"

struct G_DEMCFDCoupling{

    void get_index_of_Cell(G_StaggeredGrid& grid, ParticleSys<DeviceMemory>& ps, int gridSize,int blockSize);

};
