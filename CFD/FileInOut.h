#pragma once
#include "StaggeredGrid.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>

struct FileInOut{
    std::string outdir_;

    int solver_output_init(const char* dir);
    void output_vti(const StaggeredGrid& grid, int step);
    void output_vti_binary(const StaggeredGrid& grid, int step);
    void output_vti_binary_cellData(const StaggeredGrid& grid, int step);
};
