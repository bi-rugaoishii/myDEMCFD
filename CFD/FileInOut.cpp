#include "FileInOut.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

int FileInOut::solver_output_init(const char* dir){
    outdir_ = dir;
    struct stat st;

    /* 既に存在するか確認 */
    if(stat(dir,&st)==0)
    {
        if(S_ISDIR(st.st_mode))
        {
            printf("Output dir exists: %s\n",dir);
            return 0;
        }
    }

    /* 無ければ作成 */
    if(mkdir(dir,0755)==0)
    {
        printf("Created output dir: %s\n",dir);
        return 0;
    }

    if(errno==EEXIST)
        return 0;

    printf("ERROR: cannot create dir %s\n",dir);
    return -1;
}

void FileInOut::output_vti(const StaggeredGrid& grid, int step){
    char filename[256];
    sprintf(filename, "%s/result_%06d.vti", outdir_.c_str(), step);

    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Cannot open VTI file: %s\n", filename);
        abort();
    }

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;

    double dx = grid.dx_;
    double dy = grid.dy_;
    double dz = grid.dz_;

    const MyArray<double,3>& p     = grid.p_;
    const MyArray<double,3>& alpha = grid.alpha_;
    const MyArray<double,3>& vx    = grid.f_vx_;
    const MyArray<double,3>& vy    = grid.f_vy_;
    const MyArray<double,3>& vz    = grid.f_vz_;

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");

    fprintf(fp,
        "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" "
        "Origin=\"%.15e %.15e %.15e\" "
        "Spacing=\"%.15e %.15e %.15e\">\n",
        Nx - 1, Ny - 1, Nz - 1,
        0.5 * dx, 0.5 * dy, 0.5 * dz,
        dx, dy, dz);

    fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n", Nx - 1, Ny - 1, Nz - 1);
    fprintf(fp, "      <PointData Scalars=\"alpha\" Vectors=\"velocity\">\n");

    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"pressure\" format=\"ascii\">\n");
    for (int iz = 1; iz <= Nz; iz++) {
        for (int iy = 1; iy <= Ny; iy++) {
            for (int ix = 1; ix <= Nx; ix++) {
                fprintf(fp, "%.15e\n", p(ix, iy, iz));
            }
        }
    }
    fprintf(fp, "        </DataArray>\n");

    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"alpha\" format=\"ascii\">\n");
    for (int iz = 1; iz <= Nz; iz++) {
        for (int iy = 1; iy <= Ny; iy++) {
            for (int ix = 1; ix <= Nx; ix++) {
                fprintf(fp, "%.15e\n", alpha(ix, iy, iz));
            }
        }
    }
    fprintf(fp, "        </DataArray>\n");

    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int iz = 1; iz <= Nz; iz++) {
        for (int iy = 1; iy <= Ny; iy++) {
            for (int ix = 1; ix <= Nx; ix++) {
                double ux = 0.5 * (vx(ix - 1, iy, iz) + vx(ix, iy, iz));
                double uy = 0.5 * (vy(ix, iy - 1, iz) + vy(ix, iy, iz));
                double uz = 0.5 * (vz(ix, iy, iz - 1) + vz(ix, iy, iz));

                fprintf(fp, "%.15e %.15e %.15e\n", ux, uy, uz);
            }
        }
    }
    fprintf(fp, "        </DataArray>\n");

    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"divergence\" format=\"ascii\">\n");
    for (int iz = 1; iz <= Nz; iz++) {
        for (int iy = 1; iy <= Ny; iy++) {
            for (int ix = 1; ix <= Nx; ix++) {
                double div =
                    (vx(ix, iy, iz) - vx(ix - 1, iy, iz)) * grid.inv_dx_
                  + (vy(ix, iy, iz) - vy(ix, iy - 1, iz)) * grid.inv_dy_
                  + (vz(ix, iy, iz) - vz(ix, iy, iz - 1)) * grid.inv_dz_;

                fprintf(fp, "%.15e\n", div);
            }
        }
    }
    fprintf(fp, "        </DataArray>\n");

    fprintf(fp, "      </PointData>\n");
    fprintf(fp, "      <CellData></CellData>\n");
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </ImageData>\n");
    fprintf(fp, "</VTKFile>\n");

    fclose(fp);
    printf("VTI output: %s\n", filename);
}

static void write_vti_float_block(FILE* fp, const float* data, size_t nfloat){
    size_t nbytes_size = nfloat * sizeof(float);

    if (nbytes_size > UINT32_MAX){
        printf("VTI block is too large for UInt32 header: %zu bytes\n", nbytes_size);
        abort();
    }

    uint32_t nbytes = (uint32_t)nbytes_size;

    fwrite(&nbytes, sizeof(uint32_t), 1, fp);
    fwrite(data, sizeof(float), nfloat, fp);
}

void FileInOut::output_vti_binary_cellData(const StaggeredGrid& grid, int step){
    char filename[256];
    sprintf(filename, "%s/result_%06d.vti", outdir_.c_str(), step);

    FILE* fp = fopen(filename, "wb");
    if(fp == NULL){
        printf("Cannot open VTI file: %s\n", filename);
        abort();
    }

    const int Nx = grid.Nx_;
    const int Ny = grid.Ny_;
    const int Nz = grid.Nz_;

    const double dx = grid.dx_;
    const double dy = grid.dy_;
    const double dz = grid.dz_;

    const MyArray<double,3>& p = grid.p_;
    const MyArray<double,3>& ibm_solid_frac = grid.ibm_solid_fraction_;
    const MyArray<double,3>& alpha = grid.alpha_;
    const MyArray<double,3>& vx = grid.f_vx_;
    const MyArray<double,3>& vy = grid.f_vy_;
    const MyArray<double,3>& vz = grid.f_vz_;

    const size_t ncells = (size_t)Nx*(size_t)Ny*(size_t)Nz;
    const size_t bytes_scalar = ncells*sizeof(float);
    const size_t bytes_vector = ncells*3*sizeof(float);

    const size_t offset_pressure = 0;
    const size_t offset_alpha = offset_pressure + sizeof(uint32_t) + bytes_scalar;
    const size_t offset_velocity = offset_alpha + sizeof(uint32_t) + bytes_scalar;
    const size_t offset_ibm_solid_fraction = offset_velocity + sizeof(uint32_t) + bytes_vector;

    float* buf = (float*)malloc(sizeof(float)*ncells*3);
    if(buf == NULL){
        printf("Cannot allocate VTI output buffer\n");
        fclose(fp);
        abort();
    }

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n");
    fprintf(fp,
        "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" "
        "Origin=\"%.15e %.15e %.15e\" "
        "Spacing=\"%.15e %.15e %.15e\">\n",
        Nx, Ny, Nz, grid.origin_x_, grid.origin_y_, grid.origin_z_, dx, dy, dz);

    fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n", Nx, Ny, Nz);
    fprintf(fp, "      <PointData></PointData>\n");
    fprintf(fp, "      <CellData Scalars=\"alpha\" Vectors=\"velocity\">\n");
    fprintf(fp, "        <DataArray type=\"Float32\" Name=\"pressure\" format=\"appended\" offset=\"%zu\"/>\n", offset_pressure);
    fprintf(fp, "        <DataArray type=\"Float32\" Name=\"alpha\" format=\"appended\" offset=\"%zu\"/>\n", offset_alpha);
    fprintf(fp, "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%zu\"/>\n", offset_velocity);
    fprintf(fp, "        <DataArray type=\"Float32\" Name=\"ibm_solid_fraction\" format=\"appended\" offset=\"%zu\"/>\n", offset_ibm_solid_fraction);
    fprintf(fp, "      </CellData>\n");
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </ImageData>\n");
    fprintf(fp, "  <AppendedData encoding=\"raw\">\n_");

    size_t n = 0;

    for(int iz=1; iz<=Nz; iz++){
        for(int iy=1; iy<=Ny; iy++){
            for(int ix=1; ix<=Nx; ix++){
                buf[n++] = (float)p(ix,iy,iz);
            }
        }
    }
    write_vti_float_block(fp, buf, ncells);

    n = 0;
    for(int iz=1; iz<=Nz; iz++){
        for(int iy=1; iy<=Ny; iy++){
            for(int ix=1; ix<=Nx; ix++){
                buf[n++] = (float)alpha(ix,iy,iz);
            }
        }
    }
    write_vti_float_block(fp, buf, ncells);

    n = 0;
    for(int iz=1; iz<=Nz; iz++){
        for(int iy=1; iy<=Ny; iy++){
            for(int ix=1; ix<=Nx; ix++){
                double ux = 0.5*(vx(ix,iy,iz) + vx(ix+1,iy,iz));
                double uy = 0.5*(vy(ix,iy,iz) + vy(ix,iy+1,iz));
                double uz = 0.5*(vz(ix,iy,iz) + vz(ix,iy,iz+1));

                buf[n++] = (float)ux;
                buf[n++] = (float)uy;
                buf[n++] = (float)uz;
            }
        }
    }
    write_vti_float_block(fp, buf, ncells*3);

    n = 0;
    for(int iz=1; iz<=Nz; iz++){
        for(int iy=1; iy<=Ny; iy++){
            for(int ix=1; ix<=Nx; ix++){

                buf[n++] = (float)ibm_solid_frac(ix,iy,iz);
            }
        }
    }
    write_vti_float_block(fp, buf, ncells);

    fprintf(fp, "\n  </AppendedData>\n");
    fprintf(fp, "</VTKFile>\n");

    free(buf);
    fclose(fp);

    printf("VTI binary output: %s\n", filename);
}

void FileInOut::output_vti_binary(const StaggeredGrid& grid, int step){
    char filename[256];
    sprintf(filename, "%s/result_%06d.vti", outdir_.c_str(), step);

    FILE* fp = fopen(filename, "wb");
    if (fp == NULL) {
        printf("Cannot open VTI file: %s\n", filename);
        abort();
    }

    int Nx = grid.Nx_;
    int Ny = grid.Ny_;
    int Nz = grid.Nz_;

    double dx = grid.dx_;
    double dy = grid.dy_;
    double dz = grid.dz_;

    const MyArray<double,3>& p     = grid.p_;
    const MyArray<double,3>& alpha = grid.alpha_;
    const MyArray<double,3>& vx    = grid.f_vx_;
    const MyArray<double,3>& vy    = grid.f_vy_;
    const MyArray<double,3>& vz    = grid.f_vz_;

    size_t npoints = (size_t)Nx * (size_t)Ny * (size_t)Nz;

    size_t bytes_scalar = npoints * sizeof(float);
    size_t bytes_vector = npoints * 3 * sizeof(float);

    size_t offset_pressure   = 0;
    size_t offset_alpha      = offset_pressure + sizeof(uint32_t) + bytes_scalar;
    size_t offset_velocity   = offset_alpha    + sizeof(uint32_t) + bytes_scalar;
    size_t offset_divergence = offset_velocity + sizeof(uint32_t) + bytes_vector;

    float* buf = (float*)malloc(sizeof(float) * npoints * 3);
    if (buf == NULL){
        printf("Cannot allocate VTI output buffer\n");
        abort();
    }

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n");

    fprintf(fp,
        "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" "
        "Origin=\"%.15e %.15e %.15e\" "
        "Spacing=\"%.15e %.15e %.15e\">\n",
        Nx - 1, Ny - 1, Nz - 1,
        0.5 * dx + grid.origin_x_ , 0.5 * dy + grid.origin_y_ , 0.5 * dz + grid.origin_z_ ,
        dx, dy, dz);

    fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n", Nx - 1, Ny - 1, Nz - 1);
    fprintf(fp, "      <PointData Scalars=\"alpha\" Vectors=\"velocity\">\n");

    fprintf(fp,
        "        <DataArray type=\"Float32\" Name=\"pressure\" format=\"appended\" offset=\"%zu\"/>\n",
        offset_pressure);

    fprintf(fp,
        "        <DataArray type=\"Float32\" Name=\"alpha\" format=\"appended\" offset=\"%zu\"/>\n",
        offset_alpha);

    fprintf(fp,
        "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%zu\"/>\n",
        offset_velocity);

    fprintf(fp,
        "        <DataArray type=\"Float32\" Name=\"divergence\" format=\"appended\" offset=\"%zu\"/>\n",
        offset_divergence);

    fprintf(fp, "      </PointData>\n");
    fprintf(fp, "      <CellData></CellData>\n");
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </ImageData>\n");
    fprintf(fp, "  <AppendedData encoding=\"raw\">\n_");

    size_t n = 0;

    n = 0;
    for (int iz = 1; iz <= Nz; iz++){
        for (int iy = 1; iy <= Ny; iy++){
            for (int ix = 1; ix <= Nx; ix++){
                buf[n] = (float)p(ix, iy, iz);
                n++;
            }
        }
    }
    write_vti_float_block(fp, buf, npoints);

    n = 0;
    for (int iz = 1; iz <= Nz; iz++){
        for (int iy = 1; iy <= Ny; iy++){
            for (int ix = 1; ix <= Nx; ix++){
                buf[n] = (float)alpha(ix, iy, iz);
                n++;
            }
        }
    }
    write_vti_float_block(fp, buf, npoints);

    n = 0;
    for (int iz = 1; iz <= Nz; iz++){
        for (int iy = 1; iy <= Ny; iy++){
            for (int ix = 1; ix <= Nx; ix++){
                double ux = 0.5 * (vx(ix - 1, iy, iz) + vx(ix, iy, iz));
                double uy = 0.5 * (vy(ix, iy - 1, iz) + vy(ix, iy, iz));
                double uz = 0.5 * (vz(ix, iy, iz - 1) + vz(ix, iy, iz));

                buf[n + 0] = (float)ux;
                buf[n + 1] = (float)uy;
                buf[n + 2] = (float)uz;
                n += 3;
            }
        }
    }
    write_vti_float_block(fp, buf, npoints * 3);

    n = 0;
    for (int iz = 1; iz <= Nz; iz++){
        for (int iy = 1; iy <= Ny; iy++){
            for (int ix = 1; ix <= Nx; ix++){
                double div =
                    (vx(ix, iy, iz) - vx(ix - 1, iy, iz)) * grid.inv_dx_
                  + (vy(ix, iy, iz) - vy(ix, iy - 1, iz)) * grid.inv_dy_
                  + (vz(ix, iy, iz) - vz(ix, iy, iz - 1)) * grid.inv_dz_;

                buf[n] = (float)div;
                n++;
            }
        }
    }
    write_vti_float_block(fp, buf, npoints);

    fprintf(fp, "\n  </AppendedData>\n");
    fprintf(fp, "</VTKFile>\n");

    free(buf);
    fclose(fp);

    printf("VTI binary output: %s\n", filename);
}

