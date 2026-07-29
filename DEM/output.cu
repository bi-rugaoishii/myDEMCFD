#include "output.h"

/*
============================================================
出力
============================================================
*/
void writeParticles(ParticleSystem* ps, int step)
{
    char filename[256];
    sprintf(filename, "output_%05d.csv", step);

    FILE* fp = fopen(filename, "w");

    for (int i = 0; i < ps->N; i++)
    {
        fprintf(fp, "%f,%f %f\n", ps->x[i*DIM+0], ps->x[i*DIM+1],ps->x[i*DIM+2]);
    }

    fclose(fp);
}
/*
============================================================
VTK Binary出力（ParaView用・高速）
============================================================
*/

/* ============================================================
   エンディアン変換（little → big）
============================================================ */
void swapBytes(void* data, size_t size)
{
    char* p = (char*)data;
    size_t i;
    for (i = 0; i < size/2; i++)
    {
        char tmp = p[i];
        p[i] = p[size-1-i];
        p[size-1-i] = tmp;
    }
}
void writeParticlesVTKBinary(ParticleSystem* ps, int step)
{
    char filename[256];
    sprintf(filename, "vtkresults/particles_%05d.vtk", step);

    FILE* fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        printf("Failed to open VTK file.\n");
        return;
    }

    /* ---------- ヘッダ（ASCII） ---------- */
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "DEM particle output\n");
    fprintf(fp, "BINARY\n");
    fprintf(fp, "DATASET POLYDATA\n");

    /* ---------- POINTS ---------- */
    fprintf(fp, "POINTS %d double\n", ps->N);

    int i, k;
    double value;

    for (i = 0; i < ps->N; i++)
    {
        for (k = 0; k < 3; k++)
        {
            value = ps->x[i*DIM + k];
            swapBytes(&value, sizeof(double));
            fwrite(&value, sizeof(double), 1, fp);
        }
    }

    fprintf(fp, "\n");

    /* ---------- VERTICES ---------- */
    fprintf(fp, "VERTICES %d %d\n", ps->N, ps->N*2);

    int one = 1;
    int index;

    for (i = 0; i < ps->N; i++)
    {
        int tmp = one;
        swapBytes(&tmp, sizeof(int));
        fwrite(&tmp, sizeof(int), 1, fp);

        index = i;
        swapBytes(&index, sizeof(int));
        fwrite(&index, sizeof(int), 1, fp);
    }

    fprintf(fp, "\n");

    /* ---------- 半径 ---------- */
    fprintf(fp, "POINT_DATA %d\n", ps->N);
    fprintf(fp, "SCALARS radius double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");

    for (i = 0; i < ps->N; i++)
    {
        value = ps->r[i];
        swapBytes(&value, sizeof(double));
        fwrite(&value, sizeof(double), 1, fp);
    }

    fclose(fp);
}
/*
============================================================
VTK出力（ParaView用）
============================================================
*/
void writeParticlesDimensionalizeVTK(ParticleSystem* ps, int step)
{
    char filename[256];

    /* フォルダ付き出力 */
    sprintf(filename, "vtkresults/particles_%05d.vtk", step);

    FILE* fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Failed to open VTK file.\n");
        return;
    }

    /* --- VTKヘッダ --- */
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "DEM particle output\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET POLYDATA\n");

    /* --- 粒子位置 --- */
    fprintf(fp, "POINTS %d double\n", ps->N);
    for (int i = 0; i < ps->N; i++)
    {
        fprintf(fp, "%lf %lf %lf\n",
                ps->x[i*DIM+0]*ps->length_factor,
                ps->x[i*DIM+1]*ps->length_factor,
                ps->x[i*DIM+2]*ps->length_factor);
    }

    /* --- 各点を頂点セルとして定義 --- */
    fprintf(fp, "\nVERTICES %d %d\n", ps->N, ps->N * 2);
    for (int i = 0; i < ps->N; i++)
    {
        fprintf(fp, "1 %d\n", i);
    }

    /* --- 半径データ --- */
    fprintf(fp, "\nPOINT_DATA %d\n", ps->N);
    fprintf(fp, "SCALARS radius double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");

    for (int i = 0; i < ps->N; i++)
    {
        fprintf(fp, "%lf\n", ps->r[i]*ps->length_factor);
    }

    fclose(fp);
}

void writeParticlesVTK(ParticleSystem* ps, int step)
{
    char filename[256];

    /* フォルダ付き出力 */
    sprintf(filename, "vtkresults/particles_%05d.vtk", step);

    FILE* fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Failed to open VTK file.\n");
        return;
    }

    /* --- VTKヘッダ --- */
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "DEM particle output\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET POLYDATA\n");

    /* --- 粒子位置 --- */
    fprintf(fp, "POINTS %d double\n", ps->N);
    for (int i = 0; i < ps->N; i++)
    {
        fprintf(fp, "%lf %lf %lf\n",
                ps->x[i*DIM+0],
                ps->x[i*DIM+1],
                ps->x[i*DIM+2]);
    }

    /* --- 各点を頂点セルとして定義 --- */
    fprintf(fp, "\nVERTICES %d %d\n", ps->N, ps->N * 2);
    for (int i = 0; i < ps->N; i++)
    {
        fprintf(fp, "1 %d\n", i);
    }

    /* --- 半径データ --- */
    fprintf(fp, "\nPOINT_DATA %d\n", ps->N);
    fprintf(fp, "SCALARS radius double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");

    for (int i = 0; i < ps->N; i++)
    {
        fprintf(fp, "%lf\n", ps->r[i]);
    }

    fclose(fp);
}

