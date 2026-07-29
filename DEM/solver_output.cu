#include "solver_output.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

/*
============================================================
ディレクトリ作成（初期化時のみ）
============================================================
*/
int solver_output_init(const char* dir)
{
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

/*
============================================================
ファイル名生成
============================================================
*/
static void build_filename(
    char* out,
    const char* dir,
    int step)
{
    sprintf(out,"%s/result_%012d.bin",dir,step);
}

/*
============================================================
BIN出力

フォーマット:
[int N]
[float pos[N*3]]
[float radius[N]]
============================================================
*/
void write_header_text(const char* dir,int step,ParticleSys<HostMemory>* p,int pid){
    char filename[256];

    sprintf(filename,"%s/pid%d.txt",dir,pid);

    FILE* fp=fopen(filename,"w");

    if(fp==NULL)
    {
        printf("ERROR: cannot open %s\n",filename);
        return;
    }


    /*write headers*/
    
        fprintf(fp, "0_time(s),1_pid,2_x(m),3_y(m),4_z(m),5_vx(m/s),6_vy(m/s),7_vz(m/s),8_ax(m/s2),9_ay(m/s2),10_az(m/s2),11_angvx(rad/s),12_angvy(rad/s),13_angvz(rad/s),14_angax(rad/s2),15_angay(rad/s2),16_angaz(rad/s2),17_isActive\n");


        fclose(fp);
}

void write_initialPos_csv(const char* dir,ParticleSys<HostMemory>* p){
    char filename[256];

    sprintf(filename,"%s/initialPos.csv",dir);

    FILE* fp=fopen(filename,"a");

    if(fp==NULL){
        printf("ERROR: cannot open %s\n",filename);
        return;
    }

    for (int i=0; i<p->N; i++){
        double x=(p->x[i*3+0]);
        double y=(p->x[i*3+1]);
        double z=(p->x[i*3+2]);


        fprintf(fp,"%d 1 %f 1000. %f %f %f\n",i+1,p->r[i]*2.0,x,y,z);
    }





    fclose(fp);
}

void write_single_text(const char* dir,int step,ParticleSys<HostMemory>* p,int pid){
    char filename[256];

    sprintf(filename,"%s/pid%d.txt",dir,pid);
    int N=p->N;
    double time_factor = p->time_factor;
    double length_factor = p->length_factor;

    FILE* fp=fopen(filename,"a");

    if(fp==NULL)
    {
        printf("ERROR: cannot open %s\n",filename);
        return;
    }


    for(int i=0; i<N; i++){
        if (p->pId[i]!= pid){
            continue;
        }else{
            double dt = p->dt*time_factor; 
            double time = dt*(double)step;

            double x=(p->x[i*3+0]*length_factor);
            double y=(p->x[i*3+1]*length_factor);
            double z=(p->x[i*3+2]*length_factor);


            double ax=(p->a[i*3+0]*length_factor/(time_factor*time_factor));
            double ay=(p->a[i*3+1]*length_factor/(time_factor*time_factor));
            double az=(p->a[i*3+2]*length_factor/(time_factor*time_factor));

            double vx=(p->v[i*3+0]*length_factor/time_factor);
            double vy=(p->v[i*3+1]*length_factor/time_factor);
            double vz=(p->v[i*3+2]*length_factor/time_factor);


            double angax=(p->anga[i*3+0]/(time_factor*time_factor));
            double angay=(p->anga[i*3+1]/(time_factor*time_factor));
            double angaz=(p->anga[i*3+2]/(time_factor*time_factor));

            double angvx=(p->angv[i*3+0]/(time_factor));
            double angvy=(p->angv[i*3+1]/(time_factor));
            double angvz=(p->angv[i*3+2]/(time_factor));

            int isActive=p->isActive[i];
            fprintf(fp,"%f %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n",time,pid,x,y,z,vx,vy,vz,ax,ay,az,angvx,angvy,angvz,angax,angay,angaz,isActive);
            break;
        }
    }



    fclose(fp);
}


//void write_frame_bin_all(
//        const char* dir,
//        int step,
//        ParticleSystem* p)
//{
//    char filename[256];
//
//    build_filename(filename,dir,step);
//
//    FILE* fp=fopen(filename,"wb");
//
//    const double time_factor = ps->time_factor;
//    const double length_factor = ps->length_factor;
//    const double mass_factor = ps->mass_factor;
//    const double vel_factor = length_factor/time_factor;
//
//    if(fp==NULL)
//    {
//        printf("ERROR: cannot open %s\n",filename);
//        return;
//    }
//
//    /* check number of active particles */
//
//    int N=p->N;
//
//    /* 粒子数 */
//    fwrite(&N,sizeof(int),1,fp);
//
//    /* 変換バッファ */
//    double* x=(double*)malloc(sizeof(double)*N*3);
//    double* r  =(double*)malloc(sizeof(double)*N);
//    double* v=(double*)malloc(sizeof(double)*N*3);
//    double* m=(double*)malloc(sizeof(double)*N);
//    int* isActive=(int*)malloc(sizeof(int)*N);
//    double* angv=(double*)malloc(sizeof(double)*N*3);
//    double* deltHisx = (double*)malloc(sizeof(double)*p->N*MAX_NEI);
//    double* deltHisy = (double*)malloc(sizeof(double)*p->N*MAX_NEI);
//    double* deltHisz = (double*)malloc(sizeof(double)*p->N*MAX_NEI);
//
//    double* deltHisxWall = (double*)malloc(sizeof(double)*p->N*MAX_NEI);
//    double* deltHisyWall = (double*)malloc(sizeof(double)*p->N*MAX_NEI);
//    double* deltHiszWall = (double*)malloc(sizeof(double)*p->N*MAX_NEI);
//
//    int* indHis = (int*)malloc(sizeof(int)*p->N*MAX_NEI);
//    int* numCont = (int*)malloc(sizeof(int)*p->N);
//
//    int* indHisWall = (int*)malloc(sizeof(int)*ps->N*MAX_NEI);
//    int* numContWall = (int*)malloc(sizeof(int)*ps->N);
//    int* pId = (int*)malloc(sizeof(int)*ps->N);
//    double* refx = (double*)malloc(size);
//    double* refy = (double*)malloc(size);
//    double* refz = (double*)malloc(size);
//    int* neiList = (int*)malloc(sizeof(int)*ps->N*MAX_NEI);
//    int* numNei = (int*)malloc(sizeof(int)*ps->N);
//    int* neiListWall = (int*)malloc(sizeof(int)*ps->N*MAX_NEI);
//    int* numNeiWall = (int*)malloc(sizeof(int)*ps->N);
//
//    int i;
//    for(i=0;i<p->N;i++)
//    {
//        x[i*3+0]=(double)(p->x[i*3+0]*length_factor);
//        x[i*3+1]=(double)(p->x[i*3+1]*length_factor);
//        x[i*3+2]=(double)(p->x[i*3+2]*length_factor);
//
//        r[i]=(double)(p->r[i]*length_factor);
//
//        v[i*3+0]=(double)(p->v[i*3+0]*vel_factor);
//        v[i*3+1]=(double)(p->v[i*3+1]*vel_factor);
//        v[i*3+2]=(double)(p->v[i*3+2]*vel_factor);
//
//        m[i]=(double)(p->m[i]*mass_factor);
//        isActive[i]=(int*)(p->m[i]*mass_factor);
//        angv[i]=(double*)(p->angv[i]);
//
//    }
//
//    fwrite(pos_f,sizeof(double),N*3,fp);
//    fwrite(r_f,sizeof(double),N,fp);
//
//    free(pos_f);
//    free(r_f);
//
//    fclose(fp);
//}

void write_frame_bin(
        const char* dir,
        int step,
        ParticleSys<HostMemory>* p,const double length_factor)
{
    int N=p->N;

    char filename[256];

    build_filename(filename,dir,step);

    FILE* fp=fopen(filename,"wb");

    if(fp==NULL)
    {
        printf("ERROR: cannot open %s\n",filename);
        return;
    }


    /* check number of active particles */

    int tmpN=0;
    for(int i=0; i<N; i++){
        if(p->isActive[i]==1){
           tmpN+=1;
        }
    }
    /* 粒子数 */
    fwrite(&tmpN,sizeof(int),1,fp);

    /* float変換バッファ */
    float* pos_f=(float*)malloc(sizeof(float)*tmpN*3);
    float* r_f  =(float*)malloc(sizeof(float)*tmpN);

    int i;
    int k=0;
    for(i=0;i<N;i++)
    {
        if(p->isActive[i]==0){
            continue;
        }
        pos_f[k*3+0]=(float)(p->x[i*3+0]*length_factor);
        pos_f[k*3+1]=(float)(p->x[i*3+1]*length_factor);
        pos_f[k*3+2]=(float)(p->x[i*3+2]*length_factor);

        r_f[k]=(float)(p->r[i]*length_factor);
        k++;
    }

    fwrite(pos_f,sizeof(float),tmpN*3,fp);
    fwrite(r_f,sizeof(float),tmpN,fp);

    free(pos_f);
    free(r_f);

    fclose(fp);
}
