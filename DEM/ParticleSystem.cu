#include "ParticleSystem.h"
#include "hardCodedParameters.h"
#include "csv.h"
#include <stdio.h>
#include <random>



/* ==================== Memory Allocation =================== */
void allocate(ParticleSys<HostMemory> *ps){
    int N=ps->N;
    #define MEMBER(type,name,Np,SAVE_FLAG) ps->name = HostMemory::template allocate<type>(Np);
    #include "memberList/ParticleSystemMember_common.def"
    #undef MEMBER
}

void allocate(ParticleSys<DeviceMemory> *d_ps){
    ParticleSys<DeviceMemory> ps{};
    int N=d_ps->N;
    ps.N=d_ps->N;
    ps.dt=d_ps->dt;
    ps.mu=d_ps->mu;

    #define MEMBER(type,name,Np,SAVE_FLAG) ps.name = DeviceMemory::template allocate<type>((Np)); \
    d_ps->name = ps.name;

    #include "memberList/ParticleSystemMember_common.def"
    #undef MEMBER

    /* ==== Device Only ===== */
    cudaMalloc((void**)&ps.refreshVerletFlag,sizeof(int));
    d_ps->refreshVerletFlag = ps.refreshVerletFlag;
    
    /* ===== for sorting ===============*/
        ps.tmp_storage=NULL;
        ps.tmp_bytes=0;

        /* calculate temporarly size */

        cub::DeviceRadixSort::SortPairs(
                ps.tmp_storage,
                ps.tmp_bytes,
                ps.mortonKey,
                ps.tmpMortonKey,
                ps.mortonOrder,
                ps.tmpMortonOrder,
                N);

        cudaMalloc((void**)&ps.tmp_storage,ps.tmp_bytes);
        d_ps->tmp_storage = ps.tmp_storage;
        d_ps->tmp_bytes = ps.tmp_bytes;
        

        /* ==== Copy struct ==== */
        cudaMalloc((void**)&d_ps->d_self,sizeof(ParticleSys<DeviceMemory>));
        cudaMemcpy(d_ps->d_self,&ps,sizeof(ParticleSys<DeviceMemory>),cudaMemcpyHostToDevice);
}

void deallocate(ParticleSys<HostMemory> *ps){
    #define MEMBER(type,name,Np,SAVE_FLAG) HostMemory::deallocate(ps->name);
    #include "memberList/ParticleSystemMember_common.def"
    #undef MEMBER
}

void deallocate(ParticleSys<DeviceMemory> *ps){
    #define MEMBER(type,name,Np,SAVE_FLAG) DeviceMemory::deallocate(ps->name);
    #include "memberList/ParticleSystemMember_common.def"
    #undef MEMBER
    DeviceMemory::deallocate(ps->refreshVerletFlag);
    DeviceMemory::deallocate(ps->tmp_storage);
    DeviceMemory::deallocate(ps->d_self);
}

    


/*
   ============================================================
   Host→Device転送
   ============================================================
 */
void copyToDevice(ParticleSys<HostMemory> *ps, ParticleSys<DeviceMemory> *d_ps){
    size_t N = ps->N;

    #define MEMBER(type,name,Np,SAVE_FLAG) cudaMemcpy(d_ps->name,ps->name,(Np)*sizeof(type),cudaMemcpyHostToDevice); 
    #include "memberList/ParticleSystemMember_common.def"
    #undef MEMBER

}


/*
   ============================================================
   Device→Host転送
   ============================================================
 */
void copyFromDevice(ParticleSys<DeviceMemory>* d_ps, ParticleSys<HostMemory>* ps){
    size_t N = ps->N;

    #define MEMBER(type,name,Np,SAVE_FLAG) \
    if(SAVE_FLAG == SAVE_ON){ \
        cudaMemcpy(ps->name,d_ps->name, (Np)*sizeof(type), cudaMemcpyDeviceToHost);\
    }
    #include "memberList/ParticleSystemMember_common.def"
    #undef MEMBER
}

/*
   ============================================================
   初期化
   ============================================================
 */
void initializeTmpParticles(ParticleSys<HostMemory>* ps,cJSON *json_inlet, double r,double m,double E,double res)
{
    double k =4./3.*(E*0.5)*sqrt(r*0.5)*sqrt(0.05*r);
    for (int i = 0; i < ps->N; i++){
        ps->x[i*DIM+0] = 0.;
        ps->x[i*DIM+1] = 0.;
        ps->x[i*DIM+2] = 0.;


        /*
           ps->x[i*DIM+0] = 0.25;
           ps->x[i*DIM+1] = (double)i*1.0+0.03;
           ps->x[i*DIM+2] = 0.25;
         */

        ps->r[i] = r;
        ps->rsq[i] = ps->r[i]*ps->r[i];
        ps->invr[i] = 1./ps->r[i];
        ps->k[i] = k;
        ps->m[i] = m;
        ps->sqrtm[i] = sqrt(m);
        ps->invm[i] = 1./m;
        ps->etaconst[i]=-2.*log(res)*sqrt(ps->k[i]/(3.1415*3.1415+log(res)*log(res)));
        ps->cellId[i] = -1;
        ps->isActive[i] = 1;


        ps->numCont[i] = 0;
        ps->numContWall[i] = 0;

        ps->v[i*DIM+0] = 0.;
        ps->v[i*DIM+1] = 0.;
        ps->v[i*DIM+2] = 0.;

        /* ======== for temporarly check========= */
        /*
           if (i==1){
           ps->v[i*DIM+0] = 1.;
           ps->v[i*DIM+1] = 0.;
           ps->v[i*DIM+2] = 0.;
           }
         */
        /* ======== for temporarly check========= */

        ps->angv[i*DIM+0] = 0.;
        ps->angv[i*DIM+1] = 0.;
        ps->angv[i*DIM+2] = 0.;

        ps->anga[i*DIM+0] = 0.;
        ps->anga[i*DIM+1] = 0.;
        ps->anga[i*DIM+2] = 0.;

        ps->moi[i] = 2./5. * ps->m[i]*ps->rsq[i];
        ps->invmoi[i] = 1./ps->moi[i];

        ps->pId[i] = i;


        for (int j=0; j<MAX_NEI; j++){
            ps->indHis[i*MAX_NEI+j] = -1;
            ps->indHisWall[i*MAX_NEI+j] = -1;
            ps->indHisWallNow[i*MAX_NEI+j] = -1;

            ps->isContact[i*MAX_NEI+j] = -1;
            ps->isContactWall[i*MAX_NEI+j] = -1;

            ps->deltHisx[i*MAX_NEI+j] = 0.;
            ps->deltHisy[i*MAX_NEI+j] = 0.;
            ps->deltHisz[i*MAX_NEI+j] = 0.;
            ps->deltHisxWall[i*MAX_NEI+j] = 0.;
            ps->deltHisyWall[i*MAX_NEI+j] = 0.;
            ps->deltHiszWall[i*MAX_NEI+j] = 0.;
        }
    }
}

void initializeParticles(ParticleSys<HostMemory>* ps,cJSON *json_inlet, double r,double m,double E,double res)
{
    int max_trials = 1000; // 1粒子あたりの再配置試行回数
    int seed = 1;
    std::mt19937 mt(seed);
    std::uniform_real_distribution<double> uni(0,1);
    // intialize step //
    ps->steps = 0;

    ps->length_factor = 1.0;
    ps->mass_factor = 1.0;
    ps->time_factor = 1.0;

    //double k =4./3.*(E*0.5)*sqrt(r*0.5)*sqrt(0.05*r);
    double k =E*r*2.;
    printf("k= %e\n",k);

    if(strcmp(cJSON_GetObjectItem(json_inlet,"inputMode")->valuestring,"shuffle")==0){

        printf("shuffling particles\n");

        /* == get box for shuffling == */
        cJSON *json_shuffle = get_json_object(json_inlet,"shuffle");

        double minx_shuffle = get_json_double(json_shuffle,"minx");
        double miny_shuffle = get_json_double(json_shuffle, "miny");
        double minz_shuffle = get_json_double(json_shuffle, "minz");
        double maxx_shuffle = get_json_double(json_shuffle, "maxx");
        double maxy_shuffle = get_json_double(json_shuffle, "maxy");
        double maxz_shuffle = get_json_double(json_shuffle, "maxz");

        for (int i = 0; i < ps->N; i++){
            int trial = 0;
            while (trial < max_trials)
            {
                // ランダム配置
                /*
                   double x = (double)rand() / RAND_MAX*(0.3)-(0.35+0.15);
                   double y = (double)rand() / RAND_MAX * 1.0 + 2.8;
                //double y = -0.97;
                double z = (double)rand() / RAND_MAX*(0.3)-(0.35+0.15);
                 */

                double x = uni(mt)*(maxx_shuffle-minx_shuffle) + minx_shuffle;
                double y = uni(mt)*(maxy_shuffle-miny_shuffle) + miny_shuffle;
                double z = uni(mt)*(maxz_shuffle-minz_shuffle) + minz_shuffle;


                /* ======== for temporarly check========= */
                /*
                   double x = 0.;
                   double y = 0.021;
                   double z = 0.0;

                   if (i==0){
                   x = 0.;
                   y = 0.0;
                   z = 0.0;
                   }else{
                   x = -0.2;
                   y = 0.005;
                   z = 0.0;

                   }
                 */
                /* ======== for temporarly check========= */


                // 既存粒子との距離チェック
                int overlap = 0;
                for (int j = 0; j < i; j++)
                {
                    double dx = x - ps->x[j*DIM+0];
                    double dy = y - ps->x[j*DIM+1];
                    double dz = z - ps->x[j*DIM+2];
                    double d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 < 4*r*r) // 半径2倍以上離れているか
                    {
                        overlap = 1;
                        break;
                    }
                }

                if (!overlap)
                {
                    ps->x[i*DIM+0] = x;
                    ps->x[i*DIM+1] = y;
                    ps->x[i*DIM+2] = z;
                    break;
                }

                trial++;
            }

            if (trial == max_trials){
                printf("!!!couldn't place them!!\n");
                abort();
            };


            /*
               ps->x[i*DIM+0] = 0.25;
               ps->x[i*DIM+1] = (double)i*1.0+0.03;
               ps->x[i*DIM+2] = 0.25;
             */

            ps->r[i] = r;
            ps->rsq[i] = ps->r[i]*ps->r[i];
            ps->vol[i] = 4./3.*M_PI*ps->rsq[i]*r;
            ps->invr[i] = 1./ps->r[i];
            ps->k[i] = k;
            ps->m[i] = m;
            ps->sqrtm[i] = sqrt(m);
            ps->invm[i] = 1./m;
            ps->etaconst[i]=-2.*log(res)*sqrt(ps->k[i]/(3.1415*3.1415+log(res)*log(res)));
            ps->cellId[i] = -1;
            ps->isActive[i] = 1;


            ps->numCont[i] = 0;
            ps->numContWall[i] = 0;

            ps->v[i*DIM+0] = 0.;
            ps->v[i*DIM+1] = 0.;
            ps->v[i*DIM+2] = 0.;

            /* ======== for temporarly check========= */
            /*
               if (i==1){
               ps->v[i*DIM+0] = 1.;
               ps->v[i*DIM+1] = 0.;
               ps->v[i*DIM+2] = 0.;
               }
             */
            /* ======== for temporarly check========= */


            ps->angv[i*DIM+0] = 0.;
            ps->angv[i*DIM+1] = 0.;
            ps->angv[i*DIM+2] = 0.;

            ps->anga[i*DIM+0] = 0.;
            ps->anga[i*DIM+1] = 0.;
            ps->anga[i*DIM+2] = 0.;

            ps->moi[i] = 2./5. * ps->m[i]*ps->rsq[i];
            ps->invmoi[i] = 1./ps->moi[i];

            ps->pId[i] = i;


            for (int j=0; j<MAX_NEI; j++){
                ps->indHis[i*MAX_NEI+j] = -1;
                ps->indHisWall[i*MAX_NEI+j] = -1;
                ps->indHisWallNow[i*MAX_NEI+j] = -1;

                ps->isContact[i*MAX_NEI+j] = -1;
                ps->isContactWall[i*MAX_NEI+j] = -1;

                ps->deltHisx[i*MAX_NEI+j] = 0.;
                ps->deltHisy[i*MAX_NEI+j] = 0.;
                ps->deltHisz[i*MAX_NEI+j] = 0.;
                ps->deltHisxWall[i*MAX_NEI+j] = 0.;
                ps->deltHisyWall[i*MAX_NEI+j] = 0.;
                ps->deltHiszWall[i*MAX_NEI+j] = 0.;
            }
        }
    }else if(strcmp(cJSON_GetObjectItem(json_inlet,"inputMode")->valuestring,"file")==0){

        printf("reading particle files\n");
        cJSON *json_file=get_json_object(json_inlet,"file");

        const char *filename =get_json_string(json_file,"name");
        printf("filename = %s\n",filename);

        /* == check if file exists == */
        FILE* fp = fopen(filename, "r");
        if (!fp) {
            perror("fopen");
            abort();
            return;
        }
        fclose(fp);

        io::LineReader inRead(filename);

        int lineCounter =0;
        while(char *line =inRead.next_line()){
            lineCounter++;
        }

        lineCounter--; //skip headers


        if(lineCounter!=ps->N){
            printf("Number of particles mismatch with %d in file and %d in settings\n",lineCounter, ps->N);
            abort();
        }

        io::CSVReader<6> inCSV(filename);
        inCSV.read_header(io::ignore_no_column,"x","y","z","vx","vy","vz");
        double x,y,z,vx,vy,vz;
        int counter=0;

        double maxx=-1.e16;
        double maxy=-1.e16;
        double maxz=-1.e16;

        double minx=1.e16;
        double miny=1.e16;
        double minz=1.e16;

        while(inCSV.read_row(x,y,z,vx,vy,vz)){

            ps->x[counter*DIM+0]=x;
            ps->x[counter*DIM+1]=y;
            ps->x[counter*DIM+2]=z;

            ps->v[counter*DIM+0]=vx;
            ps->v[counter*DIM+1]=vy;
            ps->v[counter*DIM+2]=vz;

            if(maxx<x){
                maxx=x;
            }

            if(maxy<y){
                maxy=y;
            }

            if(maxz<z){
                maxz=z;
            }

            if(minx>x){
                minx=x;
            }

            if(miny>y){
                miny=y;
            }

            if(minz>z){
                minz=z;
            }

            counter++;
        }

        cJSON_AddNumberToObject(json_file,"minx",minx);
        cJSON_AddNumberToObject(json_file,"miny",miny);
        cJSON_AddNumberToObject(json_file,"minz",minz);
        cJSON_AddNumberToObject(json_file,"maxx",maxx);
        cJSON_AddNumberToObject(json_file,"maxy",maxy);
        cJSON_AddNumberToObject(json_file,"maxz",maxz);

        for(int i=0; i<ps->N; i++){
            ps->r[i] = r;
            ps->rsq[i] = ps->r[i]*ps->r[i];
            ps->invr[i] = 1./ps->r[i];
            ps->k[i] = k;
            ps->m[i] = m;
            ps->sqrtm[i] = sqrt(m);
            ps->invm[i] = 1./m;
            ps->etaconst[i]=-2.*log(res)*sqrt(ps->k[i]/(3.1415*3.1415+log(res)*log(res)));
            ps->cellId[i] = -1;
            ps->isActive[i] = 1;


            ps->numCont[i] = 0;
            ps->numContWall[i] = 0;

            ps->angv[i*DIM+0] = 0.;
            ps->angv[i*DIM+1] = 0.;
            ps->angv[i*DIM+2] = 0.;

            ps->anga[i*DIM+0] = 0.;
            ps->anga[i*DIM+1] = 0.;
            ps->anga[i*DIM+2] = 0.;

            ps->moi[i] = 2./5. * ps->m[i]*ps->rsq[i];
            ps->invmoi[i] = 1./ps->moi[i];

            ps->pId[i] = i;


            for (int j=0; j<MAX_NEI; j++){
                ps->indHis[i*MAX_NEI+j] = -1;
                ps->indHisWall[i*MAX_NEI+j] = -1;
                ps->indHisWallNow[i*MAX_NEI+j] = -1;

                ps->isContact[i*MAX_NEI+j] = -1;
                ps->isContactWall[i*MAX_NEI+j] = -1;

                ps->deltHisx[i*MAX_NEI+j] = 0.;
                ps->deltHisy[i*MAX_NEI+j] = 0.;
                ps->deltHisz[i*MAX_NEI+j] = 0.;
                ps->deltHisxWall[i*MAX_NEI+j] = 0.;
                ps->deltHisyWall[i*MAX_NEI+j] = 0.;
                ps->deltHiszWall[i*MAX_NEI+j] = 0.;
            }
        }


    }else{
        printf("no proper input mode defined!\n");
        abort();
    }

}

void nondimensionalize(ParticleSys<HostMemory>* ps, BoundingBox *box, TriangleMesh* mesh){
    /* get nondimensionalize factor */
    ps->time_factor = sqrt(ps->m[0]/ps->k[0]);
    ps->mass_factor = ps->m[0];
    ps->length_factor = ps->r[0];
    double time_factor = ps->time_factor;
    double mass_factor = ps->mass_factor;
    double length_factor = ps->length_factor;

    double inv_time_factor = 1./time_factor;
    double inv_mass_factor = 1./mass_factor;
    double inv_length_factor = 1./length_factor;
    double inv_sqrtk_factor = 1./sqrt(ps->k[0]);

    ps->dt*=inv_time_factor;

    ps->g[0]*=inv_length_factor*time_factor*time_factor;
    ps->g[1]*=inv_length_factor*time_factor*time_factor;
    ps->g[2]*=inv_length_factor*time_factor*time_factor;



    for (int i=0; i<ps->N; i++){
        ps->x[i*DIM+0]*=inv_length_factor;
        ps->x[i*DIM+1]*=inv_length_factor;
        ps->x[i*DIM+2]*=inv_length_factor;

        ps->v[i*DIM+0]*=inv_length_factor*time_factor;
        ps->v[i*DIM+1]*=inv_length_factor*time_factor;
        ps->v[i*DIM+2]*=inv_length_factor*time_factor;

        ps->r[i]*=inv_length_factor;
        ps->rsq[i]*=inv_length_factor*inv_length_factor;
        ps->invr[i]*=length_factor;

        ps->m[i]*=inv_mass_factor;
        ps->sqrtm[i]*=sqrt(inv_mass_factor);

        ps->invm[i]*=mass_factor;

        ps->moi[i]*=inv_mass_factor*inv_length_factor*inv_length_factor;
        ps->invmoi[i]*=mass_factor*length_factor*length_factor;

        ps->k[i]*=inv_sqrtk_factor*inv_sqrtk_factor;
        ps->etaconst[i]*=inv_sqrtk_factor;
    }

    /* bounding box*/

    box->minx*=inv_length_factor;
    box->miny*=inv_length_factor;
    box->minz*=inv_length_factor;
    box->maxx*=inv_length_factor;
    box->maxy*=inv_length_factor;
    box->maxz*=inv_length_factor;

    box->dx*=inv_length_factor;
    box->dy*=inv_length_factor;
    box->dz*=inv_length_factor;

    box->invdx*=length_factor;
    box->invdy*=length_factor;
    box->invdz*=length_factor;

    box->rangex*=inv_length_factor;
    box->rangey*=inv_length_factor;
    box->rangez*=inv_length_factor;
    box->skinR*=inv_length_factor;
    box->refreshThresh*=inv_length_factor;
    box->refreshThreshSq*=inv_length_factor*inv_length_factor;

    /* triangles */
    for (int i=0; i<mesh->nTri; i++){
        mesh->cx[i]*= inv_length_factor;
        mesh->cy[i]*= inv_length_factor;
        mesh->cz[i]*= inv_length_factor;

        mesh->d[i]*= inv_length_factor;

        mesh->e01x[i]*= inv_length_factor;
        mesh->e01y[i]*= inv_length_factor;
        mesh->e01z[i]*= inv_length_factor;

        mesh->e02x[i]*= inv_length_factor;
        mesh->e02y[i]*= inv_length_factor;
        mesh->e02z[i]*= inv_length_factor;

        mesh->e12x[i]*= inv_length_factor;
        mesh->e12y[i]*= inv_length_factor;
        mesh->e12z[i]*= inv_length_factor;

        mesh->d00[i]*= inv_length_factor*inv_length_factor;
        mesh->d00inv[i]*= length_factor*length_factor;
        mesh->d01[i]*= inv_length_factor*inv_length_factor;
        mesh->d11[i]*= inv_length_factor*inv_length_factor;
        mesh->d11inv[i]*=length_factor*length_factor;
        mesh->d22[i]*= inv_length_factor*inv_length_factor;
        mesh->d22inv[i]*=length_factor*length_factor;

        mesh->denom[i]*= length_factor*length_factor*length_factor*length_factor;

        mesh->minx[i]*= inv_length_factor;
        mesh->miny[i]*= inv_length_factor;
        mesh->minz[i]*= inv_length_factor;
        mesh->maxx[i]*= inv_length_factor;
        mesh->maxy[i]*= inv_length_factor;
        mesh->maxz[i]*= inv_length_factor;
    }
    for (int i=0; i<mesh->nVert; i++){
        mesh->mx[i]*= inv_length_factor;
        mesh->my[i]*= inv_length_factor;
        mesh->mz[i]*= inv_length_factor;
    }

}
