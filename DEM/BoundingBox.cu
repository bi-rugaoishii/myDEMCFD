#include "BoundingBox.h"
#include "omp.h"
#include "ParticleSystem.h"

/* =========== used for sort =========== */


int compare_int(const void *a, const void *b){
    int ia = *(const int*)a;
    int ib = *(const int*)b;

    if (ia < ib) return -1;
    if (ia > ib) return  1;
    return 0;
}

void insertSort(int *array,  int start, int end){ 
    for(int i=1; i<end; i++){
        int k=i;
        int tmpEl = array[start+i];
        while(i>0 && array[start+k-1]>tmpEl){
            array[start+k]=array[start+k-1];
            k--;
        }
        array[start+k]=tmpEl;
    }
}

// ======================================================
// ポインタスワップ版 radix sort（最速設計）
// keyPtr/indexPtr は「ポインタへのポインタ」
// ======================================================
void radixSortUint32(
    uint32_t** keyPtr,
    int**      indexPtr,
    int N,
    uint32_t*  workKey,
    int*       workIndex)
{
    const int RADIX = 256;
    const int PASS  = 4;   // 30bit Mortonなら3passでOK

    uint32_t* srcKey = *keyPtr;
    uint32_t* dstKey = workKey;

    int* srcIdx = *indexPtr;
    int* dstIdx = workIndex;

    int pass;

    for(pass=0; pass<PASS; pass++)
    {
        int count[RADIX];
        int i;

        memset(count,0,sizeof(count));

        int shift = pass*8;

        // --- histogram ---
        for(i=0;i<N;i++)
        {
            uint32_t digit = (srcKey[i] >> shift) & 0xFF;
            count[digit]++;
        }

        // --- prefix sum ---
        int sum = 0;
        for(i=0;i<RADIX;i++)
        {
            int c = count[i];
            count[i] = sum;
            sum += c;
        }

        // --- scatter ---
        for(i=0;i<N;i++)
        {
            uint32_t k = srcKey[i];
            uint32_t digit = (k >> shift) & 0xFF;

            int pos = count[digit]++;

            dstKey[pos] = k;
            dstIdx[pos] = srcIdx[i];
        }

        // --- swap pointer only ---
        {
            uint32_t* tk = srcKey;
            srcKey = dstKey;
            dstKey = tk;
        }
        {
            int* ti = srcIdx;
            srcIdx = dstIdx;
            dstIdx = ti;
        }
    }

    // 最終ポインタを返す（コピー無し）
    *keyPtr   = srcKey;
    *indexPtr = srcIdx;
}

void swap_device_ps_pointer(ParticleSys<DeviceMemory>** p, ParticleSys<DeviceMemory>** tmpP){
    ParticleSys<DeviceMemory>* tmp = *p;
    *p = *tmpP;
    *tmpP = tmp;
}

void swap_device_ps_member_pointer(ParticleSys<DeviceMemory>* p, ParticleSys<DeviceMemory>* tmp){
    double* td;
    int*    ti;
   // uint32_t* tu;


    td=p->x; p->x=tmp->x; tmp->x=td;
    td=p->v; p->v=tmp->v; tmp->v=td;

    td=p->r; p->r=tmp->r; tmp->r=td;
    td=p->rsq; p->rsq=tmp->rsq; tmp->rsq=td;
    td=p->invr; p->invr=tmp->invr; tmp->invr=td;

    td=p->m; p->m=tmp->m; tmp->m=td;
    td=p->invm; p->invm=tmp->invm; tmp->invm=td;
    td=p->sqrtm; p->sqrtm=tmp->sqrtm; tmp->sqrtm=td;
    td=p->moi; p->moi=tmp->moi; tmp->moi=td;
    td=p->invmoi; p->invmoi=tmp->invmoi; tmp->invmoi=td;
    td=p->etaconst; p->etaconst=tmp->etaconst; tmp->etaconst=td;

    td=p->angv; p->angv=tmp->angv; tmp->angv=td;

    // ---- contact history ----
    td=p->deltHisx;     p->deltHisx=tmp->deltHisx;     tmp->deltHisx=td;
    td=p->deltHisy;     p->deltHisy=tmp->deltHisy;     tmp->deltHisy=td;
    td=p->deltHisz;     p->deltHisz=tmp->deltHisz;     tmp->deltHisz=td;

    td=p->deltHisxWall; p->deltHisxWall=tmp->deltHisxWall; tmp->deltHisxWall=td;
    td=p->deltHisyWall; p->deltHisyWall=tmp->deltHisyWall; tmp->deltHisyWall=td;
    td=p->deltHiszWall; p->deltHiszWall=tmp->deltHiszWall; tmp->deltHiszWall=td;

    ti = p->isContact;      p->isContact = tmp->isContact;      tmp->isContact = ti;
    ti = p->indHis;         p->indHis = tmp->indHis;            tmp->indHis = ti;

    ti = p->isContactWall;  p->isContactWall = tmp->isContactWall; tmp->isContactWall = ti;
    ti = p->indHisWall;     p->indHisWall = tmp->indHisWall;    tmp->indHisWall = ti;

    // ---- contact number ----
    ti=p->numCont;      p->numCont=tmp->numCont;      tmp->numCont=ti;
    ti=p->numContWall;  p->numContWall=tmp->numContWall; tmp->numContWall=ti;

    // ---- cell info ----
    ti=p->cellId;       p->cellId=tmp->cellId;       tmp->cellId=ti;
    ti=p->cellx;        p->cellx=tmp->cellx;         tmp->cellx=ti;

    // ---- particle id ----
    ti=p->pId;          p->pId=tmp->pId;             tmp->pId=ti;

    ti=p->isActive;       p->isActive=tmp->isActive;       tmp->isActive=ti;

    ti=p->refreshVerletFlag;      p->refreshVerletFlag=tmp->refreshVerletFlag;      tmp->refreshVerletFlag=ti;

    // ---- morton key is swapped to p->tmpMortonkey ----
    /*
    tu=p->mortonKey;    p->mortonKey=tmp->mortonKey; tmp->mortonKey=tu;
    */
}

void swap_ps(ParticleSys<HostMemory> *p, ParticleSys<HostMemory> *tmp){
    int N = p->N;


    for (int i=0; i<N; i++){
        int src = p->mortonOrder[i];

        // --- position ---
        tmp->x[i*3+0] = p->x[src*3+0];
        tmp->x[i*3+1] = p->x[src*3+1];
        tmp->x[i*3+2] = p->x[src*3+2];

        // --- velocity ---
        tmp->v[i*3+0] = p->v[src*3+0];
        tmp->v[i*3+1] = p->v[src*3+1];
        tmp->v[i*3+2] = p->v[src*3+2];

        // --- ang velocity ---
        tmp->angv[i*3+0] = p->angv[src*3+0];
        tmp->angv[i*3+1] = p->angv[src*3+1];
        tmp->angv[i*3+2] = p->angv[src*3+2];

        // --- scalar ---
        tmp->r[i] = p->r[src];
        tmp->rsq[i] = p->rsq[src];
        tmp->invr[i] = p->invr[src];

        tmp->m[i] = p->m[src];
        tmp->sqrtm[i] = p->sqrtm[src];
        tmp->invm[i] = p->invm[src];
        tmp->moi[i] = p->moi[src];
        tmp->invmoi[i] = p->invmoi[src];
        tmp->etaconst[i] = p->etaconst[src];

        tmp->isActive[i] = p->isActive[src];

        int bi = i*MAX_NEI;
        int bsrc = src*MAX_NEI;
        for (int j=0; j<MAX_NEI; j++){
            // ---- contact history (particle) ----
            tmp->deltHisx[bi+j] = p->deltHisx[bsrc+j];
            tmp->deltHisy[bi+j] = p->deltHisy[bsrc+j];
            tmp->deltHisz[bi+j] = p->deltHisz[bsrc+j];

            tmp->isContact[bi+j] = p->isContact[bsrc+j];

            /* get the after order index of contact particle */
            tmp->indHis[bi+j] = p->indHis[bsrc+j];

            // ---- contact history (wall) ----
            tmp->deltHisxWall[bi+j] = p->deltHisxWall[bsrc+j];
            tmp->deltHisyWall[bi+j] = p->deltHisyWall[bsrc+j];
            tmp->deltHiszWall[bi+j] = p->deltHiszWall[bsrc+j];
            tmp->isContactWall[bi+j] = p->isContactWall[bsrc+j];
            tmp->indHisWall[bi+j] = p->indHisWall[bsrc+j];

        }

        // ---- number of contact ----
        tmp->numCont[i]     = p->numCont[src];
        tmp->numContWall[i] = p->numContWall[src];

        // ---- cell info ----
        tmp->cellId[i] = p->cellId[src];
        tmp->cellx[i*DIM+0]  = p->cellx[src*DIM+0];
        tmp->cellx[i*DIM+1]  = p->cellx[src*DIM+1];
        tmp->cellx[i*DIM+2]  = p->cellx[src*DIM+2];


        // ---- morton key is already swapped----
        //tmp->mortonKey[i] = p->mortonKey[src];
        tmp->pId[i] = p->pId[src];
    }


    /* swap pointers*/
    double* td;
    int*    ti;
    //uint32_t* tu;

    td=p->x; p->x=tmp->x; tmp->x=td;
    td=p->v; p->v=tmp->v; tmp->v=td;
    td=p->r; p->r=tmp->r; tmp->r=td;
    td=p->rsq; p->rsq=tmp->rsq; tmp->rsq=td;
    td=p->invr; p->invr=tmp->invr; tmp->invr=td;

    td=p->m; p->m=tmp->m; tmp->m=td;
    td=p->invm; p->invm=tmp->invm; tmp->invm=td;
    td=p->sqrtm; p->sqrtm=tmp->sqrtm; tmp->sqrtm=td;
    td=p->moi; p->moi=tmp->moi; tmp->moi=td;
    td=p->invmoi; p->invmoi=tmp->invmoi; tmp->invmoi=td;
    td=p->etaconst; p->etaconst=tmp->etaconst; tmp->etaconst=td;

    td=p->angv; p->angv=tmp->angv; tmp->angv=td;
    // ---- contact history ----
    td=p->deltHisx;     p->deltHisx=tmp->deltHisx;     tmp->deltHisx=td;
    td=p->deltHisy;     p->deltHisy=tmp->deltHisy;     tmp->deltHisy=td;
    td=p->deltHisz;     p->deltHisz=tmp->deltHisz;     tmp->deltHisz=td;

    td=p->deltHisxWall; p->deltHisxWall=tmp->deltHisxWall; tmp->deltHisxWall=td;
    td=p->deltHisyWall; p->deltHisyWall=tmp->deltHisyWall; tmp->deltHisyWall=td;
    td=p->deltHiszWall; p->deltHiszWall=tmp->deltHiszWall; tmp->deltHiszWall=td;

    ti = p->isContact;      p->isContact = tmp->isContact;      tmp->isContact = ti;
    ti = p->indHis;         p->indHis = tmp->indHis;            tmp->indHis = ti;

    ti = p->isContactWall;  p->isContactWall = tmp->isContactWall; tmp->isContactWall = ti;
    ti = p->indHisWall;     p->indHisWall = tmp->indHisWall;    tmp->indHisWall = ti;

    // ---- contact number ----
    ti=p->numCont;      p->numCont=tmp->numCont;      tmp->numCont=ti;
    ti=p->numContWall;  p->numContWall=tmp->numContWall; tmp->numContWall=ti;

    // ---- cell info ----
    ti=p->cellId;       p->cellId=tmp->cellId;       tmp->cellId=ti;
    ti=p->cellx;        p->cellx=tmp->cellx;         tmp->cellx=ti;

    // ---- particle id ----
    ti=p->pId;          p->pId=tmp->pId;             tmp->pId=ti;

    ti=p->isActive;       p->isActive=tmp->isActive;       tmp->isActive=ti;

    // ---- morton key is already swapped----
    //tu=p->mortonKey;    p->mortonKey=tmp->mortonKey; tmp->mortonKey=tu;

}


void copyToDeviceBox(BoundingBox *box, ParticleSys<HostMemory> *ps){

    size_t size = box->N;

    box->d_box.minx = box->minx;
    box->d_box.miny = box->miny;
    box->d_box.minz = box->minz;
    box->d_box.maxx = box->maxx;
    box->d_box.maxy = box->maxy;
    box->d_box.maxz = box->maxz;

    box->d_box.dx = box->dx;
    box->d_box.dy = box->dy;
    box->d_box.dz = box->dz;

    box->d_box.invdx = box->invdx;
    box->d_box.invdy = box->invdy;
    box->d_box.invdz = box->invdz;

    box->d_box.skinR = box->skinR;
    box->d_box.refreshThresh = box->refreshThresh;
    box->d_box.refreshThreshSq = box->refreshThreshSq;

    box->d_box.rangex = box->rangex;
    box->d_box.rangey = box->rangey;
    box->d_box.rangez = box->rangez;

    box->d_box.sizex = box->sizex;
    box->d_box.sizey = box->sizey;
    box->d_box.sizez = box->sizez;

    box->d_box.N = box->N;

    cudaMemcpy(box->d_box.pList,  box->pList,  sizeof(int)*ps->N, cudaMemcpyHostToDevice);
    cudaMemcpy(box->d_box.pNum,  box->pNum,  sizeof(int)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(box->d_box.usedCells, box->usedCells, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(box->d_box.pStart,  box->pStart,  sizeof(int)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(box->d_box.cellOffset,  box->cellOffset,  sizeof(int)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(box->d_box.tList,  box->tList,  sizeof(int)*size*box->MAX_TRI, cudaMemcpyHostToDevice);
    cudaMemcpy(box->d_box.tNum,  box->tNum,  sizeof(int)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(box->d_boxPtr,  &box->d_box,  sizeof(DeviceBoundingBox), cudaMemcpyHostToDevice);
}

void free_BoundingBox(BoundingBox *box, int isGPUon){

    free(box->pList);
    free(box->pNum);
    free(box->pStart);
    free(box->cellOffset);
    free(box->usedCells);

    #if USE_GPU
    if (isGPUon == 1 ){
        cudaFree(box->d_box.pList);
        cudaFree(box->d_box.tmpExSum);
        cudaFree(box->d_box.pNum);
        cudaFree(box->d_box.usedCells);
        cudaFree(box->d_box.pStart);
        cudaFree(box->d_box.cellOffset);
        cudaFree(box->d_box.tList);
        cudaFree(box->d_box.tNum);
        cudaFree(box->d_boxPtr);
    }
    #endif
}

void initialize_BoundingBox(ParticleSys<HostMemory> *p, BoundingBox *box,TriangleMesh *mesh, cJSON *json_inlet_type, int isGPUon){
    /* find boxlimits */
    calc_BoundingBoxLimits(box,mesh,json_inlet_type);

    /* find maximum radius */

    double maxr=0.;
    for (int i=0; i<p->N; i++){
        if (maxr<p->r[i]){
            maxr=p->r[i];
        }
    }

    box->rangex = box->maxx - box->minx;
    box->rangey = box->maxy - box->miny;
    box->rangez = box->maxz - box->minz;

    box->dx = maxr*4. ;
    box->dy = maxr*4. ;
    box->dz = maxr*4. ;

    /* verlet list related */
    box->skinR = maxr;
    box->refreshThresh = maxr*0.5;
    box->refreshThreshSq = box->refreshThresh*box->refreshThresh;

    box->invdx = 1./box->dx;
    box->invdy = 1./box->dy;
    box->invdz = 1./box->dz;

    box->sizex = (int)ceil(box->rangex/box->dx) +2;
    box->sizey = (int)ceil(box->rangey/box->dy) +2;
    box->sizez = (int)ceil(box->rangez/box->dz) +2;

    int sizeBox= box->sizex*box->sizey*box->sizez;
    box->N= sizeBox;

    box->pList = (int*)malloc(sizeof(int)*p->N);
    box->pNum = (int*)calloc(sizeBox,sizeof(int));
    box->pStart = (int*)calloc(sizeBox,sizeof(int));
    box->cellOffset = (int*)calloc(sizeBox,sizeof(int));

    box->usedCells = (int*)malloc(sizeof(int)*sizeBox);
    box->numUsedCells = 0;

    /* ==== for triangles ==== */

    box->MAX_TRI = 100;
    box->tList = (int*)calloc(sizeBox*box->MAX_TRI,sizeof(int));
    box->tNum = (int*)calloc(sizeBox,sizeof(int));

    #if USE_GPU
    if (isGPUon==1){
        box->d_box.N= sizeBox;
        box->d_box.MAX_TRI = box->MAX_TRI;
        cudaMalloc(&box->d_box.pList, sizeof(int)*p->N);
        cudaMalloc(&box->d_box.pNum, sizeof(int)*sizeBox);
        cudaMalloc(&box->d_box.usedCells, sizeof(int) * sizeBox);

        cudaMalloc(&box->d_box.pStart, sizeof(int)*sizeBox);
        cudaMalloc(&box->d_box.tList, sizeof(int)*sizeBox*box->MAX_TRI);
        cudaMalloc(&box->d_box.tNum, sizeof(int)*sizeBox);
        cudaMalloc(&box->d_box.cellOffset, sizeof(int)*sizeBox);

        /* for cub */

        box->d_box.tmpExSum =NULL;
        box->d_box.scanTmpBytes = 0;

        cub::DeviceScan::ExclusiveSum(
                box->d_box.tmpExSum,
                box->d_box.scanTmpBytes,
                box->d_box.pNum,
                box->d_box.pStart,
                box->d_box.N);
        cudaMalloc(&box->d_box.tmpExSum, box->d_box.scanTmpBytes);

        /* for structs*/
        cudaMalloc(&box->d_boxPtr,sizeof(DeviceBoundingBox));
    }
    #endif

}

void calc_BoundingBoxLimits(BoundingBox *box, TriangleMesh *mesh, cJSON *json_inlet_type){

    double minx_inlet = cJSON_GetObjectItem(json_inlet_type,"minx")->valuedouble;
    double miny_inlet = cJSON_GetObjectItem(json_inlet_type,"miny")->valuedouble;
    double minz_inlet = cJSON_GetObjectItem(json_inlet_type,"minz")->valuedouble;
    double maxx_inlet = cJSON_GetObjectItem(json_inlet_type,"maxx")->valuedouble;
    double maxy_inlet = cJSON_GetObjectItem(json_inlet_type,"maxy")->valuedouble;
    double maxz_inlet = cJSON_GetObjectItem(json_inlet_type,"maxz")->valuedouble;

    /* === get boxlimits ===*/
    /* gives 10 percent margin*/

    box->minx = mesh->gminx < minx_inlet ? mesh->gminx : minx_inlet;
    box->minx = box->minx*1.1;

    box->miny = mesh->gminy < miny_inlet ? mesh->gminy : miny_inlet;
    box->miny = box->miny*1.1;

    box->minz = mesh->gminz < minz_inlet ? mesh->gminz : minz_inlet;
    box->minz = box->minz*1.1;

    box->maxx = mesh->gmaxx > maxx_inlet ? mesh->gmaxx : maxx_inlet;
    box->maxx = box->maxx*1.1;

    box->maxy = mesh->gmaxy > maxy_inlet ? mesh->gmaxy : maxy_inlet;
    box->maxy = box->maxy*1.1;

    box->maxz = mesh->gmaxz > maxz_inlet ? mesh->gmaxz : maxz_inlet;
    box->maxz = box->maxz*1.1;

}

void update_neighborlist_brute(ParticleSys<HostMemory> *p,ParticleSys<HostMemory> *tmpPs, BoundingBox *box){

    double skinR = box->skinR;

    for (int i=0; i<p->N; i++){
        if(p->isActive[i]!=1){
            continue;
        }
        //particle-particle
        /* cycle through neighbor cells */
        int bi=i*DIM;
        int numNei = 0;

        for (int j=0; j<p->N; j++){
            if (i==j || p->isActive[j]!=1){
                continue;
            }else{
                int bj=j*DIM;

                Vec3 del;
                /* normal points toward particle i */
                del.x = p->x[bi+0]- p->x[bj+0];
                del.y = p->x[bi+1]- p->x[bj+1];
                del.z = p->x[bi+2]- p->x[bj+2];
                double distsq = vdot(del,del);
                double R = p->r[i]+p->r[j]+skinR;
                if (distsq<R*R){
                    p->neiList[i*MAX_NEI+numNei]=j;
                    numNei+=1;
                    if(numNei >= MAX_NEI){
                        printf("Neighbor over flow!!!!\n");
                    }
                }
            }
        }/* neighbor cell search done */
        p->numNei[i]=numNei;


        /* set reference position */
        p->refx[i]=p->x[bi+0];
        p->refy[i]=p->x[bi+1];
        p->refz[i]=p->x[bi+2];
    }
}


void update_neighborlist(ParticleSys<HostMemory> *p,ParticleSys<HostMemory> *tmpPs, BoundingBox *box){
//    update_pList_withSort_fast(p,tmpPs,box);
    update_pList_fast(p,box);

    double skinR = box->skinR;
    
    #pragma omp parallel for
    for (int i=0; i<p->N; i++){
        if(p->isActive[i]!=1){
            continue;
        }
        //particle-particle
        /* cycle through neighbor cells */
        int bi=i*DIM;
        int x=p->cellx[bi+0];
        int y=p->cellx[bi+1];
        int z=p->cellx[bi+2];
        int numNei = 0;

        for (int sx=-1; sx<=1; sx++){
            for (int sy=-1; sy<=1; sy++){
                for (int sz=-1; sz<=1; sz++){
                    int cellId = (box->sizey*(z+sz)+y+sy)*box->sizex+x+sx;

                    int start = box->pStart[cellId];
                    int end = start+box->pNum[cellId];
                    for (int k=box->pStart[cellId]; k<end; k++){
                        int j = box->pList[k];
                        if (i==j || p->isActive[j]!=1){
                            continue;
                        }else{
                            int bj=j*DIM;

                            Vec3 del;
                            /* normal points toward particle i */
                            del.x = p->x[bi+0]- p->x[bj+0];
                            del.y = p->x[bi+1]- p->x[bj+1];
                            del.z = p->x[bi+2]- p->x[bj+2];
                            double distsq = vdot(del,del);
                            double R = p->r[i]+p->r[j]+skinR;
                            if (distsq<R*R){
                                p->neiList[i*MAX_NEI+numNei]=j;
                                numNei+=1;
                                if(numNei >= MAX_NEI){
                                    printf("Neighbor over flow!!!!\n");
                                }
                            }
                        }
                    }
                }
            }
        }/* neighbor cell search done */
        p->numNei[i]=numNei;
        insertSort(p->neiList,i*MAX_NEI,numNei);


        /* set reference position */
        p->refx[i]=p->x[bi+0];
        p->refy[i]=p->x[bi+1];
        p->refz[i]=p->x[bi+2];
    }
}


void update_pList_withSort_fast(ParticleSys<HostMemory> *p, ParticleSys<HostMemory> *tmpPs,BoundingBox *box){
    printf("sorting\n");
    /* initialize */
    for (int i=0; i<box->numUsedCells; i++){
        int cid = box->usedCells[i];
        box->pNum[cid] = 0;
        box->pStart[cid] = 0;
        box->cellOffset[cid] = 0;
    }

    for(int i=0;i<p->N;i++){
        p->mortonOrder[i] = i;
    }
    /* get cellId and get mortonkey*/
    for (int i=0; i<p->N; i++){
        int bi = i*DIM;
        int dx = (int)floor((p->x[bi+0]-box->minx)*box->invdx)+1; //+1 for ghost cell
        int dy = (int)floor((p->x[bi+1]-box->miny)*box->invdy)+1; //+1 for ghost cell
        int dz = (int)floor((p->x[bi+2]-box->minz)*box->invdz)+1; //+1 for ghost cell
        p->cellx[bi+0] = dx;
        p->cellx[bi+1] = dy;
        p->cellx[bi+2] = dz;
        p->cellId[i] = (box->sizey*dz+dy)*box->sizex+dx;
        p->mortonKey[i]=morton3D((uint32_t)dx,(uint32_t)dy,(uint32_t)dz);
    }

    /* ========= sort and reorder ========= */
    radixSortUint32(&p->mortonKey,&p->mortonOrder,p->N,p->tmpMortonKey,p->tmpMortonOrder);

    swap_ps(p, tmpPs);



    /* count number of particle in cell */

    box->numUsedCells = 0;

    for (int i=0; i<p->N; i++){
        if (p->isActive[i]!=1){
            continue;
        }
        int cellId = p->cellId[i];

        if (box->pNum[cellId]==0){
            box->usedCells[box->numUsedCells] = cellId;
            box->numUsedCells += 1;
        }

        box->pNum[cellId] +=1;
    }


    int runningSum = 0;
    for (int i=0; i<box->numUsedCells; i++){
        int cid = box->usedCells[i];
        box->pStart[cid]=runningSum;
        runningSum += box->pNum[cid];
    }


    for (int i=0; i<p->N; i++){
        if (p->isActive[i]!=1){
            continue;
        }

        int cellId = p->cellId[i];
        int startId = box->pStart[cellId];
        int offset = box->cellOffset[cellId];

        box->pList[startId+offset] = i;  
        box->cellOffset[cellId] +=1;
    }
}

void update_pList_withSort(ParticleSys<HostMemory> *p, ParticleSys<HostMemory> *tmpPs,BoundingBox *box){
    /* initialize */
    for (int i=0; i<box->N; i++){
        box->pNum[i] = 0;
    }

    for (int i=0; i<box->N; i++){
        box->pStart[i] = 0;
    }

    for(int i=0;i<p->N;i++){
        p->mortonOrder[i] = i;
    }


    /* get cellId and get mortonkey*/
    for (int i=0; i<p->N; i++){
        int bi=i*DIM;
        int dx = (int)floor((p->x[bi+0]-box->minx)*box->invdx)+1; //+1 for ghost cell
        int dy = (int)floor((p->x[bi+1]-box->miny)*box->invdy)+1; //+1 for ghost cell
        int dz = (int)floor((p->x[bi+2]-box->minz)*box->invdz)+1; //+1 for ghost cell
        p->cellx[bi+0] = dx;
        p->cellx[bi+1] = dy;
        p->cellx[bi+2] = dz;
        p->cellId[i] = (box->sizey*dz+dy)*box->sizex+dx;
        p->mortonKey[i]=morton3D((uint32_t)dx,(uint32_t)dy,(uint32_t)dz);
    }

    /* ========= sort and reorder ========= */
    radixSortUint32(&p->mortonKey,&p->mortonOrder,p->N,p->tmpMortonKey,p->tmpMortonOrder);

    swap_ps(p, tmpPs);



    /* count number of particle in cell */

    for (int i=0; i<p->N; i++){
        if (p->isActive[i]!=1){
            continue;
        }
        int cellId = p->cellId[i];
        box->pNum[cellId] +=1;
    }

    box->pStart[0]=0;
    for (int i=1; i<box->N; i++){
        box->pStart[i]=box->pNum[i-1]+box->pStart[i-1];
    }

    for (int i=0; i<box->N; i++){
        box->cellOffset[i] = 0;
    }

    for (int i=0; i<p->N; i++){
        if (p->isActive[i]!=1){
            continue;
        }
        int cellId = p->cellId[i];
        int startId = box->pStart[cellId];
        int offset = box->cellOffset[cellId];


        box->pList[startId+offset] = i;  
        box->cellOffset[cellId] +=1;
    }

}

void update_tList(BoundingBox *box, TriangleMesh *mesh){
    for (int i=0; i<mesh->nTri; i++){
        int sx = (int)floor((mesh->minx[i]-box->minx)*box->invdx)+1;
        int sy = (int)floor((mesh->miny[i]-box->miny)*box->invdy)+1;
        int sz = (int)floor((mesh->minz[i]-box->minz)*box->invdz)+1;

        int ex = (int)ceil((mesh->maxx[i]-box->minx)*box->invdx)+1;
        int ey = (int)ceil((mesh->maxy[i]-box->miny)*box->invdy)+1;
        int ez = (int)ceil((mesh->maxz[i]-box->minz)*box->invdz)+1;

        for (int dz=sz; dz<=ez; dz++){
            for (int dy=sy; dy<=ey; dy++){
                for (int dx=sx; dx<=ex; dx++){
                    int cellId= (box->sizey*dz+dy)*box->sizex+dx;
                    if(cellId>=box->N){
                        printf("Cell Id is FUCKED BRO!");
                    }


                    /** put this paragraph if want to use the cell linked list triangles*/
                    /*
                       int tNum = box->tNum[cellId];
                       box->tList[cellId*box->MAX_TRI+tNum]=i;
                       if (box->tNum[cellId]>= box->MAX_TRI){
                       printf("!!!!!!!Too many triangles in a cell!!!!\n");
                       }
                     */

                    box->tNum[cellId]++;
                }
            }
        }
    }
}

void update_pList_fast(ParticleSys<HostMemory> *p, BoundingBox *box){

    /* initialize */

    #pragma omp parallel for
    for (int i=0; i<box->numUsedCells; i++){
        int cid = box->usedCells[i];
        box->pNum[cid] = 0;
        box->pStart[cid] = 0;
        box->cellOffset[cid] = 0;
    }


    /* get cellId*/
    #pragma omp parallel for
    for (int i=0; i<p->N; i++){
        if (p->isActive[i]!=1){
            continue;
        }
        int dx = (int)floor((p->x[i*DIM+0]-box->minx)*box->invdx)+1; //+1 for ghost cell
        int dy = (int)floor((p->x[i*DIM+1]-box->miny)*box->invdy)+1; //+1 for ghost cell
        int dz = (int)floor((p->x[i*DIM+2]-box->minz)*box->invdz)+1; //+1 for ghost cell
        p->cellx[i*DIM+0] = dx;
        p->cellx[i*DIM+1] = dy;
        p->cellx[i*DIM+2] = dz;
        p->cellId[i] = (box->sizey*dz+dy)*box->sizex+dx;

    }

    /* count number of particle in cell */

    box->numUsedCells = 0;

    for (int i=0; i<p->N; i++){
        if (p->isActive[i]!=1){
            continue;
        }
        int cellId = p->cellId[i];

        if (box->pNum[cellId]==0){
            box->usedCells[box->numUsedCells] = cellId;
            box->numUsedCells += 1;
        }

        box->pNum[cellId] +=1;
    }

    //qsort(box->usedCells, box->numUsedCells,sizeof(int), compare_int );

    int runningSum = 0;
    for (int i=0; i<box->numUsedCells; i++){
        int cid = box->usedCells[i];
        box->pStart[cid]=runningSum;
        runningSum += box->pNum[cid];
    }


    for (int i=0; i<p->N; i++){
        if (p->isActive[i]!=1){
            continue;
        }
        int cellId = p->cellId[i];
        int startId = box->pStart[cellId];
        int offset = box->cellOffset[cellId];

        box->pList[startId+offset] = i;  
        box->cellOffset[cellId] +=1;
    }

}

void update_pList(ParticleSys<HostMemory> *p, BoundingBox *box){
    /* initialize */
    for (int i=0; i<box->N; i++){
        box->pNum[i] = 0;
    }

    for (int i=0; i<box->N; i++){
        box->pStart[i] = 0;
    }

    /* get cellId*/
    for (int i=0; i<p->N; i++){
        if (p->isActive[i]!=1){
            continue;
        }
        int dx = (int)floor((p->x[i*DIM+0]-box->minx)*box->invdx)+1; //+1 for ghost cell
        int dy = (int)floor((p->x[i*DIM+1]-box->miny)*box->invdy)+1; //+1 for ghost cell
        int dz = (int)floor((p->x[i*DIM+2]-box->minz)*box->invdz)+1; //+1 for ghost cell
        p->cellx[i*DIM+0] = dx;
        p->cellx[i*DIM+1] = dy;
        p->cellx[i*DIM+2] = dz;
        p->cellId[i] = (box->sizey*dz+dy)*box->sizex+dx;
    }

    /* count number of particle in cell */

    for (int i=0; i<p->N; i++){
        if (p->isActive[i]!=1){
            continue;
        }
        int cellId = p->cellId[i];
        box->pNum[cellId] +=1;
    }

    box->pStart[0]=0;
    for (int i=1; i<box->N; i++){
        box->pStart[i]=box->pNum[i-1]+box->pStart[i-1];
    }

    for (int i=0; i<box->N; i++){
        box->cellOffset[i] = 0;
    }

    for (int i=0; i<p->N; i++){
        if (p->isActive[i]!=1){
            continue;
        }
        int cellId = p->cellId[i];
        int startId = box->pStart[cellId];
        int offset = box->cellOffset[cellId];


        box->pList[startId+offset] = i;  
        box->cellOffset[cellId] +=1;
    }
}

/*
   ===================================
   device  related functions
   ===================================
 */
__global__ void checkConsistency(ParticleSys<DeviceMemory>* p){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= p->N) return;

    int bi = i * DIM;

    uint32_t key = morton3D(
            (uint32_t)p->cellx[bi+0],
            (uint32_t)p->cellx[bi+1],
            (uint32_t)p->cellx[bi+2]
            );

    if(key != p->mortonKey[i]){
        printf("ERROR at i=%d pId=%d\n", i, p->pId[i]);
        printf("cellx=(%d,%d,%d) key=%u but mortonKey=%u\n",
                p->cellx[bi+0],
                p->cellx[bi+1],
                p->cellx[bi+2],
                key,
                p->mortonKey[i]);
    }
}

__global__ void check(ParticleSys<DeviceMemory>* p){
    printf("device mortonKey ptr: %p\n", p->mortonKey);
}

__global__ void d_swap_ps(ParticleSys<DeviceMemory> *p, ParticleSys<DeviceMemory> *tmp){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=p->N){
        return;
    }


    int src = p->tmpMortonOrder[i];

    // --- position ---
    tmp->x[i*3+0] = p->x[src*3+0];
    tmp->x[i*3+1] = p->x[src*3+1];
    tmp->x[i*3+2] = p->x[src*3+2];

    // --- velocity ---
    tmp->v[i*3+0] = p->v[src*3+0];
    tmp->v[i*3+1] = p->v[src*3+1];
    tmp->v[i*3+2] = p->v[src*3+2];

    // --- ang velocity ---
    tmp->angv[i*3+0] = p->angv[src*3+0];
    tmp->angv[i*3+1] = p->angv[src*3+1];
    tmp->angv[i*3+2] = p->angv[src*3+2];

    // --- scalar ---
    tmp->r[i] = p->r[src];
    tmp->rsq[i] = p->rsq[src];
    tmp->invr[i] = p->invr[src];

    tmp->m[i] = p->m[src];
    tmp->sqrtm[i] = p->sqrtm[src];
    tmp->invm[i] = p->invm[src];
    tmp->moi[i] = p->moi[src];
    tmp->invmoi[i] = p->invmoi[src];
    tmp->etaconst[i] = p->etaconst[src];

    tmp->isActive[i] = p->isActive[src];

    int bi = i*MAX_NEI;
    int bsrc = src*MAX_NEI;
    for (int j=0; j<MAX_NEI; j++){
        // ---- contact history (particle) ----
        tmp->deltHisx[bi+j] = p->deltHisx[bsrc+j];
        tmp->deltHisy[bi+j] = p->deltHisy[bsrc+j];
        tmp->deltHisz[bi+j] = p->deltHisz[bsrc+j];

        tmp->isContact[bi+j] = p->isContact[bsrc+j];

        /* get the after order index of contact particle */
        tmp->indHis[bi+j] = p->indHis[bsrc+j];

        // ---- contact history (wall) ----
        tmp->deltHisxWall[bi+j] = p->deltHisxWall[bsrc+j];
        tmp->deltHisyWall[bi+j] = p->deltHisyWall[bsrc+j];
        tmp->deltHiszWall[bi+j] = p->deltHiszWall[bsrc+j];
        tmp->isContactWall[bi+j] = p->isContactWall[bsrc+j];
        tmp->indHisWall[bi+j] = p->indHisWall[bsrc+j];

    }

    // ---- number of contact ----
    tmp->numCont[i]     = p->numCont[src];
    tmp->numContWall[i] = p->numContWall[src];

    // ---- cell info ----
    tmp->cellId[i] = p->cellId[src];
    tmp->cellx[i*DIM+0]  = p->cellx[src*DIM+0];
    tmp->cellx[i*DIM+1]  = p->cellx[src*DIM+1];
    tmp->cellx[i*DIM+2]  = p->cellx[src*DIM+2];


    // ---- morton key is refreshed here from the member tmpMortonkey ----
    //p->mortonKey[i] = p->tmpMortonKey[i];

    tmp->pId[i] = p->pId[src];

}

__global__ void dk_morton_key(ParticleSys<DeviceMemory>* p, DeviceBoundingBox* box){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=p->N){
        return;
    }

    int bi=i*DIM;
    p->mortonOrder[i] = i;
    p->mortonKey[i]=morton3D((uint32_t)p->cellx[bi+0], (uint32_t)p->cellx[bi+1], (uint32_t)p->cellx[bi+2]);

    /* for debug */
    /*
       printf("ix=%d iy=%d iz=%d, key=%d\n",p->cellx[bi+0],p->cellx[bi+1],p->cellx[bi+2],p->mortonKey[i]);
     */


}

__global__ void dk_print_N(ParticleSys<DeviceMemory>* p, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N){
        return;
    }

    int bi=i*DIM;
    /* for debug */
    printf("ix=%d iy=%d iz=%d, key=%d\n",p->cellx[bi+0],p->cellx[bi+1],p->cellx[bi+2],p->mortonKey[i]);

}

__global__ void dk_print(ParticleSys<DeviceMemory>* p){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=p->N){
        return;
    }

    int bi=i*DIM;

    printf("i=%d pId=%d ix=%u iy=%u iz=%u key=%u\n",
            i, p->pId[i], p->cellx[bi+0], p->cellx[bi+1],p->cellx[bi+2],
            p->mortonKey[i]);
}
__global__ void dk_print_tmpMorton(ParticleSys<DeviceMemory>* p){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=p->N){
        return;
    }

    int bi=i*DIM;

    printf("i=%d pId=%d ix=%u iy=%u iz=%u key=%u\n",
            i, p->pId[i], p->cellx[bi+0], p->cellx[bi+1],p->cellx[bi+2],
            p->tmpMortonKey[i]);
}

__global__ void dk_build_cellCount(ParticleSys<DeviceMemory>* p, DeviceBoundingBox* box){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=p->N || p->isActive[i]!=1){
        return;
    }

    int cellId = d_calcCellId(p,i,box);
    p->cellId[i] = cellId;

    atomicAdd(&box->pNum[cellId],1);

}

__global__ void dk_build_pList(ParticleSys<DeviceMemory>* p, DeviceBoundingBox* box){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=p->N || p->isActive[i]!=1){
        return;
    }

    int cellId = p->cellId[i];
    int offset = atomicAdd(&box->cellOffset[cellId],1);
    int index = box->pStart[cellId] + offset;
    box->pList[index] = i;
}

__device__ int d_calcCellId(ParticleSys<DeviceMemory>* p,int i, DeviceBoundingBox* box){
    int bi = i*DIM;
    int dx = (int)floor((p->x[bi+0]-box->minx)*box->invdx)+1; //+1 for ghost cell
    int dy = (int)floor((p->x[bi+1]-box->miny)*box->invdy)+1; //+1 for ghost cell
    int dz = (int)floor((p->x[bi+2]-box->minz)*box->invdz)+1; //+1 for ghost cell
    p->cellx[bi+0] = dx;
    p->cellx[bi+1] = dy;
    p->cellx[bi+2] = dz;

    /* for debug */
    /*
       printf("recalc i=%d pId=%d ix=%d iy=%d\n", i, p->pId[i], dx, dy);
     */

    return (box->sizey*dz+dy)*box->sizex+dx;
}

__global__ void k_update_neighborlist_endsort(ParticleSys<DeviceMemory> *p,DeviceBoundingBox *box){

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= p->N || p->isActive[i]!=1) return;

    double skinR = box->skinR;

    //particle-particle
    /* cycle through neighbor cells */
    int bi=i*DIM;
    int x=p->cellx[bi+0];
    int y=p->cellx[bi+1];
    int z=p->cellx[bi+2];
    int numNei = 0;

    for (int sx=-1; sx<=1; sx++){
        for (int sy=-1; sy<=1; sy++){
            for (int sz=-1; sz<=1; sz++){
                int cellId = (box->sizey*(z+sz)+y+sy)*box->sizex+x+sx;

                int start = box->pStart[cellId];
                int end = start+box->pNum[cellId];
                for (int k=box->pStart[cellId]; k<end; k++){
                    int j = box->pList[k];
                    if (i==j || p->isActive[j]!=1){
                        continue;
                    }else{
                        int bj=j*DIM;

                        Vec3 del;
                        /* normal points toward particle i */
                        del.x = p->x[bi+0]- p->x[bj+0];
                        del.y = p->x[bi+1]- p->x[bj+1];
                        del.z = p->x[bi+2]- p->x[bj+2];
                        double distsq = vdot(del,del);
                        double R = p->r[i]+p->r[j]+skinR;
                        if (distsq<R*R){
                            p->neiList[i*MAX_NEI+numNei]=j;
                            numNei+=1;
                            if(numNei >= MAX_NEI){
                                printf("Neighbor over flow!!!!\n");
                            }
                        }
                    }
                }
            }
        }
    }/* neighbor cell search done */
    p->numNei[i]=numNei;
    d_sort_neighborlist(p->neiList,i*MAX_NEI,numNei);

    /* set reference position */
    p->refx[i]=p->x[bi+0];
    p->refy[i]=p->x[bi+1];
    p->refz[i]=p->x[bi+2];
}

__global__ void k_update_neighborlist(ParticleSys<DeviceMemory> *p,DeviceBoundingBox *box){

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= p->N || p->isActive[i]!=1) return;

    double skinR = box->skinR;

    //particle-particle
    /* cycle through neighbor cells */
    int bi=i*DIM;
    int x=p->cellx[bi+0];
    int y=p->cellx[bi+1];
    int z=p->cellx[bi+2];
    int numNei = 0;

    for (int sx=-1; sx<=1; sx++){
        for (int sy=-1; sy<=1; sy++){
            for (int sz=-1; sz<=1; sz++){
                int cellId = (box->sizey*(z+sz)+y+sy)*box->sizex+x+sx;

                int start = box->pStart[cellId];
                int end = start+box->pNum[cellId];
                for (int k=box->pStart[cellId]; k<end; k++){
                    int j = box->pList[k];
                    if (i==j || p->isActive[j]!=1){
                        continue;
                    }else{
                        int bj=j*DIM;

                        Vec3 del;
                        /* normal points toward particle i */
                        del.x = p->x[bi+0]- p->x[bj+0];
                        del.y = p->x[bi+1]- p->x[bj+1];
                        del.z = p->x[bi+2]- p->x[bj+2];
                        double distsq = vdot(del,del);
                        double R = p->r[i]+p->r[j]+skinR;
                        if (distsq<R*R){
                            p->neiList[i*MAX_NEI+numNei]=j;
                            numNei+=1;
                            if(numNei >= MAX_NEI){
                                printf("Neighbor over flow!!!!\n");
                            }
                        }
                    }
                }
            }
        }
    }/* neighbor cell search done */
    p->numNei[i]=numNei;

    /* set reference position */
    p->refx[i]=p->x[bi+0];
    p->refy[i]=p->x[bi+1];
    p->refz[i]=p->x[bi+2];
}

void d_update_pList_withSort(ParticleSys<DeviceMemory> *p, ParticleSys<DeviceMemory> *tmpPs, BoundingBox *box,int gridSize, int blockSize){

    /* initialize */
    cudaMemset(box->d_box.pNum, 0, sizeof(int)*box->N);
    cudaMemset(box->d_box.pStart, 0, sizeof(int)*box->N);
    cudaMemset(box->d_box.cellOffset, 0, sizeof(int)*box->N);

    dk_build_cellCount<<<gridSize,blockSize>>>(p->d_self,box->d_boxPtr);


    dk_morton_key<<<gridSize,blockSize>>>(p->d_self,box->d_boxPtr);



    cub::DeviceRadixSort::SortPairs(
            p->tmp_storage,
            p->tmp_bytes,
            p->mortonKey,
            p->tmpMortonKey,
            p->mortonOrder,
            p->tmpMortonOrder,
            p->N);


    cudaDeviceSynchronize();
    d_swap_ps<<<gridSize,blockSize>>>(p->d_self, tmpPs->d_self);


    swap_device_ps_member_pointer(p,tmpPs);
    cudaMemcpy(p->d_self, p, sizeof(ParticleSys<DeviceMemory>), cudaMemcpyHostToDevice);
    cudaMemcpy(tmpPs->d_self, tmpPs,
            sizeof(ParticleSys<DeviceMemory>),
            cudaMemcpyHostToDevice);



    cub::DeviceScan::ExclusiveSum(box->d_box.tmpExSum,
            box->d_box.scanTmpBytes,
            box->d_box.pNum,
            box->d_box.pStart,
            box->d_box.N);

    dk_build_pList<<<gridSize,blockSize>>>(p->d_self,box->d_boxPtr);
}

void d_update_pList(ParticleSys<DeviceMemory> *p, BoundingBox *box,int gridSize, int blockSize){
    // printf("N=%d MAX_NEI=%d numCell=%d\n", p->d_group.N, p->d_group.MAX_NEI, box->N);

    /* initialize */
    cudaMemset(box->d_box.pNum, 0, sizeof(int)*box->N);
    cudaMemset(box->d_box.pStart, 0, sizeof(int)*box->N);
    cudaMemset(box->d_box.cellOffset, 0, sizeof(int)*box->N);

    dk_build_cellCount<<<gridSize,blockSize>>>(p->d_self,box->d_boxPtr);

    cub::DeviceScan::ExclusiveSum(box->d_box.tmpExSum,
            box->d_box.scanTmpBytes,
            box->d_box.pNum,
            box->d_box.pStart,
            box->d_box.N);

    dk_build_pList<<<gridSize,blockSize>>>(p->d_self,box->d_boxPtr);

}
