#include "BVH.h"
#include "BoundingBox.h"

typedef struct {
    int start;
    int end;   // [start,end)
    int node;
} BuildTask;

static inline int sphereAABBOverlapNeighbor(ParticleSys<HostMemory>* p,int i,BVH *bvh, int j, double skinR){
    double minx = bvh->minx[j];
    double miny = bvh->miny[j];
    double minz = bvh->minz[j];

    double maxx = bvh->maxx[j];
    double maxy = bvh->maxy[j];
    double maxz = bvh->maxz[j];


    int bi=i*DIM;

    double px = p->x[bi+0];
    double py = p->x[bi+1];
    double pz = p->x[bi+2];

    double dx = 0.0;
    if(px < minx) dx = minx - px;
    else if(px > maxx) dx = px - maxx;

    double dy = 0.0;
    if(py < miny) dy = miny - py;
    else if(py > maxy) dy = py - maxy;

    double dz = 0.0;
    if(pz < minz) dz = minz - pz;
    else if(pz > maxz) dz = pz - maxz;

    double R=p->r[i]+skinR;
    double Rsq = R*R;

    return (dx*dx + dy*dy + dz*dz) <= Rsq;
}

TriangleContactCache dist_triangle_neighbor(ParticleSys<HostMemory>* ps, int i, TriangleMesh* mesh, int j,double skinR){
    /* check with plane first */
    double dist = 0.;
    int bi = i*DIM;
    TriangleContactCache res;
    Vec3 ntri;
    ntri.x = mesh->nx[j];
    ntri.y = mesh->ny[j];
    ntri.z = mesh->nz[j];

    res.n.x=0.;
    res.n.y=0.;
    res.n.z=0.;
    res.dist=1.e10;

    Vec3 pos;
    pos.x = ps->x[bi+0];
    pos.y = ps->x[bi+1];
    pos.z = ps->x[bi+2];

    dist += ntri.x*pos.x;
    dist += ntri.y*pos.y;
    dist += ntri.z*pos.z;
    dist += mesh->d[j];
    double distsq = dist*dist;
    double R = ps->r[i]+skinR;
    double Rsq = R*R;
    if(distsq > Rsq){
        /* no collision */
        return res;
    }

    /* check if inside the triangle */

    /* get vertex indices */
    int v0 = mesh->tri_i0[j];
    int v1 = mesh->tri_i1[j];
    int v2 = mesh->tri_i2[j];

    Vec3 ap;
    ap.x = pos.x - mesh->mx[v0];
    ap.y = pos.y - mesh->my[v0];
    ap.z = pos.z - mesh->mz[v0];

    double d00 = mesh->d00[j];
    double d01 = mesh->d01[j];
    double d11 = mesh->d11[j];
    double denom = mesh->denom[j];

    double d20 = ap.x*mesh->e01x[j] + ap.y*mesh->e01y[j] + ap.z*mesh->e01z[j];
    double d21 = ap.x*mesh->e02x[j] + ap.y*mesh->e02y[j] + ap.z*mesh->e02z[j];
    
    double v = (d11*d20 - d01*d21)*denom;
    double w = (d00*d21 - d01*d20)*denom;
    double u = 1.0 - v - w;
    /* 1) face inside */
    if (u>0. &&  v >0. && w>0.){
        double sign= dist<0 ? -1.0 : 1.0;
        res.n = vscalar(sign,ntri);
        res.dist = sign*dist;
        res.hitAt = -1;
        return res;
    }

    /* ========== vertices ========== */

    if (v<=0. && w<=0.0){
        Vec3 tx;
        tx.x = mesh->mx[v0];
        tx.y = mesh->my[v0];
        tx.z = mesh->mz[v0];

        res.n = vsub(pos,tx);
       distsq = vdot(res.n,res.n);
       if(distsq < Rsq){
           res.dist = sqrt(distsq);
           res.n = vscalar(1./res.dist,res.n);
       }
       res.hitAt = v0; 
       return res;
    }

    if (u<=0. && w<=0.0){
        Vec3 tx;
        tx.x = mesh->mx[v1];
        tx.y = mesh->my[v1];
        tx.z = mesh->mz[v1];

        res.n = vsub(pos,tx);
        distsq = vdot(res.n,res.n);
       if(distsq < Rsq){
           res.dist = sqrt(distsq);
           res.n = vscalar(1./res.dist,res.n);
       }

       res.hitAt = v1; 
       return res;
    }

    if (u<=0. && v<=0.0){
        Vec3 tx;
        tx.x = mesh->mx[v2];
        tx.y = mesh->my[v2];
        tx.z = mesh->mz[v2];

        res.n = vsub(pos,tx);
        distsq = vdot(res.n,res.n);
        if(distsq < Rsq){
            res.dist = sqrt(distsq); 
            res.n = vscalar(1./res.dist,res.n);
        }

        res.hitAt = v2; 
        return res;
    }

    /* ========== edge ========== */
    if (u<=0.0){
        Vec3 bp;

        Vec3 tmpv;
        tmpv.x = mesh->mx[v1];
        tmpv.y = mesh->my[v1];
        tmpv.z = mesh->mz[v1];

        Vec3 tmpE;
        tmpE.x = mesh->e12x[j];
        tmpE.y = mesh->e12y[j];
        tmpE.z = mesh->e12z[j];

        bp = vsub(pos,tmpv);

        double t = vdot(bp,tmpE)*mesh->d22inv[j];
        t = fmax(0., fmin(1.0,t));

        Vec3 hitPoint = vadd(tmpv,vscalar(t,tmpE));
        res.n = vsub(pos,hitPoint);
        distsq = vdot(res.n,res.n);
        if(distsq < Rsq){
            res.dist = sqrt(distsq);
            res.n = vscalar(1./res.dist,res.n);
        }
        res.hitAt = mesh->tri_e1[j]; 
        return res;
    }

    if (v<=0.0){
        Vec3 tmpE;
        tmpE.x = mesh->e02x[j];
        tmpE.y = mesh->e02y[j];
        tmpE.z = mesh->e02z[j];

        double t = d21*mesh->d11inv[j];
        t = fmax(0., fmin(1.0,t));
        Vec3 tmpv;
        tmpv.x = mesh->mx[v0];
        tmpv.y = mesh->my[v0];
        tmpv.z = mesh->mz[v0];

        Vec3 hitPoint = vadd(tmpv,vscalar(t,tmpE));
        res.n = vsub(pos,hitPoint);
        distsq = vdot(res.n,res.n);

        if(distsq < Rsq){
            res.dist = sqrt(distsq);
            res.n = vscalar(1./res.dist,res.n);
        }
        res.hitAt = mesh->tri_e2[j]; 
        return res;
    }

    /* w=<0.0 or 0.,0.,0. */
    Vec3 tmpE;
    tmpE.x = mesh->e01x[j];
    tmpE.y = mesh->e01y[j];
    tmpE.z = mesh->e01z[j];


    double t = d20*mesh->d00inv[j];
    t = fmax(0., fmin(1.0,t));
    Vec3 tmpv;
    tmpv.x = mesh->mx[v0];
    tmpv.y = mesh->my[v0];
    tmpv.z = mesh->mz[v0];

    Vec3 hitPoint = vadd(tmpv,vscalar(t,tmpE));
    res.n = vsub(pos,hitPoint);
    distsq = vdot(res.n,res.n);
    if(distsq < Rsq){
        res.dist = sqrt(distsq);
        res.n = vscalar(1./res.dist,res.n);
    }
    res.hitAt = mesh->tri_e0[j]; 
    return res;
}

void update_neighborlist_wall_nobvh(ParticleSys<HostMemory> *p,TriangleMesh *mesh,BoundingBox *box,double skinR){
    for(int i=0; i<p->N; i++){

        if(p->isActive[i]!=1){
            continue;
        }

        /* cycle through neighbor cells */
        int bi=i*DIM;
        int x=p->cellx[bi+0];
        int y=p->cellx[bi+1];
        int z=p->cellx[bi+2];

        int numNei = 0;
        int numContWallNow = 0;
        for (int sx=-1; sx<=1; sx++){
            for (int sy=-1; sy<=1; sy++){
                for (int sz=-1; sz<=1; sz++){
                    int cellId = (box->sizey*(z+sz)+y+sy)*box->sizex+x+sx;
                    for(int j=0; j<box->tNum[cellId]; j++){
                        int indTri = box->tList[cellId*box->MAX_TRI+j];

                        // ここで球 vs 三角形の精密判定
                        TriangleContactCache tc;
                        tc = dist_triangle_neighbor(p,i,mesh,indTri,skinR); 
                        int wasHitBefore = 0;
                        for (int k=0; k<numContWallNow; k++){
                            if (indTri == p->indHisWallNow[i*MAX_NEI+k]){
                                wasHitBefore =1;
                                continue;
                            }
                        }
                        if (wasHitBefore == 1){
                            continue;
                        }

                        if(tc.dist<p->r[i]+skinR){
                            p->neiListWall[i*MAX_NEI+numNei]=indTri;
                            numNei+=1;
                            if(numNei >= MAX_NEI){
                                printf("Wall Neighbor over flow!!!!\n");
                                abort();
                            }

                            int numWall = numContWallNow;
                            p->indHisWallNow[i*MAX_NEI+numWall] = indTri;
                            numContWallNow+=1;
                        }
                    }
                }
            }
        }

        p->numNeiWall[i]=numNei;
    }
}

__device__ __forceinline__ TriangleContactCache d_dist_triangle_neighbor(ParticleSys<DeviceMemory>* ps, int i, DeviceTriangleMesh* mesh, int j,double skinR){
    /* check with plane first */
    double dist = 0.;
    int bi = i*DIM;
    TriangleContactCache res;
    Vec3 ntri;
    ntri.x = mesh->nx[j];
    ntri.y = mesh->ny[j];
    ntri.z = mesh->nz[j];

    res.n.x=0.;
    res.n.y=0.;
    res.n.z=0.;
    res.dist=1.e10;

    Vec3 pos;
    pos.x = ps->x[bi+0];
    pos.y = ps->x[bi+1];
    pos.z = ps->x[bi+2];

    dist += ntri.x*pos.x;
    dist += ntri.y*pos.y;
    dist += ntri.z*pos.z;
    dist += mesh->d[j];
    double distsq = dist*dist;
    double R = ps->r[i]+skinR;
    double Rsq = R*R;
    if(distsq > Rsq){
        /* no collision */
        return res;
    }

    /* check if inside the triangle */

    /* get vertex indices */
    int v0 = mesh->tri_i0[j];
    int v1 = mesh->tri_i1[j];
    int v2 = mesh->tri_i2[j];

    Vec3 ap;
    ap.x = pos.x - mesh->mx[v0];
    ap.y = pos.y - mesh->my[v0];
    ap.z = pos.z - mesh->mz[v0];

    double d00 = mesh->d00[j];
    double d01 = mesh->d01[j];
    double d11 = mesh->d11[j];
    double denom = mesh->denom[j];

    double d20 = ap.x*mesh->e01x[j] + ap.y*mesh->e01y[j] + ap.z*mesh->e01z[j];
    double d21 = ap.x*mesh->e02x[j] + ap.y*mesh->e02y[j] + ap.z*mesh->e02z[j];
    
    double v = (d11*d20 - d01*d21)*denom;
    double w = (d00*d21 - d01*d20)*denom;
    double u = 1.0 - v - w;
    /* 1) face inside */
    if (u>0. &&  v >0. && w>0.){
        double sign= dist<0 ? -1.0 : 1.0;
        res.n = vscalar(sign,ntri);
        res.dist = sign*dist;
        res.hitAt = -1;
        return res;
    }

    /* ========== vertices ========== */

    if (v<=0. && w<=0.0){
        Vec3 tx;
        tx.x = mesh->mx[v0];
        tx.y = mesh->my[v0];
        tx.z = mesh->mz[v0];

        res.n = vsub(pos,tx);
       distsq = vdot(res.n,res.n);
       if(distsq < Rsq){
           res.dist = sqrt(distsq);
           res.n = vscalar(1./res.dist,res.n);
       }
       res.hitAt = v0; 
       return res;
    }

    if (u<=0. && w<=0.0){
        Vec3 tx;
        tx.x = mesh->mx[v1];
        tx.y = mesh->my[v1];
        tx.z = mesh->mz[v1];

        res.n = vsub(pos,tx);
        distsq = vdot(res.n,res.n);
       if(distsq < Rsq){
           res.dist = sqrt(distsq);
           res.n = vscalar(1./res.dist,res.n);
       }

       res.hitAt = v1; 
       return res;
    }

    if (u<=0. && v<=0.0){
        Vec3 tx;
        tx.x = mesh->mx[v2];
        tx.y = mesh->my[v2];
        tx.z = mesh->mz[v2];

        res.n = vsub(pos,tx);
        distsq = vdot(res.n,res.n);
        if(distsq < Rsq){
            res.dist = sqrt(distsq); 
            res.n = vscalar(1./res.dist,res.n);
        }

        res.hitAt = v2; 
        return res;
    }

    /* ========== edge ========== */
    if (u<=0.0){
        Vec3 bp;

        Vec3 tmpv;
        tmpv.x = mesh->mx[v1];
        tmpv.y = mesh->my[v1];
        tmpv.z = mesh->mz[v1];

        Vec3 tmpE;
        tmpE.x = mesh->e12x[j];
        tmpE.y = mesh->e12y[j];
        tmpE.z = mesh->e12z[j];

        bp = vsub(pos,tmpv);

        double t = vdot(bp,tmpE)*mesh->d22inv[j];
        t = fmax(0., fmin(1.0,t));

        Vec3 hitPoint = vadd(tmpv,vscalar(t,tmpE));
        res.n = vsub(pos,hitPoint);
        distsq = vdot(res.n,res.n);
        if(distsq < Rsq){
            res.dist = sqrt(distsq);
            res.n = vscalar(1./res.dist,res.n);
        }
        res.hitAt = mesh->tri_e1[j]; 
        return res;
    }

    if (v<=0.0){
        Vec3 tmpE;
        tmpE.x = mesh->e02x[j];
        tmpE.y = mesh->e02y[j];
        tmpE.z = mesh->e02z[j];

        double t = d21*mesh->d11inv[j];
        t = fmax(0., fmin(1.0,t));
        Vec3 tmpv;
        tmpv.x = mesh->mx[v0];
        tmpv.y = mesh->my[v0];
        tmpv.z = mesh->mz[v0];

        Vec3 hitPoint = vadd(tmpv,vscalar(t,tmpE));
        res.n = vsub(pos,hitPoint);
        distsq = vdot(res.n,res.n);

        if(distsq < Rsq){
            res.dist = sqrt(distsq);
            res.n = vscalar(1./res.dist,res.n);
        }
        res.hitAt = mesh->tri_e2[j]; 
        return res;
    }

    /* w=<0.0 or 0.,0.,0. */
    Vec3 tmpE;
    tmpE.x = mesh->e01x[j];
    tmpE.y = mesh->e01y[j];
    tmpE.z = mesh->e01z[j];


    double t = d20*mesh->d00inv[j];
    t = fmax(0., fmin(1.0,t));
    Vec3 tmpv;
    tmpv.x = mesh->mx[v0];
    tmpv.y = mesh->my[v0];
    tmpv.z = mesh->mz[v0];

    Vec3 hitPoint = vadd(tmpv,vscalar(t,tmpE));
    res.n = vsub(pos,hitPoint);
    distsq = vdot(res.n,res.n);
    if(distsq < Rsq){
        res.dist = sqrt(distsq);
        res.n = vscalar(1./res.dist,res.n);
    }
    res.hitAt = mesh->tri_e0[j]; 
    return res;
}

__global__ void k_update_neighborlist_wall(ParticleSys<DeviceMemory> *p, DeviceTriangleMesh *mesh, DeviceBVH *bvh,double skinR){

    int i = blockIdx.x * blockDim.x + threadIdx.x;


    if(i >= p->N || p->isActive[i]!=1){
        return;
    }

    int stack[128];
    int sp = 0;

    stack[sp++] = 0;   // root


    int numNei = 0;
    while(sp > 0){
        int node = stack[--sp];

        if(!d_sphereAABBOverlapNeighbor(p,i,bvh,node,skinR)){
            continue;
        }

        if(bvh->tri[node] >= 0){
            int indTri = bvh->tri[node];

            // ここで球 vs 三角形の精密判定
            TriangleContactCache tc;
            tc = d_dist_triangle_neighbor(p,i,mesh,indTri,skinR); 

            if(tc.dist<p->r[i]+skinR){
                p->neiListWall[i*MAX_NEI+numNei]=indTri;
                numNei+=1;
                if(numNei >= MAX_NEI){
                    printf("Wall Neighbor over flow!!!!\n");
                }
            }
        }else{
            stack[sp++] = bvh->left[node];
            stack[sp++] = bvh->right[node];
        }
    }
    p->numNeiWall[i]=numNei;

    /* == sort for reproducibility== */
    d_sort_neighborlist(p->neiListWall,i*MAX_NEI,numNei);

}

__device__ __forceinline__ int d_sphereAABBOverlapNeighbor(ParticleSys<DeviceMemory>* p,int i,DeviceBVH *bvh, int j, double skinR){
    double minx = bvh->minx[j];
    double miny = bvh->miny[j];
    double minz = bvh->minz[j];

    double maxx = bvh->maxx[j];
    double maxy = bvh->maxy[j];
    double maxz = bvh->maxz[j];


    int bi=i*DIM;

    double px = p->x[bi+0];
    double py = p->x[bi+1];
    double pz = p->x[bi+2];

    double dx = 0.0;
    if(px < minx) dx = minx - px;
    else if(px > maxx) dx = px - maxx;

    double dy = 0.0;
    if(py < miny) dy = miny - py;
    else if(py > maxy) dy = py - maxy;

    double dz = 0.0;
    if(pz < minz) dz = minz - pz;
    else if(pz > maxz) dz = pz - maxz;

    double R=p->r[i]+skinR;
    double Rsq = R*R;

    return (dx*dx + dy*dy + dz*dz) <= Rsq;
}

void update_neighborlist_wall(ParticleSys<HostMemory> *p,TriangleMesh *mesh, BVH *bvh,double skinR){

    #pragma omp parallel for
    for(int i=0; i<p->N; i++){

        if(p->isActive[i]!=1){
            continue;
        }

        int stack[128];
        int sp = 0;

        stack[sp++] = 0;   // root


        int numNei = 0;
        while(sp > 0){
            int node = stack[--sp];

            if(!sphereAABBOverlapNeighbor(p,i,bvh,node,skinR)){
                continue;
            }

            if(bvh->tri[node] >= 0){
                int indTri = bvh->tri[node];

                // ここで球 vs 三角形の精密判定
                TriangleContactCache tc;
                tc = dist_triangle_neighbor(p,i,mesh,indTri,skinR); 

                if(tc.dist<p->r[i]+skinR){
                    p->neiListWall[i*MAX_NEI+numNei]=indTri;
                    numNei+=1;
                    if(numNei >= MAX_NEI){
                        printf("Wall Neighbor over flow!!!!\n");
                        abort();
                    }
                }
            }else{
                stack[sp++] = bvh->left[node];
                stack[sp++] = bvh->right[node];
            }
        }
        p->numNeiWall[i]=numNei;

        /* == sort for reproducibility == */
        insertSort(p->neiListWall,i*MAX_NEI,numNei);

    }
}


void initializeBVH(BVH *bvh, int numTriangles, int isGPUon){
    int maxNodes = 2*numTriangles-1;

    #define MEMBER(type, name, size) bvh->name = (type*)malloc(sizeof(type)*(size));
    #include "memberList/BVHMember_common.def"
    #undef MEMBER

    bvh->nodeCount = 0;
    #if USE_GPU
    if (isGPUon == 1){
        #define MEMBER(type, name, size) cudaMalloc(&bvh->d_bvh.name, sizeof(type)*(size));
        #include "memberList/BVHMember_common.def"
        #undef MEMBER

        bvh->d_bvh.nodeCount = bvh->nodeCount;

        /* for structs*/
        cudaMalloc(&bvh->d_bvhPtr,sizeof(DeviceBVH));
    }

    #endif
}

void free_BVH(BVH *bvh, int isGPUon){
    #define MEMBER(type, name, size) free(bvh->name);
    #include "memberList/BVHMember_common.def"
    #undef MEMBER

    #if USE_GPU
    if (isGPUon == 1){

        #define MEMBER(type, name, size) cudaFree(bvh->d_bvh.name);
        #include "memberList/BVHMember_common.def"
        #undef MEMBER

        cudaFree(bvh->d_bvhPtr);
    }
    #endif
}

void copyToDeviceBVH(BVH *bvh,  int numTriangles){

    int maxNodes = 2*numTriangles-1;

    bvh->d_bvh.nodeCount = bvh->nodeCount;

    #define MEMBER(type, name, size) cudaMemcpy(bvh->d_bvh.name, bvh->name, sizeof(type)*(size),cudaMemcpyHostToDevice);
    #include "memberList/BVHMember_common.def"
    #undef MEMBER

    cudaMemcpy(bvh->d_bvhPtr,  &bvh->d_bvh,  sizeof(DeviceBVH), cudaMemcpyHostToDevice);
}


void computeNodeAABB(int start, int end,TriangleMesh *mesh, BVH *bvh,int k){
    int startTri = mesh->sortedIndex[start];
    double minx = mesh->minx[startTri];
    double miny = mesh->miny[startTri];
    double minz = mesh->minz[startTri];
    double maxx = mesh->maxx[startTri];
    double maxy = mesh->maxy[startTri];
    double maxz = mesh->maxz[startTri];

    for(int i = start+1; i < end; i++){
        int sortedi = mesh->sortedIndex[i];
        if(mesh->minx[sortedi] < minx) minx = mesh->minx[sortedi];
        if(mesh->miny[sortedi] < miny) miny = mesh->miny[sortedi];
        if(mesh->minz[sortedi] < minz) minz = mesh->minz[sortedi];

        if(mesh->maxx[sortedi] > maxx) maxx = mesh->maxx[sortedi];
        if(mesh->maxy[sortedi] > maxy) maxy = mesh->maxy[sortedi];
        if(mesh->maxz[sortedi] > maxz) maxz = mesh->maxz[sortedi];
    }

    bvh->minx[k] = minx;
    bvh->miny[k] = miny;
    bvh->minz[k] = minz;

    bvh->maxx[k] = maxx;
    bvh->maxy[k] = maxy;
    bvh->maxz[k] = maxz;
}

int buildBVH(BVH* bvh, TriangleMesh *mesh){

    BuildTask* stack = (BuildTask*)malloc(sizeof(BuildTask) * (2*mesh->nTri));
    int sp = 0;

    int nodeCount = 0;

    int root = nodeCount++;
    stack[sp++] = (BuildTask){0, mesh->nTri, root};

    while(sp > 0){
        BuildTask t = stack[--sp];

        int start = t.start;
        int end   = t.end;
        int node  = t.node;

        computeNodeAABB(start, end,mesh, bvh, node);

        int count = end - start;

        if(count == 1)
        {
            bvh->left[node]  = -1;
            bvh->right[node] = -1;
            bvh->tri[node]   = mesh->sortedIndex[start];   // Morton順
            continue;
        }

        int mid = (start + end) / 2;

        int leftChild  = nodeCount++;
        int rightChild = nodeCount++;

        bvh->left[node]  = leftChild;
        bvh->right[node] = rightChild;
        bvh->tri[node]   = -1;

        stack[sp++] = (BuildTask){mid, end, rightChild};
        stack[sp++] = (BuildTask){start, mid, leftChild};
    }

    bvh->nodeCount = nodeCount;

    free(stack);
    return root;
}
