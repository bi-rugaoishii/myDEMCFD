#include "device_dem.h"
#define DIM 3


/*
============================================================
__device__ 関数群
============================================================
*/

/* === triangle related === */
__device__ __forceinline__
TriangleContactCache d_dist_triangle(ParticleSys<DeviceMemory>* ps, int i, DeviceTriangleMesh* mesh, int j){
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
    if(distsq > ps->rsq[i]){
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
       if(distsq < ps->rsq[i]){
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
       if(distsq < ps->rsq[i]){
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
        if(distsq < ps->rsq[i]){
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
        if(distsq < ps->rsq[i]){
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

        if(distsq < ps->rsq[i]){
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
    if(distsq < ps->rsq[i]){
        res.dist = sqrt(distsq);
        res.n = vscalar(1./res.dist,res.n);
    }
    res.hitAt = mesh->tri_e0[j]; 
    return res;
}

__device__ __forceinline__
void d_wall_collision_triangles(ParticleSys<DeviceMemory>* p,int i,DeviceBoundingBox *box, DeviceTriangleMesh* mesh){

    //particle-particle
    /* cycle through neighbor cells */
    int bi=i*DIM;
    int x=p->cellx[bi+0];
    int y=p->cellx[bi+1];
    int z=p->cellx[bi+2];

    int numContVorENow = 0;
    int numContWallNow = 0;
    for (int sx=-1; sx<=1; sx++){
        for (int sy=-1; sy<=1; sy++){
            for (int sz=-1; sz<=1; sz++){
                int cellId = (box->sizey*(z+sz)+y+sy)*box->sizex+x+sx;
                for(int j=0; j<box->tNum[cellId]; j++){
                    int indTri = box->tList[cellId*box->MAX_TRI+j];

                    int wasHitBefore = 0;
                    /* check if the triangle was already hit */
                    for (int k=0; k<numContWallNow; k++){
                        if (indTri == p->indHisWallNow[i*MAX_NEI+k]){
                            wasHitBefore =1;
                            continue;
                        }
                    }
                    if (wasHitBefore == 1){
                        continue;
                    }


                    TriangleContactCache tc;
                    tc = d_dist_triangle(p,i,mesh,indTri); 

                    if(tc.dist<p->r[i]){
                        double delmag;
                        if(tc.hitAt==-1){ //hit at face
                            delmag = p->r[i]-tc.dist;
                        }else{ //hit as face or edge
                            int end = numContVorENow;
                            int hadDuplicate=0;

                            for (int k=0; k<end; k++){
                                if(p->indHisVorENow[i*MAX_NEI+k]==tc.hitAt){
                                    hadDuplicate=1;
                                    break;
                                }
                            }

                            if(hadDuplicate ==1){
                                continue;

                            }

                            delmag = p->r[i]-tc.dist;


                            p->indHisVorENow[i*MAX_NEI+end] = tc.hitAt;
                            numContVorENow+=1;
                        }
                        if (delmag*p->invr[i]*0.5>0.1){
                            printf("overlap over 10%% with wall with %f\n!!!!\n",delmag*p->invr[i]);
                        }

                        ContactCache c;
                        c = d_calc_normal_force_wall(p,i,j,tc.n,delmag);
                        d_calc_tangential_force_wall(p,i,j,c);
                        int numWall = numContWallNow;
                        p->indHisWallNow[i*MAX_NEI+numWall] = indTri;
                        numContWallNow+=1;
                    }

                }
            }
        }
    }
    d_update_history_wall(p,i);

}

__device__ __forceinline__ void d_wall_collision_verlet(ParticleSys<DeviceMemory>* p,int i, DeviceTriangleMesh* mesh){

        if(p->isActive[i]!=1){
            return;
        }

        int end = p->numNeiWall[i];
        int ci = i*MAX_NEI;

        int numContVorENow = 0;

        for (int k=0; k<end; k++){
            int indTri = p->neiListWall[ci+k];


            // ここで球 vs 三角形の精密判定
            TriangleContactCache tc;
            tc = d_dist_triangle(p,i,mesh,indTri); 

            if(tc.dist<p->r[i]){
                double delmag;
                if(tc.hitAt==-1){ //hit at face
                    delmag = p->r[i]-tc.dist;
                }else{ //hit as face or edge
                    int end = numContVorENow;
                    int hadDuplicate=0;

                    for (int k=0; k<end; k++){
                        if(p->indHisVorENow[i*MAX_NEI+k]==tc.hitAt){
                           // printf("duplicate collision of vertex or edge!!\n");
                            hadDuplicate=1;
                            break;
                        }
                    }

                    if(hadDuplicate ==1){
                        continue;

                    }

                    delmag = p->r[i]-tc.dist;


                    p->indHisVorENow[i*MAX_NEI+end] = tc.hitAt;
                    numContVorENow+=1;
                }
                if (delmag*p->invr[i]*0.5>0.1){
                    printf("overlap over 10%% with wall!!!!\n");
                }

                ContactCache c;
                c = d_calc_normal_force_wall(p,i,indTri,tc.n,delmag);
                d_calc_tangential_force_wall(p,i,indTri,c);
            }
        }
        d_update_history_wall(p,i);
}

__device__ __forceinline__
ContactCache d_calc_normal_force_wall(ParticleSys<DeviceMemory> *p,int i,int j,Vec3 n,double delMag){

    int bi = i*DIM;

    /* get deltas */
    Vec3 del;
    del = vscalar(-delMag,n);

    /* get relative velocity (for now assumed not moving)*/ 
    Vec3 v_rel;
    v_rel.x = p->v[bi+0];
    v_rel.y = p->v[bi+1];
    v_rel.z = p->v[bi+2];

    double v_reldotn = vdot(v_rel,n);

    /* get normal relative velocity*/
    ContactCache result;
    result.vn_rel = vscalar(v_reldotn,n);


    double eta = p->etaconst[i]*p->sqrtm[i];
    result.eta = eta;
    
    double k=p->k[i];
    double r=p->r[i];

    result.fn.x = -k*del.x - eta*result.vn_rel.x;
    result.fn.y = -k*del.y - eta*result.vn_rel.y;
    result.fn.z = -k*del.z - eta*result.vn_rel.z;


    /* calculate relative tangential velocity */
    Vec3 vrot;
    vrot.x = r*p->angv[bi+0];
    vrot.y = r*p->angv[bi+1];
    vrot.z = r*p->angv[bi+2];
    vrot = vcross(vrot,n);

    /* calculate relative tangential velocity */
    Vec3 vt;
    vt=vsub(v_rel,result.vn_rel);
    result.vt=vadd(vt,vrot);

    /* add force */

    p->f[bi+0] += result.fn.x;
    p->f[bi+1] += result.fn.y;
    p->f[bi+2] += result.fn.z;

    result.n = n;

    return result;
}

__device__ __forceinline__
void d_calc_tangential_force_wall(ParticleSys<DeviceMemory> *p,int i,int j,ContactCache c){

    int bi = i*DIM;
    int ci = i*MAX_NEI;

    /* check history */

    int isInHis = 0;
    int neiInd=0;

    for (int k=0; k<p->numContWall[i]; k++){
        if (j == p->indHisWall[ci+k]){ /* if the contact particle is in the history */
            isInHis =1;
            neiInd=k;
            p->isContactWall[ci+neiInd]=1;
            break;
        }
    }

    if(isInHis == 0){
        neiInd=p->numContWall[i];
        p->isContactWall[ci+neiInd]=1;
        p->indHisWall[ci+neiInd]=j;
        p->numContWall[i] +=1;
        p->deltHisxWall[ci+neiInd] = 0.;
        p->deltHisyWall[ci+neiInd] = 0.;
        p->deltHiszWall[ci+neiInd] = 0.;
    }



    /* get the magnitude of delta t history */
    Vec3 delt_old;
    delt_old.x = p->deltHisxWall[ci+neiInd];
    delt_old.y = p->deltHisyWall[ci+neiInd];
    delt_old.z = p->deltHiszWall[ci+neiInd];

    double dt = p->dt;

    /* get new delta t*/
    Vec3 delt_new;

        
    delt_new.x =delt_old.x + c.vt.x*dt;
    delt_new.y =delt_old.y + c.vt.y*dt;
    delt_new.z =delt_old.z + c.vt.z*dt;

    double deltdotn=vdot(delt_new,c.n);
    delt_new=vsub(delt_new,vscalar(deltdotn,c.n));

    Vec3 ft;


    ft = vscalar(-p->k[i],delt_new);


    /* ========== Friction ============ */
    double ftsq = vdot(ft,ft);
    double fnsq = vdot(c.fn,c.fn);


    if(ftsq>(p->mu*p->mu)*fnsq){ /* slip */
        Vec3 t; /* tangential normal vector */
        t = vscalar(1./(sqrt(ftsq)+SMALL_NUM),ft); 

        double fnnorm = sqrt(fnsq);
        ft = vscalar(p->mu*fnnorm,t);
        delt_new = vscalar(-1./p->k[i],ft);
    }else{
        /* add damping */
        ft.x -= c.eta*c.vt.x;
        ft.y -= c.eta*c.vt.y;
        ft.z -= c.eta*c.vt.z;
    }



    /* ======== add angular acceleration ========== */
    Vec3 mom;
    mom = vcross(c.n,ft);
    mom = vscalar(p->r[i],mom);

    p->mom[bi+0] += mom.x;
    p->mom[bi+1] += mom.y;
    p->mom[bi+2] += mom.z;

    /* ======== add force ========== */
    p->f[bi+0] += ft.x;
    p->f[bi+1] += ft.y;
    p->f[bi+2] += ft.z;


    /* ====== refresh history ======= */
    p->deltHisxWall[ci+neiInd] = delt_new.x;
    p->deltHisyWall[ci+neiInd] = delt_new.y;
    p->deltHiszWall[ci+neiInd] = delt_new.z;
}


__global__
void k_wall_collision_triangles_naive(ParticleSys<DeviceMemory>* p,DeviceTriangleMesh* mesh){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p->N || p->isActive[i]!=1) return;

    //particle-triangle
    /* cycle through neighbor cells */

    int numContVorENow = 0;
    int numContWallNow = 0;
    for(int j=0; j<mesh->nTri; j++){
        int indTri = j;

        int wasHitBefore = 0;
        /* check if the triangle was already hit */
        for (int k=0; k<numContWallNow; k++){
            if (indTri == p->indHisWallNow[i*MAX_NEI+k]){
                wasHitBefore =1;
                continue;
            }
        }
        if (wasHitBefore == 1){
            continue;
        }


        TriangleContactCache tc;
        tc = d_dist_triangle(p,i,mesh,indTri); 

        if(tc.dist<p->r[i]){
            double delmag;
            if(tc.hitAt==-1){ //hit at face
                delmag = p->r[i]-tc.dist;
            }else{ //hit as face or edge
                int end = numContVorENow;
                int hadDuplicate=0;

                for (int k=0; k<end; k++){
                    if(p->indHisVorENow[i*MAX_NEI+k]==tc.hitAt){
                        //printf("duplicate collision of vertex or edge!!\n");
                        hadDuplicate=1;
                        break;
                    }
                }

                if(hadDuplicate ==1){
                    continue;

                }

                delmag = p->r[i]-tc.dist;


                p->indHisVorENow[i*MAX_NEI+end] = tc.hitAt;
                numContVorENow+=1;
            }
            if (delmag*p->invr[i]*0.5>0.1){
                printf("overlap over 10%% with wall!!!!\n");
            }

            ContactCache c;
            c = d_calc_normal_force_wall(p,i,indTri,tc.n,delmag);
            d_calc_tangential_force_wall(p,i,indTri,c);

            int numWall = numContWallNow;
            p->indHisWallNow[i*MAX_NEI+numWall] = indTri;
            numContWallNow+=1;
        }
    }
    d_update_history_wall(p,i);
}

__global__
void k_particle_collision_naive(ParticleSys<DeviceMemory>* p){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= p->N || p->isActive[i]!=1) return;

    //particle-particle
    int bi=i*DIM;
    for (int j=0; j<p->N; j++){
        if (i==j){
            continue;
        }else{
            int bj=j*DIM;

            Vec3 del;
            /* normal points toward particle i */
            del.x = p->x[bi+0]- p->x[bj+0];
            del.y = p->x[bi+1]- p->x[bj+1];
            del.z = p->x[bi+2]- p->x[bj+2];
            double distsq = vdot(del,del);
            double R = p->r[i]+p->r[j];

            if (distsq<R*R){
                double dist = sqrt(distsq);
                double delMag = R-dist;
                if (delMag*p->invr[i]*0.5>0.05){
                    printf("overlap over 5%%!!!!\n");
                    printf("overlap is %f %%\n",delMag*p->invr[i]*0.5*100.);
                }
                /* ======================================================
                   Force Calculation
                   ======================================================*/
                /* get normal direction */
                Vec3 n;
                n.x = del.x/dist;
                n.y = del.y/dist;
                n.z = del.z/dist;
                ContactCache c;
                c = d_calc_normal_force(p,i,j,n,delMag);
                d_calc_tangential_force(p,i,j,c);

            }
        }
    }
    d_update_history(p,i);
}

__device__ __forceinline__
void d_calc_tangential_force(ParticleSys<DeviceMemory> *p,int i,int j,ContactCache c){

    int bi = i*DIM;
    int ci = i*MAX_NEI;

    /* check history */

    int isInHis = 0;
    int neiInd=0;
    int pId = p->pId[j];

    for (int k=0; k<p->numCont[i]; k++){
        if (pId == p->indHis[ci+k]){ /* if the contact particle is in the history */
            isInHis =1;
            neiInd=k;
            p->isContact[ci+neiInd]=1;
            break;
        }
    }

    if(isInHis == 0){
        neiInd=p->numCont[i];
        p->isContact[ci+neiInd]=1;
        p->indHis[ci+neiInd]=pId;
        p->numCont[i] +=1;
        p->deltHisx[ci+neiInd] = 0.;
        p->deltHisy[ci+neiInd] = 0.;
        p->deltHisz[ci+neiInd] = 0.;
    }


    /* get the magnitude of delta t history */
    Vec3 delt_old;
    delt_old.x = p->deltHisx[ci+neiInd];
    delt_old.y = p->deltHisy[ci+neiInd];
    delt_old.z = p->deltHisz[ci+neiInd];


    double dt = p->dt;

    /* get new delta t*/
    Vec3 delt_new;


    delt_new.x =delt_old.x + c.vt.x*dt;
    delt_new.y =delt_old.y + c.vt.y*dt;
    delt_new.z =delt_old.z + c.vt.z*dt;

    double deltdotn=vdot(delt_new,c.n);
    delt_new=vsub(delt_new,vscalar(deltdotn,c.n));

    Vec3 ft;


    ft = vscalar(-p->k[i],delt_new);

    /* ========== Friction ============ */
    double ftsq = vdot(ft,ft);
    double fnsq = vdot(c.fn,c.fn);

    if(ftsq>(p->mu*p->mu)*fnsq){ /* slip */
        if (ftsq!=0.){
            Vec3 t;
            t = vscalar(1./(sqrt(ftsq)+SMALL_NUM),ft); 

            double fnnorm = sqrt(fnsq);
            ft = vscalar(p->mu*fnnorm,t);
            delt_new = vscalar(-1./p->k[i],ft);

            /* for debugging
               double force_factor = p->mass_factor*p->length_factor/(p->time_factor*p->time_factor);
               printf("ft after scaling %f %f %f\n", ft.x*force_factor,ft.y*force_factor,ft.z*force_factor);
               printf("ratio after scaling %f \n", sqrt(vdot(ft,ft))/(p->mu*fnnorm));
             */

        }else{
            ft.x =0.;
            ft.y =0.;
            ft.z =0.;
        }
    }else{
        /* add damping */
        ft.x -= c.eta*c.vt.x;
        ft.y -= c.eta*c.vt.y;
        ft.z -= c.eta*c.vt.z;
    }

    /* ======== add force ========== */
    p->f[bi+0] += ft.x;
    p->f[bi+1] += ft.y;
    p->f[bi+2] += ft.z;

    /* ======== add angular acceleration ========== */
    Vec3 mom;
    mom = vcross(c.n,ft);
    mom = vscalar(p->r[i],mom);

    p->mom[bi+0] += mom.x;
    p->mom[bi+1] += mom.y;
    p->mom[bi+2] += mom.z;

    /* ====== refresh history ======= */
    p->deltHisx[ci+neiInd] = delt_new.x;
    p->deltHisy[ci+neiInd] = delt_new.y;
    p->deltHisz[ci+neiInd] = delt_new.z;

    /* ========= for debug ==========*/
    /*
       double force_factor = p->mass_factor*p->length_factor/(p->time_factor*p->time_factor);
       double velocity_factor = p->length_factor/p->time_factor;
       printf("vt = %f %f %f\n",velocity_factor*c.vt.x,velocity_factor*c.vt.y,velocity_factor*c.vt.z);
       printf("vtmag = %f \n",velocity_factor*sqrt(vdot(c.vt,c.vt)));
       printf("fn = %f %f %f\n",force_factor*c.fn.x,force_factor*c.fn.y,force_factor*c.fn.z);
       printf("ft = %f %f %f\n",force_factor*ft.x,force_factor*ft.y,force_factor*ft.z);
       printf("\n");
     */
    /* ========= for debug ==========*/

}

__device__ __forceinline__
void d_update_history_wall(ParticleSys<DeviceMemory> *p,int i){

    int ci = i*MAX_NEI;

    for(int k=0; k<p->numContWall[i]; k++){
        if (p->isContactWall[ci+k]==0){ 
            /* particle contact lost */
            int last = p->numContWall[i]-1;
            p->indHisWall[ci+k] = p->indHisWall[ci+last];

            p->deltHisxWall[ci+k] = p->deltHisxWall[ci+last];
            p->deltHisyWall[ci+k] = p->deltHisyWall[ci+last];
            p->deltHiszWall[ci+k] = p->deltHiszWall[ci+last];

            p->deltHisxWall[ci+last]=0.;
            p->deltHisyWall[ci+last]=0.;
            p->deltHiszWall[ci+last]=0.;

            p->numContWall[i] -= 1;
            k -= 1; /* check if swapped particle is in contact */
        }
    }
}

__device__ __forceinline__
void d_update_history(ParticleSys<DeviceMemory> *p,int i){

    int ci = i*MAX_NEI;
    for(int k=0; k<p->numCont[i]; k++){
        if (p->isContact[ci+k]==0){ 
            /* particle contact lost */
            int last = p->numCont[i]-1;

            p->indHis[ci+k] = p->indHis[ci+last];

            p->deltHisx[ci+k] = p->deltHisx[ci+last];
            p->deltHisy[ci+k] = p->deltHisy[ci+last];
            p->deltHisz[ci+k] = p->deltHisz[ci+last];

            p->deltHisx[ci+last]=0.;
            p->deltHisy[ci+last]=0.;
            p->deltHisz[ci+last]=0.;

            p->numCont[i] -= 1;
            k -= 1; /* check if swapped particle is in contact */
        }
    }
}

__device__ __forceinline__
ContactCache d_calc_normal_force(ParticleSys<DeviceMemory>* p,int i,int j,Vec3 n,double delMag){

    int bi = i*DIM;
    int bj = j*DIM;

    /* get deltas */
    Vec3 del;
    del = vscalar(-delMag,n); // negated for the convention

    /* get relative velocity */
    Vec3 v_rel;
    v_rel.x = p->v[bi+0] - p->v[bj+0];
    v_rel.y = p->v[bi+1] - p->v[bj+1];
    v_rel.z = p->v[bi+2] - p->v[bj+2];

    double v_reldotn = vdot(v_rel,n);

    /* get relative normal velocity*/
    ContactCache result;
    result.vn_rel = vscalar(v_reldotn,n);


    double m_eff = 1./(p->invm[i]+p->invm[j]);
    double eta = p->etaconst[i]*sqrt(m_eff);
    result.eta = eta;

    double k = p->k[i];
    double ri = p->r[i];
    double rj = p->r[j];

    result.fn.x = -k*del.x - eta*result.vn_rel.x;
    result.fn.y = -k*del.y - eta*result.vn_rel.y;
    result.fn.z = -k*del.z - eta*result.vn_rel.z;

    /* calculate relative tangential velocity */
    Vec3 vrot;
    vrot.x = ri*p->angv[bi+0]+rj*p->angv[bj+0];
    vrot.y = ri*p->angv[bi+1]+rj*p->angv[bj+1];
    vrot.z = ri*p->angv[bi+2]+rj*p->angv[bj+2];
    vrot = vcross(vrot,n);

    Vec3 vt;
    vt=vsub(v_rel,result.vn_rel);
    result.vt=vadd(vt,vrot);

    /* add force */

    p->f[bi+0] += result.fn.x;
    p->f[bi+1] += result.fn.y;
    p->f[bi+2] += result.fn.z;

    result.n = n;

    return result;
}


__device__ __forceinline__
void d_particle_collision_cell_linked(ParticleSys<DeviceMemory>* p, int i, DeviceBoundingBox* box){
    //particle-particle

    /* cycle through neighbor cells */
    int bi = i*DIM;
    int x=p->cellx[bi+0];
    int y=p->cellx[bi+1];
    int z=p->cellx[bi+2];

    for (int sx=-1; sx<=1; sx++){
        for (int sy=-1; sy<=1; sy++){
            for (int sz=-1; sz<=1; sz++){
                int cellId = (box->sizey*(z+sz)+y+sy)*box->sizex+x+sx;


                int start = box->pStart[cellId];
                int end = start+box->pNum[cellId];
                for (int k=start; k<end; k++){
                    int j = box->pList[k];
                    if (i==j){
                        continue;
                    }else{
                        int bj = j*DIM;

                        Vec3 del;
                        /* normal points toward particle i */
                        del.x = p->x[bi+0]- p->x[bj+0];
                        del.y = p->x[bi+1]- p->x[bj+1];
                        del.z = p->x[bi+2]- p->x[bj+2];
                        double distsq = vdot(del,del);
                        double R = p->r[i]+p->r[j];

                        if (distsq<R*R){
                            double dist = sqrt(distsq);
                            double delMag = R-dist;
                            if(delMag*p->invr[i]*0.5>0.05){
                                printf("overlap over 5%%!!!!\n");
                                printf("delta is %f %%, dist is %f\n",delMag*p->invr[i]+0.5*100.,dist);
                            }

                            /* ======================================================
                               Force Calculation
                               ======================================================*/
                            /* get normal direction */
                            Vec3 n;
                            n.x = del.x/dist;
                            n.y = del.y/dist;
                            n.z = del.z/dist;
                            ContactCache c;
                            c = d_calc_normal_force(p,i,j,n,delMag);


                            d_calc_tangential_force(p,i,j,c);

                        }
                    }
                }
            }
        }
    }/* neighbor cell search done */

    /* update contact history */
    d_update_history(p,i);
}

__device__ __forceinline__ void d_particle_collision_verlet(ParticleSys<DeviceMemory>* p, int i ,DeviceBoundingBox *box){
    if(p->isActive[i]!=1){
        return;
    }
    //particle-particle
    /* cycle through neighbor list */

    int bi=i*DIM;
    int end = p->numNei[i];
    int ci = i*MAX_NEI;

    double xi=p->x[bi+0];
    double yi=p->x[bi+1];
    double zi=p->x[bi+2];
    double ri = p->r[i];
    double riInv = p->invr[i];


    for (int k=0; k<end; k++){

        int j = p->neiList[ci+k];



        if(p->isActive[j]!=1){
            continue;
        }

        int bj=j*DIM;

        Vec3 del;
        /* normal points toward particle i */
        del.x = xi- p->x[bj+0];
        del.y = yi- p->x[bj+1];
        del.z = zi- p->x[bj+2];
        double distsq = vdot(del,del);
        double R = ri+p->r[j];

        if (distsq<R*R){
            double dist = sqrt(distsq);
            double delMag = R-dist;
            if (delMag*riInv*0.5>0.05){
                printf("overlap over 5%% with pair %d %d!!!!\n",i,j);
                printf("overlap is %f %% with pair %d %d\n",delMag*p->invr[i]*0.5*100.,i,j);
            }

            /* ======================================================
               Force Calculation
               ======================================================*/
            /* get normal direction */
            Vec3 n;
            n.x = del.x/dist;
            n.y = del.y/dist;
            n.z = del.z/dist;
            ContactCache c;
            c = d_calc_normal_force(p,i,j,n,delMag);


            d_calc_tangential_force(p,i,j,c);

        }
    }/* neighbor search done */

    /* update contact history */
    d_update_history(p,i);
}

__device__ __forceinline__
void updateAcceleration(ParticleSys<DeviceMemory>* p, int i){
    int bi = i*DIM;


    p->a[bi+0] = (p->g[0]+p->f[bi+0]*p->invm[i]);
    p->a[bi+1] = (p->g[1]+p->f[bi+1]*p->invm[i]);
    p->a[bi+2] = (p->g[2]+p->f[bi+2]*p->invm[i]);
}

/* 速度更新（重力適用） */
__device__ __forceinline__
void updateVelocity(ParticleSys<DeviceMemory>* p,int i){
    int bi = i*DIM;
    p->v[bi+0] += p->a[bi+0] * p->dt;
    p->v[bi+1] += p->a[bi+1] * p->dt;
    p->v[bi+2] += p->a[bi+2] * p->dt;
}


/* 位置更新（オイラー法） */
__device__ __forceinline__
void updatePosition(ParticleSys<DeviceMemory>* p,int i){
    int bi = i*DIM;
    p->x[bi+0] += p->v[bi+0] * p->dt;
    p->x[bi+1] += p->v[bi+1] * p->dt;
    p->x[bi+2] += p->v[bi+2] * p->dt;
}




/*
   ============================================================
   カーネル
   ============================================================
 */
__global__ void check_g_kernel(ParticleSys<DeviceMemory>* ps,DeviceTriangleMesh *mesh){
    printf("N in kernel %d\n",ps->N);
    printf("g in kernel: %f %f %f\n",ps->g[0],ps->g[1],ps->g[2]);
    printf("mesh in kernel: %f %f %f\n",mesh->mx[0],mesh->my[0],mesh->mz[0]);
    printf("mesh in kernel: %d\n",mesh->nVert);
    /*
       printf("wall 0 in kernel: %f %f %f\n",ps->walls.n[0],ps->walls.n[1],ps->walls.n[2]);
       printf("wall 1 in kernel: %f %f %f\n",ps->walls.n[1*DIM+0],ps->walls.n[1*DIM+1],ps->walls.n[1*DIM+2]);
     */
    printf("value in kernel: %f %f %f\n",ps->r[0],ps->m[0],ps->invm[0]);
    printf("value in kernel: dt = %f, etaconst = %f\n",ps->dt,ps->etaconst[0]);
}

__global__ void k_collision_verlet_verlet(ParticleSys<DeviceMemory>* p, DeviceBoundingBox* box,DeviceTriangleMesh* mesh){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p->N || p->isActive[i]!=1) return;

    d_particle_collision_verlet(p,i,box);
    //d_particle_collision_naive(p,i);
    // d_particle_collision_cell_linked_noVec3(p,i,box);
    d_wall_collision_verlet(p,i,mesh);
}

__global__ void k_collision_verlet_triangle(ParticleSys<DeviceMemory>* p, DeviceBoundingBox* box,DeviceTriangleMesh* mesh){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p->N || p->isActive[i]!=1) return;

    d_particle_collision_verlet(p,i,box);
    //d_particle_collision_naive(p,i);
    // d_particle_collision_cell_linked_noVec3(p,i,box);
    d_wall_collision_triangles(p,i,box,mesh);

}
__global__ void k_collision_triangle(ParticleSys<DeviceMemory>* p, DeviceBoundingBox* box,DeviceTriangleMesh* mesh){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p->N || p->isActive[i]!=1) return;

    d_particle_collision_cell_linked(p,i,box);
    //d_particle_collision_naive(p,i);
    // d_particle_collision_cell_linked_noVec3(p,i,box);
    d_wall_collision_triangles(p,i,box,mesh);

}


__global__ void k_integrate(ParticleSys<DeviceMemory>* p){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p->N || p->isActive[i]!=1) return;

    updateAcceleration(p,i);
    // 処理A
    updateVelocity(p, i);

    // 処理B
    updatePosition(p, i);

    int bi = i*DIM;


    // angular acceleration
    /* ==== temporarly commented out debug ===== */
    p->anga[bi+0] = p->mom[bi+0]*p->invmoi[i];
    p->anga[bi+1] = p->mom[bi+1]*p->invmoi[i];
    p->anga[bi+2] = p->mom[bi+2]*p->invmoi[i];

    // angular velocity
    p->angv[bi+0] += p->anga[bi+0] * p->dt;
    p->angv[bi+1] += p->anga[bi+1] * p->dt;
    p->angv[bi+2] += p->anga[bi+2] * p->dt;


}

/* ============== verlet list related============  */

__global__ void k_shouldRefreshNeighborList(ParticleSys<DeviceMemory> *p, DeviceBoundingBox* box){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= p->N || p->isActive[i]!=1) return;

    double threshSq = box->refreshThreshSq;
    int bi = i*DIM;
    Vec3 del;

    del.x = p->x[bi+0]- p->refx[i];
    del.y = p->x[bi+1]- p->refy[i];
    del.z = p->x[bi+2]- p->refz[i];

    double distsq=vdot(del,del);
    if(distsq>threshSq){
        atomicOr(p->refreshVerletFlag,1);
    }
}


/* ============== check Out of Bounds ============  */
__global__ void dk_checkOoB(ParticleSys<DeviceMemory> *p,DeviceBoundingBox* box){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p->N || p->isActive[i]!=1) return;
    double x=p->x[i*DIM+0];
    double y=p->x[i*DIM+1];
    double z=p->x[i*DIM+2];
    if( x<box->minx || x>box->maxx ||y<box->miny || y>box->maxy || z<box->minz || z>box->maxz){
        p->isActive[i]=0;
    }       
}


/*
   ======================================================
   main routine 
   ======================================================
 */
void device_dem_verlet_verlet_withSort(ParticleSys<DeviceMemory> *p,ParticleSys<DeviceMemory> *tmpPs, BoundingBox *box,TriangleMesh *mesh, BVH *bvh, int gridSize, int blockSize){

    /* initialize force */
    cudaMemset(p->f, 0, sizeof(double)*DIM*p->N);
    cudaMemset(p->mom, 0, sizeof(double)*DIM*p->N);
    cudaMemset(p->isContact, 0, sizeof(int)*p->N*MAX_NEI);
    cudaMemset(p->isContactWall, 0, sizeof(int)*p->N*MAX_NEI);

    k_collision_verlet_verlet<<<gridSize,blockSize>>>(p->d_self,box->d_boxPtr,mesh->d_meshPtr);
    k_integrate<<<gridSize, blockSize>>>(p->d_self);

    dk_checkOoB<<<gridSize, blockSize>>>(p->d_self,box->d_boxPtr);

    /* ==  check if refresh of verlet list required == */

    cudaMemset(p->refreshVerletFlag, 0, sizeof(int));
    k_shouldRefreshNeighborList<<<gridSize, blockSize>>>(p->d_self,box->d_boxPtr);

    int shouldRefreshVerletFlag = 0;
    cudaMemcpy(&shouldRefreshVerletFlag,p->refreshVerletFlag,sizeof(int), cudaMemcpyDeviceToHost);

    /* == refresh neighborlist if needed == */

    if (shouldRefreshVerletFlag==1){
        printf("sorting\n");
        d_update_pList_withSort(p,tmpPs,box,gridSize, blockSize);


        k_update_neighborlist<<<gridSize, blockSize>>>(p->d_self,box->d_boxPtr);
        k_update_neighborlist_wall<<<gridSize, blockSize>>>(p->d_self,mesh->d_meshPtr,bvh->d_bvhPtr, box->skinR);
    }

    /*
       cudaError_t err = cudaGetLastError();
       printf("CUDA error = %s\n", cudaGetErrorString(err));
     */
    //    cudaDeviceSynchronize();

}

void device_dem_naive(ParticleSys<DeviceMemory> *p,BoundingBox *box, TriangleMesh *mesh, BVH *bvh, int gridSize, int blockSize){

    /* initialize force */
    cudaMemset(p->f, 0, sizeof(double)*DIM*p->N);
    cudaMemset(p->mom, 0, sizeof(double)*DIM*p->N);
    cudaMemset(p->isContact, 0, sizeof(int)*p->N*MAX_NEI);
    cudaMemset(p->isContactWall, 0, sizeof(int)*p->N*MAX_NEI);

    k_particle_collision_naive<<<gridSize,blockSize>>>(p->d_self);
    k_wall_collision_triangles_naive<<<gridSize,blockSize>>>(p->d_self,mesh->d_meshPtr);
    k_integrate<<<gridSize, blockSize>>>(p->d_self);

    dk_checkOoB<<<gridSize, blockSize>>>(p->d_self,box->d_boxPtr);


    /*
       cudaError_t err = cudaGetLastError();
       printf("CUDA error = %s\n", cudaGetErrorString(err));
     */
    //    cudaDeviceSynchronize();

}
void device_dem_verlet_verlet(ParticleSys<DeviceMemory> *p, BoundingBox *box,TriangleMesh *mesh, BVH *bvh, int gridSize, int blockSize){

    /* initialize force */
    cudaMemset(p->f, 0, sizeof(double)*DIM*p->N);
    cudaMemset(p->mom, 0, sizeof(double)*DIM*p->N);
    cudaMemset(p->isContact, 0, sizeof(int)*p->N*MAX_NEI);
    cudaMemset(p->isContactWall, 0, sizeof(int)*p->N*MAX_NEI);

    k_collision_verlet_verlet<<<gridSize,blockSize>>>(p->d_self,box->d_boxPtr,mesh->d_meshPtr);
    k_integrate<<<gridSize, blockSize>>>(p->d_self);

    dk_checkOoB<<<gridSize, blockSize>>>(p->d_self,box->d_boxPtr);

    /* ==  check if refresh of verlet list required == */

    cudaMemset(p->refreshVerletFlag, 0, sizeof(int));
    k_shouldRefreshNeighborList<<<gridSize, blockSize>>>(p->d_self,box->d_boxPtr);

    int shouldRefreshVerletFlag = 0;
    cudaMemcpy(&shouldRefreshVerletFlag,p->refreshVerletFlag,sizeof(int), cudaMemcpyDeviceToHost);

    /* == refresh neighborlist if needed == */

    if (shouldRefreshVerletFlag==1){
        /* for debug */
        printf("refreshing\n");
        d_update_pList(p,box,gridSize, blockSize);

        printf("sort neighborlist\n");
        k_update_neighborlist_endsort<<<gridSize, blockSize>>>(p->d_self,box->d_boxPtr);
        printf("sort done\n");
        k_update_neighborlist_wall<<<gridSize, blockSize>>>(p->d_self,mesh->d_meshPtr,bvh->d_bvhPtr, box->skinR);
        printf("update neighborlist\n");
    }

    /*
       cudaError_t err = cudaGetLastError();
       printf("CUDA error = %s\n", cudaGetErrorString(err));
     */
    //    cudaDeviceSynchronize();

}
void device_dem_verlet_triangles(ParticleSys<DeviceMemory> *p, BoundingBox *box,TriangleMesh *mesh, int gridSize, int blockSize){

    /* initialize force */
    cudaMemset(p->f, 0, sizeof(double)*DIM*p->N);
    cudaMemset(p->mom, 0, sizeof(double)*DIM*p->N);
    cudaMemset(p->isContact, 0, sizeof(int)*p->N*MAX_NEI);
    cudaMemset(p->isContactWall, 0, sizeof(int)*p->N*MAX_NEI);

    k_collision_verlet_triangle<<<gridSize,blockSize>>>(p->d_self,box->d_boxPtr,mesh->d_meshPtr);
    k_integrate<<<gridSize, blockSize>>>(p->d_self);
    dk_checkOoB<<<gridSize, blockSize>>>(p->d_self,box->d_boxPtr);

    /* ==  check if refresh of verlet list required == */

    cudaMemset(p->refreshVerletFlag, 0, sizeof(int));
    k_shouldRefreshNeighborList<<<gridSize, blockSize>>>(p->d_self,box->d_boxPtr);

    int shouldRefreshVerletFlag = 0;
    cudaMemcpy(&shouldRefreshVerletFlag,p->refreshVerletFlag,sizeof(int), cudaMemcpyDeviceToHost);
    /* == refresh neighborlist if needed == */

    if (shouldRefreshVerletFlag ==1){
        d_update_pList(p,box,gridSize, blockSize);
        k_update_neighborlist<<<gridSize, blockSize>>>(p->d_self,box->d_boxPtr);
    }

    /*
       cudaError_t err = cudaGetLastError();
       printf("CUDA error = %s\n", cudaGetErrorString(err));
     */
    //    cudaDeviceSynchronize();

}

void device_dem_triangles(ParticleSys<DeviceMemory> *p, BoundingBox *box,TriangleMesh *mesh, int gridSize, int blockSize){


    /* initialize force */
    cudaMemset(p->f, 0, sizeof(double)*DIM*p->N);
    cudaMemset(p->mom, 0, sizeof(double)*DIM*p->N);
    cudaMemset(p->isContact, 0, sizeof(int)*p->N*MAX_NEI);
    cudaMemset(p->isContactWall, 0, sizeof(int)*p->N*MAX_NEI);

    d_update_pList(p,box,gridSize, blockSize);
    k_collision_triangle<<<gridSize,blockSize>>>(p->d_self,box->d_boxPtr,mesh->d_meshPtr);
    k_integrate<<<gridSize, blockSize>>>(p->d_self);
    dk_checkOoB<<<gridSize, blockSize>>>(p->d_self,box->d_boxPtr);

    /*
       cudaError_t err = cudaGetLastError();
       printf("CUDA error = %s\n", cudaGetErrorString(err));
     */
    //    cudaDeviceSynchronize();

}

