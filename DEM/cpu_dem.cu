#include "cpu_dem.h"
#include "omp.h"

/*
============================================================
cpu functions
============================================================
*/

/* === triangle related === */

TriangleContactCache dist_triangle(ParticleSys<HostMemory>* ps, int i, TriangleMesh* mesh, int j){
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

static inline int sphereAABBOverlap(ParticleSys<HostMemory>* p,int i,BVH *bvh, int j){
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

    return (dx*dx + dy*dy + dz*dz) <= p->rsq[i];
}

void wall_collision_verlet(ParticleSys<HostMemory>* p,TriangleMesh* mesh){

    #pragma omp parallel for
    for(int i=0; i<p->N; i++){

        if(p->isActive[i]!=1){
            continue;
        }

        int end = p->numNeiWall[i];
        int ci = i*MAX_NEI;

        int numContVorENow = 0;

        for (int k=0; k<end; k++){
            int indTri = p->neiListWall[ci+k];


            // ここで球 vs 三角形の精密判定
            TriangleContactCache tc;
            tc = dist_triangle(p,i,mesh,indTri); 

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
                c = calc_normal_force_wall(p,i,indTri,tc.n,delmag);

                calc_tangential_force_wall(p,i,indTri,c);
            }
        }
        update_history_wall(p,i);
    }
}

void wall_collision_BVH(ParticleSys<HostMemory>* p,TriangleMesh* mesh,BVH* bvh){
    for(int i=0; i<p->N; i++){

        if(p->isActive[i]!=1){
            continue;
        }

        int stack[128];
        int sp = 0;

        stack[sp++] = 0;   // root

        int numContVorENow = 0;

        while(sp > 0){
            int node = stack[--sp];

            if(!sphereAABBOverlap(p,i,bvh,node)){
                continue;
            }

            if(bvh->tri[node] >= 0){
                int indTri = bvh->tri[node];

                // ここで球 vs 三角形の精密判定
                TriangleContactCache tc;
                tc = dist_triangle(p,i,mesh,indTri); 

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
                    c = calc_normal_force_wall(p,i,indTri,tc.n,delmag);
                    calc_tangential_force_wall(p,i,indTri,c);
                }
            }else{
                stack[sp++] = bvh->left[node];
                stack[sp++] = bvh->right[node];
            }
        }
        update_history_wall(p,i);
    }
}


void wall_collision_triangles_naive(ParticleSys<HostMemory>* p,TriangleMesh* mesh){
    #pragma omp parallel for
    for (int i=0; i<p->N; i++){
        if(p->isActive[i]!=1){
            continue;
        }
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
            tc = dist_triangle(p,i,mesh,indTri); 

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
                c = calc_normal_force_wall(p,i,indTri,tc.n,delmag);

                calc_tangential_force_wall(p,i,indTri,c);

                int numWall = numContWallNow;
                p->indHisWallNow[i*MAX_NEI+numWall] = indTri;
                numContWallNow+=1;
            }
        }
        update_history_wall(p,i);
    }
}

void wall_collision_triangles(ParticleSys<HostMemory>* p,BoundingBox *box, TriangleMesh* mesh){

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
                        tc = dist_triangle(p,i,mesh,indTri); 

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
                            c = calc_normal_force_wall(p,i,indTri,tc.n,delmag);
                            calc_tangential_force_wall(p,i,indTri,c);
                            int numWall = numContWallNow;
                            p->indHisWallNow[i*MAX_NEI+numWall] = indTri;
                            numContWallNow+=1;
                        }

                    }
                }
            }
        }
        update_history_wall(p,i);

    }
}

inline ContactCache calc_normal_force_wall(ParticleSys<HostMemory> *p,int i,int j,Vec3 n,double delMag){

    int bi = i*DIM;

    /* get deltas */
    Vec3 del;
    del = vscalar(-delMag,n); //negated for the convention

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

    result.fn.x = -p->k[i]*del.x - eta*result.vn_rel.x;
    result.fn.y = -p->k[i]*del.y - eta*result.vn_rel.y;
    result.fn.z = -p->k[i]*del.z - eta*result.vn_rel.z;


    /* calculate relative tangential velocity */
    Vec3 vrot;
    vrot.x = p->r[i]*p->angv[bi+0];
    vrot.y = p->r[i]*p->angv[bi+1];
    vrot.z = p->r[i]*p->angv[bi+2];
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

inline void update_history(ParticleSys<HostMemory> *p,int i){

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

inline void update_history_wall(ParticleSys<HostMemory> *p,int i){

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

inline void calc_tangential_force_wall(ParticleSys<HostMemory> *p,int i,int j,ContactCache c){

    int bi = i*DIM;
    int ci = i*MAX_NEI;

    /* check history */

    int isInHis = 0;
    int neiInd=0;

    for (int k=0; k<p->numContWall[i]; k++){
        if (j == p->indHisWall[ci+k]){ /* if the contact wall is in the history */
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
        if(p->numContWall[i] > MAX_NEI){
            printf("OH MY OH MY\n");
        }
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


    if(ftsq>(p->mu*p->mu)*fnsq){ // slip //
        if (ftsq!=0.){
            Vec3 t;
            t = vscalar(1./(sqrt(ftsq)+SMALL_NUM),ft); 


            double fnnorm = sqrt(fnsq);
            ft = vscalar(p->mu*fnnorm,t);
            delt_new = vscalar(-1./(p->k[i]),ft);
        }else{
            ft.x =0.;
            ft.y =0.;
            ft.z =0.;
        }
    }else{
        // add damping //
        ft.x -= c.eta*c.vt.x;
        ft.y -= c.eta*c.vt.y;
        ft.z -= c.eta*c.vt.z;
    }

    /* === debug ===*/
    /*
       if (vdot(ft,c.vt)>0.){
       printf("Pt=%e\n", vdot(ft, c.vt));
       }
     */

    /* == debug == */
    /*
       if (p->pId[i] == 0) {
       printf("forWall\n");
       double Pt = vdot(ft, c.vt);
       double fnorm = sqrt(vdot(c.fn, c.fn));
       double ft_norm = sqrt(vdot(ft, ft));
       double delt_new_norm = sqrt(vdot(delt_new,delt_new));
       double Ft_spring_mag = sqrt(vdot(vscalar(p->k[i],delt_new),vscalar(p->k[i],delt_new)));
       double Ft_damp_mag = sqrt(vdot(vscalar(c.eta,c.vt),vscalar(c.eta,c.vt)));
       Vec3 vt_normalize = vnormalize(c.vt);
       Vec3 ft_normalize = vnormalize(ft);
       printf("step %d Pt=%e delt_mag=%e |ft|=%e mu|fn|=%e\n",
       p->steps, Pt,delt_new_norm,ft_norm, p->mu * fnorm);
       printf("deltx = %e, delty = %e, deltz = %e\n", delt_new.x,delt_new.y, delt_new.z);
       printf("Ft_damp_mag = %e, Ft_spring = %e\n",Ft_damp_mag, Ft_spring_mag);
       printf("Fn = %e\n", sqrt(vdot(c.fn,c.fn)));
       printf("vt.x = %e vt.y = %e vt.z = %e\n",c.vt.x,c.vt.y,c.vt.z);
       printf("Ftnorm dot vtnorm %e\n",vdot(ft_normalize,vt_normalize));
       printf("vt dot n = %e\n", vdot(c.vt, c.n));
       Vec3 t_prev = vnormalize(delt_new);
       Vec3 t_now  = vnormalize(c.vt);
       printf("|Ft| / (mu*|Fn|) = %e\n", ft_norm / (p->mu * fnorm));
       printf("delt_new_norm dot vt_norm %e\n",vdot(t_prev,t_now));
       printf("angle delt_new vt = %e\n", acos(vdot(t_prev, t_now)));
       double align = vdot(vnormalize(delt_new), vnormalize(c.vt));
       printf("align = %e\n", align);
       printf("i=%d j=%d slot=%d \n\n",
       p->pId[i], j, neiInd);
       }
     */

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
    p->deltHisxWall[ci+neiInd] = delt_new.x;
    p->deltHisyWall[ci+neiInd] = delt_new.y;
    p->deltHiszWall[ci+neiInd] = delt_new.z;
}


inline void calc_tangential_force(ParticleSys<HostMemory> *p,int i,int j,ContactCache c){

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
            delt_new = vscalar(-1./(p->k[i]),ft);


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

    /* == debug === */
    /*
       double n2 = vdot(c.n,c.n);
       printf("at step %d, n2=%e ", p->steps,n2);
       printf("delt·n = %e\n", vdot(delt_old, c.n));
       printf("contact: i=%d j=%d delt=%e %e %e\n", i, j, delt_new.x, delt_new.y, delt_new.z);
       printf("deltnew·n = %e\n", vdot(delt_new, c.n));
       printf("\n");
     */

    /* == debug == */
    /*
       if (p->pId[i] == 0 && p->pId[j]== 1) {
       double Pt = vdot(ft, c.vt);
       double fnorm = sqrt(vdot(c.fn, c.fn));
       double ft_norm = sqrt(vdot(ft, ft));
       double delt_new_norm = sqrt(vdot(delt_new,delt_new));
       double Ft_spring_mag = sqrt(vdot(vscalar(p->k[i],delt_new),vscalar(p->k[i],delt_new)));
       double Ft_damp_mag = sqrt(vdot(vscalar(c.eta,c.vt),vscalar(c.eta,c.vt)));
       Vec3 vt_normalize = vnormalize(c.vt);
       Vec3 ft_normalize = vnormalize(ft);
       printf("step %d Pt=%e delt_mag=%e |ft|=%e mu|fn|=%e\n",
       p->steps, Pt,delt_new_norm,ft_norm, p->mu * fnorm);
       printf("deltx = %e, delty = %e, deltz = %e\n", delt_new.x,delt_new.y, delt_new.z);
       printf("Ft_damp_mag = %e, Ft_spring = %e\n",Ft_damp_mag, Ft_spring_mag);
       printf("Fn = %e\n", sqrt(vdot(c.fn,c.fn)));
       printf("vt.x = %e vt.y = %e vt.z = %e\n",c.vt.x,c.vt.y,c.vt.z);
       printf("Ftnorm dot vtnorm %e\n",vdot(ft_normalize,vt_normalize));
       printf("vt dot n = %e\n", vdot(c.vt, c.n));
       Vec3 t_prev = vnormalize(delt_new);
       Vec3 t_now  = vnormalize(c.vt);
       printf("|Ft| / (mu*|Fn|) = %e\n", ft_norm / (p->mu * fnorm));
       printf("delt_new_norm dot vt_norm %e\n",vdot(t_prev,t_now));
       printf("angle delt_new vt = %e\n", acos(vdot(t_prev, t_now)));
       double align = vdot(vnormalize(delt_new), vnormalize(c.vt));
       printf("align = %e\n", align);
       printf("i=%d j=%d slot=%d \n\n",
       p->pId[i], pId, neiInd);
       }
     */

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

inline ContactCache calc_normal_force(ParticleSys<HostMemory> *p,int i,int j,Vec3 n,double delMag){

    int bi = i*DIM;
    int bj = j*DIM;

    /* get deltas */
    Vec3 del;
    del = vscalar(-delMag,n); //negated for the convention

    /* get relative velocity */
    Vec3 v_rel;
    v_rel.x = p->v[bi+0] - p->v[bj+0];
    v_rel.y = p->v[bi+1] - p->v[bj+1];
    v_rel.z = p->v[bi+2] - p->v[bj+2];

    double v_reldotn = vdot(v_rel,n);

    /* get normal relative velocity*/
    ContactCache result;
    result.vn_rel = vscalar(v_reldotn,n);


    double m_eff = 1./(p->invm[i]+p->invm[j]);
    double eta = p->etaconst[i]*sqrt(m_eff);

    /* == debug == */
    //printf("eta=%e\n",eta);


    result.eta = eta;

    result.fn.x = -p->k[i]*del.x - eta*result.vn_rel.x;
    result.fn.y = -p->k[i]*del.y - eta*result.vn_rel.y;
    result.fn.z = -p->k[i]*del.z - eta*result.vn_rel.z;

    /* calculate relative tangential velocity */


    Vec3 vrot;
    vrot.x = (p->r[i])*p->angv[bi+0]+(p->r[j])*p->angv[bj+0];
    vrot.y = (p->r[i])*p->angv[bi+1]+(p->r[j])*p->angv[bj+1];
    vrot.z = (p->r[i])*p->angv[bi+2]+(p->r[j])*p->angv[bj+2];
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

void particle_collision_cell_linked_withSort_fastUpdate(ParticleSys<HostMemory>* p,ParticleSys<HostMemory>* tmpPs, BoundingBox *box){
    update_pList_withSort_fast(p, tmpPs, box);
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

        for (int sx=-1; sx<=1; sx++){
            for (int sy=-1; sy<=1; sy++){
                for (int sz=-1; sz<=1; sz++){
                    int cellId = (box->sizey*(z+sz)+y+sy)*box->sizex+x+sx;
                    int start = box->pStart[cellId];
                    int end = start+box->pNum[cellId];
                    for (int k=box->pStart[cellId]; k<end; k++){
                        int j = box->pList[k];
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
                                c = calc_normal_force(p,i,j,n,delMag);


                                calc_tangential_force(p,i,j,c);

                            }
                        }
                    }
                }
            }
        }/* neighbor cell search done */

        /* update contact history */
        update_history(p,i);

    }
}

void particle_collision_verlet(ParticleSys<HostMemory>* p, BoundingBox *box){

    #pragma omp parallel for
    for (int i=0; i<p->N; i++){
        if(p->isActive[i]!=1){
            continue;
        }
        //particle-particle
        /* cycle through neighbor list */

        int bi=i*DIM;
        int end = p->numNei[i];
        int ci = i*MAX_NEI;
        for (int k=0; k<end; k++){
            int j = p->neiList[ci+k];
            if (p->isActive[j] != 1){
                continue;
            }
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
                c = calc_normal_force(p,i,j,n,delMag);


                calc_tangential_force(p,i,j,c);

            }
        }/* neighbor search done */

        /* update contact history */
        update_history(p,i);
    }

    /* === debug == */
    /*
    for (int i=0; i<p->N; i++){
        if(p->isActive[i]!=1) continue;

        int bi = i*DIM;

        for (int j=0; j<p->N; j++){
            if(i==j || p->isActive[j]!=1) continue;

            int bj = j*DIM;

            Vec3 del;
            del.x = p->x[bi+0]- p->x[bj+0];
            del.y = p->x[bi+1]- p->x[bj+1];
            del.z = p->x[bi+2]- p->x[bj+2];

            double distsq = vdot(del,del);
            double R = p->r[i]+p->r[j];

            bool trueNeighbor = (distsq < R*R);

            // リストに存在するか
            bool inList = false;
            for(int k=0;k<p->numNei[i];k++){
                if(p->neiList[i*MAX_NEI+k]==j){
                    inList = true;
                    break;
                }
            }

            if(trueNeighbor && !inList){
                printf("MISS: i=%d j=%d\n", i, j);
            }
            if(!trueNeighbor && inList){
                printf("FALSE: i=%d j=%d\n", i, j);
            }
        }
    }
    */
    /* === debug done == */
}

void particle_collision_cell_linked_withSort(ParticleSys<HostMemory>* p,ParticleSys<HostMemory>* tmpPs, BoundingBox *box){
    update_pList_withSort(p, tmpPs, box);
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

        for (int sx=-1; sx<=1; sx++){
            for (int sy=-1; sy<=1; sy++){
                for (int sz=-1; sz<=1; sz++){
                    int cellId = (box->sizey*(z+sz)+y+sy)*box->sizex+x+sx;
                    int start = box->pStart[cellId];
                    int end = start+box->pNum[cellId];
                    for (int k=box->pStart[cellId]; k<end; k++){
                        int j = box->pList[k];
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
                                c = calc_normal_force(p,i,j,n,delMag);


                                calc_tangential_force(p,i,j,c);

                            }
                        }
                    }
                }
            }
        }/* neighbor cell search done */

        /* update contact history */
        update_history(p,i);

    }
}

void particle_collision_cell_linked_fastUpdate(ParticleSys<HostMemory>* p, BoundingBox *box){
    //update_pList(p,box);
    update_pList_fast(p,box);

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

        for (int sx=-1; sx<=1; sx++){
            for (int sy=-1; sy<=1; sy++){
                for (int sz=-1; sz<=1; sz++){
                    int cellId = (box->sizey*(z+sz)+y+sy)*box->sizex+x+sx;

                    int start = box->pStart[cellId];
                    int end = start+box->pNum[cellId];
                    for (int k=box->pStart[cellId]; k<end; k++){
                        int j = box->pList[k];
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
                                c = calc_normal_force(p,i,j,n,delMag);


                                calc_tangential_force(p,i,j,c);

                            }
                        }
                    }
                }
            }
        }/* neighbor cell search done */

        /* update contact history */
        update_history(p,i);

    }
}

void particle_collision_cell_linked(ParticleSys<HostMemory>* p, BoundingBox *box){
    update_pList(p,box);

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

        for (int sx=-1; sx<=1; sx++){
            for (int sy=-1; sy<=1; sy++){
                for (int sz=-1; sz<=1; sz++){
                    int cellId = (box->sizey*(z+sz)+y+sy)*box->sizex+x+sx;

                    int start = box->pStart[cellId];
                    int end = start+box->pNum[cellId];
                    for (int k=box->pStart[cellId]; k<end; k++){
                        int j = box->pList[k];
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
                                    printf("overlap over %%!!!!\n");
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
                                c = calc_normal_force(p,i,j,n,delMag);


                                calc_tangential_force(p,i,j,c);

                            }
                        }
                    }
                }
            }
        }/* neighbor cell search done */

        /* update contact history */
        update_history(p,i);

    }
}

void particle_collision_cell_linked_noVec3(ParticleSys<HostMemory>* ps, BoundingBox *box){
    update_pList(ps,box);

    for (int i=0; i<ps->N; i++){
        //particle-particle
        /* cycle through neighbor cells */
        int bi=i*DIM;
        int x=ps->cellx[bi+0];
        int y=ps->cellx[bi+1];
        int z=ps->cellx[bi+2];

        for (int sx=-1; sx<=1; sx++){
            for (int sy=-1; sy<=1; sy++){
                for (int sz=-1; sz<=1; sz++){
                    int cellId = (box->sizey*(z+sz)+y+sy)*box->sizex+x+sx;

                    int start = box->pStart[cellId];
                    int end = start+box->pNum[cellId];
                    for (int k=box->pStart[cellId]; k<end; k++){
                        int j = box->pList[k];
                        if (i==j){
                            continue;
                        }else{
                            int bj=j*DIM;
                            /* normal points toward particle i */
                            double dx = ps->x[bi+0]- ps->x[bj+0];
                            double dy = ps->x[bi+1]- ps->x[bj+1];
                            double dz = ps->x[bi+2]- ps->x[bj+2];
                            double distsq =dx*dx+dy*dy+dz*dz;
                            double R = ps->r[i]+ps->r[j];

                            if (distsq<R*R){
                                double dist = sqrt(distsq);
                                double delta = R-dist;
                                if (delta*ps->invr[i]*0.5>0.01){
                                    printf("overlap over 10%%!!!!\n");
                                    printf("delta is %f, dist is %f\n",delta,dist);
                                }
                                /* ======================================================
                                   Normal Force Calculation
                                   ======================================================*/
                                /* get normal direction */
                                double nx,ny,nz;
                                nx = dx/dist;
                                ny = dy/dist;
                                nz = dz/dist;

                                /* get deltas */
                                double delx,dely,delz;
                                delx = nx*delta;
                                dely = ny*delta;
                                delz = nz*delta;

                                /* get relative velocity */
                                double v_relx,v_rely,v_relz;
                                v_relx = ps->v[i*DIM+0] - ps->v[j*DIM+0];
                                v_rely = ps->v[i*DIM+1] - ps->v[j*DIM+1];
                                v_relz = ps->v[i*DIM+2] - ps->v[j*DIM+2];

                                double v_reldotn = v_relx*nx+v_rely*ny+v_relz*nz;

                                /* get normal relative velocity*/
                                double vn_relx,vn_rely,vn_relz;
                                vn_relx = v_reldotn*nx;
                                vn_rely = v_reldotn*ny;
                                vn_relz = v_reldotn*nz;

                                double m_eff = (ps->m[i]*ps->m[j])/(ps->m[i]+ps->m[j]);
                                double eta = ps->etaconst[i]*sqrt(m_eff);

                                ps->f[i*DIM+0]+= ps->k[i]*delx - eta*vn_relx;
                                ps->f[i*DIM+1] += ps->k[i]*dely - eta*vn_rely;
                                ps->f[i*DIM+2] += ps->k[i]*delz - eta*vn_relz;

                            }
                        }
                    }
                }
            }
        }
    }
}

void particle_collision_naive(ParticleSys<HostMemory>* p){
    for (int i=0; i<p->N; i++){
        if(p->isActive[i]!=1){
            continue;
        }
        //particle-particle
        int bi=i*DIM;
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
                    c = calc_normal_force(p,i,j,n,delMag);
                    calc_tangential_force(p,i,j,c);

                }
            }
        }
        update_history(p,i);
    }
}

/* =========== verlet list related =============== */

int shouldRefreshNeighborList(ParticleSys<HostMemory> *p, BoundingBox* box){
    int flag = 0;
    double threshSq = box->refreshThreshSq;


    for(int i=0; i<p->N; i++){
        if(p->isActive[i]!=1){
            continue;
        }
        int bi = i*DIM;
        Vec3 del;

        del.x = p->x[bi+0]- p->refx[i];
        del.y = p->x[bi+1]- p->refy[i];
        del.z = p->x[bi+2]- p->refz[i];

        double distsq=vdot(del,del);
        if(distsq>threshSq){
            flag=1;
            break;
        }
    }
    return flag;
}

double kineticEnergy(ParticleSys<HostMemory> *p){
    double kineticEne = 0.0;
    for (int i=0; i<p->N; i++){
        if(p->isActive[i]!=1){
            continue;
        }

        int bi=i*DIM;

        Vec3 v;
        v.x = p->v[bi+0];
        v.y = p->v[bi+1];
        v.z = p->v[bi+2];
        kineticEne += 0.5*vdot(v,v)+-1.*p->g[1]*p->x[bi+1];
    }

    return kineticEne;
}

/* ============== check Out of Bounds ============  */

void checkOoB(ParticleSys<HostMemory> *p, ParticleSys<HostMemory> *tmpP, BoundingBox* box){
    for(int i=0; i<p->N; i++){
        if(p->isActive[i]==0){
            continue;
        }
        double x=p->x[i*DIM+0];
        double y=p->x[i*DIM+1];
        double z=p->x[i*DIM+2];
        if( x<box->minx || x>box->maxx ||y<box->miny || y>box->maxy || z<box->minz || z>box->maxz){
            p->isActive[i]=0;
            continue;
        }       
    }

}

/* ============== dem mains ================= */

void cpu_dem_verlet_verlet(ParticleSys<HostMemory>* p, ParticleSys<HostMemory> *tmpP, BoundingBox* box,TriangleMesh *mesh,BVH *bvh, int step){


    #pragma omp parallel for
    for (int i=0; i<p->N*DIM; i++){
        p->f[i]=0.;
    }

    #pragma omp parallel for
    for (int i=0; i<p->N*DIM; i++){
        p->mom[i]=0.;
    }

    #pragma omp parallel for
    for (int i=0; i<p->N*MAX_NEI; i++){
        p->isContact[i]=0;
        p->isContactWall[i]=0;
    }



    particle_collision_verlet(p,box);


    wall_collision_verlet(p,mesh);
    //wall_collision_triangles_naive(p,mesh);

    /* update */

    #pragma omp parallel for
    for (int i = 0; i < p->N; i++){
        if (p->isActive[i]!=1){
            continue;
        }
        int bi = i*DIM;
        // acceleration
        p->a[bi+0] = p->f[bi+0]*p->invm[i]+p->g[0];
        p->a[bi+1] = p->f[bi+1]*p->invm[i]+p->g[1];
        p->a[bi+2] = p->f[bi+2]*p->invm[i]+p->g[2];

        // velocity
        p->v[bi+0] += p->a[bi+0] * p->dt;
        p->v[bi+1] += p->a[bi+1] * p->dt;
        p->v[bi+2] += p->a[bi+2] * p->dt;

        // position
        p->x[bi+0] += p->v[bi+0] * p->dt;
        p->x[bi+1] += p->v[bi+1] * p->dt;
        p->x[bi+2] += p->v[bi+2] * p->dt;

        // angular acceleration
        p->anga[bi+0] = p->mom[bi+0]*p->invmoi[i];
        p->anga[bi+1] = p->mom[bi+1]*p->invmoi[i];
        p->anga[bi+2] = p->mom[bi+2]*p->invmoi[i];

        // angular velocity
        p->angv[bi+0] += p->anga[bi+0] * p->dt;
        p->angv[bi+1] += p->anga[bi+1] * p->dt;
        p->angv[bi+2] += p->anga[bi+2] * p->dt;
    }

    int flag = shouldRefreshNeighborList(p,box);
    if (flag ==1){

        /* == debug == */
        //printf("bruteForce!\n");
        //update_neighborlist_brute(p,tmpP,box);

        update_neighborlist(p,tmpP,box);
        update_neighborlist_wall(p,mesh,bvh,box->skinR);
        //update_neighborlist_wall_nobvh(p,mesh,box,box->skinR);

    }


}

void cpu_dem_verlet_BVH(ParticleSys<HostMemory>* p, ParticleSys<HostMemory> *tmpP, BoundingBox* box,TriangleMesh *mesh,BVH *bvh, int step){
    /* initialize */
    for (int i=0; i<p->N*DIM; i++){
        p->f[i]=0.;
    }

    for (int i=0; i<p->N*DIM; i++){
        p->mom[i]=0.;
    }

    for (int i=0; i<p->N*MAX_NEI; i++){
        p->isContact[i]=0;
        p->isContactWall[i]=0;
    }



    particle_collision_verlet(p,box);

    wall_collision_BVH(p,mesh,bvh);

    /* update */
    for (int i = 0; i < p->N; i++)
    {
        if (p->isActive[i]!=1){
            continue;
        }
        int bi = i*DIM;
        // acceleration
        p->a[bi+0] = p->f[bi+0]*p->invm[i]+p->g[0];
        p->a[bi+1] = p->f[bi+1]*p->invm[i]+p->g[1];
        p->a[bi+2] = p->f[bi+2]*p->invm[i]+p->g[2];

        // velocity
        p->v[bi+0] += p->a[bi+0] * p->dt;
        p->v[bi+1] += p->a[bi+1] * p->dt;
        p->v[bi+2] += p->a[bi+2] * p->dt;

        // position
        p->x[bi+0] += p->v[bi+0] * p->dt;
        p->x[bi+1] += p->v[bi+1] * p->dt;
        p->x[bi+2] += p->v[bi+2] * p->dt;

        // angular acceleration
        p->anga[bi+0] = p->mom[bi+0]*p->invmoi[i];
        p->anga[bi+1] = p->mom[bi+1]*p->invmoi[i];
        p->anga[bi+2] = p->mom[bi+2]*p->invmoi[i];

        // angular velocity
        p->angv[bi+0] += p->anga[bi+0] * p->dt;
        p->angv[bi+1] += p->anga[bi+1] * p->dt;
        p->angv[bi+2] += p->anga[bi+2] * p->dt;
    }
    int flag = shouldRefreshNeighborList(p,box);
    if (flag ==1){
        update_neighborlist(p,tmpP,box);
    }
}

void cpu_dem_verlet_triangles(ParticleSys<HostMemory>* p, ParticleSys<HostMemory> *tmpP, BoundingBox* box,TriangleMesh *mesh, int step){
    /* initialize */
    for (int i=0; i<p->N*DIM; i++){
        p->f[i]=0.;
    }

    for (int i=0; i<p->N*DIM; i++){
        p->mom[i]=0.;
    }

    for (int i=0; i<p->N*MAX_NEI; i++){
        p->isContact[i]=0;
        p->isContactWall[i]=0;
    }



    particle_collision_verlet(p,box);

    wall_collision_triangles(p,box,mesh);

    /* update */
    for (int i = 0; i < p->N; i++)
    {
        if (p->isActive[i]!=1){
            continue;
        }
        int bi = i*DIM;
        // acceleration
        p->a[bi+0] = p->f[bi+0]*p->invm[i]+p->g[0];
        p->a[bi+1] = p->f[bi+1]*p->invm[i]+p->g[1];
        p->a[bi+2] = p->f[bi+2]*p->invm[i]+p->g[2];

        // velocity
        p->v[bi+0] += p->a[bi+0] * p->dt;
        p->v[bi+1] += p->a[bi+1] * p->dt;
        p->v[bi+2] += p->a[bi+2] * p->dt;

        // position
        p->x[bi+0] += p->v[bi+0] * p->dt;
        p->x[bi+1] += p->v[bi+1] * p->dt;
        p->x[bi+2] += p->v[bi+2] * p->dt;

        // angular acceleration
        p->anga[bi+0] = p->mom[bi+0]*p->invmoi[i];
        p->anga[bi+1] = p->mom[bi+1]*p->invmoi[i];
        p->anga[bi+2] = p->mom[bi+2]*p->invmoi[i];

        // angular velocity
        p->angv[bi+0] += p->anga[bi+0] * p->dt;
        p->angv[bi+1] += p->anga[bi+1] * p->dt;
        p->angv[bi+2] += p->anga[bi+2] * p->dt;
    }
    int flag = shouldRefreshNeighborList(p,box);
    if (flag ==1){
        update_neighborlist(p,tmpP,box);
    }
}

void cpu_dem_sort_triangles(ParticleSys<HostMemory>* ps, ParticleSys<HostMemory> *tmpPs, BoundingBox* box,TriangleMesh *mesh, int step){
    /* initialize */
    for (int i=0; i<ps->N*DIM; i++){
        ps->f[i]=0.;
    }

    for (int i=0; i<ps->N*DIM; i++){
        ps->mom[i]=0.;
    }

    for (int i=0; i<ps->N*MAX_NEI; i++){
        ps->isContact[i]=0;
        ps->isContactWall[i]=0;
    }


    int reorderFreq=100;

    if(step%(reorderFreq)==0){
        particle_collision_cell_linked_withSort_fastUpdate(ps,tmpPs,box);
    }else{
        particle_collision_cell_linked_fastUpdate(ps,box);
    }

    wall_collision_triangles(ps,box,mesh);

    /* update */
    for (int i = 0; i < ps->N; i++)
    {
        if (ps->isActive[i]!=1){
            continue;
        }
        int bi = i*DIM;
        // acceleration
        ps->a[bi+0] = ps->f[bi+0]*ps->invm[i]+ps->g[0];
        ps->a[bi+1] = ps->f[bi+1]*ps->invm[i]+ps->g[1];
        ps->a[bi+2] = ps->f[bi+2]*ps->invm[i]+ps->g[2];

        // velocity
        ps->v[bi+0] += ps->a[bi+0] * ps->dt;
        ps->v[bi+1] += ps->a[bi+1] * ps->dt;
        ps->v[bi+2] += ps->a[bi+2] * ps->dt;

        // position
        ps->x[bi+0] += ps->v[bi+0] * ps->dt;
        ps->x[bi+1] += ps->v[bi+1] * ps->dt;
        ps->x[bi+2] += ps->v[bi+2] * ps->dt;

        // angular acceleration
        ps->anga[bi+0] = ps->mom[bi+0]*ps->invmoi[i];
        ps->anga[bi+1] = ps->mom[bi+1]*ps->invmoi[i];
        ps->anga[bi+2] = ps->mom[bi+2]*ps->invmoi[i];

        // angular velocity
        ps->angv[bi+0] += ps->anga[bi+0] * ps->dt;
        ps->angv[bi+1] += ps->anga[bi+1] * ps->dt;
        ps->angv[bi+2] += ps->anga[bi+2] * ps->dt;

    }
}

void cpu_dem_naive_triangle(ParticleSys<HostMemory>* ps, BoundingBox* box, TriangleMesh *mesh){
    /* initialize */
    for (int i=0; i<ps->N*DIM; i++){
        ps->f[i]=0.;
    }

    for (int i=0; i<ps->N*DIM; i++){
        ps->mom[i]=0.;
    }

    for (int i=0; i<ps->N*MAX_NEI; i++){
        ps->isContact[i]=0;
        ps->isContactWall[i]=0;
    }



    particle_collision_naive(ps);

    wall_collision_triangles_naive(ps,mesh);

    /* update */
    for (int i = 0; i < ps->N; i++)
    {
        if (ps->isActive[i]!=1){
            continue;
        }
        int bi = i*DIM;
        // acceleration
        ps->a[bi+0] = ps->f[bi+0]*ps->invm[i]+ps->g[0];
        ps->a[bi+1] = ps->f[bi+1]*ps->invm[i]+ps->g[1];
        ps->a[bi+2] = ps->f[bi+2]*ps->invm[i]+ps->g[2];

        // velocity
        Vec3 tmp_v;
        tmp_v.x = ps->v[bi+0];
        tmp_v.y = ps->v[bi+1];
        tmp_v.z = ps->v[bi+2];

        /* === debug === */
        tmp_v.x  += ps->a[bi+0] * ps->dt;
        tmp_v.y  += ps->a[bi+1] * ps->dt;
        tmp_v.z  += ps->a[bi+2] * ps->dt;


        ps->v[bi+0] = tmp_v.x;
        ps->v[bi+1] = tmp_v.y;
        ps->v[bi+2] = tmp_v.z;


        // position
        ps->x[bi+0] += tmp_v.x * ps->dt;
        ps->x[bi+1] += tmp_v.y * ps->dt;
        ps->x[bi+2] += tmp_v.z * ps->dt;


        // angular acceleration
        ps->anga[bi+0] = ps->mom[bi+0]*ps->invmoi[i];
        ps->anga[bi+1] = ps->mom[bi+1]*ps->invmoi[i];
        ps->anga[bi+2] = ps->mom[bi+2]*ps->invmoi[i];

        // angular velocity
        ps->angv[bi+0] += ps->anga[bi+0] * ps->dt;
        ps->angv[bi+1] += ps->anga[bi+1] * ps->dt;
        ps->angv[bi+2] += ps->anga[bi+2] * ps->dt;

    }
    //printf("Energy %8.6e \n",kineticEnergy(ps));

    ps->steps +=1;

}

void cpu_dem_nosort_triangle(ParticleSys<HostMemory>* ps, ParticleSys<HostMemory> *tmpPs, BoundingBox* box,TriangleMesh *mesh){
    /* initialize */
    for (int i=0; i<ps->N*DIM; i++){
        ps->f[i]=0.;
    }

    for (int i=0; i<ps->N*DIM; i++){
        ps->mom[i]=0.;
    }

    for (int i=0; i<ps->N*MAX_NEI; i++){
        ps->isContact[i]=0;
        ps->isContactWall[i]=0;
    }



    particle_collision_cell_linked_fastUpdate(ps,box);

    wall_collision_triangles(ps,box,mesh);

    /* update */
    for (int i = 0; i < ps->N; i++)
    {
        if (ps->isActive[i]!=1){
            continue;
        }
        int bi = i*DIM;
        // acceleration
        ps->a[bi+0] = ps->f[bi+0]*ps->invm[i]+ps->g[0];
        ps->a[bi+1] = ps->f[bi+1]*ps->invm[i]+ps->g[1];
        ps->a[bi+2] = ps->f[bi+2]*ps->invm[i]+ps->g[2];

        // velocity
        ps->v[bi+0] += ps->a[bi+0] * ps->dt;
        ps->v[bi+1] += ps->a[bi+1] * ps->dt;
        ps->v[bi+2] += ps->a[bi+2] * ps->dt;

        // position
        ps->x[bi+0] += ps->v[bi+0] * ps->dt;
        ps->x[bi+1] += ps->v[bi+1] * ps->dt;
        ps->x[bi+2] += ps->v[bi+2] * ps->dt;

        // angular acceleration
        ps->anga[bi+0] = ps->mom[bi+0]*ps->invmoi[i];
        ps->anga[bi+1] = ps->mom[bi+1]*ps->invmoi[i];
        ps->anga[bi+2] = ps->mom[bi+2]*ps->invmoi[i];

        // angular velocity
        ps->angv[bi+0] += ps->anga[bi+0] * ps->dt;
        ps->angv[bi+1] += ps->anga[bi+1] * ps->dt;
        ps->angv[bi+2] += ps->anga[bi+2] * ps->dt;

    }
}
