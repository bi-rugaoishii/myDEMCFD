#include "TriangleMesh.h"

/* ========= morton key related functions ================== */
static void mortonKeyGen(TriangleMesh* mesh)
{
    const int BITS  = 10;
    const int SCALE = (1<<BITS) - 1;

    double invRangeX = SCALE / (mesh->gmaxx - mesh->gminx);
    double invRangeY = SCALE / (mesh->gmaxy - mesh->gminy);
    double invRangeZ = SCALE / (mesh->gmaxz - mesh->gminz);

    for (int i=0; i<mesh->nTri; i++)
    {
        uint32_t ix = (uint32_t)((mesh->cx[i] - mesh->gminx) * invRangeX);
        uint32_t iy = (uint32_t)((mesh->cy[i] - mesh->gminy) * invRangeY);
        uint32_t iz = (uint32_t)((mesh->cz[i] - mesh->gminz) * invRangeZ);

        mesh->mortonKey[i] = morton3D(ix,iy,iz);
    }
}

/* ========== hash key related functions used for merging ===========*/
static inline unsigned long long hash_func(long long x,long long y,long long z){
    unsigned long long h = 1469598103934665603ULL;
    h ^= (unsigned long long)x; h *= 1099511628211ULL;
    h ^= (unsigned long long)y; h *= 1099511628211ULL;
    h ^= (unsigned long long)z; h *= 1099511628211ULL;
    return h;
}

void vertex_hash_init(VertexHash* h, size_t size){
    h->size = size;
    h->table = (HashEntry*)calloc(size, sizeof(HashEntry));
}

int vertex_hash_get_or_insert(VertexHash* h,
        double x, double y, double z,
        double eps,
        int vIndex){
    long long ix = llround(x / eps);
    long long iy = llround(y / eps);
    long long iz = llround(z / eps);

    unsigned long long hash = hash_func(ix,iy,iz);
    size_t pos = hash % h->size;

    while(1){
        if(!h->table[pos].used){
            /* 新規登録 */
            h->table[pos].used  = 1;
            h->table[pos].ix    = ix;
            h->table[pos].iy    = iy;
            h->table[pos].iz    = iz;
            h->table[pos].index = vIndex;

            return h->table[pos].index;
        }

        /* 一致チェック */
        if(h->table[pos].ix == ix &&
                h->table[pos].iy == iy &&
                h->table[pos].iz == iz)
        {
            return h->table[pos].index;
        }

        /* 線形探索 */
        pos = (pos + 1) % h->size;
    }
}

int get_edge_index(int v0, int v1, int* edge,int eIndex){

    int tv0 = (v0<v1) ? v0:v1;
    int tv1 = (v0>v1) ? v0:v1;
    for (int k=0; k<eIndex; k++){
        if (edge[k*2+0]==v0 && edge[k*2+1]==v1){
            return k;
            break;
        }
    }
    edge[eIndex*2+0]= tv0;
    edge[eIndex*2+1]= tv1;
    return eIndex;

}

void sort3(int *x, int *y, int *z){
    int tmp;

    if (*x > *y){
        tmp = *x; *x = *y; *y = tmp;
    }

    if (*y > *z){
        tmp = *y; *y = *z; *z = tmp;
    }

    if (*x > *y){
        tmp = *x; *x = *y; *y = tmp;
    }
}

/* ========== Device Related===============*/
#if USE_GPU
void deviceMallocCopyTriangleMesh(TriangleMesh *mesh){
    int Nv = mesh->nVert;
    int Nt = mesh->nTri;

    mesh->d_mesh.nVert = mesh->nVert;
    mesh->d_mesh.nTri = mesh->nTri;
    mesh->d_mesh.nShift = mesh->nShift;


    #define MEMBER(type,name, size) cudaMalloc(&mesh->d_mesh.name, sizeof(type)*(size)); 
    #include "memberList/TriangleMeshMember_common.def"
    #undef MEMBER


    /* ================= malloc struct =============*/
    cudaMalloc(&mesh->d_meshPtr, sizeof(DeviceTriangleMesh));


    /* ================ memcpy ================= */
    #define MEMBER(type,name, size) cudaMemcpy(mesh->d_mesh.name,mesh->name, sizeof(type)*(size),cudaMemcpyHostToDevice); 
    #include "memberList/TriangleMeshMember_common.def"
    #undef MEMBER

    /* --- copy struct ----*/
    cudaMemcpy(mesh->d_meshPtr,  &mesh->d_mesh,  sizeof(DeviceTriangleMesh), cudaMemcpyHostToDevice);

}


#endif

/* ========== triangle mesh ===============*/
void free_TriangleMesh(TriangleMesh* mesh, int isGPUon){
    free(mesh->sortedIndex);

    free(mesh->mortonKey);
    free(mesh->cx);
    free(mesh->cy);
    free(mesh->cz);

    /* shared members */
    #define MEMBER(type,name, size) free(mesh->name); 
    #include "memberList/TriangleMeshMember_common.def"
    #undef MEMBER

    #if USE_GPU
    if(isGPUon == 1){
        #define MEMBER(type,name, size) cudaFree(mesh->d_mesh.name); 
        #include "memberList/TriangleMeshMember_common.def"
        #undef MEMBER

        cudaFree(mesh->d_meshPtr);
    }
    #endif
}

static int count_ascii_stl_triangles(FILE* fp){
    char line[256];
    int count = 0;

    while(fgets(line,256,fp))
    {
        if(strncmp(line,"facet normal",12)==0 ||
                strncmp(line," facet normal",13)==0)
        {
            count++;
        }
    }
    return count;
}


int load_ascii_stl_double(const cJSON* wallNames, TriangleMesh* mesh){
    FILE* fp;

    if (!cJSON_IsArray(wallNames)){
        printf("walls is not array\n");
        abort();
    }

    int fileNum = cJSON_GetArraySize(wallNames);


    /* count number of triangles */

    int triCount =0;
    for (int i=0; i<fileNum; i++){
        const char* filename = cJSON_GetArrayItem(wallNames,i)->valuestring;
        printf("filepath is %s\n",filename);
        fp = fopen(filename,"r");
        if(!fp){
            printf("stl open error\n");
            abort();
            return 0;
        }
        triCount += count_ascii_stl_triangles(fp);
        fclose(fp);
    }


    mesh->nTri = triCount;
    mesh->nVert  = triCount * 3;

    int Nv = mesh->nVert;
    int Nt = mesh->nTri;
    /* ===== global mesh AABB init ===== */

    mesh->gminx =  1e300;
    mesh->gminy =  1e300;
    mesh->gminz =  1e300;

    mesh->gmaxx = -1e300;
    mesh->gmaxy = -1e300;
    mesh->gmaxz = -1e300;

    mesh->sortedIndex = (int*)malloc(sizeof(int)*Nt);

    mesh->mortonKey = (uint32_t*)malloc(sizeof(uint32_t)*Nt);

    mesh->cx = (double*)malloc(sizeof(double)*Nt);
    mesh->cy = (double*)malloc(sizeof(double)*Nt);
    mesh->cz = (double*)malloc(sizeof(double)*Nt);

    /* allocate commom members */
    #define MEMBER(type,name, size) mesh->name = (type*)malloc(sizeof(type)*(size)); 
    #include "memberList/TriangleMeshMember_common.def"
    #undef MEMBER


    char line[256];

    int triIndex = 0;
    int vIndex   = 0;
    int eIndex = 0;
    mesh->nShift   = Nv;

    double nx,ny,nz;
    double x,y,z;

    for (int i=0; i<fileNum; i++){
        const char* filename = cJSON_GetArrayItem(wallNames,i)->valuestring;
        fp = fopen(filename,"r");
        if(!fp){
            printf("stl open error\n");
            abort();
            return 0;
        }

        /* ========= for merging ============= */
        VertexHash vh;
        vertex_hash_init(&vh,mesh->nVert*3);

        while(fgets(line,256,fp))
        {
            //if(sscanf(line," facet normal %lf %lf %lf",&nx,&ny,&nz)==3 ||       sscanf(line,"facet normal %lf %lf %lf",&nx,&ny,&nz)==3)
            /* need better scan */
            if(sscanf(line," facet normal %lf %lf %lf",&nx,&ny,&nz)==3){


                mesh->sortedIndex[triIndex] = triIndex; /* will be sorted later */

                /* outer loop */
                if(!fgets(line,256,fp)) break;
                int v0,v1,v2;

                /* v0 */
                if(!fgets(line,256,fp)) break;
                sscanf(line," vertex %lf %lf %lf",&x,&y,&z);
                mesh->mx[vIndex]=x;
                mesh->my[vIndex]=y;
                mesh->mz[vIndex]=z;
                v0=vertex_hash_get_or_insert(&vh,x,y,z,1e-9,vIndex);
                if (v0==vIndex){
                    vIndex   += 1; //was new  
                }



                /* v1 */
                if(!fgets(line,256,fp)) break;
                sscanf(line," vertex %lf %lf %lf",&x,&y,&z);
                mesh->mx[vIndex]=x;
                mesh->my[vIndex]=y;
                mesh->mz[vIndex]=z;
                v1=vertex_hash_get_or_insert(&vh,x,y,z,1e-9,vIndex);
                if (v1==vIndex){
                    vIndex   += 1; //was new  
                }

                /* v2 */
                if(!fgets(line,256,fp)) break;
                sscanf(line," vertex %lf %lf %lf",&x,&y,&z);
                mesh->mx[vIndex]=x;
                mesh->my[vIndex]=y;
                mesh->mz[vIndex]=z;
                v2=vertex_hash_get_or_insert(&vh,x,y,z,1e-9,vIndex);
                if (v2==vIndex){
                    vIndex   += 1; //was new  
                }

                /* reorder vertices in ascending order */
                sort3(&v0,&v1,&v2);

                /* make edge */
                int tmpEdge;
                tmpEdge = get_edge_index(v0,v1,mesh->edge,eIndex);
                if(tmpEdge==eIndex){ //new index
                    eIndex += 1;
                }
                mesh->tri_e0[triIndex] = tmpEdge+mesh->nShift;

                tmpEdge = get_edge_index(v1,v2,mesh->edge,eIndex);
                if(tmpEdge==eIndex){ //new index
                    eIndex += 1;
                }
                mesh->tri_e1[triIndex] = tmpEdge+mesh->nShift;

                tmpEdge = get_edge_index(v0,v2,mesh->edge,eIndex);
                if(tmpEdge==eIndex){ //new index
                    eIndex += 1;
                }
                mesh->tri_e2[triIndex] = tmpEdge+mesh->nShift;



                /* calculate edge vectors */
                Vec3 e01;

                e01.x = mesh->mx[v1]-mesh->mx[v0];
                e01.y = mesh->my[v1]-mesh->my[v0];
                e01.z = mesh->mz[v1]-mesh->mz[v0];

                Vec3 e02;
                e02.x = mesh->mx[v2]-mesh->mx[v0];
                e02.y = mesh->my[v2]-mesh->my[v0];
                e02.z = mesh->mz[v2]-mesh->mz[v0];

                Vec3 e12;
                e12.x = mesh->mx[v2]-mesh->mx[v1];
                e12.y = mesh->my[v2]-mesh->my[v1];
                e12.z = mesh->mz[v2]-mesh->mz[v1];

                mesh->e01x[triIndex] = e01.x;
                mesh->e01y[triIndex] = e01.y;
                mesh->e01z[triIndex] = e01.z;

                mesh->e02x[triIndex] = e02.x;
                mesh->e02y[triIndex] = e02.y;
                mesh->e02z[triIndex] = e02.z;

                mesh->e12x[triIndex] = e12.x;
                mesh->e12y[triIndex] = e12.y;
                mesh->e12z[triIndex] = e12.z;

                mesh->d00[triIndex] = vdot(e01,e01);
                mesh->d00inv[triIndex] = 1./mesh->d00[triIndex];
                mesh->d01[triIndex] = vdot(e01,e02);
                mesh->d11[triIndex] = vdot(e02,e02);
                mesh->d11inv[triIndex] = 1./mesh->d11[triIndex];
                mesh->d22[triIndex] = vdot(e12,e12);
                mesh->d22inv[triIndex] = 1./mesh->d22[triIndex];

                mesh->denom[triIndex] = mesh->d00[triIndex]*mesh->d11[triIndex]-mesh->d01[triIndex]*mesh->d01[triIndex];
                mesh->denom[triIndex] = 1./mesh->denom[triIndex];

                /* calculate normals */
                Vec3 n;
                n = vcross(e01,e02);
                n = vscalar(1./sqrt(vdot(n,n)),n);
                mesh->nx[triIndex] = n.x;
                mesh->ny[triIndex] = n.y;
                mesh->nz[triIndex] = n.z;
                mesh->d[triIndex]=-n.x*x-n.y*y-n.z*z;

                mesh->tri_i0[triIndex]=v0;
                mesh->tri_i1[triIndex]=v1;
                mesh->tri_i2[triIndex]=v2;

                /* ===== compute triangle AABB ===== */

                double x0 = mesh->mx[v0];
                double y0 = mesh->my[v0];
                double z0 = mesh->mz[v0];

                double x1 = mesh->mx[v1];
                double y1 = mesh->my[v1];
                double z1 = mesh->mz[v1];

                double x2 = mesh->mx[v2];
                double y2 = mesh->my[v2];
                double z2 = mesh->mz[v2];

                /* ==== compute center coord ===== */
                mesh->cx[triIndex] = (x0+x1+x2)/3.0;
                mesh->cy[triIndex] = (y0+y1+y2)/3.0;
                mesh->cz[triIndex] = (z0+z1+z2)/3.0;

                /* ===== compute morton key ========== */

                /* min */
                double minx = x0;
                if(x1 < minx) minx = x1;
                if(x2 < minx) minx = x2;

                double miny = y0;
                if(y1 < miny) miny = y1;
                if(y2 < miny) miny = y2;

                double minz = z0;
                if(z1 < minz) minz = z1;
                if(z2 < minz) minz = z2;

                /* max */
                double maxx = x0;
                if(x1 > maxx) maxx = x1;
                if(x2 > maxx) maxx = x2;

                double maxy = y0;
                if(y1 > maxy) maxy = y1;
                if(y2 > maxy) maxy = y2;

                double maxz = z0;
                if(z1 > maxz) maxz = z1;
                if(z2 > maxz) maxz = z2;

                mesh->minx[triIndex] = minx;
                mesh->maxx[triIndex] = maxx;

                mesh->miny[triIndex] = miny;
                mesh->maxy[triIndex] = maxy;

                mesh->minz[triIndex] = minz;
                mesh->maxz[triIndex] = maxz;

                /* ===== update global mesh AABB ===== */

                if(minx < mesh->gminx) mesh->gminx = minx;
                if(miny < mesh->gminy) mesh->gminy = miny;
                if(minz < mesh->gminz) mesh->gminz = minz;

                if(maxx > mesh->gmaxx) mesh->gmaxx = maxx;
                if(maxy > mesh->gmaxy) mesh->gmaxy = maxy;
                if(maxz > mesh->gmaxz) mesh->gmaxz = maxz;

                triIndex += 1;

                if(!fgets(line,256,fp)) break;
                if(!fgets(line,256,fp)) break;
            }
        }

        fclose(fp);
    }

    /* ============ generate morton key and check presort index ========*/

    /* === for debug === 

       printf("Triangle center coordinates\n");
       for (int i=0; i<mesh->nTri; i++){
       printf("%f %f %f\n",mesh->cx[i],mesh->cy[i],mesh->cz[i]);
       }
     */
    mortonKeyGen(mesh);
    printf("Generated triangle mortonkey and pre sort index\n");

    /* === for debug === 
       for (int i=0; i<mesh->nTri; i++){
       printf("%u %d\n",mesh->mortonKey[i],mesh->sortedIndex[i]);
       }
     */

    double checkDistance=0.;
    for (int i=1; i<mesh->nTri; i++){
        Vec3 x1;
        x1.x=mesh->cx[i]-mesh->cx[i-1];
        x1.y=mesh->cy[i]-mesh->cy[i-1];
        x1.z=mesh->cz[i]-mesh->cz[i-1];
        checkDistance+=sqrt(vdot(x1,x1));
    }
    checkDistance/=(double)mesh->nTri;
    printf("Avg distance is %f\n",checkDistance);


    /* =========== sort index by morton key ======== */
    uint32_t *tmpMortonKey = (uint32_t*)malloc(sizeof(uint32_t)*mesh->nTri);
    int *tmpSortedIndex = (int*)malloc(sizeof(int)*mesh->nTri);
    radixSortUint32(&mesh->mortonKey,&mesh->sortedIndex,mesh->nTri,tmpMortonKey,tmpSortedIndex);
    printf("Sorted triangle mortonkey and pre sort index\n");
    /* === for debug === 
       for (int i=0; i<mesh->nTri; i++){
       printf("%u %d\n",mesh->mortonKey[i],mesh->sortedIndex[i]);
       }
     */

    checkDistance=0.;
    for (int i=1; i<mesh->nTri; i++){
        Vec3 x1;
        int ind1 = mesh->sortedIndex[i];
        int ind0 = mesh->sortedIndex[i-1];
        x1.x=mesh->cx[ind1]-mesh->cx[ind0];
        x1.y=mesh->cy[ind1]-mesh->cy[ind0];
        x1.z=mesh->cz[ind1]-mesh->cz[ind0];
        checkDistance+=sqrt(vdot(x1,x1));
    }
    checkDistance/=(double)mesh->nTri;
    printf("Avg distance is %f\n",checkDistance);

    free(tmpMortonKey);
    free(tmpSortedIndex);


    printf(" final number of edges are %d \n", eIndex);


    printf("ASCII STL (double) loaded : %d triangles\n",triCount);
    return 1;
}

