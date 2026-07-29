#ifndef SOLVER_OUTPUT_H
#define SOLVER_OUTPUT_H

#include "ParticleSystem.h"


/*
============================================================
初期化

・出力ディレクトリが存在するか確認
・無ければ mkdir
・solverのmainで1回だけ呼ぶ
============================================================
*/
int solver_output_init(const char* dir);

/*
============================================================
BIN書き出し（軽量版）

前提：
solver_output_init が呼ばれていること
============================================================
*/

void write_header_text(const char* dir,int step,ParticleSys<HostMemory>* p,int pid);

void write_single_text(const char* dir,int step,ParticleSys<HostMemory>* p,int pid);
void write_single_text(const char* dir,int step,ParticleSys<HostMemory>* p,int pid);

void write_initialPos_csv(const char* dir,ParticleSys<HostMemory>* p);

void write_frame_bin_all(
    const char* dir,
    int step,
    ParticleSys<HostMemory> *p,
    const double length_factor
);

void write_frame_bin(
        const char* dir,
        int step,
        ParticleSys<HostMemory>* p, const double length_factor);


#endif
