#include "CFDTime.h"
#include "math.h"
#include "hardCodedParameters.h"
CFDTime::CFDTime(double initdt,double max_dt,double outfreqtime,double endTime,double cfl_thresh, Time_mode mode){
    mode_=mode;
    initdt_=initdt;
    max_dt_ = max_dt;
    dt_=initdt;
    current_time_ = 0.;
    current_steps_ =0;
    outStepFreq_=0;
    out_freq_time_ =outfreqtime;
    nextOut_ =outfreqtime;
    end_time_ =endTime;
    cfl_thresh_=cfl_thresh;
    isOutStep_ = false;
}


CFDTime::~CFDTime(){}

void CFDTime::updateTime(double cfl){
    
    if(current_steps_ <5){
        double dt_tmp = dt_ *cfl_thresh_/(cfl+EPS);
        dt_ = initdt_<dt_tmp? initdt_ :dt_tmp;
    }else{
        dt_ = dt_ *cfl_thresh_/(cfl+EPS);
        double timeTillNextOut = nextOut_ - current_time_;

        if(dt_>= timeTillNextOut){
            dt_ = timeTillNextOut;
        }

        if(dt_ >max_dt_){
            dt_=max_dt_;
        }

        if(fabs(current_time_+dt_-nextOut_) <= 1e-12){
            isOutStep_ = true;
            nextOut_+=out_freq_time_;
        }
    }
    /* later  put in mode */
    current_time_ += dt_;

    current_steps_+=1;
}


