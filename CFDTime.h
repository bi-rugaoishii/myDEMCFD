#pragma once
enum Time_mode{
    CONST_TIME_STEP,
    VARIBALE_TIME_STEP 
};

struct CFDTime{
    double dt_;
    double initdt_;
    double current_time_;
    int current_steps_;
    int outStepFreq_;
    double outTime_;
    double max_dt_;
    double nextOut_;
    double cfl_thresh_;
    double end_time_;
    double out_freq_time_;

    int mode_;
    bool isOutStep_;
    

    /* constructor */
    CFDTime(double initdt,double max_dt,double outfreqtime,double end_time, double cfl_thresh, Time_mode mode);

    /* destructor*/
    ~CFDTime();
    
    void updateTime(double cfl);
    


};
