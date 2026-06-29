CC = nvcc
CC_C = g++
ARCH = sm_70 
CFLAGS = -O3 -Xcompiler "-Wconversion  -Werror  -fopenmp"  
GPUFLAGS = -DUSE_GPU=1 -arch=$(ARCH) -fmad=false 
#CFLAGS = -O0 -g -G
#CFLAGS = -O3  -pg
#CFLAGS = -O0  -g 
#CFLAGS = -Xcompiler "-fsanitize=address -fno-omit-frame-pointer" -O0  -g  
CONLYFLAGS = -O3  
LIBS = -lm 
OBJS =  main.o SMACSolver.o G_SMACSolver.o CFDTime.o\
	   	pressure_solver/G_PCGSolver.o pressure_solver/G_PressureSolverBase.o pressure_solver/G_GMGSolver.o pressure_solver/G_Levels.o

PROGRAM = myCFD

all: $(PROGRAM)

$(PROGRAM): $(OBJS)
	$(CC) $(CFLAGS) $(GPUFLAGS) $(OBJS) $(LIBS)  -o $(PROGRAM)

%.o : %.cu
	$(CC) $(CFLAGS) $(GPUFLAGS) -c $< -o $@

%.o : %.cpp
	$(CC_C) $(CONLYFLAGS) -c $< -o $@

clean:
	rm -f $(PROGRAM)  $(OBJS)

