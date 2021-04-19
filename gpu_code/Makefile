###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=gcc
CC_FLAGS=-std=c11 -lm
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-arch sm_35
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Make variables ##

# Target executable name:
EXE = fft_global

# Object files:
OBJS = fft_2d.o fft_global.o

##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
fft_2d.o : fft_2d.c fft_2d.h
	$(CC) $(CC_FLAGS) -c fft_2d.c -o fft_2d.o

# Compile CUDA source files to object files:
fft_global.o : fft_global.cu cuPrintf.cu cuPrintf.cuh fft_2d.h
	$(NVCC) $(NVCC_FLAGS) -c fft_global.cu -o fft_global.o

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)
