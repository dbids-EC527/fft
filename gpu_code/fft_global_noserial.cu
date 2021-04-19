/*
    nvcc -arch sm_35 fft_global.cu ./base_code/fft_2d.c -o fft_global
*/

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <string.h>
#include "cuPrintf.cu"
#include "cuPrintf.cuh"
#include <cuComplex.h>

//My own serial fft2d code
#include "fft_2d.h"

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
 
#define PI 3.1415926535897932384
typedef double complex cplx;

//Definitions which turn on and off test printing
//#define PRINT_GPU
#define PRINT_MATRIX

//Best performance occurs when the number of pixels is divisable by the number of threads
#define BLOCK_DIM 	    16
#define GRID_DIM	      128

#define CHECK_TOL          0.05
#define MINVAL             0.0
#define MAXVAL             10.0

//Function prototypes
void initializeArray(cplx *arr, int len, int seed);
double interval(struct timespec start, struct timespec stop);
void printArray(int rowLen, cplx* data);
void runIteration(int rowLen);

/*......CUDA Device Functions......*/
// FFT kernel per thread code
__global__ void FFT_Kernel (int rowLen, cuDoubleComplex* data) 
{
  int i, j, iters;
  
  //Interleave threads over a single block of the total array
  for (iters = 0; iters < 2000; iters++) 
  {
    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < rowLen; i += blockDim.x*gridDim.x)
    {
      for (j = blockIdx.y * blockDim.y + threadIdx.y; j < rowLen; j += blockDim.y*gridDim.y)
      {
        //Reduce the current pixel
        if(i>0 && i<rowLen-1 && j>0 && j<rowLen-1)
        {
          data[i*rowLen+j] = cuCadd(data[i*rowLen+j], make_cuDoubleComplex(5, 0));
        }
      }
    }
  __syncthreads();
  }
}

/*......Host Code......*/
int main(int argc, char **argv)
{
  //Get the row length
  if (argc > 1) {
    int rowLen = atoi(argv[1]);
    printf("Running code for %dx%d matrix\n", rowLen, rowLen);
    runIteration(rowLen);
  }
  else 
  {
    printf("Running code for 1024x1024 matrix\n");
    runIteration(1024);
    printf("Running code for 2048x2048 matrix\n");
    runIteration(2048);
  }     
  
  return 0;
}

//Runs an iteration of GPU and CPU code for a given row length
void runIteration(int rowLen)
{
  // GPU Timing variables
  cudaEvent_t start, stop, start_kernel, stop_kernel;
  float elapsed_gpu, elapsed_gpu_kernel;
  
  //Serial Timing variables:
  struct timespec time_start, time_stop;

  //Define local vars for checking correctness
  int i, j, errCount = 0, zeroCount = 0;
  float currDiff, maxDiff = 0;

  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

  // Define size of matricies
  size_t allocSize = rowLen * rowLen * sizeof(cplx);

  // Allocate matricies on host memory
  cplx *h_array                    = (cplx *) malloc(allocSize);
  cplx *h_serial_array             = (cplx *) malloc(allocSize);

  // Initialize the host arrays
  printf("\nInitializing the arrays ...");
  // Arrays are initialized with a known seed for reproducability
  initializeArray(h_array, rowLen*rowLen, 2453);
  initializeArray(h_serial_array, rowLen*rowLen, 2453);
#ifdef PRINT_MATRIX  
  printf("h_array:\n");
  printArray(rowLen, h_array);
  printf("h_serial_array\n");
  printArray(rowLen, h_serial_array);
#endif
  printf("\t... done\n\n");

  // Allocate arrays on GPU global memory
  //cplx *d_array;
  //CUDA_SAFE_CALL(cudaMalloc((void **)&d_array, allocSize));
  cuDoubleComplex* d_array;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_array, rowLen*rowLen*sizeof(cuDoubleComplex)));
  for(int i = 0; i < rowLen*rowLen; i++)
  {
    d_array[i] = make_cuDoubleComplex(creal(h_array[i]), cimag(h_array[i]));
    CUDA_SAFE_CALL(cudaPeekAtLastError());
  }
  
  // Start overall GPU timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Transfer the arrays to the GPU memory
  //CUDA_SAFE_CALL(cudaMemcpy(d_array, h_array, allocSize, cudaMemcpyHostToDevice));

  // Configure the kernel
  dim3 DimGrid(GRID_DIM, GRID_DIM, 1);    
  dim3 DimBlock(BLOCK_DIM, BLOCK_DIM, 1); 
  printf("Kernal code launching\n");

#ifdef PRINT_GPU
  cudaPrintfInit();
#endif

  // Start kernel timing
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);
  cudaEventRecord(start_kernel, 0);

  // Compute the mmm for each thread
  FFT_Kernel<<<DimGrid, DimBlock>>>(rowLen, d_array);
  cudaDeviceSynchronize();

  // End kernel timing
  cudaEventRecord(stop_kernel, 0);
  cudaEventSynchronize(stop_kernel);
  cudaEventElapsedTime(&elapsed_gpu_kernel, start_kernel, stop_kernel);
  printf("\nGPU kernel time: %f (msec)\n", elapsed_gpu_kernel);   
  cudaEventDestroy(start_kernel);
  cudaEventDestroy(stop_kernel); 

  // Check for errors during launch
  CUDA_SAFE_CALL(cudaPeekAtLastError());

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(h_array, d_array, allocSize, cudaMemcpyDeviceToHost));
  
#ifdef PRINT_GPU
  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();
#endif

  // End overall GPU timing
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_gpu, start, stop);
  printf("GPU overall time: %f (msec)\n", elapsed_gpu);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Compute the results on the host
  printf("FFT_serial() start\n");
  clock_gettime(CLOCK_REALTIME, &time_start);
  fft_2d(h_serial_array, rowLen, rowLen*rowLen);
  clock_gettime(CLOCK_REALTIME, &time_stop);
  double time_spent = interval(time_start, time_stop);
  printf("FFT_serial() took %f seconds\n", time_spent);

  // Compare the results
#ifdef PRINT_MATRIX
  printf("GPU code:\n");
  printArray(rowLen, h_array);
  printf("serial code:\n");
  printArray(rowLen, h_serial_array);
#endif
  /*for(i = 0; i < rowLen; i++) {
    for(j = 0; j < rowLen; j++)
    {
        currDiff = abs(h_serial_array[i] - h_array[i]);
	      maxDiff = (maxDiff < currDiff) ? currDiff : maxDiff;
        if (currDiff > CHECK_TOL) {
            errCount++;
        }
        if (h_array[i] == 0) {
            zeroCount++;
        }
    }
  }
  
  if (errCount > 0) {
    float percentError = ((float)errCount / (float)(rowLen*rowLen)) * 100.0;
    printf("\n@ERROR: TEST FAILED: %d results did not match (%0.6f%%)\n", errCount, percentError);
  }
  else if (zeroCount > 0){
    printf("\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
  }
  else {
    printf("\nTEST PASSED: All results matched\n");
  }
  printf("MAX_DIFFERENCE = %f between serial and GPU code\n\n", maxDiff);*/
  
  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(d_array));

  free(h_serial_array);
  free(h_array);

  CUDA_SAFE_CALL(cudaDeviceReset());

  printf("Done with %dx%d matrix\n\n", rowLen, rowLen);
}

//Initiaizes the array to consistent random values
void initializeArray(cplx *arr, int len, int seed) {
  int i;
  float randNum;
  srand(seed);

  for (i = 0; i < len; i++) {
    randNum = ((float)rand()) / (float) RAND_MAX;
    arr[i] = (cplx)(MINVAL + (randNum * (MAXVAL - MINVAL)));
  }
}

//Calculates time interval for serial calculation
double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

//Print the array for testing
void printArray(int rowLen, cplx* data)
{
  long int i, j;
  for (i = 0; i < rowLen; i++)
  { 
    for (j = 0; j < rowLen; j++)
    { 
      printf("%.1f+j%.1f",creal(data[i*rowLen+j]), cimag(data[i*rowLen+j]));
    }
    printf("\n");
  }
}
