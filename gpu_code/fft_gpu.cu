/*
    nvcc -arch sm_35 fft_gpu.cu -o fft_gpu
    nvcc -arch compute_70 -code sm_70 fft_gpu.cu -o fft_gpu
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
//Maximum Threads per Block is 1024, Maximum Shared Memory is 48KB
//cuComplexDouble is 16 bytes, therefore we can have 3072 elements in shared memory at once
#define MAX_SM_ELEM_NUM	      3072
#define BLOCK_DIM 	      32   //Max of 32
#define GRID_DIM	      1   //Keep it to 1

#define CHECK_TOL          0.05
#define MINVAL             0.0
#define MAXVAL             10.0

//Function prototypes
void initializeArray(cplx *arr, int len, int seed);
double interval(struct timespec start, struct timespec stop);
void printArray(int rowLen, cplx* data);
void runIteration(int rowLen);
void show_buffer(cplx buf[], int rowLen, int n);
void transpose(cplx buf[], int rowLen);
void fft(cplx buf[], int n);
void fft_2d(cplx buf[], int rowLen, int n);

// /*......CUDA Device Functions......*/
// /**
//  * Reorders array by bit-reversing the indexes.
//  */
//  __global__ void bitrev_reorder(Cplx* __restrict__ r, Cplx* __restrict__ d, int s, size_t nthr) {
//   int id = blockIdx.x * nthr + threadIdx.x;
//   r[__brev(id) >> (32 - s)] = d[id];
// }

// //Inner part of FFT loop. Contains the procedure itself.
//  __device__ void inplace_fft_inner(Cplx* __restrict__ r, int j, int k, int m, int n) {
//   if (j + k + m / 2 < n) { 
//     Cplx t, u;
    
//     t.x = __cosf((2.0 * M_PI * k) / (1.0 * m));
//     t.y = -__sinf((2.0 * M_PI * k) / (1.0 * m));
    
//     u = r[j + k];
//     t = CplxMul(t, r[j + k + m / 2]);

//     r[j + k] = CplxAdd(u, t);
//     r[j + k + m / 2] = CplxAdd(u, CplxInv(t));
//   }
// }

// /**
//  * Middle part of FFT for small scope paralelism.
//  */
//  __global__ void inplace_fft(Cplx* __restrict__ r, int j, int m, int n, size_t nthr) {
//   int k = blockIdx.x * nthr + threadIdx.x;
//   inplace_fft_inner(r, j, k, m, n);
// }

// /**
//  * Outer part of FFT for large scope paralelism.
 
//  m = 2^iteration
//  n = number of elements in matrix
 
//  */
// __global__ void inplace_fft_outer(Cplx* __restrict__ r, int m, int n, size_t nthr) {
//   int j = (blockIdx.x * nthr + threadIdx.x) * m;
  
//   for (int k = 0; k < m / 2; k++) {
//     inplace_fft_inner(r, j, k, m, n);
//   }
// }

//Does bitwise reversal of a row of the overall matrix in the GPU
//Uses coalesced memory accesses in global memory with possible thread divergence if matrix dimensions are poorly chosen
// __global__ void reverseArrayBlockRow(int i, int rowLen, int s,  cuDoubleComplex* d_out, cuDoubleComplex* d_in)
// {
//   int rowIdx = i*rowLen;
//   int j  = (blockDim.x * blockIdx.x) + threadIdx.x + (blockDim.x*threadIdx.y);

//   //Load the given index into shared memory and do the bit order reversal in the time domain
//   __shared__ cuDoubleComplex d_shared[MAX_SM_ELEM_NUM];
//   for(; j < rowLen; j += blockDim.x*gridDim.x)
//   {  
//     if(j < rowLen)
//       d_shared[(__brev(j) >> (32 - s))] = d_in[j + rowIdx];
//       //cuPrintf("j was :%d and oidx was %d\n", j, (__brev(j) >> (32 - s)));
//   }
//   __syncthreads();

//   //Copy the data back out
//   for(j  = (blockDim.x * blockIdx.x) + threadIdx.x + (blockDim.x*threadIdx.y); j < rowLen; j += blockDim.x*gridDim.x)
//   {  
//     if(j < rowLen)
//       d_out[j + rowIdx] = d_shared[j];
//     //cuPrintf("j was :%d and oidx was %d\n", j, (__brev(j) >> (32 - s)));
//     //cuPrintf("d_shared[%d] = (%f, %f)\n", j, cuCreal(d_shared[j]), cuCreal(d_shared[j]));
//   }
// }

// FFT kernel per SM code
//Need to remove gridDim stuff if we do one block per row
__global__ void FFT_Kernel_Row(int rowIdx, int rowLen, int s,  cuDoubleComplex* d_out, cuDoubleComplex* d_in, double pi)
{
  int rowSz = rowIdx*rowLen;
  int colIdx  = (blockDim.x * blockIdx.x) + threadIdx.x + (blockDim.x*threadIdx.y);

  //Load the given index into shared memory and do the bit order reversal in the time domain
  __shared__ cuDoubleComplex d_shared[MAX_SM_ELEM_NUM];
  for(; colIdx < rowLen; colIdx += blockDim.x*gridDim.x)
  {  
    if(colIdx < rowLen)
      d_shared[(__brev(colIdx) >> (32 - s))] = d_in[colIdx + rowSz];
      //cuPrintf("j was :%d and oidx was %d\n", colIdx, (__brev(colIdx) >> (32 - s)));
  }
  __syncthreads();

  //Do the FFT itself for the row
  /*cuDoubleComplex wlen, w, u, v;
  int len, i, j;
  for (len = 2; len <= rowLen; len <<= 1)
  {
    double ang = 2 * pi / len;
    wlen = make_cuDoubleComplex(cos(ang), sin(ang));

    for (i = (blockIdx.x * blockDim.x + threadIdx.x)*len; i < rowLen; i += (blockDim.x*gridDim.x*len));
		{
			w = make_cuDoubleComplex(1, 0);
			for (j = 0; j < (len / 2); j++) 
			{
				//Compute the DFT on the correct elements
				u = d_shared[i+j];
				v = cuCmul(d_shared[i+j+(len/2)], w);
				d_shared[i+j] = cuCadd(u, v);
				d_shared[i+j+(len/2)] = cuCsub(u, v);
				w = cuCmul(w, wlen);
			}
		}
    __syncthreads();
  }
  __syncthreads();*/

  //Copy the data back out
  for(j  = (blockDim.x * blockIdx.x) + threadIdx.x + (blockDim.x*threadIdx.y); j < rowLen; j += blockDim.x*gridDim.x)
  {  
    d_out[j + rowIdx] = d_shared[j];
    //cuPrintf("j was :%d and oidx was %d\n", j, (__brev(j) >> (32 - s)));
    //cuPrintf("d_shared[%d] = (%f, %f)\n", j, cuCreal(d_shared[j]), cuCreal(d_shared[j]));
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
  double currDiff, maxDiff = 0;

  //Check that row can fit into SM
  if(rowLen > MAX_SM_ELEM_NUM)
  {
    fprintf(stderr, "The specified array will not work with shared memory\n");
    exit(EXIT_FAILURE);
  } 

  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

  // Define size of matricies
  size_t n = rowLen * rowLen;
  size_t allocSize = n * sizeof(cplx);
  
  // Allocate matricies on host memory
  cplx *h_array                    = (cplx *) malloc(allocSize);
  cplx *h_serial_array             = (cplx *) malloc(allocSize);

  // Initialize the host arrays
  printf("\nInitializing the arrays ...");
  // Arrays are initialized with a known seed for reproducability
  initializeArray(h_array, n, 2453);
  initializeArray(h_serial_array, n, 2453);
#ifdef PRINT_MATRIX  
  printf("h_array:\n");
  printArray(rowLen, h_array);
  printf("h_serial_array\n");
  printArray(rowLen, h_serial_array);
#endif
  printf("\t... done\n\n");

  //Copy double complex array to cuDoubleComplex array
  cuDoubleComplex* d = (cuDoubleComplex*) malloc(sizeof(cuDoubleComplex) * n);
  for(int i = 0; i < n; i++)
  {
    double real_part = creal(h_array[i]);
    double imag_part = cimag(h_array[i]);
    d[i] = make_cuDoubleComplex(real_part, imag_part);
    CUDA_SAFE_CALL(cudaPeekAtLastError());
  }

  // Allocate arrays on GPU global memory
  cuDoubleComplex* d_array;
  cuDoubleComplex* d_array_out;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_array, n*sizeof(cuDoubleComplex)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_array_out, n*sizeof(cuDoubleComplex)));
  
  // Start overall GPU timing
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //Transfer cuDoubleArray to device memory
  CUDA_SAFE_CALL(cudaMemcpy(d_array, d, allocSize, cudaMemcpyHostToDevice));

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

  // Compute the fft for each thread
  //FFT_Kernel<<<DimGrid, DimBlock>>>(rowLen, d_array);
  int s = (int)log2((float)rowLen);
  for(int i = 0; i < rowLen; i++)
  {
    FFT_Kernel_Row<<<DimGrid, DimBlock>>>(i, rowLen, s, d_array_out, d_array, PI);
    cudaDeviceSynchronize();
  }

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
  CUDA_SAFE_CALL(cudaMemcpy(d, d_array_out, allocSize, cudaMemcpyDeviceToHost));
  
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

  //Copy cuDoubleComplex array to double complex array  
  for(int i = 0; i < n; i++)
  {
    double real_part = cuCreal(d[i]);
    double imag_part = cuCimag(d[i]);
    h_array[i] = real_part + I*imag_part;
    CUDA_SAFE_CALL(cudaPeekAtLastError());
  }

  // Compute the results on the host
  printf("FFT_serial() start\n");
  clock_gettime(CLOCK_REALTIME, &time_start);
  fft_2d(h_serial_array, rowLen, n);
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
  for(i = 0; i < rowLen; i++) {
    for(j = 0; j < rowLen; j++)
    {
        currDiff = abs(creal(h_serial_array[i]) - creal(h_array[i]));
        maxDiff = (maxDiff < currDiff) ? currDiff : maxDiff;
        if (currDiff > CHECK_TOL) {
            errCount++;
        }
        if (h_array[i] == 0) {
            zeroCount++;
        }

        currDiff = abs(cimag(h_serial_array[i]) - cimag(h_array[i]));
        maxDiff = (maxDiff < currDiff) ? currDiff : maxDiff;
        if (currDiff > CHECK_TOL) {
            errCount++;
        }
        if (h_array[i] == 0) {
            zeroCount++;
        }
    }
  }
  
  // Free-up device and host memory
  CUDA_SAFE_CALL(cudaFree(d_array));
  CUDA_SAFE_CALL(cudaFree(d_array_out));
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
      printf("%.1f+j%.1f, ",creal(data[i*rowLen+j]), cimag(data[i*rowLen+j]));
    }
    printf("\n");
  }
}

/* Performs in place FFT on buf of size n*/
void fft(cplx buf[], int n) 
{
	//Rearrange the array such that it can be iterated upon in the correct order
	//This is called decimination-in-time or Cooley-Turkey algorithm to rearrange it first, then do nlogn iterations
	int i, j, len;
	for (i = 1, j = 0; i < n; i++) 
	{
		int bit = n >> 1;
		for (; j & bit; bit >>= 1)
				j ^= bit;
		j ^= bit;

		//swap(buf[i], buf[j]);
		cplx temp;
    if (i < j)
		{
			temp = buf[i];
			buf[i] = buf[j];
			buf[j] = temp;
		}
  }

	/*Compute the FFT for the array*/
	cplx wlen, w, u, v;
	// len goes 2, 4, ... n/2, n
	// len iterates over the array log2(n) times
  for (len = 2; len <= n; len <<= 1) 
	{
		double ang = 2 * PI / len;
		wlen = cexp(I * ang);

		/* i goes from 0 to n with stride len
		j goes from 0 to len/2 in stride 1

		The sum of i+j is used to index into the buffer 
		and determine the correct indexes at which to perform the DFT.
		For example if n = 8:
		For the first iteration len = 2, i = 0,2,4,8, j = 0 so that i + j = 0,2,4,8.  
		For the second iteration len = 4, i = 0,4, j = 0,1  so that i + j = 0,1,4,5.  
		For the final iteration len = 8, i = 0, j = 0,1,2,3 so that i + j = 0,1,2,3.
		This allows us to DFT properly for each index based on the conceptual algorithm.

		For each iteration of there are n/2 iterations as shown above,
		*/
		for (i = 0; i < n; i += len) 
		{
			w = 1;
			for (j = 0; j < (len / 2); j++) 
			{
				//Compute the DFT on the correct elements
				u = buf[i+j];
				v = buf[i+j+(len/2)] * w;
				buf[i+j] = u + v;
				buf[i+j+(len/2)] = u - v;
				w *= wlen;
			}
		}
  }
}

/* Transpose the matrix */
void transpose(cplx buf[], int rowLen)
{
	int i, j;
	cplx temp;
	for (i = 0; i < rowLen; i++)
	{
		for (j = i+1; j < rowLen; j++)
		{
			temp = buf[i*rowLen + j];
			buf[i*rowLen + j] = buf[j*rowLen + i];
			buf[j*rowLen + i] = temp;
		}
	}
}

/* Orchestrates the row-column 2D FFT algorithm */
void fft_2d(cplx buf[], int rowLen, int n)
{
	// Do rows
	int i;
	for(i = 0; i < n; i += rowLen)
	{
		fft(buf+i, rowLen);
	}

	// Transpose the matrix
	transpose(buf, rowLen);

	// Do columns
	for(i = 0; i < n; i += rowLen)
	{
		fft(buf+i, rowLen);
	}

	// Transpose back
	transpose(buf, rowLen);
}

//Print the complex arrays before and after FFT
void show_buffer(cplx buf[], int rowLen, int n) {
	int i;
	for (i = 0; i < n; i++)
	{
		if (i%rowLen == 0)
			printf("\n");

		if (!cimag(buf[i]))
			printf("%g ", creal(buf[i]));
		else
			printf("(%g,%g) ", creal(buf[i]), cimag(buf[i]));
	}
	printf("\n\n");
}
