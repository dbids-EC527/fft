#include <cuComplex.h>
#define MAX_SM_ELEM_NUM	  3072
#define BLOCK_DIM 	  16	 //Max of 32
#define GRID_DIM	  3072 	 //Max of 2147483647

// FFT kernel per SM code
__global__ void FFT_Kernel_Row(int rowLen, int logn,  cuDoubleComplex* d_out, cuDoubleComplex* d_in)
{
  for(int rowIdx = blockIdx.x; rowIdx < rowLen; rowIdx += gridDim.x)
  {
    int rowSz = rowIdx*rowLen;
    int colIdx  = threadIdx.x + (blockDim.x*threadIdx.y);

    //Load the given index into shared memory and do the bit order reversal in the time domain
    __shared__ cuDoubleComplex d_shared[MAX_SM_ELEM_NUM];
    for(; colIdx < rowLen; colIdx += blockDim.x*blockDim.y)
    {  
      d_shared[(__brev(colIdx) >> (32 - logn))] = d_in[colIdx + rowSz];
    }
    __syncthreads();

    //Do the FFT itself for the row
    InnerFFT(rowLen, &d_shared[0]);
    __syncthreads();
    
    //Copy the data from shared memory to output
    for(colIdx  = threadIdx.x + (blockDim.x*threadIdx.y); colIdx < rowLen; colIdx += blockDim.x*blockDim.y)
    {  
      d_out[colIdx + rowSz] = d_shared[colIdx];
    } 
  }
  __syncthreads();
}

//Computes the FFT for a Block that has already been loaded in bit reversed order
__device__ inline void InnerFFT(int rowLen, cuDoubleComplex* d_shared)
{
  cuDoubleComplex w, u, v;
  int len, i, j;
  for (len = 2; len <= rowLen; len <<= 1)
  {
    double ang = 2 * M_PI / len;
    for ((threadIdx.x + (blockDim.x*threadIdx.y))*len; i < rowLen; i += (blockDim.x*blockDim.y)*len)
		{
			for (j = 0; j < (len / 2); j++) 
			{
        w = make_cuDoubleComplex(cos(ang*j), sin(ang*j));
				u = d_shared[i+j];
				v = cuCmul(d_shared[i+j+(len/2)], w);
				d_shared[i+j] = cuCadd(u, v);
				d_shared[i+j+(len/2)] = cuCsub(u, v);
			}
			__syncthreads();
		}
  }
}