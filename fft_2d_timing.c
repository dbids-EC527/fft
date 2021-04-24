/*	Devin Bidstrup 03/29/21-05/07/21
 
	Must compile with:
	gcc fft_2d_timing.c ./base_code/fft_2d.c -o fft_2d_timing -lm -std=c11

  	
	Code based on a number of sources:
	*	C language with recursion:
		https://rosettacode.org/wiki/Fast_Fourier_transform#C
	*	C++ in place algorithm:
		https://cp-algorithms.com/algebra/fft.html
	*	Helpful visualization of the algorithm:
		https://towardsdatascience.com/fast-fourier-transform-937926e591cb
*/
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

//My own fft2d code
#include "base_code/fft_2d.h"
 
#define PI 3.1415926535897932384
typedef double complex cplx;

//Used to define min and max for indicies
#define MINVAL       0.0
#define MAXVAL       10.0

//Used to iterate through different array sizes
#define DELTA 32
#define BASE  32
#define ITERS 550    

//Prototypes
double interval(struct timespec start, struct timespec end);
void initializeArray(cplx *arr, int len, int seed);

int main(int argc, char *argv[])
{	
  //Serial Timing variables:
  struct timespec time_start, time_stop;
  double time_stamp[ITERS+1];
  
  //Define the buffer max sizes
  long int MAXSIZE = BASE+(ITERS+1)*DELTA;
  int rowLen = MAXSIZE;
  int n = MAXSIZE * MAXSIZE;
  printf("Doing FFT for %d different matrix sizes from %d to %ld\n",
      ITERS, DELTA, (long int) BASE+(ITERS)*DELTA);

  //Intialize two copies of the input buffer
  cplx* buf = (cplx*) calloc(n, sizeof(cplx));
  initializeArray(buf, n, 2453);
  cplx* buf_orig = (cplx*) calloc(n, sizeof(cplx));
  memcpy(buf_orig, buf, n * sizeof(cplx));

  long int iters = 0;
  for(long int i = BASE; iters < ITERS; i+=DELTA, iters++)
  {
    printf("iter %ld, size %ld\r", iters, i);
    fflush(stdout);

    //Start Timer
    clock_gettime(CLOCK_REALTIME, &time_start);

    //Run FFT
    fft_2d(buf, i, i*i);

    //Stop Timer and calculate difference
    clock_gettime(CLOCK_REALTIME, &time_stop);

    //Calculate timing difference
    time_stamp[iters] = interval(time_start, time_stop);

    //Reset the buffer to its previous values
    memcpy(buf, buf_orig, i * i * sizeof(cplx));
  }  

  //Print Timing
  printf("Done collecting measurements.\n\n");
  iters = 0;
  for(long int i = BASE; iters < ITERS; i+=DELTA, iters++)
  {
    printf("%ld, %g\n", i,  (double)time_stamp[iters]);
    
    //printf("%ld, %g\n", i,  (((double)time_stamp[iters].tv_sec) + \
                            ((double)time_stamp[iters].tv_sec)*1.0e-9));
  }
 
	return 0;
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
