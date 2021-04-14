/*	Devin Bidstrup 03/29/21-05/07/21
 
	Must compile with:
	gcc fft_2d_timing.c ./base_code/fft_2d.c -o fft_2d_timing -lm

  And run with:
  ./fft_2d_image "image_name.jpg"
	
	Code based on a number of sources:
	*	C language with recursion:
		https://rosettacode.org/wiki/Fast_Fourier_transform#C
	*	C++ in place algorithm:
		https://cp-algorithms.com/algebra/fft.html
	*	Helpful visualization of the algorithm:
		https://towardsdatascience.com/fast-fourier-transform-937926e591cb
*/

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

#define MINVAL       0.0
#define MAXVAL       10.0
#define ROWLEN       1024

//Prototypes
double interval(struct timespec start, struct timespec end);
void initializeArray(cplx *arr, int len, int seed);


int main(int argc, char *argv[])
{	
  //Serial Timing variables:
  struct timespec time_start, time_stop;
  
  //Define and print the buffer before.
	//Make buffer square and of a pow2 size
	//cplx buf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
	int rowLen = ROWLEN;
  int n = ROWLEN * ROWLEN;
  cplx* buf = (cplx*) calloc(n, sizeof(cplx));
  initializeArray(buf, n, 2453);

	//printf("Data: ");
	//show_buffer(buf, rowLen, n);
	
  //Start Timer
  clock_gettime(CLOCK_REALTIME, &time_start);

	//Run FFT
	fft_2d(buf, rowLen, n);

  //Stop Timer and calculate difference
  clock_gettime(CLOCK_REALTIME, &time_stop);
	
	//Print buffer after
	//printf("FFT result: ");
	//show_buffer(buf, rowLen, n);

  //Print Timing
  double time_spent = interval(time_start, time_stop);
  printf("FFT_serial() took %f seconds\n", time_spent);
 
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