#include <complex.h>
#include <math.h>
typedef double complex cplx;
void fft(cplx buf[], int n) 
{
  /*Do Bit Reversal of buf[]
  ..
  ..
  */

  /*Compute the FFT for the array*/
	// len goes 2, 4, ... n/2, n
  cplx w, u, v;
  for (int len = 2; len <= n; len <<= 1) 
  {
    double ang = 2 * M_PI / len;
    // n/2 iterations of i and j
		for (int i = 0; i < n; i += len) 
		{
			for (int j = 0; j < (len / 2); j++) 
			{
				//Compute the DFT on the correct elements
				w = cexp(I * ang * j);
        u = buf[i+j];
				v = buf[i+j+(len/2)] * w;
				buf[i+j] = u + v;
				buf[i+j+(len/2)] = u - v;
			}
		}
  }
}