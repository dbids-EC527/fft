/*	Devin Bidstrup 03/29/21-05/07/21
 
	Must compile with:
	gcc fft.c -o fft -lm -std=c99
	
	Code based on that found here for the C language:
	https://rosettacode.org/wiki/Fast_Fourier_transform#C

*/

#include <stdio.h>
#include <math.h>
#include <complex.h>
 
#define PI 3.1415926535897932384
typedef double complex cplx;

// Recursion function
void _fft(cplx buf[], cplx out[], int n, int step)
{
	//Recur log2(n) times
	if (step < n) {
		
		//Recur Left
		_fft(out, buf, n, step * 2);
		
		//Recure Right
		_fft(out + step, buf + step, n, step * 2);
		
		//Do DFT on current section of array
		for (int i = 0; i < n; i += 2 * step) {
			cplx t = cexp(-I * (double)PI * i / n) * out[i + step];
			buf[i / 2]     = out[i] + t;
			buf[(i + n)/2] = out[i] - t;
		}
	}
}

// First iteration of the recursive algorithm
void fft(cplx buf[], int n)
{
	//Make a copy of the buffer in out
	cplx out[n];
	for (int i = 0; i < n; i++) 
		out[i] = buf[i];
	
	//Begin the recursion
	_fft(buf, out, n, 1);
}
 
//Print the complex arrays before and after FFT
void show_buffer(cplx buf[]) {
	for (int i = 0; i < 8; i++)
		if (!cimag(buf[i]))
			printf("%g ", creal(buf[i]));
		else
			printf("(%g,%g) ", creal(buf[i]), cimag(buf[i]));
}
 
int main()
{
	//Define and print the buffer before
	cplx buf[] = {0, 0, 0, 0, 1, 1, 0, 0};
	printf("Data: ");
	show_buffer(buf);
	
	//Run FFT
	fft(buf, 8);
	
	//Print buffer after
	printf("\nFFT : ");
	show_buffer(buf);
	printf("\n");
 
	return 0;
}
