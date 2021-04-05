/*	Devin Bidstrup 03/29/21-05/07/21
 
	Must compile with:
	gcc fft.c -o fft -lm -std=c99
	
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
 
#define PI 3.1415926535897932384
typedef double complex cplx;

//Prototypes
void show_buffer(cplx buf[]);

/* Performs in place FFT on buf of size n*/
void fft(cplx buf[], int n) 
{
	//Rearrange the array such that it can be iterated upon in the correct order
	for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

		//swap(buf[i], buf[j]);
		int temp;
        if (i < j)
		{
			temp = buf[i];
			buf[i] = buf[j];
			buf[j] = temp;
		}
    }
	//show_buffer(buf);

	/*Compute the FFT for the array*/
	cplx wlen, w, u, v;
	// len goes 2, 4, ... n/2, n
	// len iterates over the array log2(n) times
    for (int len = 2; len <= n; len <<= 1) 
	{
        double ang = 2 * PI / len;
        wlen = cexp(-I * ang);

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
        for (int i = 0; i < n; i += len) 
		{
            w = 1;
            for (int j = 0; j < len / 2; j++) 
			{
				//Compute the DFT on the correct elements
                u = buf[i+j];
				v = buf[i+j+len/2] * w;
                buf[i+j] = u + v;
                buf[i+j+len/2] = u - v;
                w *= wlen;
				
				//printf("len is %d\ti + j is %d\t", len, i+j);
				//printf("w is (%g,%g)\t", creal(w), cimag(w));
				//printf("wlen is (%g,%g)\n", creal(w), cimag(w));
            }
        }
    }
}
 
int main()
{
	//Define and print the buffer before
	cplx buf[] = {0, 1, 1, 0, 0, 0, 0, 0};
	int n = 8;
	printf("Data: ");
	show_buffer(buf);
	
	//Run FFT
	fft(buf, n);
	
	//Print buffer after
	printf("FFT result: ");
	show_buffer(buf);
	printf("\n");
 
	return 0;
}

//Print the complex arrays before and after FFT
void show_buffer(cplx buf[]) {
	for (int i = 0; i < 8; i++)
		if (!cimag(buf[i]))
			printf("%g ", creal(buf[i]));
		else
			printf("(%g,%g) ", creal(buf[i]), cimag(buf[i]));
	printf("\n");
}
