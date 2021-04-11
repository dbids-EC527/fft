/*	Devin Bidstrup 03/29/21-05/07/21
 
	Must compile with:
	gcc fft_2d_image.c ./image_utils/Image.c -o fft_2d_image -lm -std=c99
	
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

//Image utilities
#include "image_utils/Image.h"
#include "image_utils/utils.h"
/*#define STB_IMAGE_IMPLEMENTATION
#include "image_utils/stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "image_utils/stb_image/stb_image_write.h"*/
 
#define PI 3.1415926535897932384
typedef double complex cplx;

//Prototypes
void show_buffer(cplx buf[], int rowLen, int n);
void transpose(cplx buf[], int rowLen);
void fft(cplx buf[], int n);
void fft_2d(cplx buf[], int rowLen, int n);

int main()
{	
	//Load image
	Image img, img_after;
	Image_load(&img, "sky.jpg");
    ON_ERROR_EXIT(img.data == NULL, "Error in loading the image");
    
    //Check for squareness
    if(img.width != img.height)
    {
    	printf("image is not square, width %d, height %d\n", img.width, img.height);
    	exit(EXIT_FAILURE);
    }
	
	//Copy image to complex buffer
	int i;
	int rowLen = img.width;
	int n = rowLen * rowLen;
	cplx* buf = (cplx*) calloc(n, sizeof(cplx));
	for (i = 0; i < n; i++)
		buf[i] = (int) img.data[i];
		
	/*int rowLen, colLen, channels;	
	size_t size;
	const char *fname = "sky.jpg";
	//cplx* buf = (cplx*) calloc(n, sizeof(cplx));
	cplx* buf = stbi_load(fname, &rowLen, &colLen, &channels, 0);
	if (rowLen != colLen)
	{
    	printf("image is not square, width %d, height %d\n", rowLen, colLen);
    	exit(EXIT_FAILURE);
    }
    int n = rowLen * rowLen;*/
	
	//Run FFT
	fft_2d(buf, rowLen, n);
	
	//Copy image from complex buffer
	Image_create(&img_after, img.width, img.height, img.channels, false);
    ON_ERROR_EXIT(img_after.data == NULL, "Error in creating the image");
	for (i = 0; i < n; i++)
		img_after.data[i] = buf[i];
	
	Image_save(&img_after, "sky_after.jpg");
	Image_free(&img);
	Image_free(&img_after);
 
	return 0;
}

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
		//show_buffer(buf, rowLen, n);
	}

	// Transpose the matrix
	transpose(buf, rowLen);
	//show_buffer(buf, rowLen, n);

	// Do columns
	for(i = 0; i < n; i += rowLen)
	{
		fft(buf+i, rowLen);
		//show_buffer(buf, rowLen, n);
	}

	// Transpose back
	transpose(buf, rowLen);
}

//Print the complex arrays before and after FFT
void show_buffer(cplx buf[], int rowLen, int n) {
	for (int i = 0; i < n; i++)
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
