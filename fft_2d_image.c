/*	Devin Bidstrup 03/29/21-05/07/21
 
	Must compile with:
	gcc fft_2d_image.c ./image_utils/Image.c -o fft_2d_image -lm -std=c99

  And run with:
  ./fft_2d_image "image_name.jpg"
	
	Code based on a number of sources:
	*	C language with recursion:
		https://rosettacode.org/wiki/Fast_Fourier_transform#C
	*	C++ in place algorithm:
		https://cp-algorithms.com/algebra/fft.html
	*	Helpful visualization of the algorithm:
		https://towardsdatascience.com/fast-fourier-transform-937926e591cb
  *	Image input/output library:
    https://solarianprogrammer.com/2019/06/10/c-programming-reading-writing-images-stb_image-libraries/
  * Logorithmic Scaling for images and test FFT images:
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/pixlog.htm
*/

#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <string.h>

//Image utilities
#include "image_utils/Image.h"
#include "image_utils/utils.h"
 
#define PI 3.1415926535897932384
typedef double complex cplx;

//Prototypes
void show_buffer(cplx buf[], int rowLen, int n);
void transpose(cplx buf[], int rowLen);
void fft(cplx buf[], int n);
void fft_2d(cplx buf[], int rowLen, int n);

int main(int argc, char *argv[])
{	
	//Check if image name has been passed on the command line
  if( argc != 2 ) 
  {
    fprintf(stderr, "ERROR: one input argument expected.\n");
    exit(EXIT_FAILURE);
  }
    
  //Load image
	Image img_rgb, img, img_after;
	Image_load(&img_rgb, argv[1]);
  ON_ERROR_EXIT(img_rgb.data == NULL, "ERROR: in loading the image");
    
  //Check for squareness
  if(img_rgb.width != img_rgb.height)
  {
    fprintf(stderr, "ERROR: image is not square, width %d, height %d\n", img_rgb.width, img_rgb.height);
    exit(EXIT_FAILURE);
  }

  //Check that length is a power of 2
	int rowLen = img_rgb.width;
  int n = rowLen * rowLen;
  if (fmod(log(rowLen) / log(2), 1))
  {
    fprintf(stderr, "ERROR: image dimensions not a power of 2, width %d, height %d\n", img_rgb.width, img_rgb.height);
    exit(EXIT_FAILURE);
  }

  //Grayscale the image if needed
  if(img_rgb.channels >= 3)
  {
    Image_to_gray(&img_rgb, &img);
  }
  else
  {
    Image_load(&img, argv[1]);
  }
	
  //Copy image to complex buffer
	int i;
	cplx* buf = (cplx*) calloc(n, sizeof(cplx));
	for (i = 0; i < n; i++)
  {
    buf[i] = img.data[i];
  }
	
	//Run FFT
	fft_2d(buf, rowLen, n);

  //Due to Dynamic Range of Image, perform a logarithmic transformation on the data
  double max_val = 0;
  double magnitude;
  for (i = 0; i < n; i++)
  {
    magnitude = cabs(buf[i]);
    max_val = (magnitude > max_val) ? magnitude : max_val;
  }
  printf("max val is %g\n", max_val);
  double c = 255.0 / (log(1 + abs(max_val)));
  printf("after transformation max val is %g\n", c * log(1 + max_val));
	
	//Copy image from complex buffer
	Image_create(&img_after, img.width, img.height, img.channels, false);
  ON_ERROR_EXIT(img_after.data == NULL, "ERROR: in creating the image");
	for (i = 0; i < n; i++)
		img_after.data[i] = c * log(1 + cabs(buf[i])); //the log transformation is actually performed here
	
  //Save the image with _after appended
  char* image_save_name = (char*) malloc(sizeof(strlen(argv[1])) + 7);
  char* file_name = strtok(argv[1], ".");
  char* extension = strtok(NULL, "\n");
  sprintf(&image_save_name[0], "%s_after.%s", file_name, extension);
  Image_save(&img_after, image_save_name);
  ON_ERROR_EXIT(img_after.data == NULL, "ERROR: in saving the image");
  printf("FFT has been saved as %s\n", image_save_name);
  free(image_save_name);

  //Free image buffers
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
            for (int j = 0; j < (len / 2); j++) 
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
