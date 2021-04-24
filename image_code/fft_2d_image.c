/*	Devin Bidstrup 03/29/21-05/07/21
 
	Must compile with:
	gcc fft_2d_image.c ./image_utils/Image.c ../base_code/fft_2d.c -o fft_2d_image -lm

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

//My own fft2d code
#include "../base_code/fft_2d.h"
 
#define PI 3.1415926535897932384
typedef double complex cplx;

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
