#include <complex.h>

#ifndef FFT_2D_H
#define FFT_2D_H

typedef double complex cplx;

void show_buffer(cplx buf[], int rowLen, int n);
void transpose(cplx buf[], int rowLen);
void fft(cplx buf[], int n);
void fft_2d(cplx buf[], int rowLen, int n);

#endif