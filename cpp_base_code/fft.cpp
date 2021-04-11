#include <vector>
#include <complex>
#include <cmath>
#include <iostream>

using namespace std;
using cd = complex<double>;
const double PI = acos(-1);

//Print the complex arrays before and after FFT
void show_buffer(vector<cd> & buf, int n) {
	for (int i = 0; i < n; i++)
		if (!imag(buf[i]))
      cout << real(buf[i]) << " ";
		else
      cout << "(" << real(buf[i]) << "," << imag(buf[i]) << ") ";
	cout << endl;
}

void fft(vector<cd> & a, bool invert) {
    int n = a.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            //cout << "w:" << "(" << real(w) << "," << imag(w) << ") " << endl;
            for (int j = 0; j < (len / 2); j++) 
            {
                cd u = a[i+j];
                //cout << "u:" << "(" << real(w) << "," << imag(w) << ") " << endl;
                cd v = a[i+j+len/2] * w;
                //cout << "v:" << "(" << real(w) << "," << imag(w) << ") " << endl;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;

                // printf("len is %d\ti + j is %d\t", len, i+j);
                // printf("w is (%g,%g)\t", real(w), imag(w));
                // printf("wlen is (%g,%g)\n", real(w), imag(w));
            }
        }
    }

    if (invert) {
        for (cd & x : a)
            x /= n;
    }
}

int main()
{
  //Define and print the buffer before
	vector<cd> buf = {{-2, 2}, {-2, 2}, {-2, 2}, {-2, 2}};
	int n = 4;
	cout << "Data: ";
	show_buffer(buf, n);
	
	//Run FFT
	fft(buf, 0);
	
	//Print buffer after
	cout << "FFT result: ";
	show_buffer(buf, n);
	cout << endl;
 
	return 0;
}