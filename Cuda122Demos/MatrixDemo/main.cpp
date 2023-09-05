#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string.h>
cudaError_t mxadd(int* c, const int* a, const int* b, int w, int h);
cudaError_t mxmul(int* c, const int* a, const int* b, int m, int q, int n);
class Matrix {
public:
	int* data = 0;
	int w = 0;
	int h = 0;

	Matrix(int w, int h) {
		this->w = w;
		this->h = h;
		data = new int[w * h];
	}
	Matrix(int w, int h, int value) {
		this->w = w;
		this->h = h;
		data = new int[w * h];
		for (int i = 0; i < w * h; i++) {
			data[i] = value;
		}
	}

	Matrix(const Matrix& other) {
		this->w = other.w;
		this->h = other.h;
		data = new int[w * h];
		memcpy(data, other.data, sizeof(int) * w * h);

	}
	~Matrix() {
		delete[] data;
	}
};

Matrix operator + (const Matrix a, const Matrix b) {
	Matrix c(a.w,a.h);

	cudaError_t cudaStatus = mxadd(c.data, a.data, b.data, a.w, a.h);
	if (cudaStatus != cudaSuccess) {
		printf("error");

	}
	return c;
}

Matrix operator * (const Matrix a, const Matrix b) {
	Matrix c(b.w, a.h);

	cudaError_t cudaStatus = mxmul(c.data, a.data, b.data, a.h, a.w, b.w);
	if (cudaStatus != cudaSuccess) {
		printf("error");

	}
	return c;
}



int main() {
	printf("hello world!!!");
	Matrix a(3, 2, 1);
	Matrix b(4, 3, 2);
	Matrix c = a * b;

	//Matrix a(5, 6, 5);
	//Matrix b(5, 6, 6);

	//Matrix c = a + b;

	return 0;
}