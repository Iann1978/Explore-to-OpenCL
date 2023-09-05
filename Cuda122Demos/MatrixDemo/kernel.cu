
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void mxaddKenel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	c[i + j * blockDim.x] = a[i + j * blockDim.x] + b[i + j * blockDim.x];
}

// a(m*q) * b(q*n) = c(m*n)
__global__ void mxmulKenel(int* c, const int* a, const int* b, int q) {


	int m = blockDim.y;
	int n = blockDim.x;
	int i = threadIdx.x;
	int j = threadIdx.y;
	int sum = 0;
	for (int k = 0; k < q; k++) {
		sum += a[k + j * q] * b[i + k * n];
	}
	c[i + j * n] = sum;
}
// a(m*q) * b(q*n) = c(m*n)
cudaError_t mxmul(int* c, const int* a, const int* b, int m, int q, int n) {

	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	cudaStatus = cudaMalloc((void**)&dev_c, m * n * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_a, m * q * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_b, q * n * sizeof(int));

	cudaStatus = cudaMemcpy(dev_a, a, m * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_b, b, q * n * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimBlock(n, m);
	mxmulKenel << <1, dimBlock >> >(dev_c, dev_a, dev_b, q);

	cudaStatus = cudaGetLastError();
	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(c, dev_c, m * n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

cudaError_t mxadd(int* c, const int* a, const int* b, int w, int h) {

	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	cudaStatus = cudaMalloc((void**)&dev_c, w * h * sizeof(int));	
	cudaStatus = cudaMalloc((void**)&dev_a, w * h * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_b, w * h * sizeof(int));

	cudaStatus = cudaMemcpy(dev_a, a, w * h * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_b, b, w * h * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimBlock(w, h);
	mxaddKenel << <1, dimBlock >> >(dev_c, dev_a, dev_b);

	cudaStatus = cudaGetLastError();
	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(c, dev_c, w * h * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}