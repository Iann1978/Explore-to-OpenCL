// https://zhuanlan.zhihu.com/p/34587739
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


void GetCudaInfo() {
	int dev = 0;
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, dev);
	printf("Using Device %d: %s\n", dev, devProp.name);
	printf("Compute capability: %d.%d\n", devProp.major, devProp.minor);
	printf("Clock rate: %d\n", devProp.clockRate);
	printf("SM count: %d\n", devProp.multiProcessorCount);
	printf("Shared Memory: %d\n", devProp.sharedMemPerBlock);
	printf("Max Threads per Block: %d\n", devProp.maxThreadsPerBlock);
	printf("Max Threads per MultiProcessor: %d\n", devProp.maxThreadsPerMultiProcessor);

	printf("Device copy overlap: ");
	if (devProp.deviceOverlap)
		printf("Enabled\n");
	else
		printf("Disabled\n");


}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKenel(int* c, const int* a, const int* b) {
	int idx = threadIdx.x;
	c[idx] = a[idx] + b[idx];
}


cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size) {
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice(0) error");
		goto Error;
	}

	cudaStatus = cudaMalloc(&dev_a, sizeof(int) * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_a error");
		goto Error;
	}

	cudaStatus = cudaMalloc(&dev_b, sizeof(int) * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_b error");
		goto Error;
	}

	cudaStatus = cudaMalloc(&dev_c, sizeof(int) * size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_c error");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, a, sizeof(int) * size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_a error");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, sizeof(int) * size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_b error");
		goto Error;
	}

	addKenel<<<1, size>>>(dev_c, dev_a, dev_b);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKenel error");
		goto Error;
	}

	cudaStatus = cudaMemcpy(c, dev_c, sizeof(int) * size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy c error");
		goto Error;
	}

Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return cudaStatus;
}


int main() {
	GetCudaInfo();

	int a[] = { 1,2,3,4,5 };
	int b[] = { 2,3,4,5,6 };
	int c[] = { 0,0,0,0,0 };

	cudaError_t cudaStatus = addWithCuda(c, a, b, 5);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	printf("{1,2,3,4,5} + {2,3,4,5,6} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return;
}

