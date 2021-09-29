
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

struct cuComplex {
    float r;
    float i;
    
    __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2(void) 
    {
        return r * r + i * i;
    }

    __device__ cuComplex operator * (const cuComplex& a) 
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__ cuComplex operator + (const cuComplex& a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

#define DIM 512
__device__ int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

__global__ void kernel(float* ptr)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;
    int juliaValue = julia(x, y);
    ptr[offset] = juliaValue;
}


bool launch_kernel(float* pos, int width, int height)
{

    int blockWidth = width < 32 ? width : 32;
    int blockHeight = height < 32 ? height : 32;
    int gridWidth = width / blockWidth;
    int gridHeight = height / blockHeight;


    cudaError_t cudaStatus = cudaSuccess;
    // execute the kernel
    dim3 grid(width, height, 1);
    dim3 block(1, 1, 1);
    kernel << < grid, block >> > (pos);


    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }


    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    return true;
Error:
    return false;
}
