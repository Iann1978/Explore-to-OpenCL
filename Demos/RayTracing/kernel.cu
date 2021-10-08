
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "RayTracing.h"


#define DIM 512

//#define INF 2e10f
//struct Sphere {
//    float	r, g, b;
//    float	radius;
//    float	x, y, z;
//    float hit(float ox, float oy, float* n) {
//        float dx = ox - x;
//        float dy = oy - y;
//        if (dx * dx + dy * dy < radius * radius) {
//            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
//            *n = dz / radius;
//            return dz + z;
//        }
//        return -INF;
//    }
//};

__global__ void kernel(float* ptr, int ticks)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float fx = x - DIM / 2;
    float fy = y - DIM / 2;
    float d = sqrtf(fx * fx + fy * fy);
    float grey = 0.5 + 0.5 * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f);
    ptr[offset] = grey;
}

#

bool launch_kernel(float* pos, int width, int height)
{

    int blockWidth = width < 32 ? width : 32;
    int blockHeight = height < 32 ? height : 32;
    int gridWidth = width / blockWidth;
    int gridHeight = height / blockHeight;

    static int ticks = 0;

    cudaError_t cudaStatus = cudaSuccess;


    // execute the kernel
    dim3 grid(DIM/16, DIM/16, 1);
    dim3 block(16, 16, 1);
    kernel << < grid, block >> > (pos, ticks++);


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
