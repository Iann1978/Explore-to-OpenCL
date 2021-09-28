
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>



__global__ void Add0001(float* vert, int width, int height)
{
    if (blockIdx.x >= width)
        return;
    if (blockIdx.y >= height)
        return;

    int idx = blockIdx.y * gridDim.x + blockIdx.x;
    //if (idx < 100)
    {
        vert[idx] = vert[idx] + 0.001 * blockIdx.x;
    }

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
    Add0001 << < grid, block >> > (pos, width, height);


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
