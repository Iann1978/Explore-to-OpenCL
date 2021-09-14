
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void moveLeft(float* vert)
{
    vert[0] = 0.2;
}

bool launch_kernel(float* pos)
{

    cudaError_t cudaStatus = cudaSuccess;
    // execute the kernel
    dim3 block(1, 1, 1);
    dim3 grid(1, 1, 1);
    moveLeft << < grid, block >> > (pos);


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
