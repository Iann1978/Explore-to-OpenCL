
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void moveleftKenel(float* vert)
{
    int idx = threadIdx.x;
    vert[idx*3] -= 0.02;
}


bool moveleft(float* vert)
{

    cudaError_t cudaStatus = cudaSuccess;
    // execute the kernel
    dim3 block(3, 1, 1);
    dim3 grid(1, 1, 1);
    moveleftKenel << < grid, block >> > (vert);


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

