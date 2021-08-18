
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>



#include <iostream>

__global__ void simulateKernel(float* dst, const float* src)
{
    int row = threadIdx.x;
    int col = threadIdx.y;
    
    float centerHeight = src[row * 10 + col];
    float centerDeitaH = 0;
    for (int drow = -1; drow <= 1; drow++)
    {
        for (int dcol = -1; dcol <= 1; dcol++)
        {
            int neighborIdx = (row + drow) * 10 + col + dcol;
            if (neighborIdx < 0)
                continue;
            if (neighborIdx >= 10 * 10)
                continue;

            float neighborHeight = src[neighborIdx];
            float deitaH = neighborHeight - centerHeight;
            centerDeitaH += deitaH / 8.0;
        }
    }
    dst[row*10+col] = src[row*10+col] + centerDeitaH;
}

cudaError_t simulate(float src[10][10], float dst[10][10])
{
    void* dev_src = 0;
    void* dev_dst = 0;
    cudaError_t cudaStatus{};

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc(&dev_src, 10 * 10 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&dev_dst, 10 * 10 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_src, src, 10 * 10 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    dim3 block(10, 10);
    
    /* First time */    {
        simulateKernel << <1, block >> > ((float*)dev_dst, (float*)dev_src);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "simulateKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching simulateKernel!\n", cudaStatus);
            goto Error;
        }
    }
   
     
    /* Second time */ {
        simulateKernel << <1, block >> > ((float*)dev_src, (float*)dev_dst);
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "simulateKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching simulateKernel!\n", cudaStatus);
            goto Error;
        }
    }

    /* Third time */ {
        simulateKernel << <1, block >> > ((float*)dev_dst, (float*)dev_src);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "simulateKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching simulateKernel!\n", cudaStatus);
            goto Error;
        }
    }


    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(dst , dev_dst, 10 * 10 * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(src, dev_src, 10 * 10 * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    return cudaStatus;
Error:
    cudaFree(dev_src);
    cudaFree(dev_dst);
    return cudaStatus;
}

#include <algorithm>
void flowtest()
{
    // Initialize src
    float src[10][10];
    for (int row = 0; row < 10; row++)
    {
        for (int col = 0; col < 10; col++)
        {
            src[row][col] = 5;
        }
    }
    src[5][5] = 10;

    // Initialzie dst
    float dst[10][10];
    for (int row = 0; row < 10; row++)
    {
        for (int col = 0; col < 10; col++)
        {
            dst[row][col] = 0;
        }
    }

    // Calculate the summary of src
    float sumSrc = 0;
    for (int row = 0; row < 10; row++)
    {
        for (int col = 0; col < 10; col++)
        {
            sumSrc += src[row][col];
        }
    }


    std::cout << "before simulate " << std::endl;
    std::cout << "src: " << std::endl;
    for (int row = 0; row < 10; row++)
    {
        for (int col = 0; col < 10; col++)
        {
            std::cout << src[row][col] << ",";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "dst: " << std::endl;
    for (int row = 0; row < 10; row++)
    {
        for (int col = 0; col < 10; col++)
        {
            std::cout << dst[row][col] << ",";
        }
        std::cout << std::endl;
    }

    simulate(src, dst);
    std::cout << std::endl;
    std::cout << std::endl;


    std::cout << "after simulate " << std::endl;
    std::cout << "src: " << std::endl;
    for (int row = 0; row < 10; row++)
    {
        for (int col = 0; col < 10; col++)
        {
            std::cout << src[row][col] << ",";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "dst: " << std::endl;
    for (int row = 0; row < 10; row++)
    {
        for (int col = 0; col < 10; col++)
        {
            std::cout << dst[row][col] << ",";
        }
        std::cout << std::endl;
    }

    // Calculate summary of dst
    float sumDst = 0;
    for (int row = 0; row < 10; row++)
    {
        for (int col = 0; col < 10; col++)
        {
            sumDst += dst[row][col];
        }
    }

    std::cout << "sumSrc:" << sumSrc << std::endl;
    std::cout << "sumDst:" << sumDst << std::endl;
    if (sumSrc == sumDst)
    {
        std::cout << "Pass sum check!!" << std::endl;
    }
    else
    {
        std::cout << "Not pass sum check!!" << std::endl;
    }
}
int main()
{

    flowtest();

  

    return 0;
}

