
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>



#include <iostream>

__global__ void simulateKernel(float* dst, float* src)
{
    int row = threadIdx.x;
    int col = threadIdx.y;
    
    float value0 = src[row * 10 + col];
    float dh = 0;
    for (int drow = -1; drow <= 1; drow++)
    {
        for (int dcol = -1; dcol <= 1; dcol++)
        {
            int index = (row + drow) * 10 + col + dcol;
            if (index < 0)
                continue;
            if (index >= 10 * 10)
                continue;

            float value1 = src[index];

            if (value1 > value0) dh +=1;
            if (value1 < value0) dh -=1;

        }
    }
    dst[row*10+col] = src[row*10+col] + dh;
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
    simulateKernel<<<1, block>>> ((float*)dev_dst, (float*)dev_src);

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

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(dst , dev_dst, 10 * 10 * sizeof(float), cudaMemcpyDeviceToHost);
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


void flowtest()
{
    float src[10][10];
    for (int row = 0; row < 10; row++)
    {
        for (int col = 0; col < 10; col++)
        {
            src[row][col] = 5;
        }
    }

    float dst[10][10];
    for (int row = 0; row < 10; row++)
    {
        for (int col = 0; col < 10; col++)
        {
            dst[row][col] = 0;
        }
    }
    src[5][5] = 10;

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

}
int main()
{

    flowtest();

  

    return 0;
}

