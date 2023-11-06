
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void moveleftKenel(float* vert)
{
    int idx = threadIdx.x;
    vert[idx*3] -= 0.02;
}

__global__ void renderKenel(unsigned char* dev_imgdata, int  img_width, int img_height, int img_channels)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * img_width + col;

    double r = double(row) / double(img_height);
    double g = double(col) / double(img_width);
    double b = 0.0;

    unsigned char ir = (unsigned char)(255.99 * r);
    unsigned char ig = (unsigned char)(255.99 * g);
    unsigned char ib = (unsigned char)(255.99 * b);

    dev_imgdata[idx * img_channels] = ir; // red
    dev_imgdata[idx * img_channels + 1] = ig; // green
	dev_imgdata[idx * img_channels + 2] = ib; // blue

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

bool render(unsigned char* dev_imgdata, int  img_width, int img_height, int img_channels)
{
    cudaError_t cudaStatus = cudaSuccess;
    // execute the kernel
    dim3 block(16, 16, 1);
    dim3 grid(32, 32, 1);
    renderKenel << < grid, block >> > (dev_imgdata, img_width, img_height, img_channels);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return false;
	}

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching renderKenel!\n", cudaStatus);
        return false;
    }

    return true;


    return false;
}

