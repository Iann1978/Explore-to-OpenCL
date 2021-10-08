
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "RayTracing.h"


#define DIM 512

#define INF 2e10f
#define SPHERES 20
#define rnd(x) (x *rand() / RAND_MAX)
struct Sphere {
    float	r, g, b;
    float	radius;
    float	x, y, z;
    __device__ float hit(float ox, float oy, float* n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / radius;
            return dz + z;
        }
        return -INF;
    }
};

__global__ void kernel(float* ptr, int ticks, Sphere* s)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float ox = x - DIM / 2;
    float oy = y - DIM / 2;

    float grey = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t > maxz)
        {
            grey = n;
            maxz = t;
        }
    }


   // float d = sqrtf(ox * ox + oy * oy);
   // float grey = 0.5 + 0.5 * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f);
    ptr[offset] = grey;
}

static Sphere* temp_s = nullptr;
bool launch_kernel(float* pos, int width, int height)
{

    int blockWidth = width < 32 ? width : 32;
    int blockHeight = height < 32 ? height : 32;
    int gridWidth = width / blockWidth;
    int gridHeight = height / blockHeight;

    static int ticks = 0;


    Sphere* s = nullptr;
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed when loading launch_kernel: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

 //   cudaStatus = cudaMalloc()
    if (temp_s == nullptr)
    {
        temp_s = new Sphere[SPHERES];
        for (int i = 0; i < SPHERES; i++)
        {
            float range = DIM;
            temp_s[i].r = rnd(1.0f);
            temp_s[i].g = rnd(1.0f);
            temp_s[i].b = rnd(1.0f);
            temp_s[i].x = rnd(range) - range / 2;
            temp_s[i].y = rnd(range) - range / 2;
            temp_s[i].z = rnd(range) - range / 2;
            temp_s[i].radius = rnd(60.0f) + 20;
        }
    }
  
    cudaStatus = cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc((void**)&dev_s, sizeof(Sphere) * SPHERES) failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy(dev_s, s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice) failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    //delete temp_s;
    //temp_s = nullptr;

    // execute the kernel
    dim3 grid(DIM/16, DIM/16, 1);
    dim3 block(16, 16, 1);
    kernel << < grid, block >> > (pos, ticks++,s);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaFree(s);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(cudaStatus));
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
