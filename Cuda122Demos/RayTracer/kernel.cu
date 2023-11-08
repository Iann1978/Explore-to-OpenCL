
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "vec3.h"
#include "ray.h"
#include "color.h"
#include "sphere.h"

__global__ void moveleftKenel(float* vert)
{
    int idx = threadIdx.x;
    vert[idx*3] -= 0.02;
}

__device__ ray getRay(int row, int col, int img_width, int img_height)
{
    vec3 origin(0.0, 0.0, 0.0);
    vec3 top_left(-1.0, 1.0, -1.0);
    vec3 center(0.0, 0.0, -1.0);
    vec3 horizontal(2.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 drow = horizontal / img_width;
    vec3 dcol = -vertical / img_height;
    vec3 start_point = top_left + drow * 0.5 + dcol * 0.5;
    ray r(origin, start_point + drow * col + dcol * row);
// 
    //return r;
    
    //ray r;
    return r;


    

}
__device__ bool hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = r.origin() - center;
    auto a = dot(r.direction(), r.direction());
    auto b = 2.0 * dot(oc, r.direction());
    auto c = dot(oc, oc) - radius * radius;
    auto discriminant = b * b - 4 * a * c;
    return (discriminant >= 0);
}

__device__ color ray_color(const ray& r, const sphere& world) {
    //if (hit_sphere(point3(0, 0, -1), 0.5, r))
        //return color(1, 0, 0);
    // 
    if (hit_sphere(world.center, world.radius, r))
        return color(1, 0, 0);
  //  hit_record rec;
  //  if (world->hit(r, 0, 10000, rec))
		//return color(1, 0, 0);
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}


__constant__ sphere dev_world;

__global__ void renderKenel(unsigned char* dev_imgdata, int  img_width, int img_height, int img_channels)
{
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    ray ry = getRay(row, col, img_width, img_height);

    //sphere s(point3(0, 0, -1), 0.5);

    color c = ray_color(ry, dev_world);
    

    //double r = double(row) / double(img_height);
    //double g = double(col) / double(img_width);
    //double r = ry.direction().x()+1.0/2.0;
    //double g = ry.direction().y()+1.0/2.0;
    //double b = 0.0;
    double r = c.x();
    double g = c.y();
    double b = c.z();

    unsigned char ir = (unsigned char)(255.99 * r);
    unsigned char ig = (unsigned char)(255.99 * g);
    unsigned char ib = (unsigned char)(255.99 * b);


    int idx = row * img_width + col;
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

bool render(unsigned char* dev_imgdata, int  img_width, int img_height, int img_channels, sphere* world)
{
    cudaError_t cudaStatus = cudaSuccess;


    cudaStatus = cudaMemcpyToSymbol( dev_world, world, sizeof(sphere));

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

