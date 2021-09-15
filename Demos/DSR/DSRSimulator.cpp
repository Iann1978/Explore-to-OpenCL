#include "DSRSimulator.h"
#include "cuda_runtime.h"
//#include "cutilSafeCall.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include <iostream>
#include "BufferExchanger.h"

//cudaGraphicsResource* vbo_res = nullptr;
float* dptr = 0;
size_t num_bytes = 0;
struct cudaGraphicsResource* cuda_pbo_resource;

DSRSimulator::DSRSimulator(BufferExchanger* changer)
    :changer(changer)
{
    
	std::cout << "DSRSimulator::DSRSimulator()" << std::endl;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, changer->pbo, cudaGraphicsMapFlagsWriteDiscard);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed!");
        goto Error;
    }
    return;
    Error:
    std::cout << "Error in DSRSimulator::DSRSimulator()" << std::endl;
}
bool launch_kernel(float* pos, int width, int height);
void DSRSimulator::runOnce()
{
    std::cout << "DSRSimulator::runOnce()" << std::endl;
    cudaError_t cudaStatus;
    cudaStatus = cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsMapResources failed!");
        goto Error;
    }


    cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_pbo_resource);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsResourceGetMappedPointer failed!");
        goto Error;
    }

    bool err = launch_kernel(dptr, changer->tex->width, changer->tex->height);


    cudaStatus = cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsUnmapResources failed!");
        goto Error;
    }

    int a = 0;
    int b = 0;
    return;
Error:
    std::cout << "Error in DSRSimulator::runOnce()" << std::endl;

}
//
//DSRSimulator::DSRSimulator(Texture* tex)
//{
//    std::cout << "DSRSimulator::DSRSimulator()" << std::endl;
//    cudaError_t cudaStatus;
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//   // cutilSafeCall
//    //cudaStatus = cudaGraphicsGLRegisterBuffer(&vbo_res,tex->texture, cudaGraphicsMapFlagsWriteDiscard);
//    cudaStatus = cudaGraphicsGLRegisterImage(&vbo_res, tex->texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
//    
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//
//    cudaStatus = cudaGraphicsMapResources(1, &vbo_res, 0);
//    //  checkCudaErrors(cudaStatus);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaGraphicsMapResources failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    float* dptr = 0;
//    size_t num_bytes = 0;
//    cudaArray* in_array;
//    cudaStatus = cudaGraphicsSubResourceGetMappedArray(&in_array, vbo_res, 0, 0);
//    //cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, vbo_res);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaGraphicsResourceGetMappedPointer failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    return;
//Error:
//    std::cout << "Error in DSRSimulator::DSRSimulator()" << std::endl;
//}
