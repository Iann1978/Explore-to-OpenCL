#include "DSRSimulator.h"
#include "cuda_runtime.h"
//#include "cutilSafeCall.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include <iostream>

cudaGraphicsResource* vbo_res = nullptr;
float* dptr = 0;
size_t num_bytes = 0;

DSRSimulator::DSRSimulator(Texture* tex)
{
    std::cout << "DSRSimulator::DSRSimulator()" << std::endl;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
   // cutilSafeCall
    //cudaStatus = cudaGraphicsGLRegisterBuffer(&vbo_res,tex->texture, cudaGraphicsMapFlagsWriteDiscard);
    cudaStatus = cudaGraphicsGLRegisterImage(&vbo_res, tex->texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }


    cudaStatus = cudaGraphicsMapResources(1, &vbo_res, 0);
    //  checkCudaErrors(cudaStatus);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsMapResources failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    float* dptr = 0;
    size_t num_bytes = 0;
    cudaArray* in_array;
    cudaStatus = cudaGraphicsSubResourceGetMappedArray(&in_array, vbo_res, 0, 0);
    //cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, vbo_res);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsResourceGetMappedPointer failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    return;
Error:
    std::cout << "Error in DSRSimulator::DSRSimulator()" << std::endl;
}
