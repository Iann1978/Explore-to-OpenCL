#include <iostream>
#include <cuda_runtime.h>
// Define a user-defined class
class MyClass {
public:
    int data;
    __device__ void printData() {
        printf("Data from constant memory: %d\n", data);
    }
};

__constant__ MyClass constantObject;  // Declare a constant object in device memory

__global__ void useConstantMemory() {
    constantObject.printData();  // Access the constant object within the kernel
}

int main() {
    MyClass hostObject;
    hostObject.data = 42;  // Set the data for the constant object
    cudaError_t cudaStatus = cudaSuccess;
    // Copy data from host to constant memory
    cudaStatus = cudaMemcpyToSymbol(constantObject, &hostObject, sizeof(MyClass));

    // Launch the kernel
    useConstantMemory << <1, 1 >> > ();

    // Synchronize to ensure the kernel finishes
    cudaDeviceSynchronize();

    return 0;
}