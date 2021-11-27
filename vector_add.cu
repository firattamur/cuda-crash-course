#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    // calculate global thread id (tid)
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // vector boundary check
    if (tid < n) {
        
        // each thread add single element
        c[tid] = a[tid] + b[tid];

    }

}

// initialize vector of size n
void vector_init(int* a, int n) {

    for (int i = 0; i < n; i++) {
        a[i] = rand() % 100;
    }

}

// check vector add function result
void check_error(int* a, int* b, int* c, int n) {

    for (int i=0; i < n; i++) {
        assert(c[i] == a[i] + b[i]);
    }

}

int main() {

    // vector size
    int n = 1 << 16;

    // host vector pointers
    int *host_a, *host_b, *host_c;

    // device(gpu) vector pointers
    int *gpu_a, *gpu_b, *gpu_c;

    // allocation size for each vector
    size_t bytes = sizeof(int) * n;
    
    // allocate host memory
    host_a = (int*) malloc(bytes);
    host_b = (int*) malloc(bytes);
    host_c = (int*) malloc(bytes);

    // allocate gpu memory
    cudaMalloc(&gpu_a, bytes);
    cudaMalloc(&gpu_b, bytes);
    cudaMalloc(&gpu_c, bytes);

    // initialize vectors a and b
    vector_init(host_a, n);
    vector_init(host_b, n);

    // host memory to device
    cudaMemcpy(gpu_a, host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, host_b, bytes, cudaMemcpyHostToDevice);

    // thread block size
    int NUM_THREADS = 256;

    // grid size
    int NUM_BLOCKS = (int) ceil(n / NUM_THREADS);

    // launch kernal on default stream
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(gpu_a, gpu_b, gpu_c, n);
    
    // device memory to host memory
    cudaMemcpy(host_c, gpu_c, bytes, cudaMemcpyDeviceToHost);

    // check result for error
    check_error(host_a, host_b, host_c, n);

    // success
    printf("Result = Pass\n");

    return 0;

}




    










    















