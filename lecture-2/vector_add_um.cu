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

    // get device id for cuda calls
    int id = cudaGetDevice(&id);

    // vector size
    int n = 1 << 16;

    // unified memory vector pointers
    int *a, *b, *c;

    // allocation size for each vector
    size_t bytes = sizeof(int) * n;
    
    // allocate memory on unified memory
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // initialize vectors a and b
    vector_init(a, n);
    vector_init(b, n);

    // thread block size
    int NUM_THREADS = 256;

    // grid size
    int NUM_BLOCKS = (int) ceil(n / NUM_THREADS);

    // pre-fetching a and b to device
    // we know we will need to a and b on device for calculations
    // pre-fetch will make things faster
    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);

    // launch kernal on default stream
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(a, b, c, n);
    
    // synchronize threads in device
    // need to prevent race condition on unified memory
    // host should not access unified memory until device is done
    cudaDeviceSynchronize();

    // pre-fetch from device to host
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    // check result for error
    check_error(a, b, c, n);

    // success
    printf("Result = Pass\n");

    return 0;

}




    










    















