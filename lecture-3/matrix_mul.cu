#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>


__global__ void matrixMul(int *a, int *b, int *c, int n) {
    
    // thread row
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // thread columm
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp = 0;

    // boundary check
    if ((row < n) && (col < n)) {

        // iterate of row and down column
        for (int k = 0; k < n; k++) {
        
            temp += a[row * n + k] * b[k * n + col];
        
        }
        
        c[row * n + col] = temp;
    }

}


void matrix_init(int *a, int *b, int n) {
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            
            a[i * n + j] = rand() / 100;
            b[i * n + j] = rand() / 100;

        }
    }

}


void check_error(int *a, int *b, int *c, int n) {
    
    int temp = 0;

    for (int i = 0; i < n; i++) {
    
        for (int j = 0; j < n; j++) {

            for (int k = 0; k < n; k++) {
                
                temp += a[i * n + k] * b[k * n + j]; 

            }

            assert(c[i * n + j] == temp);
            temp = 0;    

        }
    }

}

int main() {

    // 1024 x 1024 dim matrices
    int n = 1 << 10;

    // size in bytes
    size_t bytes = n * n * sizeof(int);

    // host pointers
    int *host_a, *host_b, *host_c;

    // allocate memory for host pointers
    host_a = (int*) malloc(bytes);
    host_b = (int*) malloc(bytes);
    host_c = (int*) malloc(bytes);

    // device pointers
    int *gpu_a, *gpu_b, *gpu_c;
    
    // allocate memory for gpu pointers
    cudaMalloc(&gpu_a, bytes);
    cudaMalloc(&gpu_b, bytes);
    cudaMalloc(&gpu_c, bytes);

    // initialize matrices
    matrix_init(host_a, host_b, n);
    
    // copy from host to gpu
    cudaMemcpy(gpu_a, host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, host_b, bytes, cudaMemcpyHostToDevice);

    // threads per block
    int BLOCK_SIZE = 16;

    // block in each dimension
    int GRID_SIZE = (int) ceil(n / BLOCK_SIZE);

    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // launch kernel
    matrixMul<<<grid, threads>>> (gpu_a, gpu_b, gpu_c, n);

    // copy result from gpu to host
    cudaMemcpy(host_c, gpu_c, bytes, cudaMemcpyDeviceToHost);

    // check result
    check_error(host_a, host_b, host_c, n); 

    // print success
    printf("Result = Pass\n");

    return 0;

}


