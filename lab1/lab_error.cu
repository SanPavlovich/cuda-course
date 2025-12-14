#include <stdio.h>
#include <stdlib.h>

#define N_BLOCKS 256
#define N_THREADS 1024

void print(double* array, size_t size) {
    for(int i=0; i < size; i++) {
        printf("%.10e ", array[i]);
    }
    printf("\n");
}

__global__ void reverse_gpu(double* array, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    size_t n_half = n/2;
    double t;
    
    while(idx < n_half) {
        t = array[idx];
        array[idx] = array[n - 1 - idx];
        array[n - 1 - idx] = t;

        idx += offset;
    }
}

int main() {
    size_t n;
    scanf("%ld", &n);
    size_t size = sizeof(double) * n;

    double* array_host = (double*)malloc(size);
    double* array_host_from_device = (double*)malloc(size);
    for(int i=0; i<n; i++)
        scanf("%lf", &array_host[i]);

    double* array_device;
    cudaMalloc(&array_device, size);
    cudaMemcpy(array_device, array_host, size, cudaMemcpyHostToDevice);
    reverse_gpu<<<N_BLOCKS, N_THREADS>>>(array_device, n);
    cudaMemcpy(array_host_from_device, array_device, size, cudaMemcpyDeviceToHost);
    
    print(array_host_from_device, n);
    
    free(array_host);
    free(array_host_from_device);
    cudaFree(array_device);
    
    return 0;
}