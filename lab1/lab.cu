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

__global__ void reverse_gpu(double* input, double* output, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    
    for (int i = idx; i < n; i += offset) {
        int reversed_idx = n - 1 - i;
        output[reversed_idx] = input[i];
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
    double* array_device_reverse;
    cudaMalloc(&array_device, size);
    cudaMalloc(&array_device_reverse, size);

    cudaMemcpy(array_device, array_host, size, cudaMemcpyHostToDevice);
    reverse_gpu<<<N_BLOCKS, N_THREADS>>>(array_device, array_device_reverse, n);
    cudaMemcpy(array_host_from_device, array_device_reverse, size, cudaMemcpyDeviceToHost);
    
    print(array_host_from_device, n);
    
    free(array_host);
    free(array_host_from_device);
    cudaFree(array_device);
    cudaFree(array_device_reverse);
    
    return 0;
}