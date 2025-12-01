#include <stdio.h>
#include <stdlib.h>


__global__ void kernel(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while(idx < n) {
        arr[idx] *= 2.0;
        idx += offset;
    }
}

int main() {
    int i, n = 10000000;
    float *arr = (float *)malloc(sizeof(float) * n);
    for(i = 0; i < n; i++)
        arr[i] = i;
    float *dev_arr;
    cudaMalloc(&dev_arr, sizeof(float) * n);
    cudaMemcpy(dev_arr, arr, sizeof(float) * n, cudaMemcpyHostToDevice);
    
    kernel<<<1024, 1024>>>(dev_arr, n);
    
    cudaMemcpy(arr, dev_arr, sizeof(float) * n, cudaMemcpyDeviceToHost);
    for(i = 0; i < 10; i++)
        printf("%f ", arr[i]);
    printf("\n");

    cudaFree(dev_arr);
    free(arr);
    return 0;
}