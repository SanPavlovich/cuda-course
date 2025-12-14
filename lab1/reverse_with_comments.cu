#include <stdio.h>
#include <stdlib.h>

#define N_BLOCKS 1024
#define N_THREADS 1024

void print(float* array, size_t size) {
    for(int i=0; i < size; i++) {
        printf("%e ", array[i]);
    }
    printf("\n");
}

void reverse_cpu(float* array, size_t n) {
    int t;
    size_t n_half = n/2;
    printf("n // 2: %ld\n", n_half);
    for(int i=0; i < n_half; i++) {
        t = array[i];
        array[i] = array[n - 1 - i];
        array[n - 1 - i] = t;
    }
}


__global__ void reverse_gpu(float* array, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    size_t n_half = n/2;

    while(idx < n_half) {
        int t;
        t = array[idx];
        array[idx] = array[n - 1 - idx];
        array[n - 1 - idx] = t;

        idx += offset;
    }
}

bool is_close(float* A, float* B, size_t n, float tol) {
    for(int i=0; i < n; i++) {
        float diff = abs((float)(A[i] - B[i]));
        if(diff > tol) {
            printf("\ndiff: %f, idx: %d", diff, i);
            return false;
        }
    }
    return true;
}

int main() {
    size_t n;
    scanf("%ld", &n);
    size_t size = sizeof(float) * n;

    float* array_host = (float*)malloc(size);
    float* array_host_from_device = (float*)malloc(size);
    for(int i=0; i<n; i++)
        scanf("%f", &array_host[i]);

    float* array_device;
    cudaMalloc(&array_device, size);
    cudaMemcpy(array_device, array_host, size, cudaMemcpyHostToDevice);
    reverse_gpu<<<N_BLOCKS, N_THREADS>>>(array_device, n);

    cudaMemcpy(array_host_from_device, array_device, size, cudaMemcpyDeviceToHost);
    
    printf("массив:\n");
    print(array_host, n);

    reverse_cpu(array_host, n);

    printf("массив reversed:\n");
    print(array_host, n);

    printf("массив reversed GPU:\n");
    print(array_host_from_device, n);

    float tol = 1e-6;
    bool result = is_close(array_host, array_host_from_device, n, tol);
    printf("Результат reverse GPU: %d\n", int(result));

    free(array_host);
    free(array_host_from_device);
    cudaFree(array_device);
    
    return 0;
}