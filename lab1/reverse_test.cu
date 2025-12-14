#include <stdio.h>
#include <stdlib.h>

#define N_BLOCKS 1024
#define N_THREADS 1024
#define WARMUP_RUNS 5
#define BENCH_RUNS 50

void print(double* array, size_t size) {
    for(int i=0; i < size; i++) {
        printf("%e ", array[i]);
    }
    printf("\n");
}

void fill_array(double* array, size_t size) {
    for(int i=0; i < size; i++)
        array[i] = (double)rand() / RAND_MAX;
}

void reverse_cpu(double* array, size_t n) {
    int t;
    size_t n_half = n/2;
    for(int i=0; i < n_half; i++) {
        t = array[i];
        array[i] = array[n - 1 - i];
        array[n - 1 - i] = t;
    }
}


__global__ void reverse_gpu(double* array, size_t n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    size_t n_half = n/2;
    int t;

    while(idx < n_half) {
        t = array[idx];
        array[idx] = array[n - 1 - idx];
        array[n - 1 - idx] = t;

        idx += offset;
    }
}

bool is_close(double* A, double* B, size_t n, float tol) {
    for(int i=0; i < n; i++) {
        double diff = abs((double)(A[i] - B[i]));
        if(diff > tol) {
            printf("\ndiff: %lf, idx: %d", diff, i);
            return false;
        }
    }
    return true;
}

template<typename T>
float runCpuFunc(
    void (*func)(T*, size_t),
    T *array, size_t n
) {
    clock_t begin = clock();
    func(array, n);
    clock_t end = clock();
    float time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
    return time_spent;
}

template<typename T>
float runKernel(
    void (*kernel)(T*, size_t), 
    T *array, size_t n
) { 
    cudaEvent_t start, stop;
    float milliseconds;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel<<<N_BLOCKS, N_THREADS>>>(array, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {
    size_t n = 33554431;
    size_t size = sizeof(double) * n;

    double* array_host = (double*)malloc(size);
    fill_array(array_host, n);

    double* array_host_from_device = (double*)malloc(size);
    double* array_device;
    double* array_device_reversed;
    cudaMalloc(&array_device, size);
    cudaMemcpy(array_device, array_host, size, cudaMemcpyHostToDevice);
    reverse_gpu<<<N_BLOCKS, N_THREADS>>>(array_device, n);

    cudaMemcpy(array_host_from_device, array_device, size, cudaMemcpyDeviceToHost);
    
    reverse_cpu(array_host, n);
    float tol = 1e-6;
    bool result = is_close(array_host, array_host_from_device, n, tol);
    printf("Результат reverse GPU: %d\n", int(result));

    // Warmup runs
    for (int i = 0; i < WARMUP_RUNS; i++) {
        runCpuFunc<double>(reverse_cpu, array_host, n);
        runKernel<double>(reverse_gpu, array_device, n);
    }

    // Benchmark runs
    float total_cpu = 0, total_kernel_gpu = 0;
    for (int i = 0; i < BENCH_RUNS; i++) {
        total_cpu += runCpuFunc<double>(reverse_cpu, array_host, n);
        total_kernel_gpu += runKernel<double>(reverse_gpu, array_device, n);
    }

    // Calculate average times
    float mean_cpu = total_cpu / BENCH_RUNS;
    float mean_gpu = total_kernel_gpu / BENCH_RUNS;

    printf("Average time for CPU: %f ms\n", mean_cpu);
    printf("Average time for GPU: %f ms\n", mean_gpu);
    
    cudaFree(array_device);
    free(array_host);
    free(array_host_from_device);
    
    return 0;
}