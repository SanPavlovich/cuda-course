#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 32
#define WARMUP_RUNS 5
#define BENCH_RUNS 40

#define CSC(call)                                       \
do {                                                    \
    cudaError_t status = call;                          \
    if(status != cudaSuccess) {                         \
        fprintf(stderr, "ERROR is %s:%d. Message %s\n", __FILE__, __LINE__, cudaGetErrorString(status));    \
    }                                                   \
} while(0)                                              \

template<typename T>
T* new_array(size_t n_rows, size_t n_cols) {
    T* array = (T*)malloc(sizeof(T) * n_rows * n_cols);
    return array;
}

template<typename T>
void free_array(T* array) {
    free(array);
}

template<typename T>
void fill_array(T* array, size_t n_rows, size_t n_cols) {
    for(int i = 0; i < n_rows; i++) {
        for(int j=0; j < n_cols; j++) {
            array[i * n_cols + j] = (T)rand() / RAND_MAX; // (rand() / RAND_MAX);
        }
    }
}

template<typename T>
void print(T* array, size_t n_rows, size_t n_cols) {
    for(int i = 0; i < n_rows; i++) {
        for(int j=0; j < n_cols; j++) {
            printf("%lf ", (float)array[i * n_cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

template<typename T>
void matmul_cpu(T* A, T* B, T* C, size_t m, size_t n, size_t k) {
    for(int m_idx = 0; m_idx < m; m_idx++) {
        for(int k_idx = 0; k_idx < k; k_idx++) {
            T sum = 0;
            for(int n_idx = 0; n_idx < n; n_idx++)
                sum += A[m_idx * n + n_idx] * B[n_idx * k + k_idx];
            C[m_idx * k + k_idx] = sum;
        }    
    }
}

template<typename T>
__global__ void matmul_gpu(T* A, T* B, T* C, size_t m, size_t n, size_t k) {
    // потоков - 32 * 32 <--> (blockDim.x, blockDim.y)
    // нужно найти номер блока, а потом внутри блока (32, 32) пройтись по потокам
    // в одномерном случае индекс массива - это blockIdx.x * blockDim.x + trheadIdx.x
    // каждый из потоков будет делать скалярное произведение по строке А и столбцу B
    // blockDim.x = blockDim.y = BLOCK_SIZE = 32
    // [blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x]
    int c_idx_k = blockIdx.x * blockDim.x + threadIdx.x;
    int c_idx_m = blockIdx.y * blockDim.y + threadIdx.y;
    if(c_idx_m < m && c_idx_k < k) {
        T sum = 0.0f;
        for(int i=0; i < n; i++)
            sum += A[c_idx_m * n + i] * B[i * k + c_idx_k];
        C[c_idx_m * k + c_idx_k] = sum;
    }
}

// template<typename T>
// __global__ void matmul_gpu_shmem(T* A, T* B, T* C, size_t m, size_t n, size_t k) {
//     // A.shape = (m, n);
//     // B.shape = (n, k);
//     // C.shape = (m, k);
//     const uint cRow = blockIdx.x;
//     const uint cCol = blockIdx.y;
    
//     __shared__ T As[BLOCK_SIZE * BLOCK_SIZE];
//     __shared__ T Bs[BLOCK_SIZE * BLOCK_SIZE];

//     const uint threadRow = threadIdx.x / BLOCK_SIZE;
//     const uint threadCol = threadIdx.x % BLOCK_SIZE;
    
//     // тут фишка с адресной арифметикой, внутри каждого потока прыгаем на нужный блок 1 раз.
//     A += cRow * BLOCK_SIZE * n; // перемещаемся на нужный блок в матрице А (поблочно прыгаем на нужную строку из блоков)
//     B += cCol * BLOCK_SIZE; // перемещаемся на нужный блок в матрице В (поблочно прыгаем на нужную колонку из блоков)
//     C += cRow * BLOCK_SIZE * k + cCol * BLOCK_SIZE; // заполняем фрагмент С поблочными умножениями, потом результат поблочного умножения прибавляем в С

//     float tmp = 0.0f; // эта переменная будет храниться внутри регистра для каждого треда 
//     for(int bkIdx = 0; bkIdx < n; bkIdx += BLOCK_SIZE) {
//         As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * n + threadCol];
//         Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * k + threadCol];

//         __syncthreads();
//         A += BLOCK_SIZE; // двигаемся вправо поблочно вдоль А (как в обычном матричном умножении)
//         B += BLOCK_SIZE * k; // двигаемся вниз поблочно вдоль В
        
//         // делаем поблочное матричное умножение всеми потоками
//         for(int dotIdx = 0; dotIdx < BLOCK_SIZE; dotIdx++) {
//             tmp += As[threadRow * BLOCK_SIZE + dotIdx] * Bs[dotIdx * BLOCK_SIZE + threadCol];
//         }
//         __syncthreads();
//     }
//     C[threadRow * k + threadCol] = tmp;
// }

template<typename T>
__global__ void matmul_gpu_shmem(T* A, T* B, T* C, size_t m, size_t n, size_t k) {
    // A.shape = (m, n)
    // B.shape = (n, k)
    // C.shape = (m, k)
    
    // Каждый блок вычисляет один тайл C размером BLOCK_SIZE x BLOCK_SIZE
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Каждый поток вычисляет один элемент C
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Глобальные индексы для элемента C
    const int row = by * BLOCK_SIZE + ty;
    const int col = bx * BLOCK_SIZE + tx;
    
    __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    T tmp = 0.0f;
    
    // Вычисляем произведение по тайлам
    for (int t = 0; t < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Загружаем тайл из A
        int aRow = row;
        int aCol = t * BLOCK_SIZE + tx;
        if (aRow < m && aCol < n) {
            As[ty][tx] = A[aRow * n + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Загружаем тайл из B
        int bRow = t * BLOCK_SIZE + ty;
        int bCol = col;
        if (bRow < n && bCol < k) {
            Bs[ty][tx] = B[bRow * k + bCol];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Вычисляем часть произведения
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            tmp += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    // Записываем результат
    if (row < m && col < k) {
        C[row * k + col] = tmp;
    }
}

template<typename T>
float runKernel(
    void (*kernel)(T*, T*, T*, size_t, size_t, size_t), 
    T *d_a, T *d_b, T *d_c, int m, int n, int k
) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    cudaEvent_t start, stop;
    float milliseconds;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}


template<typename T>
bool is_close(T* A, T* B, size_t n_rows, size_t n_cols, float tol) {
    for(int i = 0; i < n_rows; i++) {
        for(int j=0; j < n_cols; j++) {
            float diff = abs((float)(A[i * n_cols + j] - B[i * n_cols + j]));
            // printf("\ndiff: %f", diff);
            if(diff > tol) {
                return false;
            }
        }
    }
    return true;
}

void gemm_gpu_cpu_compare(int m, int n, int k) {
    float tol = 1e-3;
    int A_size = m * n * sizeof(float);
    int B_size = n * k * sizeof(float);
    int C_size = m * k * sizeof(float);

    float* A_host = new_array<float>(m, n);
    float* B_host = new_array<float>(n, k);
    float* C_host = new_array<float>(m, k);
    float* C_host_from_device = new_array<float>(m, k);
    float* C_2_host_from_device = new_array<float>(m, k);
    fill_array<float>(A_host, m, n);
    fill_array<float>(B_host, n, k);
    
    matmul_cpu<float>(A_host, B_host, C_host, m, n, k);
    
    float* A_device; 
    float* B_device; 
    float* C_device;
    float* C_2_device;
    cudaMalloc(&A_device, A_size);
    cudaMalloc(&B_device, B_size);
    cudaMalloc(&C_device, C_size);
    cudaMalloc(&C_2_device, C_size);

    cudaMemcpy(A_device, A_host, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, B_size, cudaMemcpyHostToDevice);

    // Warmup runs
    for (int i = 0; i < WARMUP_RUNS; i++) {
        runKernel<float>(matmul_gpu, A_device, B_device, C_device, m, n, k);
        runKernel<float>(matmul_gpu_shmem, A_device, B_device, C_2_device, m, n, k);
    }

    // Benchmark runs
    float total_kernel_1 = 0, total_kernel_2 = 0;
    for (int i = 0; i < BENCH_RUNS; i++) {
        total_kernel_1 += runKernel<float>(matmul_gpu, A_device, B_device, C_device, m, n, k);
        total_kernel_2 += runKernel<float>(matmul_gpu_shmem, A_device, B_device, C_2_device, m, n, k);
    }

    // Calculate average times
    float avgTimeNoUnroll = total_kernel_1 / BENCH_RUNS;
    float avgTimeUnroll = total_kernel_2 / BENCH_RUNS;

    printf("Average time for kernel naive: %f ms\n", avgTimeNoUnroll);
    printf("Average time for kernel shmem: %f ms\n", avgTimeUnroll);

    // dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 gridDim((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // matmul_gpu<float><<<gridDim, blockDim>>>(A_device, B_device, C_device, m, n, k);
    // matmul_gpu_shmem<float><<<gridDim, blockDim>>>(A_device, B_device, C_2_device, m, n, k);

    cudaMemcpy(C_host_from_device, C_device, C_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_2_host_from_device, C_2_device, C_size, cudaMemcpyDeviceToHost);
    
    bool matmul_close = is_close(C_host, C_host_from_device, m, k, tol);
    bool matmul_close_2 = is_close(C_host, C_2_host_from_device, m, k, tol);
    printf("\nМатричное умножение на CPU и GPU kernel_1 сошлось: %d", (int)matmul_close);
    printf("\nМатричное умножение на CPU и GPU kernel_2 сошлось: %d\n", (int)matmul_close_2);

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
    cudaFree(C_2_device);
    free_array<float>(C_host_from_device);
    free_array<float>(C_2_host_from_device);
    free_array<float>(A_host);
    free_array<float>(B_host);
    free_array<float>(C_host);
}

void gemm_gpu(int m, int n, int k) {
    int A_size = m * n * sizeof(float);
    int B_size = n * k * sizeof(float);
    int C_size = m * k * sizeof(float);

    float* A_host = new_array<float>(m, n);
    float* B_host = new_array<float>(n, k);
    float* C_host_from_device = new_array<float>(m, k);
    fill_array<float>(A_host, m, n);
    fill_array<float>(B_host, n, k);    
    
    float* A_device; 
    float* B_device; 
    float* C_device;
    cudaMalloc(&A_device, A_size);
    cudaMalloc(&B_device, B_size);
    cudaMalloc(&C_device, C_size);

    cudaMemcpy(A_device, A_host, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, B_size, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (k + BLOCK_SIZE - 1) / BLOCK_SIZE);
    printf("blockDim.x: %d, blockDim.y: %d, gridDim.x: %d, gridDim.y: %d\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);

    matmul_gpu<float><<<gridDim, blockDim>>>(A_device, B_device, C_device, m, n, k);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(C_host_from_device, C_device, C_size, cudaMemcpyDeviceToHost);

    // print(C_host, m, k);
    // print(C_host_from_device, m, k);

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
    free_array<float>(C_host_from_device);
    free_array<float>(A_host);
    free_array<float>(B_host);
}

int main() {
    int m = 1024;
    int n = 1024 * 3;
    int k = 1024 * 2;
    // gemm_gpu(m, n, k);
    gemm_gpu_cpu_compare(m, n, k);
    return 0;
}