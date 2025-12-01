#include <stdio.h>
#include <stdlib.h>

void print(float** array, size_t n_rows, size_t n_cols) {
    for(int i = 0; i < n_rows; i++) {
        for(int j=0; j < n_cols; j++) {
            printf("%f ", array[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void fill_array(float** array, size_t n_rows, size_t n_cols) {
    for(int i = 0; i < n_rows; i++) {
        for(int j=0; j < n_cols; j++) {
            array[i][j] = (float)rand() / RAND_MAX; //i * n_cols + j;
        }
    }
}

float** new_array(size_t n_rows, size_t n_cols) {
    float** array = (float**)malloc(sizeof(float*) * n_rows);
    for(int i = 0; i < n_rows; i++)
        array[i] = (float*)malloc(sizeof(float) * n_cols);
    return array;
}

void free_array(float** array, size_t n_rows) {
    for(int i=0; i < n_rows; i++)
        free(array[i]);
    free(array);
}

void matmul_cpu(float** A, float** B, float** C, int x, int y, int z) {
    for(int i=0; i < x; i++) {
        for(int j=0; j < z; j++) {
            float sum = 0;
            for(int k=0; k < y; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }    
    }
}

int main() {
    int m = 2;
    int n = 2;
    int k = 3;

    float** A = new_array(m, n);
    float** B = new_array(n, k);
    float** C = new_array(m, k);
    
    fill_array(A, m, n);
    fill_array(B, n, k);
    fill_array(C, m, k);
    
    print(A, m, n);
    print(B, n, k);
    
    matmul_cpu(A, B, C, m, n, k);
    print(C, m, k);
    
    free_array(A, m);
    free_array(B, n);
    free_array(C, k);
}