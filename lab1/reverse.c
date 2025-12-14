#include <stdio.h>
#include <stdlib.h>

void print(float* array, size_t size) {
    for(int i=0; i < size; i++) {
        printf("%e ", array[i]);
    }
    printf("\n");
}

void reverse(float* array, size_t n) {
    int t;
    size_t n_half = n/2;
    printf("n // 2: %ld\n", n_half);
    for(int i=0; i < n_half; i++) {
        t = array[i];
        array[i] = array[n - 1 - i];
        array[n - 1 - i] = t;
    }
}

int main() {
    size_t n;
    scanf("%ld", &n);
    size_t size = sizeof(float) * n;

    float* array = (float*)malloc(size);
    for(int i=0; i<n; i++)
        scanf("%f", &array[i]);
    
    print(array, n);
    reverse(array, n);
    print(array, n);
    free(array);
}