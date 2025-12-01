#include <stdlib.h>
#include <stdio.h>
#include <float.h>

void print(float* array, size_t size) {
    for(int i=0; i < size; i++) {
        printf("%e ", array[i]);
    }
    printf("\n");
}

void swap_elements(float* array, int idx1, int idx2) {
    int t = array[idx1];
    array[idx1] = array[idx2];
    array[idx2] = t;
}

int argmax(float* array, size_t right_border) {
    int max = array[0];
    int argmax = 0;
    if(right_border <= 1) {
        return argmax;
    }

    for(int i=0; i < right_border; i++) {
        if(array[i] > max) {
            max = array[i];
            argmax = i;
        }
    }
    return argmax;
}

void bubble_sort(float* array, size_t size) {
    for(int i=size; i > 1; i--) {
        int argm = argmax(array, i);
        if (argm != (i - 1)) {
            swap_elements(array, i - 1, argm);
        }
    }
}

int main() {
    size_t n;
    scanf("%ld", &n);
    float* a_ptr = (float*)malloc(sizeof(float) * n);
    for(int i=0; i < n; i++) {
        scanf("%f", &a_ptr[i]);
    }
    bubble_sort(a_ptr, n);
    print(a_ptr, n);
    free(a_ptr);
}