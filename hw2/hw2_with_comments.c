#include <stdlib.h> // for malloc()
#include <stdio.h>
#include <float.h>

void print(float* array, size_t size) {
    for(int i=0; i < size; i++) {
        printf("%f ", array[i]);
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
        printf("поиск максимума с правой границей: %d\n", i);
        int argm = argmax(array, i);
        printf("i: %d, argm: %d\n", i, argm);
        printf("массив до swap:\n");
        print(array, size);
        if (argm != (i - 1)) {
            swap_elements(array, i - 1, argm);
        }
        printf("массив после swap:\n");
        print(array, size);
    }
}

int main() {
    size_t n; // size_t - long unsigned int
    printf("Введите размер массива: ");
    scanf("%ld", &n);
    /* float c[n]; // так делать нельзя, это только для статической аллокации памяти на этапе компиляции.
    for(int i=0; i < n; i++) {
        scanf("%f", &c[i]);
    }
    Если сделать free(c), то вылетит предупреждение:
    warning: 'free' called on pointer to an unallocated object
    */

    float* a_ptr = (float*)malloc(sizeof(float) * n);
    for(int i=0; i < n; i++) {
        scanf("%f", &a_ptr[i]);
    }
    print(a_ptr, n);
    bubble_sort(a_ptr, n);
    printf("Результат сортировки:\n");
    print(a_ptr, n);
    // swap_elements(a_ptr, 0, 1);
    // print(a_ptr, n);
    // int argm = argmax(a_ptr, n);
    // printf("Индекс максимума: %d\n", argm);
    free(a_ptr); // освобождаем память!
}