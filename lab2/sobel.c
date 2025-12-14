#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>

#define CHANNELS 4
char base_digits[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

void print_uint8_as_hex(uint8_t num) {
    uint8_t high = num / 16;
    uint8_t low = num % 16;
    printf("%c%c", base_digits[high], base_digits[low]);
}

void print_rgb_image(uint8_t* image, size_t width, size_t height, bool hex_format) {
    for (int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            int pixel_index = (i * width + j) * CHANNELS;
            int r = image[pixel_index];
            int g = image[pixel_index + 1];
            int b = image[pixel_index + 2];
            int alpha = image[pixel_index + 3];

            if(hex_format){
                print_uint8_as_hex(r);
                print_uint8_as_hex(g);
                print_uint8_as_hex(b);
                print_uint8_as_hex(alpha);
            } else {
                printf("'%d %d %d %d'", r, g, b, alpha);
            }
            printf(" ");
        }
        printf("\n");
    }
}

int max(int a, int b) {
  return (a > b) ? a : b;
}
 
int min(int a, int b) {
  return (a > b) ? b : a;
}

float get_gray_pixel(uint8_t* input_image, size_t width, int i, int j) {
    int pixel_index = (i * width + j) * CHANNELS;
    float r = input_image[pixel_index];
    float g = input_image[pixel_index + 1];
    float b = input_image[pixel_index + 2];
    float gray = 0.299 * r + 0.587 * g + 0.114 * b;
    return gray;
};

void apply_sobel_cpu(uint8_t* input_image, uint8_t* output_image, size_t width, size_t height) {
    for (int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            int i_0 = max(min(i - 1, height - 1), 0);
            int i_1 = i;
            int i_2 = max(min(i + 1, height - 1), 0);
            int j_0 = max(min(j - 1, width - 1), 0);
            int j_1 = j;
            int j_2 = max(min(j + 1, width - 1), 0);

            float img_00 = get_gray_pixel(input_image, width, i_0, j_0);
            float img_01 = get_gray_pixel(input_image, width, i_0, j_1);
            float img_02 = get_gray_pixel(input_image, width, i_0, j_2);
            float img_10 = get_gray_pixel(input_image, width, i_1, j_0);
            float img_12 = get_gray_pixel(input_image, width, i_1, j_2);
            float img_20 = get_gray_pixel(input_image, width, i_2, j_0);
            float img_21 = get_gray_pixel(input_image, width, i_2, j_1);
            float img_22 = get_gray_pixel(input_image, width, i_2, j_2);

            float gx = (-1) * img_00 + (-2) * img_10 + (-1) * img_20 + 1 * img_02 + 2 * img_12 + 1 * img_22;
            float gy = (-1) * img_00 + (-2) * img_01 + (-1) * img_02 + 1 * img_20 + 2 * img_21 + 1 * img_22;
            float grad_norm = sqrt(gx * gx + gy * gy);
            uint8_t int_grad = max(min((int)grad_norm, 255), 0);

            for(int ch=0; ch < (CHANNELS - 1); ch++)
                output_image[(i * width + j) * CHANNELS + ch] = int_grad;
            output_image[(i * width + j) * CHANNELS + CHANNELS - 1] = 0;
        }
    }
}

int main() {
    // int a=255;
    // print_uint8_as_hex(a);
    FILE* f = fopen("test_c_2x2.bin", "rb");
    
    size_t width, height;
    fread(&width, 4, 1, f);  // 4 байта = sizeof(size_t)
    fread(&height, 4, 1, f);
    printf("width: %ld, height: %ld\n", width, height);

    int pixel_count = width * height;
    int byte_count = pixel_count * 4;  // 4 байта на пиксель (RGBA)
    
    uint8_t* img = (uint8_t*)malloc(byte_count);
    fread(img, 1, byte_count, f);
    fclose(f);

    printf("input image:\n");
    print_rgb_image(img, width, height, true);
    
    uint8_t* img_sobel = (uint8_t*)malloc(byte_count);
    apply_sobel_cpu(img, img_sobel, width, height);
    printf("output image:\n");
    print_rgb_image(img_sobel, width, height, true);

    free(img);
    free(img_sobel);
}