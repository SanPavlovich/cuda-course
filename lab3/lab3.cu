#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>

#define CHANNELS 3
#define THREADS 1024
#define BLOCKS 1024

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

typedef struct point{
    int x;
    int y;
} Point;

__constant__ float device_class_means[32 * CHANNELS]; // не более 32 классов, по 3 значения RGB в векторе среднего

__global__ void spectral_classiffy(uchar4* data, int n_classes, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    
    while(idx < w * h) {
        uchar4 pixel = data[idx];
        
        // поиск ближайшего класса по норме
        float max_sim = FLT_MIN;
        int max_sim_cls = 0;
        for(int cls=0; cls < n_classes; cls++) {
            float r_mean = device_class_means[cls * CHANNELS];
            float g_mean = device_class_means[cls * CHANNELS + 1];
            float b_mean = device_class_means[cls * CHANNELS + 2];
            float similarity = pixel.x * r_mean + pixel.y * g_mean + pixel.z * b_mean;    
            
            if(similarity > max_sim) {
                max_sim = similarity;
                max_sim_cls = cls;
            }
        }
        
        // data[idx].w = max_sim_cls;
        data[idx] = make_uchar4(pixel.x, pixel.y, pixel.z, max_sim_cls);

        idx += offset;
    }
}

int main() {
    char filepath_in[256];
    char filepath_out[256];
    scanf("%255s", filepath_in);
    scanf("%255s", filepath_out);

    int w, h;
   	FILE *fp = fopen(filepath_in, "rb");
 	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
    size_t data_size = sizeof(uchar4) * w * h;
    uchar4 *data = (uchar4 *)malloc(data_size);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    uchar4* data_host_from_device = (uchar4 *)malloc(data_size);

    size_t n_classes;
    scanf("%ld", &n_classes);

    Point** class_pixels = (Point**)malloc(sizeof(Point*) * n_classes);
    int* class_counts = (int*)malloc(sizeof(int) * n_classes);

    for(int i=0; i < n_classes; i++) {
        scanf("%d", &class_counts[i]);
        class_pixels[i] = (Point*)malloc(sizeof(Point) * class_counts[i]);
        for(int j=0; j < class_counts[i]; j++) {
            scanf("%d", &class_pixels[i][j].y);
            scanf("%d", &class_pixels[i][j].x);
        }
    }

    size_t means_size = sizeof(float) * n_classes * CHANNELS;
    float* host_class_means = (float*)malloc(means_size);
    memset(host_class_means, 0, means_size);

    for(int i=0; i < n_classes; i++) {
        // i - номер класса
        for(int j=0; j < class_counts[i]; j++) {
            int i_idx = class_pixels[i][j].x; // x - высота w при вводе
            int j_idx = class_pixels[i][j].y;
            uchar4 pixel = data[i_idx * w + j_idx];
            host_class_means[i * CHANNELS] += pixel.x; // R_mean
            host_class_means[i * CHANNELS + 1] += pixel.y; // G_mean
            host_class_means[i * CHANNELS + 2] += pixel.z; // B_mean
        
            data[i_idx * w + j_idx].w = i; // номер класса, на незаполненных местах будет стоять 255 (кладем это значение в python скрипте)
        }
    }

    // нормализуем посчитанную сумму, чтобы получить среднее
    for(int i=0; i < n_classes; i++) {
        int count = class_counts[i];
        float r = host_class_means[i * CHANNELS];
        float g = host_class_means[i * CHANNELS + 1];
        float b = host_class_means[i * CHANNELS + 2];
        r /= count;
        g /= count;
        b /= count;
        float norm = sqrt(r * r + g * g + b * b);
        r /= norm;
        g /= norm;
        b /= norm;

        host_class_means[i * CHANNELS] = r;
        host_class_means[i * CHANNELS + 1] = g;
        host_class_means[i * CHANNELS + 2] = b;
    }

    // memory copy and allocation
    cudaMemcpyToSymbol(device_class_means, host_class_means, means_size);

    uchar4* device_data;
    cudaMalloc(&device_data, data_size);
    cudaMemcpy(device_data, data, data_size, cudaMemcpyHostToDevice);
    spectral_classiffy<<<BLOCKS, THREADS>>>(device_data, n_classes, w, h);
    CSC(cudaGetLastError());
    cudaMemcpy(data_host_from_device, device_data, data_size, cudaMemcpyDeviceToHost);

    fp = fopen(filepath_out, "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data_host_from_device, sizeof(uchar4), w * h, fp);
	fclose(fp);

    free(data);
    free(data_host_from_device);
    cudaFree(device_data);
    
    return 0;
}