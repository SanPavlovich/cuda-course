#include <iostream>
#include <fstream>
#include <vector>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__constant__ float4 cuda_normed[32];

__global__ void kernel(uchar4* pixels, int num_pixels, int nc) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < num_pixels) {
        uchar4 cur_pixel = pixels[idx];
        float max_diff = 0;
        int argmax = 0;
        for (int i = 0; i < nc; ++i) { // Рассчёт дискриминантной функции и сравнение
            float cur_diff = 0;
            for (int j = 0; j < 4; ++j) {
                cur_diff += (&cur_pixel.x)[j] * (&(cuda_normed[i].x))[j];
            }
            if (cur_diff > max_diff) {
                max_diff = cur_diff;
                argmax = i;
            }
        }
        pixels[idx] = make_uchar4(pixels[idx].x, pixels[idx].y, pixels[idx].z, argmax);
        
        idx += offset;
    }
}

float4 calc_normed(uchar4* pixels, int w, int nc, std::vector<int2>& class_points) {
    // Рассчёт среднего пикселя выборки
    float4 avg_pixel = make_float4(0, 0, 0, 0);
    for (int i = 0; i < class_points.size(); ++i) {
        uchar4 cur_pixel = pixels[class_points[i].y * w + class_points[i].x];
        for (int j = 0; j < 4; ++j) {
            (&avg_pixel.x)[j] += (&cur_pixel.x)[j];
        }
    }

    // Нормировка пикселя
    float norm_coeff = 0;
    for (int i = 0; i < 4; ++i) {
        norm_coeff += (&avg_pixel.x)[i]*(&avg_pixel.x)[i];
    }
    norm_coeff = sqrtf(norm_coeff);

    for (int i = 0; i < 4; ++i) {
        (&avg_pixel.x)[i] /= norm_coeff;
    }

    return avg_pixel;
}

void print_uchar_data(uchar4* data, int w, int h, bool print_cls_only) {
    for(int i=0; i < h; i++) {
        for (int j=0; j < w; j++) {
            uchar4 pixel = data[i * w + j];
            if(print_cls_only) {
                printf("%d", pixel.w);
            } else {
                printf("[%d %d %d %d]", pixel.x, pixel.y, pixel.z, pixel.w);
            }
            printf(" ");
        }
        printf("\n");
    }
}

int main() {
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);

    // Ввод путей файлов
    std::string in_name = "test_example_1.bin", out_name = "out.bin";
    std::cin >> in_name >> out_name;

    // Чтение из файла
    std::ifstream file_read(in_name, std::ios::binary);

    if (!file_read) {
        std::cerr << "No such file: " << in_name << std::endl;
        return 0;
    }

    int w,h;

    file_read.read(reinterpret_cast<char*>(&w), sizeof(int));
    file_read.read(reinterpret_cast<char*>(&h), sizeof(int));

    uchar4 *data = new uchar4[w * h];

    file_read.read(reinterpret_cast<char*>(data), sizeof(uchar4)*w*h);

    file_read.close();

    // Рассчёт параметров для выборок и сохранение в константную память
    int nc;
    std::cin >> nc;

    std::vector<float4> cpu_normed;

    for (int i=0; i<nc; ++i) {
        int npi;
        std::cin >> npi;
        std::vector<int2> class_points;
        for (int j = 0; j < npi; ++j) {
            int cord1, cord2;
            std::cin >> cord1 >> cord2;
            class_points.push_back(make_int2(cord1,cord2));
        }
        float4 normed = calc_normed(data, w, nc, class_points);
        cpu_normed.push_back(normed);
    }

    cudaMemcpyToSymbol(cuda_normed, cpu_normed.data(), cpu_normed.size() * sizeof(float4));

    // Основная часть
    uchar4* dev_arr;
    CSC(cudaMalloc(&dev_arr, sizeof(uchar4) * w * h));
    CSC(cudaMemcpy(dev_arr, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    // // Измерение времени
    // cudaEvent_t start, stop;
    // CSC(cudaEventCreate(&start));
    // CSC(cudaEventCreate(&stop));

    // CSC(cudaEventRecord(start));

    // Вызов ядра
    kernel<<<1, 32>>>(dev_arr, w*h, nc);

    // // Окончание измерения времени
    // CSC(cudaEventRecord(stop));

    // CSC(cudaEventSynchronize(stop));
    CSC(cudaGetLastError());

    // float t;
    // CSC(cudaEventElapsedTime(&t, start, stop));
    // CSC(cudaEventDestroy(start));
    // CSC(cudaEventDestroy(stop));

    // std::cout << t << " ms\n";
    
    // Продолжение основной части
    CSC(cudaMemcpy(data, dev_arr, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_arr));

    printf("\nresult data:\n");
    print_uchar_data(data, w, h, true);

    // Запись в файл
    std::ofstream file_write(out_name, std::ios::binary);

    if (!file_write) {
        std::cerr << "Cannot create file: " << out_name << std::endl;
        return 0;
    }

    file_write.write(reinterpret_cast<char*>(&w), sizeof(int));
    file_write.write(reinterpret_cast<char*>(&h), sizeof(int));
    file_write.write(reinterpret_cast<char*>(data), sizeof(uchar4)*w*h);

    file_write.close();

    delete[] data;
}