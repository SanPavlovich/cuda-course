#include <stdio.h>
#include <stdlib.h>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__device__ float get_gray(const uchar4& p) {
    return 0.299 * p.x + 0.587 * p.y + 0.114 * p.z;
}

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int width, int height) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
   	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
    int x, y;

    int i_0, j_0, i_1, j_1, i_2, j_2;
    uchar4 p_00, p_01, p_02, p_10, p_12, p_20, p_21, p_22;
    float img_00, img_01, img_02, img_10, img_12, img_20, img_21, img_22;

    for(y = idy; y < height; y += offsety)
		for(x = idx; x < width; x += offsetx) {
            j_0 = y - 1;
            i_0 = x - 1;
            j_1 = y;
            i_1 = x;
            j_2 = y + 1;
            i_2 = x + 1;

            p_00 = tex2D< uchar4 >(tex, i_0, j_0);
            p_01 = tex2D< uchar4 >(tex, i_0, j_1);
            p_02 = tex2D< uchar4 >(tex, i_0, j_2);
            p_10 = tex2D< uchar4 >(tex, i_1, j_0);
            p_12 = tex2D< uchar4 >(tex, i_1, j_2);
            p_20 = tex2D< uchar4 >(tex, i_2, j_0);
            p_21 = tex2D< uchar4 >(tex, i_2, j_1);
            p_22 = tex2D< uchar4 >(tex, i_2, j_2);

            img_00 = get_gray(p_00);
            img_01 = get_gray(p_01);
            img_02 = get_gray(p_02);
            img_10 = get_gray(p_10);
            img_12 = get_gray(p_12);
            img_20 = get_gray(p_20);
            img_21 = get_gray(p_21);
            img_22 = get_gray(p_22);
            
            float gx = (-1) * img_00 + (-2) * img_10 + (-1) * img_20 + 1 * img_02 + 2 * img_12 + 1 * img_22;
            float gy = (-1) * img_00 + (-2) * img_01 + (-1) * img_02 + 1 * img_20 + 2 * img_21 + 1 * img_22;
            float grad_norm = sqrt(gx * gx + gy * gy);
            // int grad_int = max(min((int)(grad_norm + 0.5f), 255), 0);
            int grad_int = max(min(__float2int_rn(grad_norm), 255), 0);
            out[y * width + x] = make_uchar4(grad_int, grad_int, grad_int, 0);
        }
}

int main() {
    char filepath_in[256];
    char filepath_out[256];
    // printf("Введите путь к файлу in.data: ");
    scanf("%255s", filepath_in);
    // printf("Введите путь к файлу out.data: ");
    scanf("%255s", filepath_out);
    // printf("filepath_in: <%s>\n", filepath_in);
    // printf("filepath_out: <%s>\n", filepath_out);

    int w, h;
   	FILE *fp = fopen(filepath_in, "rb");
 	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeMirror;
    texDesc.addressMode[1] = cudaAddressModeMirror; // Clamp
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    kernel<<< dim3(16, 16), dim3(32, 32) >>>(tex, dev_out, w, h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	CSC(cudaDestroyTextureObject(tex));
	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_out));

    fp = fopen(filepath_out, "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

    free(data);
    return 0;
}