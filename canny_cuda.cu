#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>  // For time measurement

#define BLOCK_SIZE 16
#define LOW_THRESHOLD 50
#define HIGH_THRESHOLD 150

using namespace cv;
using namespace std::chrono;  // For time measurement

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Helper function to measure CUDA kernel execution time
void startTimer(cudaEvent_t &start, cudaEvent_t &stop) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

float stopTimer(cudaEvent_t &start, cudaEvent_t &stop) {
    float milliseconds = 0;
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

__global__ void gaussian_blur(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float filter[5][5] = {
        {1, 4, 7, 4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1, 4, 7, 4, 1}
    };
    
    float weight = 273.0;
    int radius = 2;
    float blur_value = 0.0;
    
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            int nx = min(max(x + j, 0), width - 1);
            int ny = min(max(y + i, 0), height - 1);
            blur_value += input[ny * width + nx] * filter[i + radius][j + radius];
        }
    }
    output[y * width + x] = blur_value / weight;
}

__global__ void sobel_filter(unsigned char *input, unsigned char *output, float *gradient, float *direction, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    
    float grad_x = 0;
    float grad_y = 0;
    
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int nx = min(max(x + j, 0), width - 1);
            int ny = min(max(y + i, 0), height - 1);
            grad_x += input[ny * width + nx] * gx[i + 1][j + 1];
            grad_y += input[ny * width + nx] * gy[i + 1][j + 1];
        }
    }
    gradient[y * width + x] = sqrt(grad_x * grad_x + grad_y * grad_y);
    direction[y * width + x] = atan2f(grad_y, grad_x);
    output[y * width + x] = min(255, (int)gradient[y * width + x]);
}

__global__ void non_maximum_suppression(float *gradient, float *direction, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        float angle = direction[y * width + x] * (180.0 / M_PI);
        angle = fmod(angle + 180.0, 180.0);
        
        float q = 255, r = 255;
        if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
            q = gradient[y * width + (x + 1)];
            r = gradient[y * width + (x - 1)];
        } else if (angle >= 22.5 && angle < 67.5) {
            q = gradient[(y + 1) * width + (x - 1)];
            r = gradient[(y - 1) * width + (x + 1)];
        } else if (angle >= 67.5 && angle < 112.5) {
            q = gradient[(y + 1) * width + x];
            r = gradient[(y - 1) * width + x];
        } else if (angle >= 112.5 && angle < 157.5) {
            q = gradient[(y - 1) * width + (x - 1)];
            r = gradient[(y + 1) * width + (x + 1)];
        }
        
        if (gradient[y * width + x] >= q && gradient[y * width + x] >= r) {
            output[y * width + x] = gradient[y * width + x];
        } else {
            output[y * width + x] = 0;
        }
    }
}

__global__ void double_threshold(float *gradient, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float pixel = gradient[y * width + x];
    if (pixel >= HIGH_THRESHOLD) {
        output[y * width + x] = 255;
    } else if (pixel >= LOW_THRESHOLD) {
        output[y * width + x] = 128;
    } else {
        output[y * width + x] = 0;
    }
}

__global__ void edge_tracking_hysteresis(unsigned char *image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    if (image[y * width + x] == 128) { // Weak edge
        if (image[(y - 1) * width + (x - 1)] == 255 || image[(y - 1) * width + x] == 255 ||
            image[(y - 1) * width + (x + 1)] == 255 || image[y * width + (x - 1)] == 255 ||
            image[y * width + (x + 1)] == 255 || image[(y + 1) * width + (x - 1)] == 255 ||
            image[(y + 1) * width + x] == 255 || image[(y + 1) * width + (x + 1)] == 255) {
            image[y * width + x] = 255;
        } else {
            image[y * width + x] = 0;
        }
    }
}

int main(int argc, char **argv) {
    // Start total time measurement
    auto start_total = high_resolution_clock::now();
    
    if (argc != 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return -1;
    }

    // Start image loading time measurement
    auto start_load = high_resolution_clock::now();
    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    if (image.empty()) {
        printf("Failed to load image\n");
        return -1;
    }
    auto end_load = high_resolution_clock::now();
    auto duration_load = duration_cast<milliseconds>(end_load - start_load);
    printf("Time taken to load image: %lld ms\n", duration_load.count());

    int width = image.cols;
    int height = image.rows;
    size_t img_size = width * height * sizeof(unsigned char);
    size_t grad_size = width * height * sizeof(float);

    // Start memory allocation time measurement
    auto start_alloc = high_resolution_clock::now();
    unsigned char *d_input, *d_blur, *d_sobel, *d_nms, *d_final;
    float *d_gradient, *d_direction;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, img_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_blur, img_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sobel, img_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_nms, img_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_final, img_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_gradient, grad_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_direction, grad_size));
    auto end_alloc = high_resolution_clock::now();
    auto duration_alloc = duration_cast<milliseconds>(end_alloc - start_alloc);
    printf("Time taken for CUDA memory allocation: %lld ms\n", duration_alloc.count());
    
    // Memory transfer time (host to device)
    auto start_h2d = high_resolution_clock::now();
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, image.data, img_size, cudaMemcpyHostToDevice));
    auto end_h2d = high_resolution_clock::now();
    auto duration_h2d = duration_cast<milliseconds>(end_h2d - start_h2d);
    printf("Time taken for host to device transfer: %lld ms\n", duration_h2d.count());

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // CUDA kernel timing events
    cudaEvent_t start, stop;
    float kernel_time = 0.0f;
    float total_kernel_time = 0.0f;

    // Gaussian Blur
    printf("Applying Gaussian Blur...\n");
    startTimer(start, stop);
    gaussian_blur<<<gridSize, blockSize>>>(d_input, d_blur, width, height);
    kernel_time = stopTimer(start, stop);
    total_kernel_time += kernel_time;
    printf("Gaussian Blur kernel execution time: %.2f ms\n", kernel_time);
    
    // Memory copy for Gaussian Blur result
    auto start_blur_copy = high_resolution_clock::now();
    Mat blurred(height, width, CV_8UC1);
    CHECK_CUDA_ERROR(cudaMemcpy(blurred.data, d_blur, img_size, cudaMemcpyDeviceToHost));
    auto end_blur_copy = high_resolution_clock::now();
    auto duration_blur_copy = duration_cast<milliseconds>(end_blur_copy - start_blur_copy);
    printf("Time taken to copy Gaussian Blur result: %lld ms\n", duration_blur_copy.count());
    imwrite("gaussian_blur.png", blurred);
    
    // Sobel Filter
    printf("Applying Sobel Filter...\n");
    startTimer(start, stop);
    sobel_filter<<<gridSize, blockSize>>>(d_blur, d_sobel, d_gradient, d_direction, width, height);
    kernel_time = stopTimer(start, stop);
    total_kernel_time += kernel_time;
    printf("Sobel Filter kernel execution time: %.2f ms\n", kernel_time);
    
    // Memory copy for Sobel result
    auto start_sobel_copy = high_resolution_clock::now();
    Mat sobel(height, width, CV_8UC1);
    CHECK_CUDA_ERROR(cudaMemcpy(sobel.data, d_sobel, img_size, cudaMemcpyDeviceToHost));
    auto end_sobel_copy = high_resolution_clock::now();
    auto duration_sobel_copy = duration_cast<milliseconds>(end_sobel_copy - start_sobel_copy);
    printf("Time taken to copy Sobel Filter result: %lld ms\n", duration_sobel_copy.count());
    imwrite("sobel.png", sobel);

    // Non-Maximum Suppression
    printf("Applying Non-Maximum Suppression...\n");
    startTimer(start, stop);
    non_maximum_suppression<<<gridSize, blockSize>>>(d_gradient, d_direction, d_nms, width, height);
    kernel_time = stopTimer(start, stop);
    total_kernel_time += kernel_time;
    printf("Non-Maximum Suppression kernel execution time: %.2f ms\n", kernel_time);
    
    // Memory copy for NMS result
    auto start_nms_copy = high_resolution_clock::now();
    Mat nms(height, width, CV_8UC1);
    CHECK_CUDA_ERROR(cudaMemcpy(nms.data, d_nms, img_size, cudaMemcpyDeviceToHost));
    auto end_nms_copy = high_resolution_clock::now();
    auto duration_nms_copy = duration_cast<milliseconds>(end_nms_copy - start_nms_copy);
    printf("Time taken to copy Non-Maximum Suppression result: %lld ms\n", duration_nms_copy.count());
    imwrite("nms.png", nms);
    
    // Double Thresholding
    printf("Applying Double Thresholding...\n");
    startTimer(start, stop);
    double_threshold<<<gridSize, blockSize>>>(d_gradient, d_final, width, height);
    kernel_time = stopTimer(start, stop);
    total_kernel_time += kernel_time;
    printf("Double Thresholding kernel execution time: %.2f ms\n", kernel_time);
    
    // Memory copy for threshold result
    auto start_threshold_copy = high_resolution_clock::now();
    Mat thresholded(height, width, CV_8UC1);
    CHECK_CUDA_ERROR(cudaMemcpy(thresholded.data, d_final, img_size, cudaMemcpyDeviceToHost));
    auto end_threshold_copy = high_resolution_clock::now();
    auto duration_threshold_copy = duration_cast<milliseconds>(end_threshold_copy - start_threshold_copy);
    printf("Time taken to copy Double Thresholding result: %lld ms\n", duration_threshold_copy.count());
    imwrite("double_threshold.png", thresholded);
    
    // Edge Tracking with Hysteresis
    printf("Applying Edge Tracking with Hysteresis...\n");
    startTimer(start, stop);
    edge_tracking_hysteresis<<<gridSize, blockSize>>>(d_final, width, height);
    kernel_time = stopTimer(start, stop);
    total_kernel_time += kernel_time;
    printf("Edge Tracking with Hysteresis kernel execution time: %.2f ms\n", kernel_time);
    
    // Memory copy for final result
    auto start_final_copy = high_resolution_clock::now();
    Mat hysteresis(height, width, CV_8UC1);
    CHECK_CUDA_ERROR(cudaMemcpy(hysteresis.data, d_final, img_size, cudaMemcpyDeviceToHost));
    auto end_final_copy = high_resolution_clock::now();
    auto duration_final_copy = duration_cast<milliseconds>(end_final_copy - start_final_copy);
    printf("Time taken to copy final result: %lld ms\n", duration_final_copy.count());
    imwrite("hysteresis.png", hysteresis);

    // Free CUDA memory
    auto start_free = high_resolution_clock::now();
    cudaFree(d_input);
    cudaFree(d_blur);
    cudaFree(d_sobel);
    cudaFree(d_nms);
    cudaFree(d_final);
    cudaFree(d_gradient);
    cudaFree(d_direction);
    auto end_free = high_resolution_clock::now();
    auto duration_free = duration_cast<milliseconds>(end_free - start_free);
    printf("Time taken to free CUDA memory: %lld ms\n", duration_free.count());

    // End total time measurement
    auto end_total = high_resolution_clock::now();
    auto duration_total = duration_cast<milliseconds>(end_total - start_total);
    printf("\n========================================\n");
    printf("Total kernel execution time: %.2f ms\n", total_kernel_time);
    printf("Total execution time: %lld ms\n", duration_total.count());
    printf("========================================\n");
    

    return 0;
}
