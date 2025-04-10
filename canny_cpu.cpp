#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <chrono>  // For time measurement

#define LOW_THRESHOLD 50
#define HIGH_THRESHOLD 150

using namespace cv;
using namespace std::chrono;  // For time measurement

// Function prototypes
void gaussian_blur_cpu(unsigned char *input, unsigned char *output, int width, int height);
void sobel_filter_cpu(unsigned char *input, unsigned char *output, float *gradient, float *direction, int width, int height);
void non_maximum_suppression_cpu(float *gradient, float *direction, unsigned char *output, int width, int height);
void double_threshold_cpu(float *gradient, unsigned char *output, int width, int height);
void edge_tracking_hysteresis_cpu(unsigned char *image, int width, int height);

void gaussian_blur_cpu(unsigned char *input, unsigned char *output, int width, int height) {
    float filter[5][5] = {
        {1, 4, 7, 4, 1},
        {4, 16, 26, 16, 4},
        {7, 26, 41, 26, 7},
        {4, 16, 26, 16, 4},
        {1, 4, 7, 4, 1}
    };
    
    float weight = 273.0;
    int radius = 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float blur_value = 0.0;
            
            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    int nx = std::min(std::max(x + j, 0), width - 1);
                    int ny = std::min(std::max(y + i, 0), height - 1);
                    blur_value += input[ny * width + nx] * filter[i + radius][j + radius];
                }
            }
            output[y * width + x] = blur_value / weight;
        }
    }
}

void sobel_filter_cpu(unsigned char *input, unsigned char *output, float *gradient, float *direction, int width, int height) {
    int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float grad_x = 0;
            float grad_y = 0;
            
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int nx = std::min(std::max(x + j, 0), width - 1);
                    int ny = std::min(std::max(y + i, 0), height - 1);
                    grad_x += input[ny * width + nx] * gx[i + 1][j + 1];
                    grad_y += input[ny * width + nx] * gy[i + 1][j + 1];
                }
            }
            gradient[y * width + x] = sqrt(grad_x * grad_x + grad_y * grad_y);
            direction[y * width + x] = atan2(grad_y, grad_x);
            output[y * width + x] = std::min(255, (int)gradient[y * width + x]);
        }
    }
}

void non_maximum_suppression_cpu(float *gradient, float *direction, unsigned char *output, int width, int height) {
    // Initialize output to zeros
    memset(output, 0, width * height * sizeof(unsigned char));
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
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
}

void double_threshold_cpu(float *gradient, unsigned char *output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float pixel = gradient[y * width + x];
            if (pixel >= HIGH_THRESHOLD) {
                output[y * width + x] = 255;
            } else if (pixel >= LOW_THRESHOLD) {
                output[y * width + x] = 128;
            } else {
                output[y * width + x] = 0;
            }
        }
    }
}

void edge_tracking_hysteresis_cpu(unsigned char *image, int width, int height) {
    // Create a copy of the image for checking
    unsigned char* temp = new unsigned char[width * height];
    memcpy(temp, image, width * height * sizeof(unsigned char));
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            if (temp[y * width + x] == 128) {  // Weak edge
                if (temp[(y - 1) * width + (x - 1)] == 255 || temp[(y - 1) * width + x] == 255 ||
                    temp[(y - 1) * width + (x + 1)] == 255 || temp[y * width + (x - 1)] == 255 ||
                    temp[y * width + (x + 1)] == 255 || temp[(y + 1) * width + (x - 1)] == 255 ||
                    temp[(y + 1) * width + x] == 255 || temp[(y + 1) * width + (x + 1)] == 255) {
                    image[y * width + x] = 255;
                } else {
                    image[y * width + x] = 0;
                }
            }
        }
    }
    
    delete[] temp;
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
    
    // Allocate host memory
    unsigned char *h_input = image.data;
    unsigned char *h_blur = new unsigned char[width * height];
    unsigned char *h_sobel = new unsigned char[width * height];
    unsigned char *h_nms = new unsigned char[width * height];
    unsigned char *h_final = new unsigned char[width * height];
    float *h_gradient = new float[width * height];
    float *h_direction = new float[width * height];
    
    auto end_alloc = high_resolution_clock::now();
    auto duration_alloc = duration_cast<milliseconds>(end_alloc - start_alloc);
    printf("Time taken for memory allocation: %lld ms\n", duration_alloc.count());

    // Gaussian Blur
    printf("Applying Gaussian Blur...\n");
    auto start_blur = high_resolution_clock::now();
    gaussian_blur_cpu(h_input, h_blur, width, height);
    auto end_blur = high_resolution_clock::now();
    auto duration_blur = duration_cast<milliseconds>(end_blur - start_blur);
    printf("Gaussian Blur execution time: %lld ms\n", duration_blur.count());
    
    // Save Gaussian Blur result
    Mat blurred(height, width, CV_8UC1, h_blur);
    imwrite("cpu_gaussian_blur.png", blurred);
    
    // Sobel Filter
    printf("Applying Sobel Filter...\n");
    auto start_sobel = high_resolution_clock::now();
    sobel_filter_cpu(h_blur, h_sobel, h_gradient, h_direction, width, height);
    auto end_sobel = high_resolution_clock::now();
    auto duration_sobel = duration_cast<milliseconds>(end_sobel - start_sobel);
    printf("Sobel Filter execution time: %lld ms\n", duration_sobel.count());
    
    // Save Sobel result
    Mat sobel(height, width, CV_8UC1, h_sobel);
    imwrite("cpu_sobel.png", sobel);

    // Non-Maximum Suppression
    printf("Applying Non-Maximum Suppression...\n");
    auto start_nms = high_resolution_clock::now();
    non_maximum_suppression_cpu(h_gradient, h_direction, h_nms, width, height);
    auto end_nms = high_resolution_clock::now();
    auto duration_nms = duration_cast<milliseconds>(end_nms - start_nms);
    printf("Non-Maximum Suppression execution time: %lld ms\n", duration_nms.count());
    
    // Save NMS result
    Mat nms(height, width, CV_8UC1, h_nms);
    imwrite("cpu_nms.png", nms);
    
    // Double Thresholding
    printf("Applying Double Thresholding...\n");
    auto start_threshold = high_resolution_clock::now();
    double_threshold_cpu(h_gradient, h_final, width, height);
    auto end_threshold = high_resolution_clock::now();
    auto duration_threshold = duration_cast<milliseconds>(end_threshold - start_threshold);
    printf("Double Thresholding execution time: %lld ms\n", duration_threshold.count());
    
    // Save threshold result
    Mat thresholded(height, width, CV_8UC1, h_final);
    imwrite("cpu_double_threshold.png", thresholded);
    
    // Edge Tracking with Hysteresis
    printf("Applying Edge Tracking with Hysteresis...\n");
    auto start_hysteresis = high_resolution_clock::now();
    edge_tracking_hysteresis_cpu(h_final, width, height);
    auto end_hysteresis = high_resolution_clock::now();
    auto duration_hysteresis = duration_cast<milliseconds>(end_hysteresis - start_hysteresis);
    printf("Edge Tracking with Hysteresis execution time: %lld ms\n", duration_hysteresis.count());
    
    // Save final result
    Mat hysteresis(height, width, CV_8UC1, h_final);
    imwrite("cpu_hysteresis.png", hysteresis);

    // Free memory
    auto start_free = high_resolution_clock::now();
    delete[] h_blur;
    delete[] h_sobel;
    delete[] h_nms;
    delete[] h_final;
    delete[] h_gradient;
    delete[] h_direction;
    auto end_free = high_resolution_clock::now();
    auto duration_free = duration_cast<milliseconds>(end_free - start_free);
    printf("Time taken to free memory: %lld ms\n", duration_free.count());

    // Calculate kernel times sum
    auto total_kernel_time = duration_blur + duration_sobel + duration_nms + duration_threshold + duration_hysteresis;
    printf("\nTotal kernel execution time: %lld ms\n", total_kernel_time.count());

    // End total time measurement
    auto end_total = high_resolution_clock::now();
    auto duration_total = duration_cast<milliseconds>(end_total - start_total);
    printf("\n========================================\n");
    printf("Total execution time: %lld ms\n", duration_total.count());
    printf("========================================\n");

    return 0;
}
