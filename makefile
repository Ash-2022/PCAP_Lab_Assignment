# Compiler and flags
CXX = g++
CFLAGS = -I/usr/include/opencv4 -O3
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

# Target executable
TARGET_CPU = canny_cpu_benchmark
TARGET_CUDA = canny_cuda_benchmark

# Source files
SRC_CPU = canny_cpu.cpp
SRC_CUDA = canny_cuda.cu

# Image file to process
IMAGE = Test_Image.jpg

# Build all targets
all: $(TARGET_CPU) $(TARGET_CUDA)

# Build CPU version
$(TARGET_CPU): $(SRC_CPU)
	$(CXX) -o $(TARGET_CPU) $(SRC_CPU) $(CFLAGS) $(LDFLAGS)

# Build CUDA version
$(TARGET_CUDA): $(SRC_CUDA)
	nvcc -o $(TARGET_CUDA) $(SRC_CUDA) $(CFLAGS) $(LDFLAGS)

# Run CPU version
run_cpu: $(TARGET_CPU)
	./$(TARGET_CPU) $(IMAGE)

# Run CUDA version
run_cuda: $(TARGET_CUDA)
	./$(TARGET_CUDA) $(IMAGE)

# Run both and compare
compare: $(TARGET_CPU) $(TARGET_CUDA)
	@echo "Running CPU implementation..."
	./$(TARGET_CPU) $(IMAGE) > cpu_benchmark.txt
	@echo "\nRunning CUDA implementation..."
	./$(TARGET_CUDA) $(IMAGE) > cuda_benchmark.txt
	@echo "\nComparison complete. Results saved to cpu_benchmark.txt and cuda_benchmark.txt"

# Generate performance report
benchmark_cpu: $(TARGET_CPU)
	@echo "Running CPU Canny edge detection benchmark..."
	@./$(TARGET_CPU) $(IMAGE) > cpu_benchmark_report.txt
	@echo "Benchmark results saved to cpu_benchmark_report.txt"

# Generate performance report for CUDA
benchmark_cuda: $(TARGET_CUDA)
	@echo "Running CUDA Canny edge detection benchmark..."
	@./$(TARGET_CUDA) $(IMAGE) > cuda_benchmark_report.txt
	@echo "Benchmark results saved to cuda_benchmark_report.txt"

# Clean up generated files
clean:
	rm -f $(TARGET_CPU) $(TARGET_CUDA) *.png *_benchmark*.txt
