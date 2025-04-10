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
IMAGES = Test_Image.jpg 4000x5000_Image.jpg 4000x6000_Image.jpg

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

# Run CPU and CUDA versions on all images and compare
compare: $(TARGET_CPU) $(TARGET_CUDA)
	@for img in $(IMAGES); do \
		base=$$(basename $$img .jpg); \
		echo "Running CPU implementation on $$img..."; \
		./$(TARGET_CPU) $$img > cpu_benchmark_$$base.txt; \
		echo "Running CUDA implementation on $$img..."; \
		./$(TARGET_CUDA) $$img > cuda_benchmark_$$base.txt; \
		echo "Comparison for $$img complete. Results saved to cpu_benchmark_$$base.txt and cuda_benchmark_$$base.txt"; \
		echo ""; \
	done

# Benchmark all images with CPU
benchmark_cpu_all: $(TARGET_CPU)
	@for img in $(IMAGES); do \
		out=cpu_benchmark_$$(basename $$img .jpg).txt; \
		echo "Benchmarking $$img with CPU..."; \
		./$(TARGET_CPU) $$img > $$out; \
		echo "Saved to $$out"; \
	done

# Benchmark all images with CUDA
benchmark_cuda_all: $(TARGET_CUDA)
	@for img in $(IMAGES); do \
		out=cuda_benchmark_$$(basename $$img .jpg).txt; \
		echo "Benchmarking $$img with CUDA..."; \
		./$(TARGET_CUDA) $$img > $$out; \
		echo "Saved to $$out"; \
	done

# Run both CPU and CUDA benchmarks for all images
benchmark_all: benchmark_cpu_all benchmark_cuda_all
	@echo "All benchmarks completed."


# Clean up generated files
clean:
	rm -f $(TARGET_CPU) $(TARGET_CUDA) *.png *_benchmark*.txt
