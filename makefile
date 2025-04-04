# Compiler and flags
NVCC = nvcc
CFLAGS = -I/usr/include/opencv4
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

# Target executable
TARGET = canny_cuda

# Source files
SRC = canny_cuda.cu

# Image file to process
IMAGE = Test_Image.jpg

# Build the executable
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC) $(CFLAGS) $(LDFLAGS)

# Run the executable with a test image
run: $(TARGET)
	./$(TARGET) $(IMAGE)

# Clean up generated files
clean:
	rm -f $(TARGET) *.png
