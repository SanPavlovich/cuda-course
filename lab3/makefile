# NVCC = nvcc
# NVCC_FLAGS = -O3 -std=c++17 -Xcompiler="-Wall -Wextra"
# TARGET = spectral_angle_cuda

# all: $(TARGET)

# $(TARGET): main.cu
# 	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) main.cu

# clean:
# 	rm -f $(TARGET) *.o output.txt

# run: $(TARGET)
# 	./$(TARGET) example_1_input.txt output.txt

# test: $(TARGET)
# 	@echo "Running test..."
# 	./$(TARGET) example_1_input.txt output.txt
# 	@echo -e "\nFirst few pixels of output:"
# 	@head -5 output.txt

# .PHONY: all clean run test
compile_c:
	gcc lab.c -o labc -lm
run_c:
	./labc