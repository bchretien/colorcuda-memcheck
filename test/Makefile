NVCC      = nvcc
CXX       = $(NVCC)
LINK      = $(CXX)
NVCCFLAGS = -g -G -lineinfo

# Program-specific
TARGET= check_bug
SRC = check_bug.cu
OBJ = check_bug.cu.o

# Build Rules
.SUFFIXES: .cu

$(TARGET): $(OBJ) 
	$(LINK) -o $(TARGET) $(OBJ)

$(OBJ): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $(OBJ) -c $< 

clean:
	rm -f $(OBJ) $(TARGET)
