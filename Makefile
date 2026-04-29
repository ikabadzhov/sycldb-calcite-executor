export PATH := /media/ACPP/AdaptiveCpp-25.10.0/install/bin:$(PATH)
export ACPP_CPU_CXX := /usr/bin/g++
export ACPP_CLANG := /usr/bin/clang++-16
export ACPP_CUDA_PATH := /usr/local/cuda-12.6
export ACPP_ROCM_PATH := /opt/rocm-6.3.3

CXX := acpp
CXXFLAGS := -std=c++20 -O3 -fPIC --acpp-targets="generic" \
	-I. -Iapp -Ibenchmark -Iexecutor -Iruntime -Ikernels -Igen-cpp -Ioperations -Imodels -I/usr/local/include \
	-I/home/ivan/.local/lib/python3.12/site-packages/pyarrow/include
LDFLAGS := -L/usr/local/lib -lthrift -Wl,-rpath=/usr/local/lib \
	-L/home/ivan/.local/lib/python3.12/site-packages/pyarrow -larrow -lparquet

SRC := main.cpp \
	$(wildcard app/*.cpp) \
	$(wildcard benchmark/*.cpp) \
	$(wildcard executor/*.cpp) \
	$(wildcard kernels/*.cpp) \
	$(wildcard operations/*.cpp) \
	$(wildcard runtime/*.cpp) \
	gen-cpp/CalciteServer.cpp \
	gen-cpp/calciteserver_types.cpp

OBJ := $(SRC:.cpp=.o)

HEADERS := gen-cpp/CalciteServer.h gen-cpp/calciteserver_types.h common.hpp \
	$(wildcard app/*.hpp) \
	$(wildcard benchmark/*.hpp) \
	$(wildcard executor/*.hpp) \
	$(wildcard runtime/*.hpp) \
	$(wildcard kernels/*.hpp) \
	$(wildcard operations/*.hpp) \
	$(wildcard models/*.hpp)
TARGET := client

.PHONY: clean run-ssb

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run-ssb: $(TARGET)
	./$(TARGET) --benchmark-ssb --suite benchmark/ssb_queries.txt

clean:
	-rm client
	-rm $(OBJ)
	-rm q*.res
