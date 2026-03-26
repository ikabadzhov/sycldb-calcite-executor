CXX := icpx -fsycl -fsycl-embed-ir -Wall -fsycl-targets=spir64,nvptx64-nvidia-cuda,amdgcn-amd-amdhsa -Xsycl-target-backend=nvptx64-nvidia-cuda "--offload-arch=sm_89" -Xsycl-target-backend=amdgcn-amd-amdhsa "--offload-arch=gfx90a"
CXXFLAGS := -std=c++20 -O3 -I. -Iapp -Ibenchmark -Iexecutor -Iruntime -Ikernels -Igen-cpp -Ioperations -Imodels -I/usr/local/include
LDFLAGS := -L/usr/local/lib -lthrift -Wl,-rpath=/usr/local/lib

SRC := main.cpp \
	$(wildcard app/*.cpp) \
	$(wildcard benchmark/*.cpp) \
	$(wildcard executor/*.cpp) \
	$(wildcard kernels/*.cpp) \
	$(wildcard operations/*.cpp) \
	$(wildcard runtime/*.cpp) \
	gen-cpp/CalciteServer.cpp \
	gen-cpp/calciteserver_types.cpp
HEADERS := gen-cpp/CalciteServer.h gen-cpp/calciteserver_types.h common.hpp \
	$(wildcard app/*.hpp) \
	$(wildcard benchmark/*.hpp) \
	$(wildcard executor/*.hpp) \
	$(wildcard runtime/*.hpp) \
	$(wildcard kernels/*.hpp) \
	$(wildcard operations/*.hpp) \
	$(wildcard models/*.hpp)
TARGET := client

.PHONY: clean benchmark benchmark-nvidia-staged benchmark-simultaneous

$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

benchmark: $(TARGET)
	bash benchmark_all_devices.sh

benchmark-nvidia-staged: $(TARGET)
	bash benchmark_nvidia_staged.sh

benchmark-simultaneous: $(TARGET)
	bash benchmark_all_devices_simultaneous.sh

clean:
	-rm client
	-rm q*.res
