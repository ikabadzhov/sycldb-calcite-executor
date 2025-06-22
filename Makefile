CXX = clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wall
CXXFLAGS = -std=c++17 -g -Ikernels -Igen-cpp -I/usr/local/include
LDFLAGS = -L/usr/local/lib -lthrift -Wl,-rpath=/usr/local/lib

SRC = main.cpp gen-cpp/CalciteServer.cpp gen-cpp/calciteserver_types.cpp
TARGET = client

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)
	./client | grep '^>>>'
	./client ./queries/transformed/q11.sql | grep '^>>>'
	./client ./queries/transformed/q41.sql | grep '^>>>'

clean:
		-rm client