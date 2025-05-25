INCLUDES := -Ikernels -Igen-cpp

client: main.o gen-cpp/CalciteServer.o gen-cpp/calciteserver_types.o
	g++ -g -o client $^ -lthrift $(INCLUDES)

%.o: %.cpp
	g++ -g -c -o $@ $< $(INCLUDES)