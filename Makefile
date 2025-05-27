INCLUDES := -Ikernels -Igen-cpp -I/usr/local/include -L/usr/local/lib -Wl,-rpath=/usr/local/lib

client: main.o gen-cpp/CalciteServer.o gen-cpp/calciteserver_types.o
		g++ -g -o client $^ -lthrift $(INCLUDES)

%.o: %.cpp
		g++ -g -c -o $@ $< $(INCLUDES)

clean:
		-rm client
		-rm ./**.o
		-rm ./**/*.o