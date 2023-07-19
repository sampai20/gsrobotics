CC = g++
CFLAGS = -g -Wall -std=c++11 -O3 -Wall -shared -std=c++11 `python3 -m pybind11 --includes` -I ./
SRCS = tracking_class.cpp 
PROG = src/lib/find_marker`python3-config --extension-suffix`

$(PROG):
	mkdir -p src/lib
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS)

clean:
	rm src/lib/find_marker
