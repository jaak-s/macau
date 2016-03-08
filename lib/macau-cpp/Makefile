CXX=g++
CXXFLAGS=-std=c++11 -Wall -Wno-unknown-pragmas -DNDEBUG -O3 -fopenmp -fstrict-aliasing
LINK.o=$(CXX) $(CXXFLAGS)

INCLUDES := -I ../eigen3 -I ../libfastsparse
LIBLOCS := -L/usr/lib/openblas-base -L/opt/OpenBLAS/lib -L/usr/local/lib
LDFLAGS := -lm -lopenblas -lpthread

all: test
test: tests

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

OBJS = chol.o linop.o tests.o
tests: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o tests $(OBJS) $(LDFLAGS)

clean:
	rm -f tests *.o
