CXX=g++
CXXFLAGS=-std=c++11 -Wall -Wno-unknown-pragmas -O3 -fopenmp -fstrict-aliasing
LINK.o=$(CXX) $(CXXFLAGS)

INCLUDES := -I ../eigen3 -I ../libfastsparse
LIBLOCS := -L/opt/OpenBLAS/lib -L/usr/local/lib -L/usr/lib/openblas-base
#LDFLAGS := -lm -lopenblas -lpthread -lgfortran
LDFLAGS := -lm -lblas -llapack -lpthread -lgfortran

all: test
test: tests

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

OBJS = mvnormal.o chol.o linop.o latentprior.o latentpriorvb.o tests.o
tests: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o tests $(OBJS) $(LIBLOCS) $(LDFLAGS)

clean:
	rm -f tests *.o
