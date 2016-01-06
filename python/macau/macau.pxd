cdef extern from "hello.h":
    int hello()

cdef extern from "macau.h":
    cdef cppclass Macau:                                                                                               
        Macau()
        Macau(int num_latent)
        void setPrecision(double p)
        void setSamples(int burnin, int nsamples)
        void setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols)
        void setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols)
        double getRmseTest()
        void run()

