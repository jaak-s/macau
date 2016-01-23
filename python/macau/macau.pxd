cdef extern from "Eigen/Dense" namespace "Eigen":
    cdef cppclass MatrixXd:
        MatrixXd()
        MatrixXd(int nrow, int ncol)
        int rows()
        int cols()
        double* data()
    cdef cppclass VectorXd:
        VectorXd()
        VectorXd(int n)
        int size()
        double* data()
    T Map[T](double* x, int nrows, int ncols)

cdef extern from "hello.h":
    void hello(double* x, double* y, int n, int k)
    MatrixXd getx()
    void eigenQR(double* X, int nrow, int ncol)

cdef extern from "linop.h":
    void At_mul_A_blas(MatrixXd & A, double* AtA)

cdef extern from "latentprior.h":
    cdef cppclass BPMFPrior:
        MatrixXd Lambda
        VectorXd mu
        BPMFPrior(int num_latent)

cdef extern from "macau.h":
    cdef cppclass Macau:                                                                                               
        BPMFPrior prior_u
        BPMFPrior prior_m
        Macau()
        Macau(int num_latent)
        void setPrecision(double p)
        void setSamples(int burnin, int nsamples)
        void setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols)
        void setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols)
        double getRmseTest()
        void run()

