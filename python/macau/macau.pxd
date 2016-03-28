from libcpp cimport bool

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
    void hello2(double* x, double* y, int n, int k)
    MatrixXd getx()
    void eigenQR(double* X, int nrow, int ncol)
    void At_mul_A_eig(MatrixXd & A, MatrixXd & C)

cdef extern from "linop.h":
    cdef cppclass SparseFeat:
        SparseFeat()
        SparseFeat(int nrow, int ncol, long nnz, int* rows, int* cols)
    void At_mul_A_blas(MatrixXd & A, double* AtA)
    int solve_blockcg(MatrixXd & out, SparseFeat & K, double reg, MatrixXd & B, double tol)

cdef extern from "latentprior.h":
    cdef cppclass ILatentPrior:
        pass
    cdef cppclass BPMFPrior(ILatentPrior):
        MatrixXd Lambda
        VectorXd mu
        BPMFPrior()
        BPMFPrior(int num_latent)
    cdef cppclass MacauPrior[FType](ILatentPrior):
        MacauPrior()
        MacauPrior(int nlatent, FType & Fmat, bool comp_FtF)

cdef extern from "macau.h":
    cdef cppclass Macau:
        Macau()
        Macau(int num_latent)
        void addPrior(ILatentPrior* prior)
        void setPrecision(double p)
        void setSamples(int burnin, int nsamples)
        void setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols)
        void setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols)
        double getRmseTest()
        void run()

