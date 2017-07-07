from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "<memory>" namespace "std" nogil:
    ctypedef void* nullptr_t;

    cdef cppclass unique_ptr[T]:
        unique_ptr()
        unique_ptr(T*)

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
    cdef cppclass SparseDoubleFeat:
        SparseDoubleFeat()
        SparseDoubleFeat(int nrow, int ncol, long nnz, int* rows, int* cols, double* vals)
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
        MacauPrior(int nlatent, unique_ptr[FType] & Fmat, bool comp_FtF)
        void setLambdaBeta(double lb)
        void setTol(double t)
    MacauPrior[MatrixXd]* make_dense_prior(int nlatent, double* ptr, int nrows, int ncols, bool colMajor, bool comp_FtF)

cdef extern from "macauoneprior.h":
    cdef cppclass MacauOnePrior[FType](ILatentPrior):
        MacauOnePrior()
        MacauOnePrior(int nlatent, unique_ptr[FType] & Fmat)
        void setLambdaBeta(double lb)

cdef extern from "macau.h":
    cdef cppclass Macau:
        void addPrior(unique_ptr[ILatentPrior] & prior)
        void setSamples(int burnin, int nsamples)
        void setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols)
        void setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols)
        void setRelationData(int* idx, int nmodes, double* values, int nnz, int* dims)
        void setRelationDataTest(int* idx, int nmodes, double* values, int nnz, int* dims)
        void setVerbose(bool v)
        double getRmseTest()
        VectorXd getPredictions()
        VectorXd getStds()
        MatrixXd getTestData()
        void run() except *
        void setSaveModel(bool save)
        void setSavePrefix(string pref)
    cdef cppclass MacauX[DType](Macau):
        MacauX()
        MacauX(int num_latent)
    Macau* make_macau_probit(int nmodes, int num_latent)
    Macau* make_macau_fixed(int nmodes, int num_latent, double precision)
    Macau* make_macau_adaptive(int nmodes, int num_latent, double sn_init, double sn_max)
