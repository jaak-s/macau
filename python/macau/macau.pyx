cimport cython
import numpy as np
cimport numpy as np
import scipy as sp
import timeit
import numbers
import pandas as pd
import signal

class MacauResult(object):
  def __init__(self):
    pass
  def __repr__(self):
    s = ("Matrix factorization results\n" +
         "Test RMSE:        %.4f\n" % self.rmse_test +
         "Matrix size:      [%d x %d]\n" % (self.Yshape[0], self.Yshape[1]) +
         "Number of train:  %d\n" % self.ntrain +
         "Number of test:   %d\n" % self.ntest  +
         "To see predictions on test set see '.prediction' field.")
    return s

cpdef mul_blas(np.ndarray[np.double_t, ndim=2] x, np.ndarray[np.double_t, ndim=2] y):
    if y.shape[0] != y.shape[1]:
        raise ValueError("y must be square matrix.")
    if x.shape[1] != y.shape[0]:
        raise ValueError("num columns in x and y must be the same.")
    hello(&x[0,0], &y[0,0], x.shape[1], x.shape[0])

cpdef mul_blas2(np.ndarray[np.double_t, ndim=2] x, np.ndarray[np.double_t, ndim=2] y):
    if y.shape[0] != y.shape[1]:
        raise ValueError("y must be square matrix.")
    if x.shape[1] != y.shape[0]:
        raise ValueError("num columns in x and y must be the same.")
    hello2(&x[0,0], &y[0,0], x.shape[1], x.shape[0])

cpdef py_getx():
    cdef MatrixXd m = getx()
    cdef np.ndarray[np.double_t, ndim=2] A = matview(&m)
    #print(A)
    return A.copy()

cpdef py_eigenQR(np.ndarray[np.double_t, ndim=2] x):
    eigenQR(& x[0,0], x.shape[0], x.shape[1])

cpdef blas_AtA(np.ndarray[np.double_t, ndim=2] X, np.ndarray[np.double_t, ndim=2] AtA):
    cdef MatrixXd Xeig   = Map[MatrixXd](&X[0,0],   X.shape[0],   X.shape[1])
    cdef MatrixXd AtAeig = Map[MatrixXd](&AtA[0,0], AtA.shape[0], AtA.shape[1])
    #At_mul_A_blas(Xeig, & AtA[0,0])
    At_mul_A_eig(Xeig, AtAeig)
    return 0

cdef SparseFeat* sparse2SparseBinFeat(X):
    X = X.tocoo(copy=False)
    cdef np.ndarray[int] irows = X.row.astype(np.int32, copy=False)
    cdef np.ndarray[int] icols = X.col.astype(np.int32, copy=False)
    return new SparseFeat(X.shape[0], X.shape[1], irows.shape[0], & irows[0], & icols[0])

cdef SparseDoubleFeat* sparse2SparseDoubleFeat(X):
    X = X.tocoo(copy=False)
    cdef np.ndarray[int] irows = X.row.astype(np.int32, copy=False)
    cdef np.ndarray[int] icols = X.col.astype(np.int32, copy=False)
    cdef np.ndarray[np.double_t] vals = X.data.astype(np.double, copy=False)
    return new SparseDoubleFeat(X.shape[0], X.shape[1], irows.shape[0], & irows[0], & icols[0], & vals[0])

cpdef blockcg(X, np.ndarray[np.double_t, ndim=2] B, double reg, double tol = 1e-6):
    if not np.isfortran(B):
        raise ValueError("B must have order='F' (fortran)")
    if B.shape[1] != X.shape[1]:
        raise ValueError("B.shape[1] must equal X.shape[1]")
    X = X.tocoo(copy=False)
    cdef np.ndarray[np.double_t, ndim=2] result = np.zeros( (B.shape[0], B.shape[1]), order='F' )
    cdef MatrixXd result_eig = Map[MatrixXd](&result[0,0], result.shape[0], result.shape[1])

    cdef MatrixXd Beig = Map[MatrixXd](&B[0,0], B.shape[0], B.shape[1])
    cdef MatrixXd out  = Map[MatrixXd](&B[0,0], B.shape[0], B.shape[1])
    cdef np.ndarray[int] irows = X.row.astype(np.int32, copy=False)
    cdef np.ndarray[int] icols = X.col.astype(np.int32, copy=False)
    print("Sparse [%d x %d] matrix" % (X.shape[0], X.shape[1]))
    cdef SparseFeat* K = new SparseFeat(X.shape[0], X.shape[1], irows.shape[0], & irows[0], & icols[0])
    print("Running block-cg")
    cdef double start = timeit.default_timer()
    cdef int niter = solve_blockcg(result_eig, K[0], reg, Beig, tol)
    cdef double end = timeit.default_timer()

    cdef np.ndarray[np.double_t] v = np.zeros(2)
    v[0] = niter
    v[1] = end - start
    del K
    return v
    #cdef np.ndarray[np.double_t] ivals = X.data.astype(np.double, copy=False)

cdef matview(MatrixXd *A):
    cdef int nrow = A.rows()
    cdef int ncol = A.cols()
    if nrow == 0:
      return np.zeros( (nrow, ncol) )
    cdef np.double_t[:,:] view = <np.double_t[:nrow:1, :ncol]> A.data()
    return np.asarray(view)

cdef vecview(VectorXd *v):
    cdef int size = v.size()
    if size == 0:
      return np.zeros( 0 )
    cdef np.double_t[:] view = <np.double_t[:size]> v.data()
    return np.asarray(view)

def make_train_test(Y, ntest):
    if type(Y) not in [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        raise ValueError("Unsupported Y type: %s" + type(Y))
    if not isinstance(ntest, numbers.Real) or ntest < 0:
        raise ValueError("ntest has to be a non-negative number (number or ratio of test samples).")
    Y = Y.tocoo(copy = False)
    if ntest < 1:
        ntest = Y.nnz * ntest
    ntest = int(round(ntest))
    rperm = np.random.permutation(Y.nnz)
    train = rperm[ntest:]
    test  = rperm[0:ntest]
    Ytrain = sp.sparse.coo_matrix( (Y.data[train], (Y.row[train], Y.col[train])), shape=Y.shape )
    Ytest  = sp.sparse.coo_matrix( (Y.data[test],  (Y.row[test],  Y.col[test])),  shape=Y.shape )
    return Ytrain, Ytest

cdef ILatentPrior* make_prior(side, int num_latent, int max_ff_size, double lambda_beta, double tol) except NULL:
    if (side is None) or side == ():
        return new BPMFPrior(num_latent)
    if type(side) not in [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        raise ValueError("Unsupported side information type: '%s'" % type(side).__name__)

    cdef bool compute_ff = (side.shape[1] <= max_ff_size)

    ## binary CSR
    cdef unique_ptr[SparseFeat] sf_ptr
    cdef MacauPrior[SparseFeat]* sf_prior
    if (side.data == 1).all():
        sf_ptr   = unique_ptr[SparseFeat]( sparse2SparseBinFeat(side) )
        sf_prior = new MacauPrior[SparseFeat](num_latent, sf_ptr, compute_ff)
        sf_prior.setLambdaBeta(lambda_beta)
        sf_prior.setTol(tol)
        return sf_prior

    ## double CSR
    cdef unique_ptr[SparseDoubleFeat] sdf_ptr
    sdf_ptr = unique_ptr[SparseDoubleFeat]( sparse2SparseDoubleFeat(side) )
    cdef MacauPrior[SparseDoubleFeat]* sdf_prior = new MacauPrior[SparseDoubleFeat](num_latent, sdf_ptr, compute_ff)
    sdf_prior.setLambdaBeta(lambda_beta)
    sdf_prior.setTol(tol)
    return sdf_prior

cdef ILatentPrior* make_one_prior(side, int num_latent, double lambda_beta) except NULL:
    if (side is None) or side == ():
        return new BPMFPrior(num_latent)
    if type(side) not in [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        raise ValueError("Unsupported side information type: '%s'" % type(side).__name__)

    ## binary CSR
    cdef unique_ptr[SparseFeat] sf_ptr
    cdef MacauOnePrior[SparseFeat]* sf_prior
    if (side.data == 1).all():
        sf_ptr   = unique_ptr[SparseFeat]( sparse2SparseBinFeat(side) )
        sf_prior = new MacauOnePrior[SparseFeat](num_latent, sf_ptr)
        sf_prior.setLambdaBeta(lambda_beta)
        return sf_prior

    ## double CSR
    cdef unique_ptr[SparseDoubleFeat] sdf_ptr
    sdf_ptr = unique_ptr[SparseDoubleFeat]( sparse2SparseDoubleFeat(side) )
    cdef MacauOnePrior[SparseDoubleFeat]* sdf_prior = new MacauOnePrior[SparseDoubleFeat](num_latent, sdf_ptr)
    sdf_prior.setLambdaBeta(lambda_beta)
    return sdf_prior

## API functions:
## 1) F'F
## 2) F*X (X is a matrix)
## 3) F'X (X is a matrix)
## 4) solve(A, b, sym_pos=True) where A is posdef
def bpmf(Y,
         Ytest      = None,
         num_latent = 10,
         precision  = 1.0,
         burnin     = 50,
         nsamples   = 400,
         **keywords):
    return macau(Y,
                 Ytest = Ytest,
                 num_latent = num_latent,
                 precision  = precision,
                 burnin     = burnin,
                 nsamples   = nsamples,
                 **keywords)

def remove_nan(Y):
    if not np.any(np.isnan(Y.data)):
        return Y
    idx = np.where(np.isnan(Y.data) == False)[0]
    return sp.sparse.coo_matrix( (Y.data[idx], (Y.row[idx], Y.col[idx])), shape = Y.shape )

def prepare_Y(Y, Ytest):
    if type(Y) not in [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        raise ValueError("Y must be either coo, csr or csc (from scipy.sparse)")
    Y = Y.tocoo(copy = False)
    Y = remove_nan(Y)
    if Ytest is not None:
        if isinstance(Ytest, numbers.Real):
            Y, Ytest = make_train_test(Y, Ytest)
        if type(Ytest) not in [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
            raise ValueError("Ytest must be either coo, csr or csc (from scipy.sparse)")
        if Ytest.shape != Y.shape:
            raise ValueError("Ytest and Y must have the same shape")
        Ytest = Ytest.tocoo(copy = False)
        Ytest = remove_nan(Ytest)
    return Y, Ytest

def macau(Y,
          Ytest      = None,
          side       = [],
          lambda_beta = 5.0,
          num_latent = 10,
          precision  = 1.0,
          burnin     = 50,
          nsamples   = 400,
          univariate = False,
          tol        = 1e-6,
          sn_max     = 10.0,
          save_prefix= None,
          verbose    = True):
    Y, Ytest = prepare_Y(Y, Ytest)

    cdef np.ndarray[int] irows = Y.row.astype(np.int32, copy=False)
    cdef np.ndarray[int] icols = Y.col.astype(np.int32, copy=False)
    cdef np.ndarray[np.double_t] ivals = Y.data.astype(np.double, copy=False)

    ## side information
    if not side:
        side = [None, None]
    if type(side) not in [list, tuple]:
        raise ValueError("Parameter 'side' must be a tuple or a list.")
    if len(side) != 2:
        raise ValueError("If specified 'side' must contain 2 elements.")

    cdef int D = np.int32(num_latent)
    cdef unique_ptr[ILatentPrior] prior_u
    cdef unique_ptr[ILatentPrior] prior_v
    if univariate:
        prior_u = unique_ptr[ILatentPrior](make_one_prior(side[0], D, lambda_beta))
        prior_v = unique_ptr[ILatentPrior](make_one_prior(side[1], D, lambda_beta))
    else:
        prior_u = unique_ptr[ILatentPrior](make_prior(side[0], D, 10000, lambda_beta, tol))
        prior_v = unique_ptr[ILatentPrior](make_prior(side[1], D, 10000, lambda_beta, tol))

    cdef Macau *macau = new Macau(D)
    macau.addPrior(prior_u)
    macau.addPrior(prior_v)
    macau.setRelationData(&irows[0], &icols[0], &ivals[0], irows.shape[0], Y.shape[0], Y.shape[1]);
    macau.setSamples(np.int32(burnin), np.int32(nsamples))
    macau.setVerbose(verbose)

    if isinstance(precision, str):
      if precision == "adaptive" or precision == "sample":
        macau.setAdaptivePrecision(np.float64(1.0), np.float64(sn_max))
      elif precision == "probit":
        macau.setProbit()
      else:
        raise ValueError("Parameter 'precision' has to be either a number or \"adaptive\" for adaptive precision, or \"probit\" for binary matrices.")
    else:
      macau.setPrecision(np.float64(precision))

    cdef np.ndarray[int] trows, tcols
    cdef np.ndarray[np.double_t] tvals

    if Ytest is not None:
        trows = Ytest.row.astype(np.int32, copy=False)
        tcols = Ytest.col.astype(np.int32, copy=False)
        tvals = Ytest.data.astype(np.double, copy=False)
        macau.setRelationDataTest(&trows[0], &tcols[0], &tvals[0], trows.shape[0], Y.shape[0], Y.shape[1])

    if save_prefix is None:
        macau.setSaveModel(0)
    else:
        if type(save_prefix) != str:
            raise ValueError("Parameter 'save_prefix' has to be a string (str) or None.")
        macau.setSaveModel(1)
        macau.setSavePrefix(save_prefix)

    macau.run()
    ## restoring Python default signal handler
    signal.signal(signal.SIGINT, signal.default_int_handler)

    cdef VectorXd yhat_raw     = macau.getPredictions()
    cdef VectorXd yhat_sd_raw  = macau.getStds()
    cdef MatrixXd testdata_raw = macau.getTestData()

    cdef np.ndarray[np.double_t] yhat    = vecview( & yhat_raw ).copy()
    cdef np.ndarray[np.double_t] yhat_sd = vecview( & yhat_sd_raw ).copy()
    cdef np.ndarray[np.double_t, ndim=2] testdata = matview( & testdata_raw ).copy()

    df = pd.DataFrame({
      "row" : pd.Series(testdata[:,0], dtype='int'),
      "col" : pd.Series(testdata[:,1], dtype='int'),
      "y"   : pd.Series(testdata[:,2]),
      "y_pred" : pd.Series(yhat),
      "y_pred_std" : pd.Series(yhat_sd)
    })

    result = MacauResult()
    result.rmse_test  = macau.getRmseTest()
    result.Yshape     = Y.shape
    result.ntrain     = Y.nnz
    result.ntest      = Ytest.nnz if Ytest is not None else 0
    result.prediction = pd.DataFrame(df)

    del macau

    return result

