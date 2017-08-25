cimport cython
import numpy as np
cimport numpy as np
import scipy as sp
import scipy.sparse
import timeit
import numbers
import pandas as pd
import signal
import sys

class MacauResult(object):
  def __init__(self):
    pass
  def __repr__(self):
    s = ("Matrix factorization results\n" +
         "Test RMSE:        %.4f\n" % self.rmse_test +
         "Matrix size:      [%s]\n" % " x ".join([np.str(x) for x in self.Yshape]) +
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
    """Splits a sparse matrix Y into a train and a test matrix.
       Y      scipy sparse matrix (coo_matrix, csr_matrix or csc_matrix)
       ntest  either a float below 1.0 or integer.
              if float, then indicates the ratio of test cells
              if integer, then indicates the number of test cells
       returns Ytrain, Ytest (type coo_matrix)
    """
    if type(Y) not in [scipy.sparse.coo.coo_matrix, scipy.sparse.csr.csr_matrix, scipy.sparse.csc.csc_matrix]:
        raise TypeError("Unsupported Y type: %s" + type(Y))
    if not isinstance(ntest, numbers.Real) or ntest < 0:
        raise TypeError("ntest has to be a non-negative number (number or ratio of test samples).")
    Y = Y.tocoo(copy = False)
    if ntest < 1:
        ntest = Y.nnz * ntest
    ntest = int(round(ntest))
    rperm = np.random.permutation(Y.nnz)
    train = rperm[ntest:]
    test  = rperm[0:ntest]
    Ytrain = scipy.sparse.coo_matrix( (Y.data[train], (Y.row[train], Y.col[train])), shape=Y.shape )
    Ytest  = scipy.sparse.coo_matrix( (Y.data[test],  (Y.row[test],  Y.col[test])),  shape=Y.shape )
    return Ytrain, Ytest

def make_train_test_df(Y, ntest):
    """Splits rows of dataframe Y into a train and a test dataframe.
       Y      pandas dataframe
       ntest  either a float below 1.0 or integer.
              if float, then indicates the ratio of test cells
              if integer, then indicates the number of test cells
       returns Ytrain, Ytest (type coo_matrix)
    """
    if type(Y) != pd.core.frame.DataFrame:
        raise TypeError("Y should be DataFrame.")
    if not isinstance(ntest, numbers.Real) or ntest < 0:
        raise TypeError("ntest has to be a non-negative number (number or ratio of test samples).")

    ## randomly spliting train-test
    if ntest < 1:
        ntest = Y.shape[0] * ntest
    ntest  = int(round(ntest))
    rperm  = np.random.permutation(Y.shape[0])
    train  = rperm[ntest:]
    test   = rperm[0:ntest]
    return Y.iloc[train], Y.iloc[test]

cdef ILatentPrior* make_prior(side, int num_latent, int max_ff_size, double lambda_beta, double tol) except NULL:
    if side is None:
        return new BPMFPrior(num_latent)
    if type(side) not in [scipy.sparse.coo.coo_matrix, scipy.sparse.csr.csr_matrix, scipy.sparse.csc.csc_matrix, np.ndarray]:
        raise TypeError("Unsupported side information type: '%s'" % type(side).__name__)

    cdef bool compute_ff = (side.shape[1] <= max_ff_size)
    
    ## dense side information
    cdef MacauPrior[MatrixXd]* dense_prior
    cdef np.ndarray[np.double_t, ndim=2] X
    cdef bool colMajor
    if type(side) == np.ndarray:
        if len(side.shape) != 2:
            raise TypeError("Side information must have 2 dimensions (got %d)." % len(side.shape))
        X = side.astype(np.float64, copy=False)
        colMajor = np.isfortran(side)
        dense_prior = make_dense_prior(num_latent, &X[0, 0], side.shape[0], side.shape[1], colMajor, compute_ff)
        dense_prior.setLambdaBeta(lambda_beta)
        dense_prior.setTol(tol)
        return dense_prior

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
    if type(side) not in [scipy.sparse.coo.coo_matrix, scipy.sparse.csr.csr_matrix, scipy.sparse.csc.csc_matrix]:
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
    return scipy.sparse.coo_matrix( (Y.data[idx], (Y.row[idx], Y.col[idx])), shape = Y.shape )

class Data:
    def __init__(self, Y, Ytest):
        matrix_types = [scipy.sparse.coo.coo_matrix, scipy.sparse.csr.csr_matrix, scipy.sparse.csc.csc_matrix]
        if type(Y) in matrix_types:
            if Ytest is None:
                Ytest = scipy.sparse.coo_matrix(Y.shape, np.float64)
            if isinstance(Ytest, numbers.Real):
                Y, Ytest = make_train_test(Y, Ytest)
            if type(Ytest) not in matrix_types:
                raise ValueError("When Y is a sparse matrix Ytest must be too.")
            if Y.shape != Ytest.shape:
                raise ValueError("Y (%d x %d) and Ytest (%d x %d) must have the same shape." %
                                 (Y.shape[0], Y.shape[1], Ytest.shape[0], Ytest.shape[1]))
            Y = Y.tocoo(copy = False)
            Y = remove_nan(Y)
            Ytest = Ytest.tocoo(copy = False)
            Ytest = remove_nan(Ytest)
            self.shape = np.array(Y.shape, dtype=np.int32)
            self.idxTrain = [Y.row, Y.col]
            self.valTrain = Y.data
            self.idxTest  = [Ytest.row, Ytest.col]
            self.valTest  = Ytest.data
            self.colnames   = np.array(["row", "col"], dtype=np.object)

        elif type(Y) == pd.core.frame.DataFrame:
            if Ytest is None:
                Ytrain = Y
                Ytest  = Y[0:0]
            if isinstance(Ytest, numbers.Real):
                Ytrain, Ytest = make_train_test_df(Y, Ytest)
            else:
                Ytrain = Y

            if type(Ytest) != pd.core.frame.DataFrame:
                raise TypeError("When Y is a DataFrame Ytest must be too.")
            if (Y.columns != Ytest.columns).any():
                raise ValueError("Columns of Y and Ytest must be the same.")
            if (Y.dtypes != Ytest.dtypes).any():
                raise TypeError("Y.dtypes and Ytest.dtypes must be the same.")
            int_cols   = list(filter(lambda c: Ytrain[c].dtype==np.int64 or Ytrain[c].dtype==np.int32, Ytrain.columns))
            float_cols = list(filter(lambda c: Ytrain[c].dtype==np.float32 or Ytrain[c].dtype==np.float64, Ytrain.columns))
            if len(int_cols) > 10:
                raise ValueError("Y has too many index(int) columns (%d), maximum is 10." % len(int_cols))
            if len(int_cols) < 2:
                raise ValueError("Y must have at least 2 index (int) columns.")
            if len(float_cols) != 1:
                raise ValueError("Y has %d float columns but must have exactly 1 value column." % len(float_cols))
            if Ytrain.shape[0] == 0:
                raise ValueError("Ytrain must not be empty.")
            value_col = float_cols[0]
            self.colnames = int_cols

            if Ytest.shape[0] > 0:
                self.shape = np.array(np.maximum([Y[c].max() for c in int_cols],
                                                 [Ytest[c].max() for c in int_cols]) + 1, dtype=np.int32)
            else:
                self.shape = np.array([Y[c].max() for c in int_cols], dtype=np.int32) + 1

            self.idxTrain = [np.array(Y[c],     dtype=np.int32) for c in int_cols]
            self.idxTest  = [np.array(Ytest[c], dtype=np.int32) for c in int_cols]
            self.valTrain = np.array(Y[value_col],     dtype=np.float64)
            self.valTest  = np.array(Ytest[value_col], dtype=np.float64)

        else:
            raise TypeError("Unsupported Y type: %s" + type(Y))

cdef np.ndarray idx_matrix(idxList):
    cdef np.ndarray idx = np.zeros([len(idxList[0]), len(idxList)], dtype=np.int32, order='F')
    for i in range(len(idxList)):
        idx[:, i] = idxList[i]
    return idx

cdef setData(Macau* macau, data):
    ## training data
    cdef np.ndarray[int] dims = np.array(data.shape, dtype=np.int32)
    cdef np.ndarray[int, ndim=2] idx  = idx_matrix(data.idxTrain)
    cdef np.ndarray[np.double_t] ivals = data.valTrain.astype(np.double, copy=False)

    macau.setRelationData(&idx[0,0], len(data.shape), &ivals[0], idx.shape[0], &dims[0])

    ## testing data
    cdef np.ndarray[int, ndim=2] te_idx   = idx_matrix(data.idxTest)
    cdef np.ndarray[np.double_t] te_ivals = data.valTest.astype(np.double, copy=False)
    if te_idx.shape[0] > 0:
        macau.setRelationDataTest(&te_idx[0,0], len(data.shape), &te_ivals[0], te_idx.shape[0], &dims[0])

cdef setSidePriors(Macau* macau, side, int D, double lambda_beta, double tol, bool univariate):
    cdef unique_ptr[ILatentPrior] prior
    for s in side:
        if univariate:
            prior = unique_ptr[ILatentPrior](make_one_prior(s, D, lambda_beta))
        else:
            prior = unique_ptr[ILatentPrior](make_prior(s, D, 10000, lambda_beta, tol))
        macau.addPrior(prior)

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
    """
    Matrix and tensor factorization with side information.
      Y          training data to factorize (sparse matrix or DataFrame)
      Ytest      either:
                 number below 1.0, how much training data to move to test
                 sparse matrix or DataFrame of test data
      side       list of side information matrices.
                 If Y is matrix, then need 2 side matrices (first for rows, second for columns)
      lambda_beta  initial precision (regularization) for the link matrix
                   connecting side matrices to Y
      num_latent   number of latent dimensions
      precision    precision of observations of Y, can be
                   scalar     - specifies the precision (1 / noise_variance)
                   "adaptive" - automatically learn precision
                   "probit"   - probit model for binary matrices
      burnin       number of Gibbs samples to drop (burn-in)
      nsamples     number of Gibbs samples to collect
      univariate   whether to use univariate sampler (faster than standard sampler)
      tol          error tolerance for conjugate gradient solver
      sn_max       maximum signal-to-noise ratio for observation precision
                   only used if precision="adaptive"
      save_prefix  prefix for model files or None if not saving the model
      verbose      whether to print output for each Gibbs iteration
    """
    data = Data(Y, Ytest)

    ## side information
    if not side:
        side = [None for i in range(len(data.shape))]
    if type(side) not in [list, tuple]:
        raise ValueError("Parameter 'side' must be a tuple or a list.")
    if len(side) != len(data.shape):
        raise ValueError("Length of 'side' is %d but must be equal to the number of data dimensions (%d)." % 
                (len(side), len(data.shape)) )

    cdef int D = np.int32(num_latent)
    cdef Macau *macau

    ## choosing the noise model
    cdef int Nmodes = len(data.shape)
    if isinstance(precision, str):
      if precision == "adaptive" or precision == "sample":
          macau = make_macau_adaptive(Nmodes, D, np.float64(1.0), np.float64(sn_max))
      elif precision == "probit":
          if univariate == True:
              raise ValueError("Univariate sampler for probit model is not yet implemented.")
          macau = make_macau_probit(Nmodes, D)
      else:
          raise ValueError("Parameter 'precision' has to be either a number or \"adaptive\" for adaptive precision, or \"probit\" for binary matrices.")
    else:
      macau = make_macau_fixed(Nmodes, D, np.float64(precision))

    setData(macau, data)
    setSidePriors(macau, side, D, np.float64(lambda_beta), np.float64(tol), np.bool(univariate))

    macau.setSamples(np.int32(burnin), np.int32(nsamples))
    macau.setVerbose(verbose)

    if save_prefix is None:
        macau.setSaveModel(0)
    else:
        if type(save_prefix) != str:
            raise ValueError("Parameter 'save_prefix' has to be a string (str) or None.")
        macau.setSaveModel(1)
        if sys.version_info[0] >= 3:
            macau.setSavePrefix(save_prefix.encode())
        else:
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

    df = pd.DataFrame(testdata[:, :-1], columns=data.colnames, dtype='int')
    df["y"]      = pd.Series(testdata[:, -1])
    df["y_pred"] = pd.Series(yhat)
    df["y_pred_std"] = pd.Series(yhat_sd)

    result = MacauResult()
    result.rmse_test  = macau.getRmseTest()
    result.Yshape     = data.shape
    result.ntrain     = data.valTrain.shape[0]
    result.ntest      = data.valTest.shape[0] if Ytest is not None else 0
    result.prediction = pd.DataFrame(df)

    del macau

    return result

