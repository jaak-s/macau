cimport cython                                                                                                         
import numpy as np
cimport numpy as np
import scipy as sp
import scipy.linalg

cimport macau

cpdef test(np.ndarray[np.double_t] x, int nrows, int ncols):
    return hello(&x[0], nrows, ncols)

cpdef xtest():
    return hellotest()

cpdef py_getx():
    cdef MatrixXd m = getx()
    cdef np.ndarray[np.double_t, ndim=2] A = matview(&m)
    print(A)

#cpdef mysolve(np.ndarray[np.double_t, ndim=2] A, np.ndarray[np.double_t] b):
#    if A.shape[0] != A.shape[1]:
#        raise ValueError("A is not square")
#    if A.shape[0] != b.shape[0]:
#        raise ValueError("b length is not the same as A nrows.")
#    cdef np.ndarray[np.double_t] x = np.zeros(b.shape[0])
#    solve(&A[0,0], &b[0], &x[0], A.shape[0])
#    return x

cdef matview(MatrixXd *A):
    cdef int nrow = A.rows()
    cdef int ncol = A.cols()
    cdef np.double_t[:,:] view = <np.double_t[:nrow:1, :ncol]> A.data()
    return np.asarray(view)

cdef vecview(VectorXd *v):
    cdef int size = v.size()
    cdef np.double_t[:] view = <np.double_t[:size]> v.data()
    return np.asarray(view)

## API functions:
## 1) F'F
## 2) F*X (X is a matrix)
## 3) F'X (X is a matrix)
## 4) solve(A, b, sym_pos=True) where A is posdef

## solves A * X = B for X, where A is sym_pos
cdef api void py_solve_sympos(MatrixXd *cA, MatrixXd *cB, MatrixXd *cX):
    cdef np.ndarray[np.double_t, ndim=2] A = matview(cA)
    cdef np.ndarray[np.double_t, ndim=2] B = matview(cB)
    cdef np.ndarray[np.double_t, ndim=2] X = matview(cX)
    X[:] = scipy.linalg.solve(A, B, sym_pos=True, check_finite=False)


cdef api double test1(VectorXd *x):
    cdef np.ndarray[np.double_t] myx = vecview(x)
    return myx[0]

def bpmf(Y,
         Ytest = None,
         num_latent = 10,
         precision  = 1.0, 
         burnin     = 50,
         nsamples   = 400):
    if type(Y) != sp.sparse.coo.coo_matrix:
        raise ValueError("Y must be scipy.sparse.coo.coo_matrix")
    if Ytest != None:
        if type(Ytest) != sp.sparse.coo.coo_matrix:
            raise ValueError("Ytest must be scipy.sparse.coo.coo_matrix")
        if Ytest.shape != Y.shape:
            raise ValueError("Ytest and Y must have the same shape")

    cdef np.ndarray[int] irows = Y.row.astype(np.int32, copy=False)
    cdef np.ndarray[int] icols = Y.col.astype(np.int32, copy=False)
    cdef np.ndarray[np.double_t] ivals = Y.data.astype(np.double, copy=False)

    cdef Macau macau
    macau = Macau(np.int32(num_latent))
    macau.setPrecision(np.float64(precision))
    macau.setRelationData(&irows[0], &icols[0], &ivals[0], irows.shape[0], Y.shape[0], Y.shape[1]);
    macau.setSamples(np.int32(burnin), np.int32(nsamples))

    cdef np.ndarray[int] trows, tcols
    cdef np.ndarray[np.double_t] tvals

    if Ytest != None:
        trows = Ytest.row.astype(np.int32, copy=False)
        tcols = Ytest.col.astype(np.int32, copy=False)
        tvals = Ytest.data.astype(np.double, copy=False)
        macau.setRelationDataTest(&trows[0], &tcols[0], &tvals[0], trows.shape[0], Y.shape[0], Y.shape[1])

    macau.run()
    #print("rows=%d, cols=%d" % ( macau.prior_u.Lambda.rows(), macau.prior_u.Lambda.cols()))
    #cdef np.ndarray[np.double_t, ndim=2] L = matview(&macau.prior_u.Lambda)
    #cdef np.ndarray[np.double_t] mu = vecview(&macau.prior_u.mu)
    return dict(rmse_test = macau.getRmseTest())

