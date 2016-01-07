cimport cython                                                                                                         
import numpy as np
cimport numpy as np
import scipy as sp

cimport macau

cpdef test():
    return hello()

def bpmf(Y,
         Ytest = None,
         num_latent = 10,
         precision  = 1.0, 
         burnin     = 50,
         nsamples   = 400):
    if type(Y) != sp.sparse.coo.coo_matrix:
        raise ValueError("Y must be scipy.sparse.coo.coo_matrix")

    cdef np.ndarray[int] irows = Y.row.astype(np.int32, copy=False)
    cdef np.ndarray[int] icols = Y.col.astype(np.int32, copy=False)
    cdef np.ndarray[np.double_t] ivals = Y.data.astype(np.double, copy=False)

    cdef Macau macau
    macau = Macau(np.int32(num_latent))
    macau.setPrecision(np.float64(precision))
    macau.setRelationData(&irows[0], &icols[0], &ivals[0], irows.shape[0], Y.shape[0], Y.shape[1]);
    macau.setSamples(np.int32(burnin), np.int32(nsamples))
    macau.run()
    return dict(rmse_test = macau.getRmseTest())                                                                                                         

