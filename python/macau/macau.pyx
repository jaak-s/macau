cimport cython                                                                                                         
import numpy as np
cimport numpy as np

cimport macau

cpdef test():
    return hello()

def bpmf(np.ndarray rows,
         np.ndarray cols,
         np.ndarray values,
         num_latent = 10,
         precision  = 1.0, 
         burnin     = 50,
         nsamples   = 400):
    if rows.shape[0] != cols.shape[0]:
        raise ValueError("rows and cols must have the same length.")
    if rows.shape[0] != values.shape[0]:
        raise ValueError("rows and values must have the same length.")

    cdef np.ndarray[int] irows = rows.astype(np.int32, copy=False)
    cdef np.ndarray[int] icols = cols.astype(np.int32, copy=False)
    cdef np.ndarray[np.double_t] ivals = values.astype(np.double, copy=False)

    cdef Macau macau
    macau = Macau(np.int32(num_latent))
    macau.setPrecision(np.float64(precision))
    macau.setRelationData(&irows[0], &icols[0], &ivals[0], rows.shape[0], rows.max() + 1, cols.max() + 1);
    macau.setSamples(np.int32(burnin), np.int32(nsamples))
    macau.run()
    return dict(rmse_test = macau.getRmseTest())                                                                                                         

